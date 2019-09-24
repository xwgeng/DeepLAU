import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, normal

from beam import Beam
from util import reverse_padded_sequence


class LAUCell(nn.Module):
    def __init__(self, ninp, nhid):
       super(LAUCell, self).__init__() 
       self.i2gate = nn.Linear(ninp, 3 * nhid)
       self.h2gate = nn.Linear(nhid, 3 * nhid)
       self.i2i = nn.Linear(ninp, nhid)
       self.i2h = nn.Linear(ninp, nhid)
       self.h2h = nn.Linear(nhid, nhid)
       self.igLN = nn.LayerNorm(3 * nhid)
       self.hgLN = nn.LayerNorm(3 * nhid)
       self.iiLN = nn.LayerNorm(nhid)
       self.ihLN = nn.LayerNorm(nhid)
       self.hhLN = nn.LayerNorm(nhid)

    def forward(self, inp, hidden):
        gate = F.sigmoid(self.igLN(self.i2gate(inp)) + self.hgLN(self.h2gate(hidden)))
        r, z, g = torch.chunk(gate, 3, dim=-1)
        Hx = self.iiLN(self.i2i(inp))
        xh = self.ihLN(self.i2h(inp))
        hh = self.hhLN(self.h2h(hidden))

        hidden_m = F.tanh((1.0 - r) * xh + r * hh)
        hidden = ((1.0 - z) * hidden + z * hidden_m) * (1.0 - g) + g * Hx
        return hidden 


class LAUEncoder(nn.Module):
    def __init__(self, nlayer, ninp, nhid, ntok, padding_idx, emb_dropout, hid_dropout):
        super(LAUEncoder, self).__init__()
        self.nlayer = nlayer
        self.nhid = nhid
        self.emb = nn.Embedding(ntok, ninp, padding_idx=padding_idx)
        self.lau_cells = nn.ModuleList([LAUCell(ninp, nhid) for _ in xrange(nlayer)])
        self.enc_emb_dp = nn.Dropout(emb_dropout)
        self.enc_hid_dp = nn.Dropout(hid_dropout)

    def forward(self, inp, mask):
        inp = self.enc_emb_dp(self.emb(inp))
        inp = [v.squeeze(1) for v in torch.split(inp, 1, dim=1)]
        
        max_len = len(inp)
        for i in xrange(self.nlayer):
            output = []    
            hidden = inp[0].new_zeros(inp[0].size(0), self.nhid)
            for k in xrange(max_len):
                idx = k if i % 2 == 0 else (max_len - k - 1)
                hidden = self.lau_cells[i](inp[k], hidden) * mask[:, idx].unsqueeze(-1)
                output.append(hidden)
            inp = output[::-1]
        if self.nlayer % 2 == 0:
            output = output[::-1]
        output = torch.stack(output, dim=1)
        output = self.enc_hid_dp(output)
        return output
        

class Attention(nn.Module):
    def __init__(self, nhid, ncontext, natt):
        super(Attention, self).__init__()
        self.h2a = nn.Linear(nhid, natt)
        self.s2a = nn.Linear(ncontext, natt)
        self.a2o = nn.Linear(natt, 1)
        self.hLN = nn.LayerNorm(natt)
        self.sLN = nn.LayerNorm(natt)
            
    def forward(self, hidden, mask, context):
        shape = context.size()
        attn_h = self.sLN(self.s2a(context.view(-1, shape[2])))
        attn_h = attn_h.view(shape[0], shape[1], -1)
        attn_h += self.hLN(self.h2a(hidden)).unsqueeze(1).expand_as(attn_h)
        logit = self.a2o(F.tanh(attn_h)).view(shape[0], shape[1])
        if mask.any():
            logit.data.masked_fill_(1 - mask, -float('inf'))
        softmax = F.softmax(logit, dim=1)
        output = torch.bmm(softmax.unsqueeze(1), context).squeeze(1)
        return output


class LAUDecoder(nn.Module):
    def __init__(self, nlayer, ninp, nhid, ntok, enc_ncontext, natt, nreadout, readout_dropout):
        super(LAUDecoder, self).__init__()
        self.nlayer = nlayer
        self.lau_cells = nn.ModuleList([LAUCell(ninp, nhid) for _ in xrange(nlayer + 1)])
        self.enc_attn = Attention(nhid, enc_ncontext, natt)
        self.e2o = nn.Linear(ninp, nreadout)
        self.h2o = nn.Linear(nhid, nreadout)
        self.c2o = nn.Linear(enc_ncontext, nreadout)
        self.eLN = nn.LayerNorm(nreadout)
        self.hLN = nn.LayerNorm(nreadout)
        self.cLN = nn.LayerNorm(nreadout)
        self.readout_dp = nn.Dropout(readout_dropout)

    def forward(self, emb, hidden_m, enc_mask, enc_context):
        hidden_m[0] = self.lau_cells[0](emb, hidden_m[0])
        attn_enc = self.enc_attn(hidden_m[0], enc_mask, enc_context)
        hidden = [] 
        inp = attn_enc
        for i in xrange(1, self.nlayer + 1):
            inp = self.lau_cells[i](inp, hidden_m[i - 1])
            hidden.append(inp)
        proj = F.tanh(self.eLN(self.e2o(emb)) + self.hLN(self.h2o(hidden[-1])) + self.cLN(self.c2o(attn_enc)))
        output = self.readout_dp(proj)
        return output, hidden


class LAUModel(nn.Module):
    def __init__(self, opt):
        super(LAUModel, self).__init__()
        self.dec_nhid = opt.dec_nhid
        self.dec_sos = opt.dec_sos
        self.dec_eos = opt.dec_eos
        self.dec_pad = opt.dec_pad
        self.enc_pad = opt.enc_pad
        self.dec_nlayer = opt.dec_nlayer
        self.tied_emb = opt.tied_emb

        self.encoder = LAUEncoder(opt.enc_nlayer, opt.enc_ninp, opt.enc_nhid, opt.enc_ntok, opt.enc_pad, opt.enc_emb_dropout, opt.enc_hid_dropout)
        self.decoder = LAUDecoder(opt.dec_nlayer, opt.dec_ninp, opt.dec_nhid, opt.dec_ntok, opt.enc_nhid, opt.dec_natt, opt.nreadout, opt.readout_dropout)
        self.emb = nn.Embedding(opt.dec_ntok, opt.dec_ninp, padding_idx=opt.dec_pad)
        self.dec_emb_dp = nn.Dropout(opt.dec_emb_dropout)
        self.e2d = nn.Linear(opt.enc_nhid, opt.dec_nhid)
        self.eLN = nn.LayerNorm(opt.dec_nhid)
        if not self.tied_emb:
            self.proj = nn.Linear(opt.dec_nhid, opt.dec_ntok)

    def forward(self, src, src_mask, f_trg, f_trg_mask, b_trg=None, b_trg_mask=None):
        enc_context = self.encoder(src, src_mask)
        enc_context = enc_context.contiguous()
        
        avg_enc_context = enc_context.sum(1)
        enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)
        avg_enc_context = avg_enc_context / enc_context_len

        attn_mask = src_mask.byte()
        init_hidden = F.tanh(self.eLN(self.e2d(avg_enc_context)))
        hidden = [init_hidden.clone() for _ in xrange(self.dec_nlayer)]

        inp = self.dec_emb_dp(self.emb(f_trg))

        loss = 0
        for i in xrange(f_trg.size(1) - 1):
            output, hidden = self.decoder(inp[:, i, :], hidden, attn_mask, enc_context)
            logit = torch.matmul(output, self.emb.weight.t()) if self.tied_emb else self.proj(output)
            loss += F.cross_entropy(logit, f_trg[:, i+1], reduce=False) * f_trg_mask[:, i+1]
        w_loss = loss.sum() / f_trg_mask[:, 1:].sum()
        loss = loss.mean()
        return loss.unsqueeze(0), w_loss.unsqueeze(0)

    def beamsearch(self, src, src_mask, beam_size=8, normalize=False, max_len=None, min_len=None):
        max_len = src.size(1) * 3 if max_len is None else max_len
        min_len = src.size(1) / 2 if min_len is None else min_len

        enc_context = self.encoder(src, src_mask)
        enc_context = enc_context.contiguous()
        
        avg_enc_context = enc_context.sum(1)
        enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)
        avg_enc_context = avg_enc_context / enc_context_len

        attn_mask = src_mask.byte()
        init_hidden = F.tanh(self.eLN(self.e2d(avg_enc_context)))
        hidden = [init_hidden.clone() for _ in xrange(self.dec_nlayer)]

        prev_beam = Beam(beam_size)
        prev_beam.candidates = [[self.dec_sos]]
        prev_beam.scores = [0]
        f_done = (lambda x: x[-1] == self.dec_eos)

        valid_size = beam_size

        hyp_list = []
        for k in xrange(max_len):
            candidates = prev_beam.candidates
            input = src.new_tensor(map(lambda cand: cand[-1], candidates))
            input = self.dec_emb_dp(self.emb(input))
            output, hidden = self.decoder(input, hidden, attn_mask, enc_context)
            logit = torch.matmul(output, self.emb.weight.t()) if self.tied_emb else self.proj(output)
            log_prob = F.log_softmax(logit, dim=1)
            if k < min_len:
                log_prob[:, self.dec_eos] = -float('inf')
            if k == max_len - 1:
                eos_prob = log_prob[:, self.dec_eos].clone()
                log_prob[:, :] = -float('inf')
                log_prob[:, self.dec_eos] = eos_prob
            next_beam = Beam(valid_size)
            done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
            hyp_list.extend(done_list)
            valid_size -= len(done_list)

            if valid_size == 0:
                break

            beam_remain_ix = src.new_tensor(remain_list)
            enc_context = enc_context.index_select(0, beam_remain_ix)
            attn_mask = attn_mask.index_select(0, beam_remain_ix)
            hidden = [h.index_select(0, beam_remain_ix) for h in hidden]
            prev_beam = next_beam
        score_list = [hyp[1] for hyp in hyp_list]
        hyp_list = [hyp[0][1: hyp[0].index(self.dec_eos)] if self.dec_eos in hyp[0] else hyp[0][1:] for hyp in hyp_list]                                
        if normalize:
            for k, (hyp, score) in enumerate(zip(hyp_list, score_list)):
                if len(hyp) > 0:
                    score_list[k] = score_list[k] / len(hyp)
        score = hidden[0].new_tensor(score_list)
        sort_score, sort_ix = torch.sort(score)
        output = []
        for ix in sort_ix.tolist():
            output.append((hyp_list[ix], score[ix].item()))
        return output
