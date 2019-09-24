import cPickle

import torch


def convert_data(batch, vocab, device, reverse=False, unk=None, pad=None, sos=None, eos=None):
    max_len = max(len(x) for x in batch)
    padded = []
    for x in batch:
        if reverse:
            padded.append(
                ([] if eos is None else [eos]) +
                list(x[::-1]) +
                ([] if sos is None else [sos]))
        else:
            padded.append(
                ([] if sos is None else [sos]) +
                list(x) +
                ([] if eos is None else [eos]))
        padded[-1] = padded[-1] + [pad] * max(0, max_len - len(x))
        padded[-1] = map(lambda v: vocab['stoi'][v] if v in vocab['stoi'] else vocab['stoi'][unk], padded[-1])
    padded = torch.LongTensor(padded).to(device)
    mask = padded.ne(vocab['stoi'][pad]).float()
    return padded, mask


def convert_str(batch, vocab):
    output = []
    for x in batch:
        output.append(map(lambda v: vocab['itos'][v], x))
    return output


def invert_vocab(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k
    return v


def load_vocab(path):
    f = open(path, 'rb')
    vocab = cPickle.load(f)
    f.close()
    return vocab


def sort_batch(batch):
    batch = zip(*batch)
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    batch = zip(*batch)
    return batch


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs
