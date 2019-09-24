Deep Neural Machine Translation with Linear Associate Unit
=====================================================================

### Installation
The following packages are needed:
* [Pytorch-0.4.0](https://github.com/pytorch/pytorch)
* NLTK

### Preparation
To obtain vocabulary for training, run:
```
python scripts/buildvocab.py --corpus /path/to/train.cn --output /path/to/cn.voc3.pkl \
--limit 30000 --groundhog
python scripts/buildvocab.py --corpus /path/to/train.en --output /path/to/en.voc3.pkl \
--limit 30000 --groundhog
```

### Training
Training the DeepLAU model on Chinese-English translation datasets as follows:
```
python train.py \
--src_vocab /path/to/cn.voc3.pkl --trg_vocab /path/to/en.voc3.pkl \
--train_src corpus/train.cn-en.cn --train_trg corpus/train.cn-en.en \
--src_max_len 50 --trg_max_len 50 \
--valid_src corpus/nist02/nist02.cn \
--valid_trg corpus/nist02/nist02.en0 corpus/nist02/nist02.en1 corpus/nist02/nist02.en2 corpus/nist02/nist02.en3 \
--eval_script scripts/validate.sh \
--model LAUModel \
--optim Adam \
--batch_size 128 \
--half_epoch \
--cuda \
--info Adam-half_epoch 
```
### Evaluation
```
python translate.py \
--src_vocab /path/to/cn.voc3.pkl --trg_vocab /path/to/en.voc3.pkl \
--test_src corpus/nist03/nist03.cn \
--test_trg corpus/nist03/nist02.en0 corpus/nist03/nist03.en1 corpus/nist03/nist03.en2 corpus/nist03/nist03.en3 \
--eval_script scripts/validate.sh \
--model LAUModel \
--name LAUModel.best.pt \
--cuda 
```
The evaluation metric for Chinese-English we use is case-insensitive BLEU. We use the `muti-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder) to compute the BLEU:
```
perl scripts/multi-bleu.perl -lc corpus/nist03/nist03.en < nist03.translated
```
### Results on Chinese-English translation
The trainining dataset consists of 1.25M billingual sentence pairs extracted from LDC corpora. Use NIST 2002(MT02) as tuning set for hyper-parameter optimization and model selection, and NIST 2003(MT03), 2004 (MT04), 2005(MT05), 2006(MT06) and 2008(MT08) as test sets. The beam size is set to 10.

|MT02|MT03|MT04|MT05|MT06|MT08|Ave.|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|40.65|38.32|40.48|37.97|36.98|28.11|36.37|
