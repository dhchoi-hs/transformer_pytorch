# Lanaguage Model(LM) Using transformer encoder/decoder
- Implement pre-train models using encoder/decoder for NLP task.

## Introduction
- pre-training task for encoder/decoder stacks of transformer.
- each of pre-trained language models will be fine-tuned by merging into transformer.

## Preliminaries
TBD

## Methods
- tokenization
  * a vocabulary dictionary with korean-english
  1. BPE of char-level
  2. BPE of byte-level
- models
  1. LM using encoder stacks only
     - Use shared embedding to output instead of linear regression
  2. LM using decoder stacks only

## Experiments
- training
  - encoder stacks 

- results

## Lessons Learned
1. loss not decreasing issue in implement
   - According doc of [torch.nn.CrossEntropyLoss()](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), input logits should not be normalized to use. remove softmax function attached end of linear transformation. __Read the documentation carefully before using the any api.__
2. loss value got NaN after a certain epoch on training.
   - high input values makes infinity value in softmax. To prevent getting infinity, subtract max value of input to all input values.
2. loss not decreasing issue in tuning hyper-parameter
   - high weight decay(in this task, 0.01 causes issue.) of optimizer cause loss not decreasing and acc not increasing. lower and find proper the weight decay. 0.001 or 0.0001 ...
 
## Conclusion and Constraints
TBD

## References
* Entire architecture of transformer: [Attention is all you need](https://arxiv.org/abs/1706.03762)
* BPE: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909v5)
* Masked Language Model: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* Masked Language Model-2: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* training dataset: [AI-hub; 한국어-영어 번역(병렬) 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=126)