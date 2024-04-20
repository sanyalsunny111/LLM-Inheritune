# GPT2 experiments with Inheritune 

## Introduction
We analyze Inheritune in a slightly different setting where we assume full access to the pre-training data. 
We observe that a much smaller target model can be extracted if we use the full pre-training data. We ran controlled experiments with GPT2-large and GPT2-medium LLMs. Utilizing Inheritune we show that for GPT2-large we can keep 50% of the layers and 45% of parameters while for a GPT2-medium, we keep 33% layers and 28% parameters without compromising the validation loss (log perplexity). Intriguingly, we also observe that these smaller models derived with \method{} exhibit lower validation loss to their same-sized counterparts are trained from scratch for 2x the number of training steps. Moreover, these smaller models derived with Inheritune exhibit a similar convergence pattern to their larger counterparts.
## Data Preparation
Prepare the [OpenWebText](https://huggingface.co/datasets/openwebtext) data following [nanoGPT](https://github.com/karpathy/nanoGPT/):
```
$ python data/openwebtext/prepare.py
```

## Training with Inheritune Script for Our GPT-2-large 18 layer variant

##### 1 round of training
To train a GPT2-large (our variant 18 layers) model with Inheritune use the following command:
```bash
torchrun --standalone --nproc_per_node=3 train_inheritune.py
```


## Dependencies
- [pytorch](https://pytorch.org) 2.0
- transformers
- datasets
- tiktoken
- wandb

## Cite
If you find this work helpful, please consider citing us:

```
@inproceedings{Sanyal2024pretraining,
  title  = {Pre-training Small Base LMs with Fewer Tokens},
  author = {Sunny Sanyal and sujay sanghavi and Alex Dimakis},
  year   = {2024},
  url    = {https://openreview.net/forum?id=SmNlrStwHW}
}
```

## Acknowledgement
The training code is mainly adapted from [Sophia](https://github.com/Liuhong99/Sophia/) and [nanoGPT](https://github.com/karpathy/nanoGPT/).