

## Setup

Clone the repo:

```bash
git clone https://github.com/sanyalsunny111/LLM-Inheritune.git
cd LLM-Inheritune
```

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Install with all dependencies (including quantization, sentencepiece, tokenizers for Llama models, etc.):

```bash
pip install -r requirements-all.txt
```


```bash
pip uninstall -y torch torchvision torchaudio torchtext
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

You are all set! 🎉

&nbsp;

## Prepare reference model

```bash
export HF_HOME=sample/huggingface
```

Download the model and convert it to lit format from huggingface format:

```bash
python scripts/download.py --repo_id openlm-research/open_llama_3b --access_token writeyourtokenhere
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b
```

&nbsp;

## Prepare 1B tokens for training

Download the data using git lfs:

```bash
# Make sure you have git-lfs installed (https://git-lfs.com):
sudo apt install git-lfs
```

```bash
# download the 1 billion token subset and tokenize
pip install huggingface_hub sentencepiece

python scripts/prepare_redpajama.py --source_path data/RedPajama-Data-1T-Sample \
  --checkpoint_dir checkpoints/openlm-research/open_llama_3b/ --destination_path data/lit-redpajama-sample --sample True
```
The tokenization process will take a while.

&nbsp;

## Train a small base LM

```bash
# download the 1 billion token subset and tokenize
bash scripts/train_train.sh
```

&nbsp;

## Evaluation

Install the evaluation harness

```bash
pip install https://github.com/EleutherAI/lm-evaluation-harness/archive/refs/heads/master.zip -U
```

Run MMLU evaluation:

```bash
python eval/lm_eval_harness.py --checkpoint_dir "checkpoints/openlm-research/our-1_5b/" \
--eval_tasks "[hendrycksTest-*]" --num_fewshot 5 --save_filepath "out/results_mmlu_.json"
```


&nbsp;

## License

Lit-GPT is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.
