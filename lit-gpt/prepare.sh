# NOTE: specify your own hugging face token to download the models
export HF_HOME=sample/huggingface

#download llama2 model
python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf --access_token writeyourtokenhere
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf

# download openllama model
python scripts/download.py --repo_id openlm-research/open_llama_3b --checkpoint_dir /sample/checkpoints/openlm-research/open_llama_3b
python scripts/convert_hf_checkpoint.py --checkpoint_dir /sample/checkpoints/openlm-research/open_llama_3b

# download dataset and tokenize
python scripts/prepare_redpajama.py --source_path data/RedPajama-Data-1T-Sample --checkpoint_dir checkpoints/openlm-research/open_llama_3b/ --destination_path data/lit-redpajama-sample --sample True

#python scripts/prepare_redpajama.py --source_path /sample/data/RedPajama-Data-1T-Sample --checkpoint_dir /sample/checkpoints/meta-llama/Llama-2-7b-hf/ --destination_path data/lit-redpajama-sample --sample True
