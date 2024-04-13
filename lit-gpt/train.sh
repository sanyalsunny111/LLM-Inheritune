export CUDA_VISIBLE_DEVICES=0

python finetune/full_redpj_epoch.py --precision "bf16-true" --train_data_dir /sample/data/lit-redpajama --checkpoint_dir /sample/checkpoints/openlm-research/open_llama_3b/

