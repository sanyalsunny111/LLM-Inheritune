export CUDA_VISIBLE_DEVICES=0,1,2


torchrun --standalone --nproc_per_node=3 train_inheritune.py