export CUDA_VISIBLE_DEVICES=0

export HF_HOME=/location/huggingface

python eval/lm_eval_harness.py --checkpoint_dir "/sample/checkpoints/openlm-research/open_llama_3b/" --eval_tasks "[wsc,logiqa,sciq,arc_easy]" --num_fewshot 0


python eval/lm_eval_harness.py --checkpoint_dir "/sample/checkpoints/openlm-research/open_llama_3b/" --eval_tasks "[openbookqa,winogrande,hellaswag,boolq,arc_easy,piqa]" --num_fewshot 0

python eval/lm_eval_harness.py --checkpoint_dir "/sample/checkpoints/openlm-research/open_llama_3b/" --eval_tasks "[hendrycksTest-*]" --save_filepath "/out/results_mmlu.json"
