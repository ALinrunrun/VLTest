import os

os.system("CUDA_VISIBLE_DEVICES=1 python perturber.py \
    --output_path=outputs\
    --task=vqa \
    --perturb_mode=joint \
    --max_samples=100 \
    --pert_budget=10 \
    --pict_top_p_ratio=0.01\
    --pict_pert_time=1 \
    --pict_pert_mode=uniform \
    --text_variant_num=1 \
    --text_max_word=1 \
    --text_max_word_attempts=1 \
    --seed 123456")