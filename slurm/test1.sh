#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=100g
#SBATCH --gres=gpu:2
#SBATCH --job-name=resformer1
#SBATCH --output=resformer1.out

source /scratch/itee/uqwma4/miniconda3/etc/profile.d/conda.sh
conda activate resformer
export PYTHONPATH="/scratch/itee/uqwma4/resformer"
echo $PYTHONPATH

python -m torch.distributed.launch \
    --nproc_per_node 2 \
    ../image_classification/main.py \
    --data-path "/scratch/itee/uqwma4/data/imagenet" \
    --model resformer_small_patch16 \
    --output_dir "output" \
    --batch-size 128 \
    --pin-mem \
    --input-size 224 160 128 \
    --auto-resume \
    --distillation-type 'smooth-l1' \
    --distillation-target cls \
    --sep-aug

# --data-path  "D:\Projects\SFCViT\data\imagenet"  --model resformer_small_patch16  --output_dir "output" --batch-size 128 --pin-mem --input-size 224 160 128 --auto-resume  --distillation-type 'smooth-l1' --distillation-target cls --sep-aug

#python -m torch.distributed.launch \
#    --nproc_per_node 8 \
#    ../image_classification/main.py \
#    --data-path YOUR_DATA_PATH \
#    --model resformer_small_patch16 \
#    --output_dir YOUR_OUTPUT_PATH \
#    --batch-size 128 \
#    --pin-mem \
#    --input-size 224 160 128 \
#    --auto-resume \
#    --distillation-type 'smooth-l1' \
#    --distillation-target cls \
#    --sep-aug
