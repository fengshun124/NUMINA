#!/usr/bin/env bash

which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export CUDA_VISIBLE_DEVICES=3

# --------------------------------- #
# Parameters to be changed manually #
# --------------------------------- #
export NCCL_P2P_DISABLE=1 # ban NCCL P2P function
export OMP_STACKSIZE=256M # set OpenMP (Open Multi-Processing) thread stack size environment variable
ulimit -n 4096            # set the maximum number of open files to 4096
cot=True                 # whether or not to perform cot ablation study
ls=False                  # Whether or not to perform ls ablation study
num_workers=0             # used to specify the number of worker threads or processes to be created
batch_size=2
lr=2e-6
# evaluate=False
# pretrained_path=""

# --------------------------- #
# choose the model to be used #
# --------------------------- #
llama_model_path=""


# --------------------------------- #
# turn on when evaluating the model #
# --------------------------------- #
# evaluate=True
# pretrained_path=""

# ------------------------ #
# normal training datasets #
# ------------------------ #
# train_tag="NONUM-scanqa-PM-train_set#NONUM-scanqa-FV-train_set#NUM-quantity-NI-train_set#NUM-distance-NI-train_set#NUM-volume-NI-train_set#NUM-quantity-FV-train_set#NUM-distance-FV-train_set#NUM-volume-FV-train_set"
# val_tag="NONUM-scanqa-PM-val_set#NONUM-scanqa-FV-val_set#NUM-quantity-NI-val_set#NUM-distance-NI-val_set#NUM-volume-NI-val_set#NUM-quantity-FV-val_set#NUM-distance-FV-val_set#NUM-volume-FV-val_set"

# --------------------------- #
# ls ablation study datasets  #
# --------------------------- #
# train_tag="ls-NUM-quantity_compare-FV-train_set#ls-NUM-distance_compare-FV-train_set#ls-NUM-volume_compare-FV-train_set"
# val_tag="ls-NUM-quantity_compare-FV-val_set#ls-NUM-distance_compare-FV-val_set#ls-NUM-volume_compare-FV-val_set"

# --------------------------- #
# cot ablation study datasets
# --------------------------- #
train_tag="cot-NUM-volume-NI-train_set#cot-NUM-quantity-FV-train_set#cot-NUM-distance-FV-train_set#cot-NUM-volume-FV-train_set"
val_tag="cot-NUM-volume-NI-val_set#cot-NUM-quantity-FV-val_set#cot-NUM-distance-FV-val_set#cot-NUM-volume-FV-val_set"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost
# export OMP_NUM_THREADS=4

epoch=3
train_emb=True
train_img_proj=True
add_img_token=True
add_scene_token=False
no_obj=False
input_dim=1024 # 1024
bidirection=False
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=False
feat_fusion=False
fuse_with_id=False
config=""
max_grad_norm=0.01
seed=42
use_location_token=False
model_name=$(basename "$llama_model_path")
echo "Used model is $model_name"

debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    enable_wandb=False
    gpu_num=4
    do_save=True
    other_info="chatscene"
fi

tag="${train_tag}__${val_tag}__${other_info}"

OUTPUT_DIR=outputs/"$(date +"%Y%m%d_%H%M%S")"_"$model_name"_lr"$lr"
mkdir -p ${OUTPUT_DIR}

# torchrun --nproc_per_node=4 \
python tasks/train_numina.py \
    "$(dirname $0)/${config}config_numina.py" \
    output_dir "$OUTPUT_DIR" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.add_scene_token "$add_scene_token" \
    model.add_img_token "$add_img_token" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    cot "$cot" \
    ls "$ls" \
    model.train_emb "$train_emb" \
    model.train_img_proj "$train_img_proj" \
    train_tag "$train_tag" \
    val_tag "$val_tag" \
    model.no_obj "$no_obj" \
    segmentor "$segmentor" \
    pc_encoder "$pc_encoder" \
    model.input_dim "$input_dim" \
    model.bidirection "$bidirection" \
    optimizer.different_lr.enable "$different_lr" \
    model.max_obj_num "$max_obj_num" \
    lora.lora_r "$lora_r" \
    lora.lora_alpha "$lora_alpha" \
    model.add_pos_emb "$add_pos_emb" \
    model.feat_fusion "$feat_fusion" \
    optimizer.max_grad_norm "$max_grad_norm" \
    seed "$seed" \
    model.fuse_with_id "$fuse_with_id" \
    model.llama_model_path "$llama_model_path" \
    model.use_location_token "$use_location_token"
