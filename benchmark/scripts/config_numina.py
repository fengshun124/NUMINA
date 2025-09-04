# ========================= data ==========================
numina_train_set_root = ""  # root of the NUMINA train set
numina_val_set_root = ""  # root of the NUMINA validation set

num_ni = "NUM-NI"
num_fv = "NUM-FV"
nonum_fv = "NONUM-FV"
nonum_pm = "NONUM-PM"
ls_ablation = "logical-consistency-ablation"
cot_ablation = "chain-of-thought-ablation"

anno_root = "annotations"  # annotation dir
pc_encoder = "uni3d"
segmentor = "mask3d"
version = ""

gt_feat_file = f"{anno_root}/scannet_gt_{pc_encoder}_feats.pt"
seg_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats.pt"
seg_all_feat_file = f"{anno_root}/scannet_{segmentor}_{pc_encoder}_feats_all.pt"
gt_img_feat_file = f"{anno_root}/scannet_gt_videofeats.pt"
seg_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats.pt"
seg_all_img_feat_file = f"{anno_root}/scannet_{segmentor}_videofeats_all.pt"
gt_train_attr_file = f"{anno_root}/scannet_train_attributes.pt"
gt_val_attr_file = f"{anno_root}/scannet_val_attributes.pt"
seg_train_attr_file = f"{anno_root}/scannet_{segmentor}_train_attributes.pt"
seg_val_attr_file = f"{anno_root}/scannet_{segmentor}_val_attributes.pt"
seg_all_attr_file = f"{anno_root}/scannet_{segmentor}_all_attributes.pt"


train_tag='NUM-quantity-FV-train_set'
val_tag='NUM-quantity-FV-val_set'

train_file_dict = {
    'NONUM-scanqa-PM-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{nonum_pm}/NONUM-scanqa-PM-train_set.json'
    ],
    'NONUM-scanqa-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{nonum_fv}/NONUM-scanqa-FV-train_set.json'
    ],
    'NUM-quantity-NI-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{num_ni}/NUM-quantity-NI-train_set.json'
    ],
    'NUM-distance-NI-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{num_ni}/NUM-distance-NI-train_set.json'
    ],
    'NUM-volume-NI-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{num_ni}/NUM-volume-NI-train_set.json'
    ],
    'NUM-quantity-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{num_fv}/NUM-quantity-FV-train_set.json'
    ],
    'NUM-distance-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{num_fv}/NUM-distance-FV-train_set.json'
    ],
    'NUM-volume-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{num_fv}/NUM-volume-FV-train_set.json'
    ],
    'cot-NUM-quantity-NI-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{cot_ablation}/cot-NUM-quantity-NI-train_set.json'
    ],
    'cot-NUM-distance-NI-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{cot_ablation}/cot-NUM-distance-NI-train_set.json'
    ],
    'cot-NUM-volume-NI-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{cot_ablation}/cot-NUM-volume-NI-train_set.json'
    ],
    'cot-NUM-quantity-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{cot_ablation}/cot-NUM-quantity-FV-train_set.json'
    ],
    'cot-NUM-distance-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{cot_ablation}/cot-NUM-distance-FV-train_set.json'
    ],
    'cot-NUM-volume-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{cot_ablation}/cot-NUM-volume-FV-train_set.json'
    ],
    'ls-NUM-quantity_compare-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{ls_ablation}/ls-NUM-quantity_compare-FV-train_set.json'
    ],
    'ls-NUM-distance_compare-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{ls_ablation}/ls-NUM-distance_compare-FV-train_set.json'
    ],
    'ls-NUM-volume_compare-FV-train_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f'{numina_train_set_root}/{ls_ablation}/ls-NUM-volume_compare-FV-train_set.json'
    ]
}


val_file_dict = {
    'NONUM-scanqa-PM-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{nonum_pm}/NONUM-scanqa-PM-val_set.json'
    ],
    'NONUM-scanqa-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{nonum_fv}/NONUM-scanqa-FV-val_set.json'
    ],
    'NUM-quantity-NI-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{num_ni}/NUM-quantity-NI-val_set.json'
    ],
    'NUM-distance-NI-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{num_ni}/NUM-distance-NI-val_set.json'
    ],
    'NUM-volume-NI-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{num_ni}/NUM-volume-NI-val_set.json'
    ],
    'NUM-quantity-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{num_fv}/NUM-quantity-FV-val_set.json'
    ],
    'NUM-distance-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{num_fv}/NUM-distance-FV-val_set.json'
    ],
    'NUM-volume-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{num_fv}/NUM-volume-FV-val_set.json'
    ],   
    'cot-NUM-quantity-NI-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{cot_ablation}/cot-NUM-quantity-NI-val_set.json'
    ],
    'cot-NUM-distance-NI-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{cot_ablation}/cot-NUM-distance-NI-val_set.json'
    ],
    'cot-NUM-volume-NI-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{cot_ablation}/cot-NUM-volume-NI-val_set.json'
    ],
    'cot-NUM-quantity-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{cot_ablation}/cot-NUM-quantity-FV-val_set.json'
    ],
    'cot-NUM-distance-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{cot_ablation}/cot-NUM-distance-FV-val_set.json'
    ],
    'cot-NUM-volume-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{cot_ablation}/cot-NUM-volume-FV-val_set.json'
    ],
    'ls-NUM-quantity_compare-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{ls_ablation}/ls-NUM-quantity_compare-FV-val_set.json'
    ],
    'ls-NUM-distance_compare-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{ls_ablation}/ls-NUM-distance_compare-FV-val_set.json'
    ],
    'ls-NUM-volume_compare-FV-val_set': [
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f'{numina_val_set_root}/{ls_ablation}/ls-NUM-volume_compare-FV-val_set.json'
    ]     
}


num_workers = 0 # 32
batch_size = 32
cot=False
ls=False

# ========================= model ==========================
model = dict(
    llama_model_path="llm/vicuna-7b-v1.5",
    input_dim=1024,
    img_input_dim=1024,
    attr_dim=512,
    scene_dim=256,
    pos_dim=128,
    encoder_num_layers=3,
    low_resource=False,
    system_path="prompts/system_numina.txt",  # "prompts/system.txt"
    instruction_path="prompts/instruction.txt", 
    max_txt_len=384,  # set too short for cot prediction to show all the text, defalut is 128
    end_sym="</s>",
    role=("USER", "ASSISTANT"),
    add_scene_token=True,
    add_img_token=True,
    use_lora=True,
    train_emb=True,
    train_img_proj=False,
    no_obj=False,
    max_obj_num=200,
    bidirection=False,
    add_pos_emb=False,
    feat_fusion=False,
    fuse_with_id=False,
    use_objid=True,
    use_location_token=False
)


swanlab = dict(
    enable=True,
    project="NUMINA",
    experiment_name="vicuna13b-cot"
)

lora = dict(
    lora_target_modules=[
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.05
)

optimizer = dict(
    opt="adamW",
    lr=5e-3,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    scaler_enable=False,
    max_grad_norm=5,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["model.embed_tokens"],
        lr=[5e-4],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False


# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="huanghaifeng",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="Scene-LLM",
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "outputs/tmp"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
# resume = True
debug = False
log_freq = 50  # 20
# eval_freq = 500
seed = 42

save_latest = False  # False
do_save = True
auto_resume = True
pretrained_path = ""
img_projector_path = ""

debug=False
gpu_num=1

