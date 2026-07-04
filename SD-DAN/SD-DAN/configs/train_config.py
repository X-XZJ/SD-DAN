from torch import optim

from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict

# Commonly changed training configurations
num_epochs = 12   # train epochs
batch_size = 2    # total_batch_size = #GPU x batch_size
num_workers = 4   # workers for pytorch DataLoader
pin_memory = True # whether pin_memory for pytorch DataLoader
print_freq = 50   # frequency to print logs
starting_epoch = 0
max_norm = 0.1    # clip gradient norm

output_dir = None  # path to save checkpoints, default for None: checkpoints/{model_name}
find_unused_parameters = False  # useful for debugging distributed training

# define dataset for train
coco_path = "/media/omnisky/lmy/home/ll_salience_detr/salience-detr6/datasets/processed_apple_data"
train_transform = presets.detr
train_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/train",
    ann_file=f"{coco_path}/annotations_train.json",
    transforms=train_transform,
    train=True,
)
test_dataset = CocoDetection(
    img_folder=f"{coco_path}/images/val",
    ann_file=f"{coco_path}/annotations_val.json",
    transforms=None,
)

# model config to train
model_path = "configs/relation_detr/relation_detr_resnet50_800_1333.py"

resume_from_checkpoint = None

# 学习率配置
learning_rate = 1e-4  # 基础学习率
threshold_lr = 1e-3   # threshold 的学习率

optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# 定义参数组函数 - 接受 model 作为第一个参数
def param_dicts(model):
    return param_dict.finetune_backbone_and_linear_projection(
        model=model,
        lr=learning_rate,
        threshold_lr=threshold_lr
    )