import sys
sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision

# Architecture
vocab_size = 500
    
# Training
epochs = 30
batch_size = 64
accumulated_steps = 1
precision = torch.float16
callback_path = "callbacks/LRW/EffConfCE"

# Model    
model = nnet.VisualEfficientConformerCE(vocab_size=vocab_size)
model.compile()

# Datasets
crop_size = (88, 88)
collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])
training_dataset = nnet.datasets.LRW(
    batch_size=batch_size,
    collate_fn=collate_fn,
    mode="train",
    video_transform=nn.Sequential(
        torchvision.transforms.RandomCrop(crop_size),
        torchvision.transforms.RandomHorizontalFlip()
    )
)
evaluation_dataset = nnet.datasets.LRW(
    batch_size=batch_size,
    collate_fn=collate_fn,
    mode="val",
    video_transform=torchvision.transforms.CenterCrop(crop_size)
)