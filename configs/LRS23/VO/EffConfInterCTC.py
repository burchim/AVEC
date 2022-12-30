import sys
sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision

# Architecture
vocab_size = 256
interctc_blocks = [3, 6, 9]
loss_weights = [0.5/3, 0.5/3, 0.5/3, 0.5]

# lrw_pretrained
lrw_pretrained = True
lrw_checkpoint = "callbacks/LRW/EffConfCE/checkpoints_epoch_30_step_57247.ckpt"

# Beam Search
beamsearch = True
tokenizer_path="datasets/LRS3/tokenizerbpe256.model"
ngram_path="datasets/LRS3/6gram_lrs23.arpa"
ngram_offset = 100
beam_size = 16
ngram_alpha = 0.6
ngram_beta = 1.0
ngram_tmp = 1.0
neural_config_path = "configs/LRS23/LM/GPT-Small.py"
neural_checkpoint = "checkpoints_epoch_10_step_2860.ckpt"
neural_alpha = 0.6
neural_beta = 1.0
test_augments = torchvision.transforms.RandomHorizontalFlip(p=1.0)
test_time_aug = True

# Training
batch_size = 16
accumulated_steps = 4
eval_training = False
precision = torch.float16
recompute_metrics=True
callback_path = "callbacks/LRS23/VO/EffConfInterCTC"

# Model
model = nnet.VisualEfficientConformerInterCTC(vocab_size=vocab_size, interctc_blocks=interctc_blocks, test_augments=test_augments if test_time_aug else None)
model.compile(
    losses=None if test_time_aug else nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    decoders={"outputs": nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path) if not beamsearch else nnet.CTCBeamSearchDecoder(tokenizer_path=tokenizer_path, beam_size=beam_size, ngram_path=ngram_path, ngram_tmp=ngram_tmp, ngram_alpha=ngram_alpha, ngram_beta=ngram_beta, ngram_offset=ngram_offset, neural_config_path=neural_config_path, neural_checkpoint=neural_checkpoint, neural_alpha=neural_alpha, neural_beta=neural_beta, test_time_aug=test_time_aug)},
    metrics={"outputs": nnet.WordErrorRate()},
    loss_weights=loss_weights
)

# Load Pretrained
if lrw_pretrained:
    lrw_checkpoint = torch.load(lrw_checkpoint, map_location=model.device)
    for key, value in lrw_checkpoint["model_state_dict"].copy().items():
        if not "front_end" in key:
            lrw_checkpoint["model_state_dict"].pop(key)
    model.encoder.front_end.load_state_dict({key.replace(".module.", ".").replace("encoder.front_end.", ""):value for key, value in lrw_checkpoint["model_state_dict"].items()})

# Datasets
video_max_length = 400
crop_size = (88, 88)
collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0, "padding": True}, {"axis": 3}], targets_params=({"axis": 2, "padding": True}, {"axis": 5}))
training_video_transform = nn.Sequential(
    torchvision.transforms.RandomCrop(crop_size),
    torchvision.transforms.RandomHorizontalFlip(),
    nnet.Permute(dims=(2, 3, 0, 1)),
    nnet.TimeMaskSecond(T_second=0.4, num_mask_second=1.0, fps=25.0, mean_frame=True),
    nnet.Permute(dims=(2, 3, 0, 1))
)
evaluation_video_transform = torchvision.transforms.CenterCrop(crop_size)
training_dataset = nnet.datasets.MultiDataset(
    batch_size=batch_size,
    collate_fn=collate_fn,
    datasets=[
    nnet.datasets.LRS(
        batch_size=None,
        collate_fn=None,
        version="LRS2",
        mode="pretrain+train+val",
        video_max_length=video_max_length,
        video_transform=training_video_transform
    ),
    nnet.datasets.LRS(
        batch_size=None,
        collate_fn=None,
        version="LRS3",
        mode="pretrain+trainval",
        video_max_length=video_max_length,
        video_transform=training_video_transform
    )
])
evaluation_dataset = [
    nnet.datasets.LRS(
        batch_size=batch_size,
        collate_fn=collate_fn,
        version="LRS2",
        mode="test",
        video_transform=evaluation_video_transform
    ),
    nnet.datasets.LRS(
        batch_size=batch_size,
        collate_fn=collate_fn,
        version="LRS3",
        mode="test",
        video_transform=evaluation_video_transform
    )
]