import sys
sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision
import importlib

# Architecture
vocab_size = 256
v_interctc_blocks = [3, 6]
a_interctc_blocks = [8, 11]
f_interctc_blocks = [2]
loss_weights={
    "v_ctc_2": 0.5 / 3,
    "v_ctc_5": 0.5 / 3,
    "a_ctc_7": 0.5 / 3,
    "a_ctc_10": 0.5 / 3,
    "f_ctc_1": 0.5 / 3,
    "outputs": 0.5
}

# lrw_pretrained
lrw_pretrained = True
lrw_checkpoint = "callbacks/LRW/EffConfCE/checkpoints_epoch_30_step_57247.ckpt"

# Arch
ff_ratio = 4
conv_stride = 2
drop_rate = 0.1
attn_drop_rate = 0.0
kernel_size = 15
num_heads = 4
max_pos_encoding = 10000

# Beam Search
beamsearch = True
beam_size = 16
tokenizer_path="datasets/LRS3/tokenizerbpe256.model"
ngram_path="datasets/LRS3/6gram_lrs23.arpa"
ngram_offset = 100
ngram_alpha = 0.6
ngram_beta = 1.0
ngram_tmp = 1.0
neural_config_path = "configs/LRS23/LM/GPT-Small.py"
neural_checkpoint = "checkpoints_epoch_10_step_2860.ckpt"
neural_alpha = 0.6
neural_beta = 1.0

# Training
batch_size = 16
accumulated_steps = 4
eval_training = False
precision = torch.float16
recompute_metrics=True
callback_path = "callbacks/LRS23/AV/EffConfInterCTC"

# Model
model = nnet.AudioVisualEfficientConformerInterCTC(vocab_size=vocab_size, v_interctc_blocks=v_interctc_blocks, a_interctc_blocks=a_interctc_blocks, f_interctc_blocks=f_interctc_blocks)
model.compile(
    losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    decoders={"outputs": nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path) if not beamsearch else nnet.CTCBeamSearchDecoder(tokenizer_path=tokenizer_path, beam_size=beam_size, ngram_path=ngram_path, ngram_tmp=ngram_tmp, ngram_alpha=ngram_alpha, ngram_beta=ngram_beta, ngram_offset=ngram_offset, neural_config_path=neural_config_path, neural_checkpoint=neural_checkpoint, neural_alpha=neural_alpha, neural_beta=neural_beta)},
    metrics={"outputs": nnet.WordErrorRate()},
    loss_weights=loss_weights
)

# Load Pretrained
if lrw_pretrained:
    lrw_checkpoint = torch.load(lrw_checkpoint, map_location=model.device)
    for key, value in lrw_checkpoint["model_state_dict"].copy().items():
        if not "front_end" in key:
            lrw_checkpoint["model_state_dict"].pop(key)
    model.encoder.video_encoder.front_end.load_state_dict({key.replace(".module.", ".").replace("encoder.front_end.", ""):value for key, value in lrw_checkpoint["model_state_dict"].items()})

# Datasets
align = True
video_max_length = 400
collate_fn=nnet.CollateFn(inputs_params=[{"axis": 0, "padding": True}, {"axis": 3}, {"axis": 1, "padding": True}, {"axis": 4}], targets_params=({"axis": 2, "padding": True}, {"axis": 5}))
crop_size = (88, 88)
training_video_transform=nn.Sequential(
    torchvision.transforms.RandomCrop(crop_size),
    torchvision.transforms.RandomHorizontalFlip(),
    nnet.Permute(dims=(2, 3, 0, 1)),
    nnet.TimeMaskSecond(T_second=0.4, num_mask_second=1.0, fps=25.0, mean_frame=True),
    nnet.Permute(dims=(2, 3, 0, 1))
)
evaluation_video_transform=torchvision.transforms.CenterCrop(crop_size)
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
        video_transform=training_video_transform,
        align=align
    ),
    nnet.datasets.LRS(
        batch_size=None,
        collate_fn=None,
        version="LRS3",
        mode="pretrain+trainval",
        video_max_length=video_max_length,
        video_transform=training_video_transform,
        align=align
    )
])

evaluation_dataset = [
    nnet.datasets.LRS(
        batch_size=batch_size,
        collate_fn=collate_fn,
        version="LRS2",
        mode="test",
        video_transform=evaluation_video_transform,
        align=align
    ),
    nnet.datasets.LRS(
        batch_size=batch_size,
        collate_fn=collate_fn,
        version="LRS3",
        mode="test",
        video_transform=evaluation_video_transform,
        align=align
    )
]