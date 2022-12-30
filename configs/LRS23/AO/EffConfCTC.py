import sys
sys.path.append("../../")

# Imports
import nnet
import torch

# Architecture
vocab_size = 256
interctc_blocks = []
loss_weights = None
att_type = "patch"

# Training
batch_size = 16
accumulated_steps = 4
eval_training = False
precision = torch.float16
recompute_metrics=True
callback_path = "callbacks/LRS23/AO/EffConfCTC"

# Beam Search
beam_search = True
tokenizer_path="datasets/LRS3/tokenizerbpe256.model"
ngram_path="datasets/LRS3/6gram_lrs23.arpa"
ngram_offset = 100
beam_size=16 
ngram_alpha=0.6
ngram_beta=1.0
ngram_tmp=1.0
neural_config_path = "configs/LRS23/LM/GPT-Small.py"
neural_checkpoint = "checkpoints_epoch_10_step_2860.ckpt"
neural_alpha = 0.6
neural_beta = 1.0

# Model
model = nnet.AudioEfficientConformerInterCTC(vocab_size=vocab_size, att_type=att_type, interctc_blocks=interctc_blocks)
model.compile(
    losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    metrics=nnet.WordErrorRate(),
    decoders=nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path) if not beam_search else nnet.CTCBeamSearchDecoder(tokenizer_path=tokenizer_path, beam_size=beam_size, ngram_path=ngram_path, ngram_tmp=ngram_tmp, ngram_alpha=ngram_alpha, ngram_beta=ngram_beta, ngram_offset=ngram_offset, neural_config_path=neural_config_path, neural_checkpoint=neural_checkpoint, neural_alpha=neural_alpha, neural_beta=neural_beta),
    loss_weights=loss_weights
)

# Datasets
load_video = False
audio_max_length = 16 * 16000
collate_fn=nnet.CollateFn(inputs_params=[{"axis": 1, "padding": True}, {"axis": 4}], targets_params=({"axis": 2, "padding": True}, {"axis": 5}))
training_dataset = nnet.datasets.MultiDataset(
    batch_size=batch_size,
    collate_fn=collate_fn,
    datasets=[
    nnet.datasets.LRS(
        batch_size=None,
        collate_fn=None,
        version="LRS2",
        mode="pretrain+train+val",
        audio_max_length=audio_max_length,
        load_video=load_video
    ),
    nnet.datasets.LRS(
        batch_size=None,
        collate_fn=None,
        version="LRS3",
        mode="pretrain+trainval",
        audio_max_length=audio_max_length,
        load_video=load_video
    )
])
evaluation_dataset = [
    nnet.datasets.LRS(
        batch_size=batch_size,
        collate_fn=collate_fn,
        version="LRS2",
        mode="test",
        load_video=load_video
    ),
    nnet.datasets.LRS(
        batch_size=batch_size,
        collate_fn=collate_fn,
        version="LRS3",
        mode="test",
        load_video=load_video
    )
]