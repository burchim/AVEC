import sys
sys.path.append("../../")

# Imports
import nnet
import torch

# Architecture
vocab_size=1024
model="GPT-Small"
max_pos_encoding=2048
pad_token = 0
sos_token = vocab_size
eos_token = vocab_size

# Training
batch_size = 128
accumulated_steps = 2
tokenizer_path = "datasets/LRS3/tokenizerbpe1024.model"
precision = torch.float16
callback_path = "callbacks/LibriSpeechCorpus/GPT-Small"

# Model
model = nnet.GPT(vocab_size=vocab_size+1, padding_idx=pad_token, max_pos_encoding=max_pos_encoding, model=model, pos_embedding=nnet.SinPosEmbedding)
model.compile()

# Datasets
label_max_length = 100
collate_fn=nnet.CollateFn(inputs_params=[{"axis": 0, "padding": True, "start_token": sos_token, "padding_value": pad_token}], targets_params=[{"axis": 0, "padding": True, "end_token": eos_token, "padding_value": -1}])
training_dataset = nnet.datasets.CorpusLM(
    collate_fn=collate_fn,
    batch_size=batch_size,
    tokenizer_path=tokenizer_path,
    max_length=label_max_length,
    corpus_path="datasets/LibriSpeechCorpus/librispeech-lm-norm.txt" # https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
)
evaluation_dataset = [
    nnet.datasets.CorpusLM(
        collate_fn=collate_fn,
        batch_size=batch_size,
        tokenizer_path=tokenizer_path,
        corpus_path="datasets/LRS2/corpus_test.txt"
    ),
    nnet.datasets.CorpusLM(
        collate_fn=collate_fn,
        batch_size=batch_size,
        tokenizer_path=tokenizer_path,
        corpus_path="datasets/LRS3/corpus_test.txt"
    )
]