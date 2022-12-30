# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# Other
import os
import sentencepiece as spm
import importlib

# Neural Nets
from nnet.module import Module

# CTC Decode
try:
    from ctcdecode import CTCBeamDecoder
except Exception as e:
    print(e)

###############################################################################
# Decoders
###############################################################################

class IdentityDecoder(nn.Module):

    def __init__(self):
        super(IdentityDecoder, self).__init__()

    def forward(self, outputs, from_logits=True):

        return outputs.tolist()

class ThresholdDecoder(nn.Module):

    def __init__(self, threshold=0.5):
        super(ThresholdDecoder, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = torch.where(outputs >= self.threshold, 1, 0).squeeze(dim=-1).tolist()
        else:
            tokens = outputs.tolist()

        return tokens

class ArgMaxDecoder(nn.Module):

    def __init__(self, axis=-1):
        super(ArgMaxDecoder, self).__init__()
        self.axis = axis

    def forward(self, outputs, from_logits=True):

        if from_logits:
            # Softmax -> Log -> argmax
            tokens = outputs.softmax(dim=self.axis).argmax(axis=self.axis).tolist()
        else:
            tokens = outputs.tolist()

        return tokens

class CTCGreedySearchDecoder(nn.Module):

    def __init__(self, tokenizer_path, blank_token=0):
        super(CTCGreedySearchDecoder, self).__init__()

        # Load Tokenizer
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

        # Blank Token
        self.blank_token = blank_token

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = self.greedy_search(*outputs)
        else:
            tokens = outputs[0].tolist()

        return self.tokenizer.decode(tokens)

    def greedy_search(self, logits, logits_len):

        # Argmax (B, T, V) -> (B, T)
        preds = logits.argmax(dim=-1)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):

            # Slice Preds
            preds_b = preds[b, :logits_len[b]]

            # Remove Consecutives
            preds_b = preds_b.unique_consecutive(dim=-1)

            # Remove Blanks
            preds_b = [token.item() for token in preds_b if token != self.blank_token]

            # Append Prediction
            batch_pred_list.append(preds_b)

        return batch_pred_list

class CTCBeamSearchDecoder(Module):

    """
    
        Parameter beam_alpha specifies amount of importance to place on the N-gram language model, 
        and beam_beta is a penalty term to consider the sequence length in the scores. 
        Larger alpha means more importance on the LM and less importance on the acoustic model. 
        Negative values for beta will give penalty to longer sequences and make the decoder to prefer shorter predictions, 
        while positive values would result in longer candidates.
    
    """

    def __init__(self, tokenizer_path, beam_size=16, ngram_path=None, ngram_tmp=1.0, ngram_alpha=0.6, ngram_beta=1.0, ngram_offset=100, neural_config_path=None, neural_checkpoint=None, neural_alpha=0.6, neural_beta=1.0, num_processes=8, test_time_aug=False):
        super(CTCBeamSearchDecoder, self).__init__()

        # Load Tokenizer
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

        # Params
        self.beam_size = beam_size
        self.test_time_aug = test_time_aug

        # Ngram
        self.ngram_path = ngram_path
        self.ngram_tmp = ngram_tmp
        self.ngram_alpha = ngram_alpha
        self.ngram_beta = ngram_beta
        self.ngram_offset = ngram_offset
        self.num_processes = num_processes

        # Neural Rescorer
        self.neural_alpha = neural_alpha
        self.neural_beta = neural_beta
        if neural_config_path is not None:
            neural_config = importlib.import_module(neural_config_path.replace(".py", "").replace("/", "."))
            self.register_module_buffer("neural_rescorer", neural_config.model)
            self.neural_rescorer.load(os.path.join(neural_config.callback_path, neural_checkpoint))
            self.neural_tokenizer = spm.SentencePieceProcessor(neural_config.tokenizer_path)
            self.neural_pad_token = neural_config.pad_token
            self.neural_bos_token = torch.tensor([neural_config.sos_token], dtype=torch.long)
            self.neural_eos_token = torch.tensor([neural_config.eos_token], dtype=torch.long)
        else:
            self.neural_rescorer = None

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = self.beam_search(*outputs)
        else:
            tokens = outputs[0].tolist()

        return self.tokenizer.decode(tokens)

    def beam_search(self, logits, logits_len, verbose=False):

        # test_time_aug
        if self.test_time_aug:
            batch_size, num_augments = logits.shape[0], logits.shape[1] # b, Naug
            logits = logits.flatten(start_dim=0, end_dim=1)
            logits_len = logits_len.flatten(start_dim=0, end_dim=1)
        else:
            batch_size, num_augments = logits.shape[0], 1

        # Beam Search Decoder
        decoder = CTCBeamDecoder(
            [chr(idx + self.ngram_offset) for idx in range(self.tokenizer.vocab_size())],
            model_path=self.ngram_path,
            alpha=self.ngram_alpha,
            beta=self.ngram_beta,
            cutoff_top_n=self.tokenizer.vocab_size(),
            cutoff_prob=1.0,
            beam_width=self.beam_size,
            num_processes=self.num_processes,
            blank_id=0,
            log_probs_input=True
        )

        # Apply Temperature
        logits = logits / self.ngram_tmp

        # Softmax -> Log
        logP = logits.log_softmax(dim=-1)

        # Beam Search Decoding: (B, Beam, N), (B, Beam), (B, Beam, N), (B, Beam)
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(logP, logits_len)

        # Neural Rescoring
        if self.neural_rescorer is not None:

            # Convert to neural LM ids (B, Beam, sos + N + eos)
            preds_ids_neural = [[torch.cat([self.neural_bos_token, torch.tensor(self.neural_tokenizer.encode(self.tokenizer.decode(beam_results[b][beam][:out_lens[b][beam]].tolist())), dtype=torch.long), self.neural_eos_token], dim=0) for beam in range(self.beam_size)] for b in range(logits.size(0))]

            # Pad ids (B, Beam, sos + N + eos)
            preds_ids_neural_pad = [torch.nn.utils.rnn.pad_sequence(preds_ids_neural[b], batch_first=True, padding_value=self.neural_pad_token) for b in range(logits.size(0))]

            # Forward Ids (B, Beam, sos + N + eos, V)
            neural_results = [self.neural_rescorer(preds_ids_neural_pad[b].to(self.device)).cpu() for b in range(logits.size(0))]

            # Softmax -> Log -> Neg (B, Beam, sos + N + eos, V)
            neural_results = [- neural_results[b].log_softmax(dim=-1) for b in range(logits.size(0))]

            # Compute Neural Scores and Lengths (B, beam)
            neural_scores = torch.zeros(logits.size(0), self.beam_size)
            neural_lengths = torch.zeros(logits.size(0), self.beam_size)
            for b in range(logits.size(0)):
                for beam in range(self.beam_size):
                    length_pred = len(preds_ids_neural[b][beam][1:]) # N + eos
                    for t in range(length_pred):
                        neural_scores[b][beam] += neural_results[b][beam][t][preds_ids_neural[b][beam][t+1]]
                    neural_lengths[b][beam] += self.neural_beta * length_pred

            # Rescore Predictions
            total_scores = beam_scores + self.neural_alpha * neural_scores - self.neural_beta * neural_lengths

            # (B, Beam) -> (b, Naug * Beam)
            total_scores = total_scores.reshape(batch_size, num_augments * self.beam_size)
            beam_results = beam_results.reshape(batch_size, num_augments * self.beam_size, -1)
            out_lens = out_lens.reshape(batch_size, num_augments * self.beam_size)

            # Best Ids (b,)
            best_ids = total_scores.argmin(dim=-1)

        else:

            # (B, Beam) -> (b, Naug)
            beam_scores = beam_scores.reshape(batch_size, num_augments, self.beam_size)[:, :, 0]
            beam_results = beam_results.reshape(batch_size, num_augments, self.beam_size, -1)[:, :, 0]
            out_lens = out_lens.reshape(batch_size, num_augments, self.beam_size)[:, :, 0]

            # Best Ids (b,)
            best_ids = beam_scores.argmin(dim=-1)

        # Batch Pred List
        batch_pred_list = [beam_results[b][best_ids[b]][:out_lens[b][best_ids[b]]].tolist() for b in range(batch_size)]

        return batch_pred_list

###############################################################################
# Decoder Dictionary
###############################################################################

decoder_dict = {
    "Threshold": ThresholdDecoder,
    "ArgMax": ArgMaxDecoder,
    "CTCGreedySearchDecoder": CTCGreedySearchDecoder,
    "CTCBeamSearch": CTCBeamSearchDecoder
}
