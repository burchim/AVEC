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
import torchaudio

###############################################################################
# Audio Preprocessing
###############################################################################

class AudioPreprocessing(nn.Module):

    """Audio Preprocessing

    Computes mel-scale log filter banks spectrogram

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT frame size, creates n_fft // 2 + 1 frequency bins.
        win_length_ms: FFT window length in ms, must be <= n_fft
        hop_length_ms: length of hop between FFT windows in ms
        n_mels: number of mel filter banks
        normalize: whether to normalize mel spectrograms outputs
        mean: training mean
        std: training std

    Shape:
        Input: (batch_size, audio_len)
        Output: (batch_size, n_mels, audio_len // hop_length + 1)
    
    """

    def __init__(self, sample_rate=16000, n_fft=512, win_length_ms=25, hop_length_ms=10, n_mels=80, normalize=False, mean=0, std=1):
        super(AudioPreprocessing, self).__init__()

        self.win_length = int(sample_rate * win_length_ms) // 1000
        self.hop_length = int(sample_rate * hop_length_ms) // 1000
        self.Spectrogram = torchaudio.transforms.Spectrogram(n_fft, self.win_length, self.hop_length)
        self.MelScale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min=0, f_max=8000, n_stft=n_fft // 2 + 1)
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def forward(self, x, lengths=None):

        with torch.cuda.amp.autocast(enabled=False):

            dtype = x.dtype
            x = x.float()

            # Short Time Fourier Transform (B, T) -> (B, n_fft // 2 + 1, T // hop_length + 1)
            x = self.Spectrogram(x)

            x_prev = x

            # Mel Scale (B, n_fft // 2 + 1, T // hop_length + 1) -> (B, n_mels, T // hop_length + 1)
            x = self.MelScale(x)
            
            # Energy log, autocast disabled to prevent float16 overflow
            x = (x.float() + 1e-9).log().type(x.dtype)

            # Compute Sequence lengths 
            if lengths is not None:
                lengths = torch.div(lengths, self.hop_length, rounding_mode='floor') + 1

            # Normalize
            if self.normalize:
                x = (x - self.mean) / self.std
            
        x = x.type(dtype)

        return (x, lengths) if lengths != None else x 

class SpecAugment(nn.Module):

    """Spectrogram Augmentation

    Args:
        spec_augment: whether to apply spec augment
        mF: number of frequency masks
        F: maximum frequency mask size
        mT: number of time masks
        pS: adaptive maximum time mask size in %

    References:
        SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition, Park et al.
        https://arxiv.org/abs/1904.08779

        SpecAugment on Large Scale Datasets, Park et al.
        https://arxiv.org/abs/1912.05533

    """

    def __init__(self, mF, F, mT, pS):
        super(SpecAugment, self).__init__()

        self.mF = mF
        self.F = F
        self.mT = mT
        self.pS = pS

    def forward(self, samples, lengths):

        # Spec Augment
        if self.training:
        
            # Frequency Masking
            for _ in range(self.mF):
                samples = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.F, iid_masks=False).forward(samples)

            # Time Masking
            for b in range(samples.size(0)):
                T = int(self.pS * lengths[b])
                for _ in range(self.mT):
                    samples[b:b+1, :, :lengths[b]] = torchaudio.transforms.TimeMasking(time_mask_param=T).forward(samples[b:b+1, :, :lengths[b]])

        return samples