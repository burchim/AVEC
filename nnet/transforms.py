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

# Other
import numpy as np
import collections 
import skimage.transform
import sys

# Face Detectors
try:
    from ibug.face_detection import RetinaFacePredictor
except Exception as e:
    print(e)
try:
    from ibug.face_alignment import FANPredictor
except Exception as e:
    print(e)

###############################################################################
# Transforms
###############################################################################

class NormalizeVideo(nn.Module):

    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).reshape(len(mean), 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).reshape(len(std), 1, 1, 1), persistent=False)

    def forward(self, x):

        x = (x - self.mean) / self.std

        return x

class DenormalizeVideo(nn.Module):

    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).reshape(len(mean), 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).reshape(len(std), 1, 1, 1), persistent=False)

    def forward(self, x):

        x = x * self.std + self.mean

        return x

def video_to_images(videos):

    # (B, C, T, H, W) -> (B*T, C, H, W)
    return videos.transpose(1, 2).flatten(start_dim=0, end_dim=1)

def images_to_videos(images, video_frames):

    #(B*T, C, H, W) -> (B, C, T, H, W)
    assert images.size(0) % video_frames == 0
    return images.reshape(images.size(0) // video_frames, video_frames, images.size(1), images.size(2), images.size(3)).transpose(1, 2)

class VideoToImages(nn.Module):

    """ Transform Batch of Videos to Batch of Images """

    def __init__(self):
        super().__init__()

    def forward(self, videos):
        return video_to_images(videos)

class ImagesToVideos(nn.Module):

    """ Transform Batch of Images to Batch of Videos """

    def __init__(self, video_frames=None):
        super().__init__()
        self.video_frames = video_frames

    def forward(self, images, video_frames=None):

        # default video_frames
        if video_frames == None:
            video_frames = self.video_frames

        # assert not None
        assert video_frames != None

        return images_to_videos(images, video_frames)

class TimeMaskSecond(nn.Module):

    def __init__(self, T_second, num_mask_second, fps, mean_frame=False):
        super().__init__()

        self.T = int(T_second * fps)
        self.num_mask_second = num_mask_second
        self.mean_frame = mean_frame
        self.fps = fps

    def forward(self, x):

        mT = int(x.shape[-1] / self.fps * self.num_mask_second)

        # Mask
        for _ in range(mT):
            x = torchaudio.functional.mask_along_axis(x, mask_param=self.T, mask_value=x.mean() if self.mean_frame else 0.0, axis=2)

        return x

class BabbleNoise(nn.Module):

    def __init__(self, noise_file_path="datasets/NoiseX/babble/babble.flac", SNR_db=[-5, 0, 5, 10, 15, 20, None], to_sample_rate=16000):
        super().__init__()

        self.noise, self.from_sample_rate = torchaudio.load(noise_file_path)
        self.noise = self.noise[:1]
        if to_sample_rate != None:
            self.noise = torchaudio.functional.resample(self.noise, orig_freq=self.from_sample_rate, new_freq=to_sample_rate)
        self.SNR_db = SNR_db

    def forward(self, x):

        # Select SNR
        SNR_db = self.SNR_db[torch.randint(0, len(self.SNR_db), ())]

        # Transform
        if SNR_db != None:

            # db to power
            SNR = 10 ** (SNR_db / 10)

            # power to ampl
            SNR = SNR ** 0.5

            # Selet Noise
            pos = torch.randint(0, self.noise.shape[-1] - x.shape[-1] + 1, ())
            noise = self.noise[:, pos:pos + x.shape[-1]]

            # Power
            x_power = (x ** 2).sum() / x.shape[-1]
            noise_power = (noise ** 2).sum() / noise.shape[-1]

            # Scale Noise
            noise = 1 / SNR * noise * torch.sqrt(x_power / noise_power)

            # Add Noise
            x = x + noise

        return x

def align_video_to_audio(video, audio):

    Tv, H, W, C = video.shape # (Tv, H, W, Cv)
    Ta = audio.shape[0] # (Ta,)

    padding = Ta // (160*2*2) + 1 - Tv
    padding_left = padding // 2
    padding_right = padding // 2 + padding % 2

    video = torch.cat([video.new_zeros(padding_left, H, W, C), video, video.new_zeros(padding_right, H, W, C)], dim=0)

    return video

class LipDetectCrop(nn.Module):
    
    def __init__(
        self, 
        mean_face_landmarks_path="datasets/LRS3/20words_mean_face.npy",
        start_idx=48,
        stop_idx = 68,
        crop_width = 96,
        crop_height = 96,
        window_margin = 12,
        STD_SIZE = (256, 256),
        stablePntsIDs = [33, 36, 39, 42, 45]
    ):
        super().__init__()
        
        self.mean_face_landmarks = np.load(mean_face_landmarks_path)
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.window_margin = window_margin
        self.STD_SIZE = STD_SIZE
        self.stablePntsIDs = stablePntsIDs
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self.face_detector = RetinaFacePredictor(device=device, threshold=0.8, model=RetinaFacePredictor.get_model('resnet50'))
            self.landmark_detector = FANPredictor(device=device, model=None)
        except:
            pass
        
    def detect_landmarks(self, video, verbose=0):
        
        video_landmarks = []

        for i, frame in enumerate(video):
            if verbose:
                sys.stdout.write("\r{}/{}".format(i+1, video.shape[0]))
            frame = frame.numpy()
            detected_faces = self.face_detector(frame, rgb=True)
            landmarks, scores = self.landmark_detector(frame, detected_faces, rgb=True)
            if len(landmarks) > 0:
                video_landmarks.append(landmarks[0])
            else:
                video_landmarks.append(None)
            
        return video_landmarks

    def landmarks_interpolate(self, landmarks):
            """landmarks_interpolate.
            :param landmarks: List, the raw landmark (in-place)
            """
            valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
            if not valid_frames_idx:
                return None
            for idx in range(1, len(valid_frames_idx)):
                if valid_frames_idx[idx] - valid_frames_idx[idx-1] == 1:
                    continue
                else:
                    landmarks = self.linear_interpolate(landmarks, valid_frames_idx[idx-1], valid_frames_idx[idx])
            valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
            # -- Corner case: keep frames at the beginning or at the end failed to be detected.
            if valid_frames_idx:
                landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
                landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
            valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
            assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
            return landmarks

    def linear_interpolate(self, landmarks, start_idx, stop_idx):
            """linear_interpolate.
            :param landmarks: ndarray, input landmarks to be interpolated.
            :param start_idx: int, the start index for linear interpolation.
            :param stop_idx: int, the stop for linear interpolation.
            """
            start_landmarks = landmarks[start_idx]
            stop_landmarks = landmarks[stop_idx]
            delta = stop_landmarks - start_landmarks
            for idx in range(1, stop_idx-start_idx):
                landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
            return landmarks

    def warp_img(self, src, dst, img, std_size):
            """warp_img.
            :param src: ndarray, source coordinates.
            :param dst: ndarray, destination coordinates. 
            :param img: ndarray, an input image.
            :param std_size: tuple (rows, cols), shape of the output image generated.
            """
            tform = skimage.transform.estimate_transform('similarity', src, dst)  # find the transformation matrix
            warped = skimage.transform.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
            warped = warped * 255  # note output from wrap is double image (value range [0,1])
            warped = warped.astype('uint8')
            return warped, tform

    def apply_transform(self, transform, img, std_size):
            """apply_transform.
            :param transform: Transform object, containing the transformation parameters \
                            and providing access to forward and inverse transformation functions.
            :param img: ndarray, an input image.
            :param std_size: tuple (rows, cols), shape of the output image generated.
            """
            warped = skimage.transform.warp(img, inverse_map=transform.inverse, output_shape=std_size)
            warped = warped * 255  # note output from wrap is double image (value range [0,1])
            warped = warped.astype('uint8')
            return warped

    def cut_patch(self, img, landmarks, height, width, threshold=5):
            """cut_patch.
            :param img: ndarray, an input image.
            :param landmarks: ndarray, the corresponding landmarks for the input image.
            :param height: int, the distance from the centre to the side of of a bounding box.
            :param width: int, the distance from the centre to the side of of a bounding box.
            :param threshold: int, the threshold from the centre of a bounding box to the side of image.
            """
            center_x, center_y = np.mean(landmarks, axis=0)

            if center_y - height < 0:                                                
                center_y = height                                                    
            if center_y - height < 0 - threshold:                                    
                raise Exception('too much bias in height')                           
            if center_x - width < 0:                                                 
                center_x = width                                                     
            if center_x - width < 0 - threshold:                                     
                raise Exception('too much bias in width')                            
                                                                                    
            if center_y + height > img.shape[0]:                                     
                center_y = img.shape[0] - height                                     
            if center_y + height > img.shape[0] + threshold:                         
                raise Exception('too much bias in height')                           
            if center_x + width > img.shape[1]:                                      
                center_x = img.shape[1] - width                                      
            if center_x + width > img.shape[1] + threshold:                          
                raise Exception('too much bias in width')                            
                                                                                    
            cutted_img = np.copy(img[ int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                                int(round(center_x) - round(width)): int(round(center_x) + round(width))])
            return cutted_img

    def crop_patch(self, video, landmarks):

            """Crop mouth patch
            :param str video_pathname: pathname for the video_dieo
            :param list landmarks: interpolated landmarks
            """

            frame_idx = 0
            num_frames = video.shape[0]
            frame_gen = video
            margin = min(num_frames, self.window_margin)
            while True:
                try:
                    #frame = frame_gen.__next__() ## -- BGR
                    frame = frame_gen[frame_idx]
                except StopIteration:
                    break
                if frame_idx == 0:
                    q_frame, q_landmarks = collections.deque(), collections.deque()
                    sequence = []

                q_landmarks.append(landmarks[frame_idx])
                q_frame.append(frame)
                if len(q_frame) == margin:
                    smoothed_landmarks = np.mean(q_landmarks, axis=0)
                    cur_landmarks = q_landmarks.popleft()
                    cur_frame = q_frame.popleft()

                    # -- affine transformation
                    trans_frame, trans = self.warp_img( smoothed_landmarks[self.stablePntsIDs, :],
                                                self.mean_face_landmarks[self.stablePntsIDs, :],
                                                cur_frame,
                                                self.STD_SIZE)
                    trans_landmarks = trans(cur_landmarks)

                    # -- crop mouth patch
                    sequence.append( self.cut_patch( trans_frame,
                                                trans_landmarks[self.start_idx:self.stop_idx],
                                                self.crop_height//2,
                                                self.crop_width//2,))
                if frame_idx == len(landmarks)-1:
                    while q_frame:

                        cur_frame = q_frame.popleft()

                        # -- transform frame
                        trans_frame = self.apply_transform( trans, cur_frame, self.STD_SIZE)

                        # -- transform landmarks
                        trans_landmarks = trans(q_landmarks.popleft())

                        # -- crop mouth patch
                        sequence.append( self.cut_patch( trans_frame,
                                                    trans_landmarks[self.start_idx:self.stop_idx],
                                                    self.crop_height//2,
                                                    self.crop_width//2,))
                    return np.array(sequence)
                frame_idx += 1
            return None
        
    def forward(self, video, verbose=0):
        
        landmarks = self.detect_landmarks(video, verbose=verbose)
        preprocessed_landmarks = self.landmarks_interpolate(landmarks)
        video_crop = self.crop_patch(video.numpy(), preprocessed_landmarks)
        video_crop = torch.tensor(video_crop)
        
        return video_crop