import os
import glob  # 新增
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, hr_path, lr_path, segment_length):
        self.hr_files = sorted(glob.glob(os.path.join(hr_path, "*.wav")))  # 使用 glob 模块
        self.lr_files = sorted(glob.glob(os.path.join(lr_path, "*.wav")))
        self.segment_length = segment_length

    def __getitem__(self, index):
        hr_file = self.hr_files[index]
        lr_file = self.lr_files[index]

        hr_audio, _ = sf.read(hr_file)
        lr_audio, _ = sf.read(lr_file)

        # 如果音频长度不一样，裁剪或填充它们
        if len(hr_audio) > self.segment_length:
            hr_audio = hr_audio[:self.segment_length]
        elif len(hr_audio) < self.segment_length:
            hr_audio = np.pad(hr_audio, (0, self.segment_length - len(hr_audio)))

        if len(lr_audio) > self.segment_length:
            lr_audio = lr_audio[:self.segment_length]
        elif len(lr_audio) < self.segment_length:
            lr_audio = np.pad(lr_audio, (0, self.segment_length - len(lr_audio)))

        # 添加一个额外的维度以表示通道数
        hr = torch.from_numpy(hr_audio).float().unsqueeze(0)
        lr = torch.from_numpy(lr_audio).float().unsqueeze(0)

        return hr, lr

    def __len__(self):
        return len(self.hr_files)

def preprocess(audio, segment_length):
    num_segments = len(audio) // segment_length
    segments = np.array_split(audio, num_segments)

    # 如果最后一个片段的长度小于segment_length，用0填充
    if len(segments[-1]) < segment_length:
        pad = np.zeros(segment_length - len(segments[-1]))
        segments[-1] = np.concatenate([segments[-1], pad])

    # 对所有片段进行堆叠，然后转换为PyTorch张量
    segments_tensor = np.stack([segment for segment in segments if len(segment) == segment_length])
    return torch.from_numpy(segments_tensor).float()