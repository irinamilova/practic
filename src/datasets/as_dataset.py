import numpy as np
import torch
import os
import torchaudio
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

def protocol(x):
    print(ROOT_PATH)
    print(ROOT_PATH / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{x}.txt")
    return ROOT_PATH / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{x}.txt"
# train.trn, dev.trl, eval.trl

direct = lambda x: (ROOT_PATH / f"ASVspoof2019_LA_{x}" / "flac")
# train, dev, eval


def load_protocol(protocol_path, base_dir):
    protocol = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            label = parts[4]
            path = os.path.join(base_dir, f"{file_id}.flac")
            protocol.append((path, label, file_id))
    return protocol


def extract_logmel_features(waveform, sr, n_fft=512, hop_length=160, n_mels=64):
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )(waveform)
    log_mel = torch.log1p(spec)
    return log_mel.squeeze(0)


class AntiSpoofDataset(BaseDataset):
    def __init__(self, protocol_name, name, max_len=600):
        self.file_list = load_protocol(protocol(protocol_name), direct(name))
        self.label_map = {"spoof": 0, "bonafide": 1}
        self.max_len = max_len

    def __getitem__(self, idx):
        filepath, label, _ = self.file_list[idx]
        waveform, sr = torchaudio.load(filepath)
        features = extract_logmel_features(waveform, sr)
        features = features[:, : self.max_len]
        if features.shape[1] < self.max_len:
            pad = self.max_len - features.shape[1]
            features = F.pad(features, (0, pad), "constant", 0)
        features = features.unsqueeze(0)
        features = features.unsqueeze(0)
        #print("features")
        #print(features.shape)
        return {"data_object": features, "labels": self.label_map[label]}

    def __len__(self):
        return len(self.file_list)
