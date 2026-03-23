import os
import json
import numpy as np
import soundfile as sf
import librosa
from torch.utils.data import Dataset
import torch
class DOADataset_regression_stft(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 1))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        stft_cat, ild = self.compute_ipd_ild(wav[:sr * 10, ].T, sr, wav_name=audio_name)

        stft_cat = stft_cat.astype(np.float32)

        ild = ild.astype(np.float32)
        label = np.int64(label)
        stft_cat = torch.from_numpy(stft_cat)  # [1, F, T]
        ild = torch.from_numpy(ild).unsqueeze(0)  # [1, F, T]
        return stft_cat, ild, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]
        stft = np.stack([stft_l_sel, stft_r_sel], axis=0)
        stft_cat = np.concatenate([np.real(stft), np.imag(stft)], axis=0)

        ratio = np.abs(stft_l_sel) / (np.abs(stft_r_sel) + 1e-8)
        ratio = np.clip(ratio, 1e-3, 1e3)
        ild = 20 * np.log10(ratio)

        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            stft_cat = np.pad(stft_cat, ((0, 0), (0, pad_width)), mode='constant')
            ild = np.pad(ild, ((0, 0), (0, pad_width)), mode='constant')

        return stft_cat, ild
class DOADataset_5_stft(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 5))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        stft_cat = self.compute_ipd_ild(wav[:sr * 10, ].T, sr, wav_name=audio_name)

        stft_cat = stft_cat.astype(np.float32)
        label = np.int64(label)
        stft_cat = torch.from_numpy(stft_cat)  # [1, F, T]
        return stft_cat, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]
        stft = np.stack([stft_l_sel, stft_r_sel], axis=0)
        stft_cat = np.concatenate([np.real(stft), np.imag(stft)], axis=0)
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            stft_cat = np.pad(stft_cat, ((0, 0), (0, pad_width)), mode='constant')

        return stft_cat
class DOADataset_10_stft(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 10))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        stft_cat = self.compute_ipd_ild(wav[:sr * 10, ].T, sr, wav_name=audio_name)

        stft_cat = stft_cat.astype(np.float32)
        label = np.int64(label)
        stft_cat = torch.from_numpy(stft_cat)  # [1, F, T]
        return stft_cat, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]
        stft = np.stack([stft_l_sel, stft_r_sel], axis=0)
        stft_cat = np.concatenate([np.real(stft), np.imag(stft)], axis=0)
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            stft_cat = np.pad(stft_cat, ((0, 0), (0, pad_width)), mode='constant')

        return stft_cat
class DOADataset_15_stft(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 15))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        stft_cat = self.compute_ipd_ild(wav[:sr * 10, ].T, sr, wav_name=audio_name)

        stft_cat = stft_cat.astype(np.float32)
        label = np.int64(label)
        stft_cat = torch.from_numpy(stft_cat)  # [1, F, T]
        return stft_cat, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]
        stft = np.stack([stft_l_sel, stft_r_sel], axis=0)
        stft_cat = np.concatenate([np.real(stft), np.imag(stft)], axis=0)
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            stft_cat = np.pad(stft_cat, ((0, 0), (0, pad_width)), mode='constant')

        return stft_cat

class DOADataset_20_stft(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 20))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        stft_cat = self.compute_ipd_ild(wav[:sr * 10, ].T, sr, wav_name=audio_name)

        stft_cat = stft_cat.astype(np.float32)
        label = np.int64(label)
        stft_cat = torch.from_numpy(stft_cat)  # [1, F, T]
        return stft_cat, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]
        stft = np.stack([stft_l_sel, stft_r_sel], axis=0)
        stft_cat = np.concatenate([np.real(stft), np.imag(stft)], axis=0)
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            stft_cat = np.pad(stft_cat, ((0, 0), (0, pad_width)), mode='constant')

        return stft_cat
        
        
class DOADataset_5_ipd(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 5))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        ipd = ipd.astype(np.float32)
        label = np.int64(label)
        ipd = torch.from_numpy(ipd).unsqueeze(0)  # [1, F, T]
        return ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]

        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
       
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            ipd = np.pad(ipd, ((0, 0), (0, pad_width)), mode='constant')
        

        return ipd
class DOADataset_10_ipd(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 10))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        ipd = ipd.astype(np.float32)
        label = np.int64(label)
        ipd = torch.from_numpy(ipd).unsqueeze(0)  # [1, F, T]
        return ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]

        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
       
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            ipd = np.pad(ipd, ((0, 0), (0, pad_width)), mode='constant')
        

        return ipd      
        
class DOADataset_10_ipd_F30(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 10))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        ipd = ipd.astype(np.float32)
        label = np.int64(label)
        ipd = torch.from_numpy(ipd).unsqueeze(0)  # [1, F, T]
        return ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:31, top_indices]
        stft_r_sel = stft_r[:31, top_indices]

        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
       
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            ipd = np.pad(ipd, ((0, 0), (0, pad_width)), mode='constant')
        

        return ipd     
class DOADataset_10_ipd_F257_1k(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 10))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        ipd = ipd.astype(np.float32)
        label = np.int64(label)
        ipd = torch.from_numpy(ipd).unsqueeze(0)  # [1, F, T]
        return ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=4096, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:257, top_indices]
        stft_r_sel = stft_r[:257, top_indices]

        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
       
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            ipd = np.pad(ipd, ((0, 0), (0, pad_width)), mode='constant')
        

        return ipd
class DOADataset_10_ipd_F257_1k_h_w(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 10))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        ipd = ipd.astype(np.float32)
        label = np.int64(label)
        ipd = torch.from_numpy(ipd).unsqueeze(0)  # [1, F, T]
        return ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=4096, hop_length=4096, win_length=1024, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:257, top_indices]
        stft_r_sel = stft_r[:257, top_indices]

        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
       
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            ipd = np.pad(ipd, ((0, 0), (0, pad_width)), mode='constant')
        

        return ipd     
        
class DOADataset_10_ipd_F257_1k_h_w_2(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 10))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        ipd = ipd.astype(np.float32)
        label = np.int64(label)
        ipd = torch.from_numpy(ipd).unsqueeze(0)  # [1, F, T]
        return ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=4096, hop_length=1024, win_length=4096, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:257, top_indices]
        stft_r_sel = stft_r[:257, top_indices]

        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
       
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            ipd = np.pad(ipd, ((0, 0), (0, pad_width)), mode='constant')
        

        return ipd       
class DOADataset_15_ipd(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 15))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        ipd = ipd.astype(np.float32)
        label = np.int64(label)
        ipd = torch.from_numpy(ipd).unsqueeze(0)  # [1, F, T]
        return ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]

        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
       
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            ipd = np.pad(ipd, ((0, 0), (0, pad_width)), mode='constant')
        

        return ipd        
class DOADataset_20_ipd(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 20))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        ipd = ipd.astype(np.float32)
        label = np.int64(label)
        ipd = torch.from_numpy(ipd).unsqueeze(0)  # [1, F, T]
        return ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]

        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
       
        if len(top_indices) < self.top_T:
            pad_width = self.top_T - len(top_indices)
            ipd = np.pad(ipd, ((0, 0), (0, pad_width)), mode='constant')
        

        return ipd


class DOADataset_5_stft_ipd(Dataset):
    def __init__(self, jsonl_path, audio_root, top_T=128, max_angle=180):
        with open(jsonl_path, "r") as f:
            self.items = [json.loads(line) for line in f.readlines()]
        self.audio_root = audio_root
        self.top_T = top_T
        self.max_angle = max_angle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_name = item["audio_name"]
        wav_path = os.path.join(self.audio_root, audio_name + ".wav")
        wav_path = os.path.normpath(wav_path)

        angle = item["start_angle"]
        angle = float(angle[0]) if isinstance(angle, list) else float(angle)
        angle = angle % 360
        if angle > self.max_angle:
            angle = 360 - angle
        label = int(round(angle / 5))

        wav, sr = sf.read(wav_path)
        if wav.ndim == 1 or wav.shape[1] != 2:
            raise ValueError("Expected stereo audio.")

        stft_cat_ipd = self.compute_ipd_ild(wav[:sr * 10,].T, sr, wav_name=audio_name)

        stft_cat_ipd = stft_cat_ipd.astype(np.float32)

        label = np.int64(label)
        stft_cat_ipd = torch.from_numpy(stft_cat_ipd)  # [5, F, T]

        return stft_cat_ipd, label

    def compute_ipd_ild(self, wav, sr, n_fft=512, hop_length=160, win_length=400, wav_name=None):
        stft_l = librosa.stft(wav[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_r = librosa.stft(wav[1], n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        energy = np.mean(np.abs(stft_l) ** 2 + np.abs(stft_r) ** 2, axis=0)

        if len(energy) >= self.top_T:
            top_indices = np.argsort(energy)[-self.top_T:]
            top_indices = np.sort(top_indices)
        else:
            top_indices = np.arange(len(energy))

        stft_l_sel = stft_l[:, top_indices]
        stft_r_sel = stft_r[:, top_indices]
        stft = np.stack([stft_l_sel,stft_r_sel],axis=0)
        ipd = np.angle(stft_l_sel / (stft_r_sel + 1e-8))
        stft_cat_ipd = np.concatenate([np.real(stft),np.imag(stft),ipd[np.newaxis,:,:]],axis=0)

        return stft_cat_ipd
