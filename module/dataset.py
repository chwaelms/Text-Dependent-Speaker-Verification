import collections
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from .augment import WavAugment


def load_audio(filename, second=2):
    sample_rate, waveform = wavfile.read(filename)
    audio_length = waveform.shape[0]
    
    if second <= 0:
        return waveform.astype(np.float64).copy()

    length = np.int64(sample_rate * second)

    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap') # 오디오 짧을 경우 반복해서 채워짐
        waveform = waveform.astype(np.float64)
    else:
        start = np.int64(random.random()*(audio_length-length))
        waveform =  waveform[start:start+length].astype(np.float64)
    return waveform.copy()

def normalize_audio(waveform):
    return waveform / (np.max(np.abs(waveform))+1e-8)

class Train_Dataset(Dataset):
    def __init__(self, train_csv_path, second=3, pairs=True, aug=False, **kwargs):
        self.second = second
        self.pairs = pairs

        df = pd.read_csv(train_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values
        self.labels, self.paths = shuffle(self.labels, self.paths)
        self.aug = aug
        if aug:
            self.wav_aug = WavAugment()

        print("Train Dataset  {} speakers".format(len(set(self.labels))))
        print("Train Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform_1 = load_audio(self.paths[index], self.second)
        waveform_1 = normalize_audio(waveform_1)
        waveform_aug_1 = self.wav_aug(waveform_1.copy()) if self.aug else waveform_1.copy()
        
        if self.pairs == True:
            waveform_2 = load_audio(self.paths[index], self.second)
            waveform_2 = normalize_audio(waveform_2)
            waveform_aug_2 = self.wav_aug(waveform_2.copy()) if self.aug else waveform_2
            return torch.FloatTensor(waveform_1), torch.FloatTensor(waveform_aug_1), torch.FloatTensor(waveform_2), torch.FloatTensor(waveform_aug_2), self.labels[index]
        else:
            return torch.FloatTensor(waveform_1), torch.FloatTensor(waveform_aug_1), self.labels[index]
    
    def __len__(self):
        return len(self.paths)


class Semi_Dataset(Dataset):
    def __init__(self, label_csv_path, unlabel_csv_path, second=2, pairs=True, aug=False, **kwargs):
        self.second = second
        self.pairs = pairs

        df = pd.read_csv(label_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values

        self.aug = aug
        if aug:
            self.wav_aug = WavAugment()

        df = pd.read_csv(unlabel_csv_path)
        self.u_paths = df["utt_paths"].values
        self.u_paths_length = len(self.u_paths)

        if label_csv_path != unlabel_csv_path:
            self.labels, self.paths = shuffle(self.labels, self.paths)
            self.u_paths = shuffle(self.u_paths)

        # self.labels = self.labels[:self.u_paths_length]
        # self.paths = self.paths[:self.u_paths_length]
        print("Semi Dataset load {} speakers".format(len(set(self.labels))))
        print("Semi Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform_l = load_audio(self.paths[index], self.second)
        waveform_1 = normalize_audio(waveform_1)

        idx = np.random.randint(0, self.u_paths_length)
        waveform_u_1 = load_audio(self.u_paths[idx], self.second)
        
        waveform_aug_u_1 = self.wav_aug(waveform_u_1.copy()) if self.aug else waveform_u_1.copy()

        if self.pairs == False:
            return torch.FloatTensor(waveform_l), self.labels[index], torch.FloatTensor(waveform_u_1), torch.FloatTensor(waveform_aug_u_1)
        else:
            waveform_u_2 = load_audio(self.u_paths[idx], self.second)
            waveform_2 = normalize_audio(waveform_2)
            waveform_aug_u_2 = self.wav_aug(waveform_u_2.copy()) if self.aug else waveform_u_2
            return torch.FloatTensor(waveform_l), self.labels[index], torch.FloatTensor(waveform_u_1), torch.FloatTensor(waveform_aug_u_1), torch.FloatTensor(waveform_u_2), torch.FloatTensor(waveform_aug_u_2)

    def __len__(self):
        return len(self.paths)


class Evaluation_Dataset(Dataset):
    def __init__(self, paths, second=-1, apply_rir=False, rir_csv_path=None, num_aug=1, **kwargs):
        self.paths = paths
        self.second = second
        self.apply_rir = apply_rir
        self.num_aug = num_aug
        self.wav_augment = WavAugment(rir_csv_path) if apply_rir else None
        print("load {} utterance".format(len(self.paths)))

    def __getitem__(self, index):
        waveform = load_audio(self.paths[index], self.second)
        waveform = normalize_audio(waveform)

        aug_waveform = []
        if self.apply_rir and self.wav_augment:
            for _ in range(self.num_aug):
                # aug_waveform = self.wav_augment.reverberate(waveform.copy())
                aug_waveform.append(torch.FloatTensor(self.wav_augment.reverberate(waveform.copy())))
        else: aug_waveform = []
        
        return torch.FloatTensor(waveform), aug_waveform, self.paths[index]

    def __len__(self):
        return len(self.paths)


import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
