import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from scipy.io import wavfile
from scipy import signal
import soundfile

def compute_dB(waveform):
    """
    Args:
        x (numpy.array): Input waveform (#length).
    Returns:
        numpy.array: Output array (#length).
    """
    val = max(0.0, np.mean(np.power(waveform, 2)))
    dB = 10*np.log10(val+1e-4)
    return dB

class WavAugment(object):
    def __init__(self, noise_csv_path="data/noise.csv", rir_csv_path="data/rir.csv"):
        self.noise_paths = pd.read_csv(noise_csv_path)["utt_paths"].values
        self.noise_names = pd.read_csv(noise_csv_path)["speaker_name"].values
        self.rir_paths = pd.read_csv(rir_csv_path)["utt_paths"].values

    def __call__(self, waveform):
        idx = np.random.randint(0, 10)
        if idx == 0:
            waveform = self.add_gaussian_noise(waveform)
            waveform = self.add_real_noise(waveform)

        if idx == 1 or idx == 2 or idx == 3:
            waveform = self.add_real_noise(waveform)

        if idx == 4 or idx == 5 or idx == 6:
            waveform = self.reverberate(waveform)

        if idx == 7:
            waveform = self.change_volum(waveform)
            waveform = self.reverberate(waveform)

        if idx == 8:
            waveform = self.change_volum(waveform)
            waveform = self.add_real_noise(waveform)

        if idx == 9:
            waveform = self.add_gaussian_noise(waveform)
            waveform = self.reverberate(waveform)

        return waveform

    def add_gaussian_noise(self, waveform):
        """
        Args:
            x (numpy.array): Input waveform array (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        snr = np.random.uniform(low=10, high=25)
        clean_dB = compute_dB(waveform)
        noise = np.random.randn(len(waveform))
        noise_dB = compute_dB(noise)
        noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        waveform = (waveform + noise)
        return waveform

    def change_volum(self, waveform):
        """
        Args:
            x (numpy.array): Input waveform array (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        volum = np.random.uniform(low=0.5, high=2)
        waveform = waveform * volum
        return waveform

    def add_real_noise(self, waveform):
        """
        Args:
            x (numpy.array): Input length (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        clean_dB = compute_dB(waveform)

        idx = np.random.randint(0, len(self.noise_paths))
        sample_rate, noise = wavfile.read(self.noise_paths[idx])
        noise = noise.astype(np.float64)

        snr = np.random.uniform(15, 25)

        noise_length = len(noise)
        audio_length = len(waveform)

        if audio_length >= noise_length:
            shortage = audio_length - noise_length
            noise = np.pad(noise, (0, shortage), 'wrap')
        else:
            start = np.random.randint(0, (noise_length-audio_length))
            noise = noise[start:start+audio_length]

        # noise_dB = compute_dB(noise)
        # noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        # waveform = (waveform + noise)
        # return waveform
        # RMS 기반 스케일링
        clean_rms = np.sqrt(np.mean(waveform**2))
        noise_rms = np.sqrt(np.mean(noise**2))
        desired_noise_rms = clean_rms / (10**(snr/20))
        
        noise = noise * (desired_noise_rms / (noise_rms + 1e-6))

        # 노이즈 추가
        noisy_waveform = waveform + noise

        # 클리핑 방지
        max_val = np.max(np.abs(noisy_waveform))
        if max_val > 1:
            noisy_waveform = noisy_waveform / max_val

        return noisy_waveform
        

    def reverberate(self, waveform):
        """
        Args:
            x (numpy.array): Input length (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        audio_length = len(waveform)
        idx = np.random.randint(0, len(self.rir_paths))

        path = self.rir_paths[idx]
        rir, sample_rate = soundfile.read(path)
        
        if len(rir.shape) > 1:
            rir = rir[:, 0]
        if len(waveform.shape) > 1:
            waveform = waveform[:, 0]
                
        # rir = rir/np.sqrt(np.sum(rir**2))
        rir = rir - np.mean(rir)
        rir = rir / (np.max(np.abs(rir)) + 1e-6)  # RMS 기반 정규화
                
        # waveform과 RIR 길이 맞추기
        if len(rir) > audio_length:
            rir = rir[:audio_length]
        elif len(rir) < audio_length:
            rir = np.pad(rir, (0, audio_length - len(rir)), 'constant')               

        reverbed = signal.convolve(waveform, rir, mode='full')
        reverbed = reverbed[:audio_length]
        
        output_scale = np.clip(np.std(waveform) / (np.std(reverbed) + 1e-6), 0.1, 0.5)
        reverbed = reverbed * output_scale
        reverbed = reverbed / (np.max(np.abs(reverbed)) + 1e-6)
        # return waveform[:audio_length]
        return reverbed
