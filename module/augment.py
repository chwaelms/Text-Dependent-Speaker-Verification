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
    


# if __name__ == "__main__":
#     aug = WavAugment()
#     # audio_file="/media/nvidia/673500ef-8e0c-403d-8b57-fbb7ce138a2c/mfa_conformer-master_Transfer/module/SV0001_2_02_F0661.wav"
#     audio_file="/media/lim/fd385c31-b3b5-49a0-bf16-7c7bdede52b9/home/lim/cofla/Dataset/Dataset_send/dependent/SLR58/train/SPEECHDATA/wav/SV0001/SV0001_2_02_F0661.wav"
#     sample_rate, waveform = wavfile.read(audio_file)
#     waveform = waveform.astype(np.float64)

#     gaussian_noise_wave = aug.add_gaussian_noise(waveform)
#     print(gaussian_noise_wave.dtype)
#     wavfile.write("gaussian_noise_wave.wav", 16000, gaussian_noise_wave.astype(np.int16))

#     real_noise_wave = aug.add_real_noise(waveform)
#     print(real_noise_wave.dtype)
#     wavfile.write("real_noise_wave.wav", 16000, real_noise_wave.astype(np.int16))

#     change_volum_wave = aug.change_volum(waveform)
#     print(change_volum_wave.dtype)
#     wavfile.write("change_volum_wave.wav", 16000, change_volum_wave.astype(np.int16))

#     reverberate_wave = aug.reverberate(waveform)
#     print(reverberate_wave.dtype)
#     wavfile.write("reverberate_wave.wav", 16000, reverberate_wave.astype(np.int16))

#     reverb_noise_wave = aug.reverberate(waveform)
#     reverb_noise_wave = aug.add_real_noise(waveform)
#     print(reverb_noise_wave.dtype)
#     wavfile.write("reverb_noise_wave.wav", 16000, reverb_noise_wave.astype(np.int16))

#     noise_reverb_wave = aug.add_real_noise(waveform)
#     noise_reverb_wave = aug.reverberate(waveform)
#     print(noise_reverb_wave.dtype)
#     wavfile.write("noise_reverb_wave.wav", 16000, reverb_noise_wave.astype(np.int16))

#     a = torch.FloatTensor(noise_reverb_wave)
#     print(a.dtype)

# if __name__ == "__main__":
#     aug = WavAugment()
#     # Load audio file
#     audio_file="/media/lim/fd385c31-b3b5-49a0-bf16-7c7bdede52b9/home/lim/cofla/Dataset/Dataset_send/dependent/SLR58/train/SPEECHDATA/wav/SV0001/SV0001_2_02_F0661.wav"
#     sample_rate, waveform = wavfile.read(audio_file)
#     waveform = waveform.astype(np.float64)
    
#     # Normalize original audio
#     waveform = waveform / np.max(np.abs(waveform))
    
#     # Apply EXTREME augmentation techniques for dramatic effect
#     # 1. Gaussian noise - MUCH stronger noise
#     np.random.seed(42)  # For reproducibility
#     clean_dB = compute_dB(waveform)
#     noise = np.random.randn(len(waveform))
#     noise_dB = compute_dB(noise)
#     snr = 0  # Very low SNR for extreme noise effect (0dB = noise as loud as signal)
#     noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
#     gaussian_noise_wave = waveform + noise
#     # Normalize to prevent clipping
#     gaussian_noise_wave = gaussian_noise_wave / max(1.0, np.max(np.abs(gaussian_noise_wave)))
    
#     # 2. Volume change - EXTREME change
#     volume_factor = 3.0  # Much larger volume change
#     change_volum_wave = waveform * volume_factor
#     # No normalization to show the extreme effect
    
#     # 3. Real noise - MUCH stronger with multiple layers
#     # First layer of real noise with original function
#     real_noise_wave = aug.add_real_noise(waveform.copy())
    
#     # Second layer of real noise - apply again (different random noise will be selected)
#     real_noise_wave = aug.add_real_noise(real_noise_wave)
    
#     # Third layer of real noise - and again!
#     real_noise_wave = aug.add_real_noise(real_noise_wave)
    
#     # Apply high-pass filter to emphasize high-frequency noise (makes it more noticeable)
#     from scipy import signal
#     b, a = signal.butter(4, 0.1, 'highpass')
#     high_freq_noise = signal.filtfilt(b, a, real_noise_wave)
    
#     # Add boosted high-frequency noise back to the signal
#     real_noise_wave = real_noise_wave + 0.5 * high_freq_noise
    
#     # Final normalization to prevent clipping
#     real_noise_wave = real_noise_wave / max(1.0, np.max(np.abs(real_noise_wave)))
    
#     # 4. Reverberation - extreme echo
#     reverberate_wave = aug.reverberate(waveform.copy())
#     # Double reverberation for extreme effect
#     reverberate_wave = aug.reverberate(reverberate_wave)
#     # Normalize
#     reverberate_wave = reverberate_wave / max(1.0, np.max(np.abs(reverberate_wave)))
    
#     # Import required libraries
#     import matplotlib.pyplot as plt
#     import librosa
#     import librosa.display
    
#     # Define the blue color similar to the image
#     WAVEFORM_COLOR = '#1F77B4'  # Dodger blue
    
#     # Create a figure with side-by-side comparisons
#     plt.figure(figsize=(15, 20))
    
#     # Functions for clearer plotting
#     def plot_waveform(ax, signal, title):
#         times = np.arange(len(signal)) / sample_rate
#         ax.plot(times, signal, color=WAVEFORM_COLOR, linewidth=1)
#         ax.set_title(title, fontsize=14)
#         ax.set_ylabel('Amplitude')
#         ax.set_ylim([-1.2, 1.2])
#         ax.grid(True, alpha=0.3)
    
#     # 1. Original vs Gaussian Noise
#     plt.subplot(5, 1, 1)
#     times = np.arange(len(waveform)) / sample_rate
#     plt.plot(times, gaussian_noise_wave, color=WAVEFORM_COLOR, label='Gaussian Noise')
#     plt.title('Gaussian Noise', fontsize=14)
#     plt.ylabel('Amplitude')
#     plt.ylim([-1.2, 1.2])
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # 2. Original vs Volume Change
#     plt.subplot(5, 1, 2)
#     plt.plot(times, change_volum_wave, color=WAVEFORM_COLOR, label='Volume Change')
#     plt.title('Volume Change', fontsize=14)
#     plt.ylabel('Amplitude')
#     plt.ylim([-1.2, 1.2])  # Wider range for volume change
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # 3. Original vs Real Noise
#     plt.subplot(5, 1, 3)
#     plt.plot(times, real_noise_wave, color=WAVEFORM_COLOR, label='Real Noise')
#     plt.title('Real Noise', fontsize=14)
#     plt.ylabel('Amplitude')
#     plt.ylim([-1.2, 1.2])
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # 4. Original vs Reverberation
#     plt.subplot(5, 1, 4)
#     plt.plot(times, reverberate_wave, color=WAVEFORM_COLOR, label='Reverberation')
#     plt.title('Reverberation', fontsize=14)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.ylim([-1.2, 1.2])
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     plt.subplot(5, 1, 5)
#     plt.plot(times, waveform, color=WAVEFORM_COLOR, label='Original')
#     plt.title('Original', fontsize=14)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.ylim([-1.2, 1.2])
#     plt.grid(True, alpha=0.3)
#     plt.legend()
        

#     plt.tight_layout()
#     plt.savefig('extreme_augmentation_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # Create zoomed in view of a short segment
#     plt.figure(figsize=(15, 20))
    
#     # Select a 0.2-second segment where there's clear activity
#     # First, find a high-energy section
#     window_size = int(0.1 * sample_rate)  # 100ms window
#     energy = np.array([np.sum(waveform[i:i+window_size]**2) for i in range(0, len(waveform)-window_size, window_size)])
#     high_energy_window = np.argmax(energy) * window_size
    
#     segment_start = high_energy_window
#     segment_end = high_energy_window + int(0.1 * sample_rate)  # 200ms segment
    
#     # Extract segments
#     waveform_segment = waveform[segment_start:segment_end]
#     gaussian_segment = gaussian_noise_wave[segment_start:segment_end]
#     volume_segment = change_volum_wave[segment_start:segment_end]
#     noise_segment = real_noise_wave[segment_start:segment_end]
#     reverb_segment = reverberate_wave[segment_start:segment_end]
    
#     # Create time axis for the segment
#     segment_times = np.arange(len(waveform_segment)) / sample_rate + (segment_start/sample_rate)
    
#     # 1. Original vs Gaussian Noise (zoomed)
#     plt.subplot(4, 1, 1)
#     plt.plot(segment_times, waveform_segment, color='gray', alpha=0.7, label='Original')
#     plt.plot(segment_times, gaussian_segment, color=WAVEFORM_COLOR, label='Gaussian Noise')
#     plt.title('Original vs Gaussian Noise (200ms Detail)', fontsize=14)
#     plt.ylabel('Amplitude')
#     plt.ylim([-5.5, 5.5])
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # 2. Original vs Volume Change (zoomed)
#     plt.subplot(4, 1, 2)
#     plt.plot(segment_times, waveform_segment, color='gray', alpha=0.7, label='Original')
#     plt.plot(segment_times, volume_segment, color=WAVEFORM_COLOR, label='Volume Change')
#     plt.title('Original vs Volume Change (200ms Detail)', fontsize=14)
#     plt.ylabel('Amplitude')
#     plt.ylim([-5.5, 5.5])  # Wider range for volume change
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # 3. Original vs Real Noise (zoomed)
#     plt.subplot(4, 1, 3)
#     plt.plot(segment_times, waveform_segment, color='gray', alpha=0.7, label='Original')
#     plt.plot(segment_times, noise_segment, color=WAVEFORM_COLOR, label='Real Noise')
#     plt.title('Original vs Real Noise (200ms Detail)', fontsize=14)
#     plt.ylabel('Amplitude')
#     plt.ylim([-5.5, 5.5])
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # 4. Original vs Reverberation (zoomed)
#     plt.subplot(4, 1, 4)
#     plt.plot(segment_times, waveform_segment, color='gray', alpha=0.7, label='Original')
#     plt.plot(segment_times, reverb_segment, color=WAVEFORM_COLOR, label='Reverberation')
#     plt.title('Original vs Reverberation (200ms Detail)', fontsize=14)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.ylim([-5.5, 5.5])
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('extreme_augmentation_detail.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # Create spectrogram comparison
#     plt.figure(figsize=(20, 15))
    
#     # Function to plot spectrogram differences
#     def plot_spec_comparison(ax, original, modified, title):
#         D_diff = librosa.amplitude_to_db(
#             np.abs(librosa.stft(modified)) - np.abs(librosa.stft(original)),
#             ref=np.max
#         )
#         img = librosa.display.specshow(D_diff, sr=sample_rate, x_axis='time', y_axis='log', ax=ax)
#         plt.colorbar(img, ax=ax, format='%+2.0f dB')
#         ax.set_title(f'{title} - Difference from Original', fontsize=14)
    
#     # 1. Gaussian noise spectrogram difference
#     plt.subplot(2, 2, 1)
#     plot_spec_comparison(plt.gca(), waveform, gaussian_noise_wave, 'Gaussian Noise')
    
#     # 2. Volume change spectrogram difference
#     plt.subplot(2, 2, 2)
#     plot_spec_comparison(plt.gca(), waveform, change_volum_wave, 'Volume Change')
    
#     # 3. Real noise spectrogram difference
#     plt.subplot(2, 2, 3)
#     plot_spec_comparison(plt.gca(), waveform, real_noise_wave, 'Real Noise')
    
#     # 4. Reverberation spectrogram difference
#     plt.subplot(2, 2, 4)
#     plot_spec_comparison(plt.gca(), waveform, reverberate_wave, 'Reverberation')
#     plt.xlabel('Time (s)')
    
#     plt.tight_layout()
#     plt.savefig('extreme_augmentation_spectrograms.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
#     # Save audio files
#     wavfile.write("original.wav", sample_rate, (waveform * 32767).astype(np.int16))
#     wavfile.write("extreme_gaussian_noise.wav", sample_rate, (gaussian_noise_wave * 32767).astype(np.int16))
#     wavfile.write("extreme_volume_change.wav", sample_rate, (np.clip(change_volum_wave, -1, 1) * 32767).astype(np.int16))
#     wavfile.write("extreme_real_noise.wav", sample_rate, (real_noise_wave * 32767).astype(np.int16))
#     wavfile.write("extreme_reverb.wav", sample_rate, (reverberate_wave * 32767).astype(np.int16))
    
#     print("Visualization complete! Check the extreme augmentation comparison images.")