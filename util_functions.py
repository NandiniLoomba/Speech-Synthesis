import numpy as np
import librosa
import os, copy
from scipy import signal
import params as hp
import torch as t

def generate_spectrograms(fpath):
  
  
    y, sr = librosa.load(fpath, sr=hp.sr)

    y, _ = librosa.effects.trim(y)

    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    mag = np.abs(linear) 

    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def generateWavsfromSpectrograms(mag):
   
    mag = mag.T

    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    mag = np.power(10.0, mag * 0.05)
    wav = griffin_lim_algo(mag**hp.power)

    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

def griffin_lim_algo(spectrogram):
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = spectrogram_inversion(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = spectrogram_inversion(X_best)
    y = np.real(X_t)

    return y

def spectrogram_inversion(spectrogram):
   
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def get_positional_table(d_pos_vec, n_position=1024):
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return t.from_numpy(position_enc).type(t.FloatTensor)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
 
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  

    if padding_idx is not None:
      
        sinusoid_table[padding_idx] = 0.

    return t.FloatTensor(sinusoid_table)

def guided_attention(N, T, g=0.2):
  
    W = np.zeros((N, T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N)) ** 2 / (2 * g * g))
    return W
