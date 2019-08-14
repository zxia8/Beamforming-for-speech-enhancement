# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:09:47 2018

@author: a-kojima

"""
import numpy as np
from scipy.fftpack import fft, ifft
import numpy.matlib as npm
from scipy import signal as sg


def get_3dim_spectrum_from_data(wav_data, frame, shift, fftl):
    """
    dump_wav : channel_size * speech_size (2dim)
    """
    len_sample, len_channel_vec = len(wav_data[:, 0]), 4
    dump_wav = wav_data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    window = sg.hanning(fftl + 1, 'periodic')[: - 1]
    multi_window = npm.repmat(window, len_channel_vec, 1)    
    st = 0
    ed = frame
    number_of_frame = np.int((len_sample - frame) / shift)
    spectrums = np.zeros((len_channel_vec, number_of_frame, np.int(fftl / 2) + 1), dtype=np.complex64)
    for ii in range(0, number_of_frame):       
        multi_signal_spectrum = fft(dump_wav[:, st:ed],
                                    n=fftl, axis=1)[:, 0:np.int(fftl / 2) + 1]  # channel * number_of_bin
        spectrums[:, ii, :] = multi_signal_spectrum
        st = st + shift
        ed = ed + shift
    return spectrums, len_sample


def spec2wav(spectrogram, sampling_frequency, fftl, frame_len, shift_len):
    n_of_frame, fft_half = np.shape(spectrogram)    
    hanning = sg.hanning(fftl + 1, 'periodic')[: - 1]    
    cut_data = np.zeros(fftl, dtype=np.complex64)
    result = np.zeros(sampling_frequency * 4200 * 5, dtype=np.float32)
    start_point = 0
    end_point = start_point + frame_len
    for ii in range(0, n_of_frame):
        half_spec = spectrogram[ii, :]        
        cut_data[0:np.int(fftl / 2) + 1] = half_spec.T   
        cut_data[np.int(fftl / 2) + 1:] =  np.flip(np.conjugate(half_spec[1:np.int(fftl / 2)]), axis=0)
        cut_data2 = np.real(ifft(cut_data, n=fftl))        
        result[start_point:end_point] = result[start_point:end_point] + np.real(cut_data2 * hanning.T)
        start_point = start_point + shift_len
        end_point = end_point + shift_len
    return result[0:end_point - shift_len]



