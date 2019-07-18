# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:49:02 2019

@author: a-kojima
"""
import numpy as np
# import soundfile as sf
from scipy.io import wavfile as wf
from beamformer import complexGMM_mvdr as cgmm
import sys
import os

SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 512
FFT_SHIFT = 128
NUMBER_EM_ITERATION = 20
MIN_SEGMENT_DUR = 2
# INPUT_ARRAYS = "./../../channels_4"
# SOURCE_PATH = "./../../sample_data/dev"
# ENHANCED_PATH = "./../../"
INPUT_ARRAYS = sys.argv[1]
SOURCE_PATH = sys.argv[2]
ENHANCED_PATH = sys.argv[3]

# IS_MASK_PLOT = False


def file_dict(input_arrays):

    file_dic = {}
    with open(input_arrays, 'r') as f:
        file_group = f.read().split("\n")
    f.close()
    file_group.remove('')

    for file in file_group:
        a = file.split()
        file_dic[a.pop(0)] = a

    return file_dic


# def separate_into_4s(list):
#     data_store = []
#     vehicle = []
#     while len(list) != 0:
#         for i in range(4):
#             vehicle.append(list.pop(0))
#         data_store.append(vehicle)
#         vehicle = []
#     return data_store
#
#
# def naming(c_list):
#     name = ''
#     for i in range(7):
#         name += c_list[i]
#     name += '.wav'
#     return name


def multi_channel_read(list, path):  # list: file_dict[key]
    for i in range(len(list)):
        list[i] = path + '/' + list[i]

    # wav, _ = sf.read(list[0], dtype='float32')
    _, wav = wf.read(list[0])
    wav_multi = np.zeros((len(wav), 4), dtype=np.float32)
    wav_multi[:, 0] = wav
    for i in range(4):
        # wav_multi[:, i] = sf.read(list[i])[0]
        _, wav_multi[:, i] = wf.read(list[i])
    print("read done")
    return wav_multi


inp = file_dict(INPUT_ARRAYS)


for key in inp:
    print(1)
    multi_channels_data = multi_channel_read(inp[key], SOURCE_PATH)
    print(2)
    cgmm_beamformer = cgmm.complexGMM_mvdr(SAMPLING_FREQUENCY, FFT_LENGTH, FFT_SHIFT, NUMBER_EM_ITERATION, MIN_SEGMENT_DUR)
    print(3)
    complex_spectrum, R_x, R_n, noise_mask, speech_mask = cgmm_beamformer.get_spatial_correlation_matrix(multi_channels_data)
    print(4)
    beamformer, steering_vector = cgmm_beamformer.get_mvdr_beamformer(R_x, R_n)
    print(5)
    enhanced_speech = cgmm_beamformer.apply_beamformer(beamformer, complex_spectrum)
    print(6)
    # sf.write(ENHANCED_PATH + '/' + naming(char_list), enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, SAMPLING_FREQUENCY)
    wf.write(ENHANCED_PATH + '/' + key, SAMPLING_FREQUENCY, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65)
    print(7)
# if IS_MASK_PLOT:
#     pl.figure()
#     pl.subplot(2, 1, 1)
#     pl.imshow(np.real(noise_mask).T, aspect='auto', origin='lower', cmap='hot')
#     pl.title('noise mask')
#     pl.subplot(2, 1, 2)
#     pl.imshow(np.real(speech_mask).T, aspect='auto', origin='lower', cmap='hot')
#     pl.title('speech mask')
#     pl.show()
