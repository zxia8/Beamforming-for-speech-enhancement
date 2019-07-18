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
# SOURCE_PATH = "./../../sample_data/dev"
# ENHANCED_PATH = "./../../"
SOURCE_PATH = sys.argv[1]
ENHANCED_PATH = sys.argv[2]
IS_MASK_PLOT = False


def file_list():
    path = SOURCE_PATH
    path_list = os.listdir(path)
    f_list = []
    p_list = []
    for filename in path_list:
        p_list.append(path + '/' + filename)
        f_list.append(filename)
    p_list.sort()
    f_list.sort()

    return p_list, f_list


def separate_into_4s(list):
    data_store = []
    vehicle = []
    while len(list) != 0:
        for i in range(4):
            vehicle.append(list.pop(0))
        data_store.append(vehicle)
        vehicle = []
    return data_store


def naming(c_list):
    name = ''
    for i in range(7):
        name += c_list[i]
    name += '.wav'
    return name


def multi_channel_read(list):
    # wav, _ = sf.read(list[0], dtype='float32')
    _, wav = wf.read(list[0])
    wav_multi = np.zeros((len(wav), 4), dtype=np.float32)
    wav_multi[:, 0] = wav
    for i in range(4):
        # wav_multi[:, i] = sf.read(list[i])[0]
        wav_multi[:, i] = wf.read(list[i])[1]
    return wav_multi


flist, name = file_list()

data = separate_into_4s(flist)
names = separate_into_4s(name)
print()
for i in range(len(data)):
    char_list = []
    for char in names[i][0]:
        char_list.append(char)
    if len(char_list) == 15:

        multi_channels_data = multi_channel_read(data[i])

        cgmm_beamformer = cgmm.complexGMM_mvdr(SAMPLING_FREQUENCY, FFT_LENGTH, FFT_SHIFT, NUMBER_EM_ITERATION, MIN_SEGMENT_DUR)

        complex_spectrum, R_x, R_n, noise_mask, speech_mask = cgmm_beamformer.get_spatial_correlation_matrix(multi_channels_data)

        beamformer, steering_vector = cgmm_beamformer.get_mvdr_beamformer(R_x, R_n)

        enhanced_speech = cgmm_beamformer.apply_beamformer(beamformer, complex_spectrum)

        # sf.write(ENHANCED_PATH + '/' + naming(char_list), enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, SAMPLING_FREQUENCY)
        wf.write(ENHANCED_PATH + '/' + naming(char_list), SAMPLING_FREQUENCY, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65)

# if IS_MASK_PLOT:
#     pl.figure()
#     pl.subplot(2, 1, 1)
#     pl.imshow(np.real(noise_mask).T, aspect='auto', origin='lower', cmap='hot')
#     pl.title('noise mask')
#     pl.subplot(2, 1, 2)
#     pl.imshow(np.real(speech_mask).T, aspect='auto', origin='lower', cmap='hot')
#     pl.title('speech mask')
#     pl.show()
