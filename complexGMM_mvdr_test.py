# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:49:02 2019

@author: a-kojima
"""
import copy
import pickle
from pprint import pprint

import numpy as np
# from memory_profiler import profile
from scipy.io import wavfile as wf
import os
import re

from beamformer import complexGMM_mvdr as cgmm
import sys
from beamformer import util
from beamformer import minimum_variance_distortioless_response as mvdr
import time


class MyTimer:
    def __init__(self):
        self.startTime = {}

    def start(self, label = None):
        self.startTime[label] = time.clock()

    def stopPrint(self, label = None):
        duration = time.clock() - self.startTime[label]
        msg = 'TIME (%s): %.2f' % (label, duration)
        print(msg, file=sys.stderr)


# @profile
def multi_channel_read(list, path):  # list: file_dict[key]
    """

    :param list: file_dict[key] is a list, which is names of the same audio with 4 channels
    :param path: the file dir
    :return: a list in which stores the chunks from file reading. chunks are len(data_chunk) * 4 channels
    """
    if path != '':
        for z in range(len(list)):
            list[z] = path + '/' + list[z]

    # wav, _ = sf.read(list[0], dtype='float32')
    _, wav = wf.read(list[0])
    wav_multi = np.zeros((len(wav), 4), dtype=np.float16)  # float32 takes too much memory
    wav_multi[:, 0] = wav
    os.system("echo column 1 done")

    _, wav1 = wf.read(list[1])
    os.system("echo wav 2 done")

    wav_multi[:, 1] = wav1
    os.system("echo column 2 done")

    _, wav2 = wf.read(list[2])
    os.system("echo wav 3 done")
    wav_multi[:, 2] = wav2
    os.system("echo column 3 done")

    _, wav3 = wf.read(list[3])
    os.system("echo wav 4 done")

    wav_multi[:, 3] = wav3
    os.system("echo read done")

    # wav_multi_sep = np.split(wav_multi, 2)
    return wav_multi


def noise_handling(matrix):
    noise_1d = (matrix[:, 0] + matrix[:, 1] + matrix[:, 2] + matrix[:, 3])/4
    return noise_1d


def file_dict(input_arrays):
    """

    :param input_arrays: the channels_4 file in run_MVDR.sh (same with the one in run_beamformit.sh)
    :return: a dictionary with structure: {'S01_U01':'S01_U01 files with all channels',
                                           'S01_U02': 'S01_U01 files with all channels', ...}
    """
    file_dic = {}
    with open(input_arrays, 'r') as f:
        file_group = f.read().split("\n")
    f.close()
    if '' in file_group:
        file_group.remove('')

    for file in file_group:
        a = file.split()
        file_dic[a.pop(0)] = a

    return file_dic


def name2vec(name_string):
    """
    read name to a vector contains information may of use
    :param name_string: a name string formatted S*_U*.CH*_$noise/speech_$index
    :return: a list formatted [S*, U*, CH*, $noise/speech, $index]
    """
    name = re.sub('\W', ' ', name_string)
    name = re.sub('_', ' ', name).split(' ')
    if 'wav' in name:
        name.remove('wav')

    return name


# @profile
def do_cgmm_mvdr(audio):
    """
    Doing the cgmm_mvdr algorithm
    :return: no return
    """
    oo = MyTimer()
    oo.start("all")
    oo.start("init")
    cgmm_beamformer = cgmm.complexGMM_mvdr(SAMPLING_FREQUENCY, FFT_LENGTH, FFT_SHIFT, NUMBER_EM_ITERATION,
                                           MIN_SEGMENT_DUR)

    complex_spectrum_audio, _ = util.get_3dim_spectrum_from_data(audio,
                                                                 cgmm_beamformer.fft_length,
                                                                 cgmm_beamformer.fft_shift,
                                                                 cgmm_beamformer.fft_length)
    number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum_audio)

    R_noise = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
    R_noisy = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
    for f in range(0, number_of_bins):
        for t in range(0, number_of_frames):
            h = np.multiply.outer(complex_spectrum_audio[:, t, f], np.conj(complex_spectrum_audio[:, t, f]).T)
            R_noisy[:, :, f] = R_noisy[:, :, f] + h
        R_noisy[:, :, f] = R_noisy[:, :, f] / number_of_frames
        R_noise[:, :, f] = np.eye(number_of_channels, number_of_channels, dtype=np.complex64)
    R_xn = copy.deepcopy(R_noisy)
    oo.stopPrint("init")

    oo.start("em")
    frame = 0
    R_n = np.zeros((number_of_channels, number_of_channels, number_of_bins), dtype=np.complex64)
    # R_noise = np.zeros((4,4,129), dtype=np.complex64)
    # R_noisy = np.zeros((4,4,129), dtype=np.complex64)
    # for i in range(len(multi_channels_data)):
    for i in range(5):
        oo.start("chunk " + str(i))
        os.system("echo ---- chunk " + str(i + 1) + ' ----')
        R_noise, R_noisy, R_n, nf = cgmm_beamformer.get_spatial_correlation_matrix(
            multi_channels_data[i], R_noise, R_noisy, R_n)
        frame += nf
        oo.stopPrint("chunk " + str(i))

    oo.stopPrint("em")
    oo.start("mask")

    R_x = R_xn - R_n
    oo.stopPrint("mask")

    f = open('R_n.pkl', 'wb')
    pickle.dump(R_n, f)
    f.close()
    f = open('R_x.pkl', 'wb')
    pickle.dump(R_x, f)
    f.close()

    os.system("echo mask estimation done")
    oo.start("bmf")
    beamformer, steering_vector = cgmm_beamformer.get_mvdr_beamformer(R_x, R_n)
    os.system("echo bmf done")

    enhanced_speech = cgmm_beamformer.apply_beamformer(beamformer, complex_spectrum_audio)
    os.system("echo enhan done")

    # sf.write(ENHANCED_PATH + '/' + naming(char_list),
    # enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, SAMPLING_FREQUENCY)
    outname = '_'.join(name2vec(inputli[0])[:1])
    wf.write(ENHANCED_PATH + '/' + outname + ".wav",
                 SAMPLING_FREQUENCY, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65)
    os.system("echo all done")
    oo.stopPrint("bmf")
    oo.stopPrint("all")
    # if IS_MASK_PLOT:
    #     pl.figure()
    #     pl.subplot(2, 1, 1)
    #     pl.imshow(np.real(noise_mask).T, aspect='auto', origin='lower', cmap='hot')
    #     pl.title('noise mask')
    #     pl.subplot(2, 1, 2)
    #     pl.imshow(np.real(speech_mask).T, aspect='auto', origin='lower', cmap='hot')
    #     pl.title('speech mask')
    #     pl.show()


# def do_mvdr():
#     """
#     Doing the simple mvdr algorithm
#     :return: no return
#     """
#
#     for i in range(len(multi_channels_data)):
#         complex_spectrum, _ = util.get_3dim_spectrum_from_data(multi_channels_data[i], FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
#
#         mvdr_beamformer = mvdr.minimum_variance_distortioless_response(MIC_ANGLE_VECTOR, MIC_DIAMETER,
#                                                                        sampling_frequency=SAMPLING_FREQUENCY,
#                                                                        fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)
#
#         steering_vector = mvdr_beamformer.get_sterring_vector(LOOK_DIRECTION)
#
#         spatial_correlation_matrix = mvdr_beamformer.get_spatial_correlation_matrix(multi_channels_data[i])
#
#         beamformer = mvdr_beamformer.get_mvdr_beamformer(steering_vector, spatial_correlation_matrix)
#
#         enhanced_speech.extend(mvdr_beamformer.apply_beamformer(beamformer, complex_spectrum))
#
#     wf.write(ENHANCED_PATH + '/' + key + ".wav", SAMPLING_FREQUENCY, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65)


if __name__ == '__main__':
    t = MyTimer()

    '''
    parameters for beamforming
    '''
    SAMPLING_FREQUENCY = 16000
    FFT_LENGTH = 256
    FFT_SHIFT = 128
    NUMBER_EM_ITERATION = 2
    MIN_SEGMENT_DUR = 2

    '''
    args from .sh files
    '''
    # INPUT_ARRAYS = "./../../channels_4"
    # SOURCE_PATH = "./../../fsdownload"
    # CHUNK_PATH = "./../../audio_chunk/dev"
    # ENHANCED_PATH = "./../../"
    # LINE = 2

    SOURCE_PATH = "/fastdata/acs18zx/CHiME5/audio/dev"
    CHUNK_PATH = "/fastdata/acs18zx/CHiME5/audio_chunk/dev"
    ENHANCED_PATH = "."
    LINE = 2
    # INPUT_ARRAYS = sys.argv[1]
    # SOURCE_PATH = sys.argv[2]
    # ENHANCED_PATH = sys.argv[3]
    # LINE = sys.argv[4]

    '''
    parameters for simple mvdr
    '''
    MIC_ANGLE_VECTOR = np.array([0, 90, 180, 270])
    LOOK_DIRECTION = 0
    MIC_DIAMETER = 0.1

    '''
    get file dictionary (see comment for file_dict())
    get current target files (i.e. S01_U01_CH*.wav)
    '''
    # inp = file_dict(INPUT_ARRAYS)
    # a = list(inp.keys())
    # key = a[int(LINE) - 1]  # i.e. S01_U01
    inputli = ['S02_U02.CH1.wav', 'S02_U02.CH2.wav', 'S02_U02.CH3.wav', 'S02_U02.CH4.wav']
    data_name_list = []
    data_dir_list = []
    for n in inputli:
        data_name_list.append(name2vec(n))
        data_dir_list.append(CHUNK_PATH + '/' + str(name2vec(n)[0]) + '/' +
                             str(name2vec(n)[0] + '_' + name2vec(n)[1]) + '.' + name2vec(n)[2] + '/' + '_speech_')
    '''
    prepare data for bmf (see comment for multi_channel_read())
    '''
    audio = multi_channel_read(inputli, SOURCE_PATH)
    multi_channels_data = []
    data_list = []
    for dir in data_dir_list:
        ch = []
        for files in os.walk(dir):
            for i in range(len(files[2])):
                ch.append(dir + '/' + files[2][i])
        data_list.append(ch)
    input_data_list = []

    for i in range(len(data_list[0])):
        input_data_list.append([data_list[0][i], data_list[1][i], data_list[2][i], data_list[3][i]])
    for i in range(len(input_data_list)):
        data = multi_channel_read(input_data_list[i], '')
        multi_channels_data.append(data)

    # if len(multi_channels_data[0]) + len(multi_channels_data[1]) == len(audio):
    #     print("jdsnbvfhshvdbsuhs")
    #     print("jdsnbvfhshvdbsuhs")
    #     print("jdsnbvfhshvdbsuhs")
    #     print("jdsnbvfhshvdbsuhs")
    #     print("jdsnbvfhshvdbsuhs")
    #     print("jdsnbvfhshvdbsuhs")
    os.system("echo data reading done")

    # IS_MASK_PLOT = False

    '''
    run algorithm
    '''
    do_cgmm_mvdr(audio)
    # do_mvdr()

