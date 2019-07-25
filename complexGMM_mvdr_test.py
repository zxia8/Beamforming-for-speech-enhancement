# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:49:02 2019

@author: a-kojima
"""
import numpy as np
# from memory_profiler import profile
from scipy.io import wavfile as wf
import os

from beamformer import complexGMM_mvdr as cgmm
import sys
from beamformer import util
from beamformer import minimum_variance_distortioless_response as mvdr


# @profile
def multi_channel_read(list, path):  # list: file_dict[key]
    """

    :param list: file_dict[key] is a list, which is same audio with 4 channels
    :param path: the file dir
    :return: a list in which stores the chunks from file reading. chunks are len(data_chunk) * 4 channels
    """
    for i in range(len(list)):
        list[i] = path + '/' + list[i]

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

    wav_multi = np.array_split(wav_multi, 43)
    return wav_multi


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


# @profile
def do_cgmm_mvdr():
    """
    Doing the cgmm_mvdr algorithm
    :return: no return
    """

    cgmm_beamformer = cgmm.complexGMM_mvdr(SAMPLING_FREQUENCY, FFT_LENGTH, FFT_SHIFT, NUMBER_EM_ITERATION,
                                           MIN_SEGMENT_DUR)
    os.system("echo init done")
    for i in range(len(multi_channels_data)):
        complex_spectrum, R_x, R_n, noise_mask, speech_mask = cgmm_beamformer.get_spatial_correlation_matrix(
            multi_channels_data[i])
        os.system("echo mask estimation done")

        beamformer, steering_vector = cgmm_beamformer.get_mvdr_beamformer(R_x, R_n)
        os.system("echo bmf done")

        enhanced_speech.extend(cgmm_beamformer.apply_beamformer(beamformer, complex_spectrum))
        os.system("echo enhan done")

        # sf.write(ENHANCED_PATH + '/' + naming(char_list),
        # enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, SAMPLING_FREQUENCY)

    wf.write(ENHANCED_PATH + '/' + key + ".wav",
                 SAMPLING_FREQUENCY, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65)
    os.system("echo all done")

    # if IS_MASK_PLOT:
    #     pl.figure()
    #     pl.subplot(2, 1, 1)
    #     pl.imshow(np.real(noise_mask).T, aspect='auto', origin='lower', cmap='hot')
    #     pl.title('noise mask')
    #     pl.subplot(2, 1, 2)
    #     pl.imshow(np.real(speech_mask).T, aspect='auto', origin='lower', cmap='hot')
    #     pl.title('speech mask')
    #     pl.show()


def do_mvdr():
    """
    Doing the simple mvdr algorithm
    :return: no return
    """

    for i in range(len(multi_channels_data)):
        complex_spectrum, _ = util.get_3dim_spectrum_from_data(multi_channels_data[i], FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)

        mvdr_beamformer = mvdr.minimum_variance_distortioless_response(MIC_ANGLE_VECTOR, MIC_DIAMETER,
                                                                       sampling_frequency=SAMPLING_FREQUENCY,
                                                                       fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

        steering_vector = mvdr_beamformer.get_sterring_vector(LOOK_DIRECTION)


        spatial_correlation_matrix = mvdr_beamformer.get_spatial_correlation_matrix(multi_channels_data[i])

        beamformer = mvdr_beamformer.get_mvdr_beamformer(steering_vector, spatial_correlation_matrix)

        enhanced_speech.extend(mvdr_beamformer.apply_beamformer(beamformer, complex_spectrum))

    wf.write(ENHANCED_PATH + '/' + key + ".wav", SAMPLING_FREQUENCY, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65)


if __name__ == '__main__':
    '''
    parameters for beamforming
    '''
    SAMPLING_FREQUENCY = 16000
    FFT_LENGTH = 512
    FFT_SHIFT = 128
    NUMBER_EM_ITERATION = 20
    MIN_SEGMENT_DUR = 2

    '''
    args from .sh files
    '''
    # INPUT_ARRAYS = "./../../channels_4"
    # SOURCE_PATH = "./../../sample_data/dev"
    # ENHANCED_PATH = "./../../"
    # LINE = 2

    INPUT_ARRAYS = sys.argv[1]
    SOURCE_PATH = sys.argv[2]
    ENHANCED_PATH = sys.argv[3]
    LINE = sys.argv[4]

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
    inp = file_dict(INPUT_ARRAYS)
    a = list(inp.keys())
    key = a[int(LINE) - 1]  # i.e. S01_U01

    '''
    prepare data for bmf (see comment for multi_channel_read())
    '''
    multi_channels_data = multi_channel_read(inp[key], SOURCE_PATH)
    os.system("echo data reading done")

    # IS_MASK_PLOT = False

    '''
    run different algorithm
    '''
    enhanced_speech = []
    # do_cgmm_mvdr()
    do_mvdr()

