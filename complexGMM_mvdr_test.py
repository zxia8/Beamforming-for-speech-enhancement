# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:49:02 2019

@author: a-kojima
"""
import numpy as np
# from memory_profiler import profile
from scipy.io import wavfile as wf


from beamformer import complexGMM_mvdr as cgmm
import sys


# @profile
def multi_channel_read(list, path):  # list: file_dict[key]
    for i in range(len(list)):
        list[i] = path + '/' + list[i]

    # wav, _ = sf.read(list[0], dtype='float32')
    _, wav = wf.read(list[0])
    wav_multi = np.zeros((len(wav), 4), dtype=np.float16)
    wav_multi[:, 0] = wav
    print("column 1 done")

    _, wav1 = wf.read(list[1])
    print("wav 2 done")

    wav_multi[:, 1] = wav1
    print("column 2 done")

    _, wav2 = wf.read(list[2])
    print("wav 3 done")
    wav_multi[:, 2] = wav2
    print("column 3 done")

    _, wav3 = wf.read(list[3])
    print("wav 4 done")

    wav_multi[:, 3] = wav3
    print("read done")

    wav_multi = np.array_split(wav_multi, 100)
    return wav_multi


def file_dict(input_arrays):

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
def main():

    inp = file_dict(INPUT_ARRAYS)
    a = list(inp.keys())
    key = a[LINE-1]
    enhanced_speech = []
    print(inp[key])


    multi_channels_data = multi_channel_read(inp[key], SOURCE_PATH)
    print("data reading done")

    cgmm_beamformer = cgmm.complexGMM_mvdr(SAMPLING_FREQUENCY, FFT_LENGTH, FFT_SHIFT, NUMBER_EM_ITERATION,
                                           MIN_SEGMENT_DUR)
    print("init done")
    for i in range(len(multi_channels_data)):
        complex_spectrum, R_x, R_n, noise_mask, speech_mask = cgmm_beamformer.get_spatial_correlation_matrix(
            multi_channels_data[i])
        print("mask estimation done")

        beamformer, steering_vector = cgmm_beamformer.get_mvdr_beamformer(R_x, R_n)
        print("beamforming done")

        enhanced_speech.extend(cgmm_beamformer.apply_beamformer(beamformer, complex_spectrum))
        print("enhancing done")

        # sf.write(ENHANCED_PATH + '/' + naming(char_list),
        # enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, SAMPLING_FREQUENCY)

    wf.write(ENHANCED_PATH + '/' + key + ".wav",
                 SAMPLING_FREQUENCY, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65)
    print("output done")
    print("all done")

    # if IS_MASK_PLOT:
    #     pl.figure()
    #     pl.subplot(2, 1, 1)
    #     pl.imshow(np.real(noise_mask).T, aspect='auto', origin='lower', cmap='hot')
    #     pl.title('noise mask')
    #     pl.subplot(2, 1, 2)
    #     pl.imshow(np.real(speech_mask).T, aspect='auto', origin='lower', cmap='hot')
    #     pl.title('speech mask')
    #     pl.show()


if __name__ == '__main__':
    SAMPLING_FREQUENCY = 16000
    FFT_LENGTH = 512
    FFT_SHIFT = 128
    NUMBER_EM_ITERATION = 20
    MIN_SEGMENT_DUR = 2

    # INPUT_ARRAYS = "./../../channels_4"
    # SOURCE_PATH = "./../../sample_data/eval"
    # ENHANCED_PATH = "./../../"
    # LINE = 1

    INPUT_ARRAYS = sys.argv[1]
    SOURCE_PATH = sys.argv[2]
    ENHANCED_PATH = sys.argv[3]
    LINE = sys.argv[4]

    # IS_MASK_PLOT = False

    main()
