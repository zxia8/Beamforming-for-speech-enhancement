import copy
# import pickle
# from pprint import pprint

import numpy as np
from scipy.io import wavfile as wf
import os
import re
import sys
from beamformer import util
from beamformer import minimum_variance_distortioless_response as mvdr


# @profile
def multi_channel_read(list, path):
    """
    :param list: names of the same audio with 4 channels
    :param path: the file dir
    :return: a matrix in which stores the input data. The matrix size is len(data_chunk) * 4 channels
    """
    if path != '':
        for z in range(len(list)):
            list[z] = path + '/' + list[z]

    # wav, _ = sf.read(list[0], dtype='float32')
    _, wav = wf.read(list[0])
    wav_multi = np.zeros((len(wav), 4), dtype=np.float16)
    wav_multi[:, 0] = wav
    # os.system("echo column 1 done")

    _, wav1 = wf.read(list[1])
    # os.system("echo wav 2 done")

    wav_multi[:, 1] = wav1
    # os.system("echo column 2 done")

    _, wav2 = wf.read(list[2])
    # os.system("echo wav 3 done")
    wav_multi[:, 2] = wav2
    # os.system("echo column 3 done")

    _, wav3 = wf.read(list[3])
    # os.system("echo wav 4 done")

    wav_multi[:, 3] = wav3
    os.system("echo read done")

    # wav_multi_sep = np.array_split(wav_multi, 200)
    return wav_multi


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
def do_mvdr():
    """
    Doing the simple mvdr algorithm
    :return: no return
    """
    complex_spectrum, _ = util.get_3dim_spectrum_from_data(audio,
    FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)

    mvdr_beamformer = mvdr.minimum_variance_distortioless_response(MIC_ANGLE_VECTOR, MIC_DIAMETER,
                                                                   sampling_frequency=SAMPLING_FREQUENCY,
                                                                   fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)

    steering_vector = mvdr_beamformer.get_sterring_vector(LOOK_DIRECTION)

    spatial_correlation_matrix = mvdr_beamformer.get_spatial_correlation_matrix(audio)

    beamformer = mvdr_beamformer.get_mvdr_beamformer(steering_vector, spatial_correlation_matrix)

    enhanced_speech = mvdr_beamformer.apply_beamformer(beamformer, complex_spectrum)

    wf.write(outpath + '/' + outname,
                 SAMPLING_FREQUENCY, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65)


if __name__ == '__main__':

    '''
    parameters for beamforming
    '''
    SAMPLING_FREQUENCY = 16000
    FFT_LENGTH = 256
    FFT_SHIFT = 128

    '''
    args from .sh files
    '''
    # INPUT_ARRAYS = "file_name"
    # SOURCE_PATH = "./../../"
    # CHUNK_PATH = "./../../audio_chunks"
    # ENHANCED_PATH = "./../.."
    # LINE = 0
    INPUT_ARRAYS = "./Beamforming-for-speech-enhancement/file_name"
    SOURCE_PATH = "/fastdata/acs18zx/CHiME5/audio"
    ENHANCED_PATH = "/data/acs18zx/new3/kaldi/egs/chime5/s5/enhan"
    LINE = int(sys.argv[1])

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
    with open(INPUT_ARRAYS, 'r') as f:
        a = f.readlines()
    f.close()
    inputli = a[LINE].split()
    folder = inputli.pop(0)
    mic = name2vec(inputli[0])[1].lower()
    outpath = ENHANCED_PATH + '/' + folder + '_mvdr_' + mic
    outname = str(name2vec(inputli[0])[0] + '_' + name2vec(inputli[0])[1] + '.wav')
    print(outpath + '/' + outname)
    '''
    prepare data for bmf (see comment for multi_channel_read())
    '''
    audio = multi_channel_read(inputli, SOURCE_PATH + '/' + folder)

    if not os.path.exists(outpath):
        os.mkdir(outpath)
    os.system("echo data reading done")

    '''
    run algorithm
    '''
    do_mvdr()
