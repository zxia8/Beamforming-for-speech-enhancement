import os
import re
import sys

from scipy.io import wavfile as wf
import numpy as np


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


def multi_channel_read(path):  # list: file_dict[key]

    # wav, _ = sf.read(list[0], dtype='float32')
    _, wav = wf.read(path)
    wav_multi = np.zeros((len(wav), 1), dtype=np.float16)
    wav_multi[:, 0] = wav

    return wav_multi


a = {}
del_list = []
# dr = sys.argv[1]
dir_to_check = './../../audio_chunks/dev/S02'


for root, _, files in os.walk(dir_to_check):
    n = name2vec(root)
    if n[-2] == 'speech' and n[-4] == 'CH1':
        for f in files:
            fullpath = os.path.join(root, f)
            vec = name2vec(fullpath)
            a[fullpath] = []
            f_name = []
            os.system('echo ' + f)
            for i in range(1, 5):
                vec[-3] = "CH" + str(i)
                file_name = dir_to_check + '/' + vec[-11] + '_' + vec[-10] + '.' + \
                              vec[-3] + '/' + '_' + vec[-2] + '_' + '/' + vec[-11] + '_' + vec[-10] + '.' + \
                              vec[-3] + '_' + vec[-2] + '_' + vec[-1] + '.wav'
                c = max(multi_channel_read(file_name))
                d = min(multi_channel_read(file_name))
                a[fullpath].append(abs(c-d))

            if os.path.getsize(fullpath) < 100 * 1024 or max(a[fullpath]) < 6300:
                os.system('echo del')
                for i in range(1, 5):
                    vec[-3] = "CH" + str(i)
                    os.remove(dir_to_check + '/' +
                                    vec[-11] + '_' + vec[-10] + '.' +
                                    vec[-3] + '/' + '_' + vec[-2] + '_' + '/' + vec[-11] + '_' + vec[-10] + '.' +
                                    vec[-3] + '_' + vec[-2] + '_' + vec[-1] + '.wav')
