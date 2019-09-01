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

    return name


dir = sys.argv[1]
dirtocheck = '/data/acs18zx/new3/kaldi/egs/chime5/s5/enhan/' + dir

for root, _, files in os.walk(dirtocheck):
    for f in files:
        if 'wav' in name2vec(f):
            fullpath = os.path.join(root, f)
            _, wav = wf.read(fullpath)
            wav = np.array(wav)
            p = np.dot(np.linalg.norm(wav), np.linalg.norm(wav).T)
            if str(p) == 'nan':
                os.system("echo " + f)
