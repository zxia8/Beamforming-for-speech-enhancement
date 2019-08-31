import os
import re
import sys


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


dirtocheck = sys.argv[1]
for root, _, files in os.walk(dirtocheck):
    for f in files:
        temp_name = f + "_temp.wav"

        os.system("mv " + f + " " + temp_name)
        print("mv " + f + " " + temp_name)
        # os.system("sox " + temp_name + " -t wav -r 16000 -b 16 " + f)
        # os.system("rm " + temp_name)s
