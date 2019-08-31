import os
import re


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


dirtocheck = '/fastdata/acs18zx/CHiME5/audio/train2'
for root, _, files in os.walk(dirtocheck):
    for f in files:
        name = name2vec(f)
        fullpath = os.path.join(root, f)
        for i in (1, 5):
            os.system("scp " + fullpath + ' /fastdata/acs18zx/CHiME5/audio/retrain/' + name[0] + '_' + name[1] + '.CH' + str(i) + '.wav')
        os.system("rm " + fullpath)
