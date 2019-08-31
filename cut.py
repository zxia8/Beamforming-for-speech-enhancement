s = 'S18'
u = ['U01', 'U02', 'U03', 'U04', 'U05', 'U06']
ch = ['CH1', 'CH2', 'CH3', 'CH4']
inli = []
outli = []

f = open('cut.sh', 'w')
f.write('#!/bin/bash\n')

for c_item in ch:
    for u_item in u:
        infile = '/fastdata/acs18zx/CHiME5/audio_chunks/train/S18/' + s + '_' + u_item + '.' + c_item + '/_speech_/' + s + '_' + u_item + '.' + c_item + '_speech_821.wav'
        for i in range(1, 12):
            outfile = '/fastdata/acs18zx/CHiME5/audio_chunks/train/S18/' + s + '_' + u_item + '.' + c_item + '/_speech_/' + s + '_' + u_item + '.' + c_item + '_speech_' + str(9999-i) + '.wav'
            if i == 1:
                f.write('sox ' + infile + ' ' + outfile + ' trim 0:00:00.00 =0:10:00.00\n')
            elif i <= 5 and i != 1:
                f.write('sox ' + infile + ' ' + outfile + ' trim 0:'
                        + str((i-1)*10) + ":00.00 =" + '0:' + str(i*10) + ':00.00\n')
            elif i == 6:
                f.write('sox ' + infile + ' ' + outfile + ' trim 0:50:00.00 =1:00:00:00\n')
            elif i == 7:
                f.write('sox ' + infile + ' ' + outfile + ' trim 1:00:00.00 =1:10:00:00\n')
            elif i > 7:
                f.write('sox ' + infile + ' ' + outfile + ' trim 1:'
                        + str((i - 7) * 10) + ":00.00 =" + '1:' + str((i - 6) * 10) + ':00.00\n')
        f.write('rm ' + infile + '\n')
f.close()
