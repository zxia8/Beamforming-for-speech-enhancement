from scipy.io import wavfile as wf
import numpy as np
import math

_, worn = wf.read('./../../S02_P08_sample.wav')
_, bmf = wf.read('./../../S02_U02_sample_bmf.wav')
_, cgmm = wf.read('./../../S02_U02_sample_cgmm.wav')
_, ori = wf.read('./../../S02_U02_sample_ori.wav')

worn = np.array(worn[:, 0])
bmf = np.array(bmf)
cgmm = np.array(cgmm)
ori = np.array(ori)

p_b = np.dot(np.linalg.norm(bmf), np.linalg.norm(bmf).T)
p_c = np.dot(np.linalg.norm(cgmm), np.linalg.norm(cgmm).T)
p_o = np.dot(np.linalg.norm(ori), np.linalg.norm(ori).T)
p_w = np.dot(np.linalg.norm(worn), np.linalg.norm(worn).T)

p_b_noise = np.dot(np.linalg.norm(bmf - worn), np.linalg.norm(bmf - worn).T)
p_c_noise = np.dot(np.linalg.norm(cgmm - worn), np.linalg.norm(cgmm - worn).T)
p_o_noise = np.dot(np.linalg.norm(ori - worn), np.linalg.norm(ori - worn).T)

snr_b = 10 * math.log(p_w/p_b_noise, 10)
snr_c = 10 * math.log(p_w/p_c_noise, 10)
snr_o = 10 * math.log(p_w/p_o_noise, 10)

print("original file snr: " + str(snr_o))
print("beamformit snr: " + str(snr_b))
print("cgmm snr: " + str(snr_c))
