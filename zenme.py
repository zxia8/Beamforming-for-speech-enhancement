from scipy.io import wavfile as wf

_, wav = wf.read('./../../audio_chunk/dev/S02/S02_U02.CH1/_speech_/S02_U02.CH1_speech_1.wav')
_, wav2 = wf.read('./../../sample_data/eval/S02_U02.CH1.wav')
print(1)
