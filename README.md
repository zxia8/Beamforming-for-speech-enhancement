# Beamforming-for-speech-enhancement
- 修改原作者AkojimaSLP的波束形成器，用于替换CHiME-5中语音增强部分的Beamformit（delay sum 波束形成，表现十分弟弟）
  Edit AkojimaSLP's beamformer to replace Beamformit in kaldi-CHiME-5 which is a simple delay_sum beamformer, performs not satisfying.
- 修改算法获得更高的可懂度（intelligibility）更适合长距离语音识别
  Edit algorithm to get signal with higher intelligibility, suitable for long distance microphone SR
- 修改算法提高对长音频的运行效率（guided beamforming）
  Edit algorithm to reduce computational cost, suitable for large audio files (use guided beamforming).
