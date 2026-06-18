import librosa

y, sr = librosa.load("voice.mp3", sr=None)

print(sr)      # サンプリング周波数
print(y.shape) # サンプル数
D = librosa.stft(y)

print(D.shape)

import matplotlib.pyplot as plt
import librosa.display

# S_db = librosa.amplitude_to_db(abs(D))

# librosa.display.specshow(
#     S_db,
#     sr=sr,
#     x_axis='time',
#     y_axis='hz'
# )

# plt.colorbar()
# plt.show()

S = abs(D)
phase = D / (S + 1e-8)
S_engine = S.copy()

n_freq = S.shape[0]

for harmonic in range(2, 6):
    target = S_engine[harmonic::harmonic]
    n = target.shape[0]

    target += 0.3 * S[:n]
D_new = S_engine * phase

y_new = librosa.istft(D_new)

import soundfile as sf

sf.write("engine.wav", y_new, int(sr))
