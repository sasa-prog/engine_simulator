import librosa
import numpy as np
import matplotlib.pyplot as plt
from engine_simulator.signal_processor import process_engine, effect_engine, normalize_engine
from engine_simulator.audio_utils import calculate_rpm
import soundfile as sf


y, sr = librosa.load("./silen2.mp3", sr=None)
audio_duration_sec = librosa.get_duration(y=y, sr=sr)

print(f"Audio duration: {audio_duration_sec:.2f} sec")

f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=80,
    fmax=400,
)

# 無声音(NaN)を補間
f0 = np.nan_to_num(f0, nan=0)

rpm, rms = calculate_rpm(f0, y)
times_rpm = np.arange(len(rpm)) / 60
t_audio = np.arange(len(y)) / sr
rpm_interp = np.interp(
    t_audio,
    times_rpm,
    rpm
)
rms_interp = np.interp(
    t_audio,
    np.arange(len(rms)) / (len(rms) / audio_duration_sec),
    rms
)
plt.plot(rpm)
plt.ylabel("RPM")
plt.xlabel("Time (frames)")
plt.show()

engine, rpm_interp, phase = process_engine(rpm_interp, rms_interp, sr)

# engine = effect_engine(np.asarray(engine), rpm_interp, phase, sr)

# rms_out = np.sqrt(np.mean(engine**2))
# engine *= 0.25 / rms_out
# engine = np.tanh(engine)

# engine = normalize_engine(engine, rpm_interp, sr)

# engine *= 1.2
# engine = np.clip(engine, -1, 1)
engine /= np.max(np.abs(engine))

print(f"Generated audio duration: {len(engine) / sr:.2f} sec")

sf.write("engine.wav", engine, int(sr))
