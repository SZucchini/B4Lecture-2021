import os
import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

out_dir = "./out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 音声ファイルの読み込み
fname = "./sound/source.aifc"
y, sr = librosa.load(fname)
time = np.arange(0, len(y)) / sr
# 窓幅・オーバーラップ・窓関数の設定
window = 1024
overlap = 512
win_func = signal.hamming(window)

# フーリエ変換
spec = []
fft_data = []
for idx in range(0, len(y), window-overlap):
    frame = y[idx:idx+window]
    if len(frame) == window:
        frame = frame * win_func
        fft_res = np.fft.rfft(frame)
        # 逆変換用のデータを準備
        fft_data.append(fft_res)
        res = np.log(np.abs(fft_res) ** 2)
        spec.append(res)

# 逆変換
ifft_res = []
for i in range(len(fft_data)):
    f = np.fft.irfft(fft_data[i])
    if i == 0:
        ifft_res.extend(f)
    else:
        ifft_res.extend(f[overlap:window])
# 欠損値をゼロ埋め
for i in range(len(time) - len(ifft_res)):
    ifft_res.append(0)

# データのプロット
fig, axes = plt.subplots(3, 1)
axes[0].plot(time, y)
im = axes[1].imshow(np.array(spec).T, extent=[0, time[-1], 0, (sr/2) / 1000], aspect="auto")
cbar = fig.colorbar(im, ax = axes[1])
axes[2].plot(time, ifft_res)

# レイアウトの調整
fig.tight_layout()
plt.subplots_adjust(left=0.1, right=1.0, bottom=0.1, top=0.9)
fig.set_figheight(5.2)
fig.set_figwidth(5.5)
fig.canvas.draw()

# 幅を揃える
ax0pos = axes[0].get_position()
ax1pos = axes[1].get_position()
ax2pos = axes[2].get_position()
axes[0].set_position([ax0pos.x0, ax0pos.y0, ax1pos.width, ax1pos.height])
# axes[1].set_position([ax1pos.x0, ax1pos.y0, ax1pos.width, ax1pos.height])
axes[2].set_position([ax2pos.x0, ax2pos.y0, ax1pos.width, ax1pos.height])

# タイトル
axes[0].set_title("Original signal")
axes[1].set_title("Spectrogram")
axes[2].set_title("Re-synthesized signal")

# ラベル
axes[0].set_ylabel("Magnitude")
axes[1].set_ylabel("Frequency[kHz]")
axes[2].set_ylabel("Magnitude")
axes[2].set_xlabel("Time[s]")

# 表示・保存
plt.show(block=True)
fig.savefig(f'{out_dir}/result.png')

