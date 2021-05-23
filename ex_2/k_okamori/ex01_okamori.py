import numpy as np
import soundfile as sf # 音声データの読み取り
import matplotlib.pyplot as plt # 出力データのプロット

# 定数の定義
NFFT = 1024 # frame の大きさ
OVERLAP = int(NFFT / 2) # half shift を採用
window = np.hamming(NFFT) # 窓関数はハミング窓を使用

# 音声データのロード
def wav_load(fname):
    wave, samplerate = sf.read(fname)
    return wave, samplerate

# 時間信号グラフの描画
def draw_signal(ax, title, time, wave):
    ax.plot(time, wave)
    plt.title(title) # タイトルの追加
    plt.ylabel("Magnitude") # y軸ラベルの追加

# スペクトログラムグラフの描画
def draw_spectrogram(ax, title, spec, wave_length, samplerate):
    spec = np.log(np.square(np.abs(spec))) # 
    ax.imshow(spec.T, extent = [0, wave_length, 0, samplerate / 2],
    aspect = "auto")
    plt.title(title) # タイトルの追加
    plt.ylabel("Frequency[kHz]") # y軸ラベルの追加
    

# 短時間フーリエ変換(STFT)
def stft(wave):
    spec = np.zeros([len(wave) // OVERLAP - 1, int(NFFT / 2) + 1],
    dtype = np.complex) # spec:出力データ
    for idx in range(0, len(wave), OVERLAP):
        frame = wave[idx : idx + NFFT] # frame の切り出し
        if len(frame) == NFFT: # frame が全部切り出せるときのみ変換
            windowed = window * frame # 窓関数をかける
            result = np.fft.rfft(windowed) #フーリエ変換
            for i in range(spec.shape[1]):
                spec[int(idx / OVERLAP)][i] = result[i] # 計算結果を出力データに追加
    return spec

# 短時間フーリエ逆変換(ISTFT)
def i_stft(spec, wave_size):
    wave = np.zeros(wave_size) # wave:出力データ
    for idx in range(wave_size // OVERLAP - 2):
        frame = np.fft.irfft(spec[idx]) # frame の切り出し
        wave[idx * OVERLAP : idx * OVERLAP + NFFT] += np.real(frame) #フーリエ逆変換
    return wave


# main
def main():
    fig, axes = plt.subplots(3, 1, sharex = True) # グラフ領域の確保
    wave, samplerate = wav_load("sound.wav") # 音声データのロード
    wave_size = len(wave) # wave の要素数
    wave_length = wave_size / samplerate # wave の長さ
    time = np.arange(0, wave_size) / samplerate # 横軸を time に設定
    draw_signal(axes[0], "Original signal", time, wave) # 元データの描画
    spec = stft(wave) # 短時間フーリエ変換
    draw_spectrogram(axes[1], "Spectrogram", spec,
    wave_length, samplerate) # スペクトログラムデータの描画
    wave = i_stft(spec, wave_size) # 短時間フーリエ逆変換
    draw_signal(axes[2], "Re-synthesized signal", time, wave) # 復元データの描画
    # グラフの調整
    
    ax_0_pos = axes[0].get_position()
    ax_1_pos = axes[1].get_position()
    ax_2_pos = axes[2].get_position()
    axes[0].set_position([ax_0_pos.x0, ax_0_pos.y0, ax_1_pos.width, ax_1_pos.height])
    axes[1].set_position([ax_1_pos.x0, ax_1_pos.y0, ax_1_pos.width, ax_1_pos.height])
    axes[2].set_position([ax_2_pos.x0, ax_2_pos.y0, ax_1_pos.width, ax_1_pos.height])
    plt.show()
    plt.savefig('ex01_okamori1.png')
    print("ok")
    return

main()