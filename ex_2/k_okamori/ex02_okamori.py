import sys
import numpy as np
import soundfile as sf  # 音声データの読み取り
import matplotlib.pyplot as plt  # 出力データのプロット

# 定数の定義

NFFT = 1024  # frame の大きさ
OVERLAP = int(NFFT / 2)  # half shift を採用
WINDOW = np.hamming(NFFT)  # STFTの窓関数はハミング窓を使用

# 畳み込み
#   f1, f2: 畳み込む二つの関数


def convolution(f1, f2):
    result = np.zeros(f1.size + f2.size)  # 計算結果を格納する配列
    for i in range(f1.size):
        result[i: i + f2.size] += f1[i] * f2  # 畳み込み計算
    return result[f2.size:]

# 時間信号グラフの描画
#   ax: 描画するグラフの領域
#   title: グラフのタイトル
#   time: x座標のデータ
#   wave: y座標のデータ


def draw_signal(ax, title, time, wave):
    ax.plot(time, wave)
    plt.title(title)  # タイトルの追加
    plt.ylabel("Magnitude")  # y軸ラベルの追加

# フィルタの周波数特性のグラフの描画
#   ax: 描画するグラフの領域(振幅特性)
#   ax: 描画するグラフの領域(位相特性)
#   filter: フィルタのデータ


def draw_filter(ax_amp, ax_phase, filter):
    filter_w = np.fft.rfft(filter)  # フィルタのフーリエ変換
    filter_amp = 20 * np.log10(np.abs(filter_w))  # 振幅の計算
    filter_phase = np.degrees(np.unwrap(np.angle(filter_w)))  # 位相の計算
    ax_amp.plot(filter_amp)     # 振幅特性の描画
    ax_phase.plot(filter_phase)  # 位相特性の描画
    ax_amp.set_title("Amplitude Characteristics")  # タイトルの追加(振幅特性)
    ax_phase.set_title("Phase Characteristics")   # タイトルの追加(位相特性)
    ax_amp.set_xlabel("frequency[Hz]")   # x軸ラベルの追加(振幅特性)
    ax_phase.set_xlabel("frequency[Hz]")  # x軸ラベルの追加(位相特性)
    ax_amp.set_ylabel("amplitube[dB]")   # y軸ラベルの追加(振幅特性)
    ax_phase.set_ylabel("phase[degree]")  # y軸ラベルの追加(位相特性)


# スペクトログラムグラフの描画
#   spec: スペクトログラムのデータ
#   wave_length: 音源の長さ
#   samplerate: サンプルレート
def draw_spectrogram(fig, ax, title, spec, wave_length, samplerate):
    spec = np.log(np.square(np.abs(spec)))
    im = ax.imshow(spec.T, extent=[0, wave_length, 0, samplerate / 2],
                   cmap="rainbow", aspect="auto", origin="lower")  # スペクトログラムの描画
    fig.colorbar(im, ax=ax)
    ax.set_title(title)             # タイトルの追加
    ax.set_xlabel("time[s]")        # x軸ラベルの追加
    ax.set_ylabel("Frequency[kHz]")  # y軸ラベルの追加


# 短時間フーリエ変換(STFT)
def stft(wave):
    spec = np.zeros([len(wave) // OVERLAP - 1, int(NFFT / 2) + 1],
                    dtype=np.complex)  # spec:出力データ
    for idx in range(0, len(wave), OVERLAP):
        frame = wave[idx: idx + NFFT]  # frame の切り出し
        if len(frame) == NFFT:  # frame が全部切り出せるときのみ変換
            windowed = WINDOW * frame  # 窓関数をかける
            result = np.fft.fft(windowed)  # フーリエ変換
            for i in range(spec.shape[1]):
                spec[int(idx / OVERLAP)][i] = result[i]  # 計算結果を出力データに追加
    return spec

# フィルタの作成
#   samplerate: サンプルレート
#   f1 から f2 までの周波数を通す
#   (f1, f2) = (0, b)             : LPF
#   (f1, f2) = (a, samplerate / 2): HPF
#   (f1, f2) = (a, b)             : BPF
#   f2 < f1 の場合は f2 から f1 までの周波数を遮断する BEF となる


def create_filter(f1, f2, samplerate):
    fd = np.zeros(samplerate)
    if f1 > f2:  # 以下，矩形波の作成
        fd[: f2] = 1
        fd[f1: samplerate - f1] = 1
        fd[samplerate - f2:] = 1
    else:
        fd[f1: f2] = 1
        fd[samplerate - f2: samplerate - f1] = 1
    filter = (np.fft.ifft(fd)).real  # 逆フーリエ変換
    filter = filter.reshape(2, samplerate // 2)
    filter = filter[[1, 0], :]
    filter = filter.reshape(-1)
    filter = filter * np.hamming(samplerate)  # 窓関数をかける

    return filter

# main
#   agrv[1], argv[2]: フィルタ作成における f1, f2
#   詳細は create_filter を参照


def main():
    f1 = int(sys.argv[1])
    f2 = int(sys.argv[2])

    fig = plt.figure()  # 以下，グラフ領域の確保
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    wave, samplerate = sf.read("sound.wav")  # 音声データのロード
    wave_size = len(wave)  # wave_size: wave の要素数
    wave_length = wave_size / samplerate  # wave_length: wave の長さ

    filter = create_filter(f1, f2, samplerate)  # フィルタ作成
    draw_filter(ax1, ax2, filter)  # フィルタの周波数特性の描画

    spec_orignal = stft(wave)  # 短時間フーリエ変換(元データ)
    wave_filtered = convolution(wave, filter)  # 畳み込み
    spec_filtered = stft(wave_filtered)  # 短時間フーリエ変換(フィルタ後)

    draw_spectrogram(fig, ax3, "Spectrogram (Original)", spec_orignal,
                     wave_length, samplerate)  # スペクトログラムの描画(元データ)
    draw_spectrogram(fig, ax4, "Spectrogram (Filtered)", spec_filtered,
                     wave_length, samplerate)  # スペクトログラムの描画(フィルタ後)

    fig.tight_layout()  # レイアウト調整
    plt.savefig('ex02_spec.png')  # 保存

    sf.write("out.wav", wave_filtered, samplerate,
             format="WAV", subtype='PCM_16')  # 変換後音声データの書き出し


if __name__ == "__main__":
    main()
