import numpy as np
import matplotlib.pyplot as plt
import wave
import librosa

# 畳み込み演算
def convolution(x, h):
    """
    paramerters
    --
    x:array_like
      first input
    h:array_like
      second input

    returns
    --
    out:numply ndarray
       Discreate, Liner convolution of a and v.
    """

    x_len = len(x)
    h_len = len(h)
    out = np.zeros(x_len + h_len - 1)

    for i in range(x_len):
        out[i : i + h_len] += x[i] * h

    return out[:x_len]


# バンドパスフィルタ
def BPF(fl, fh, sr, N):
    """
    parameters
    --
    fl:int
       low edge frequency
    fh:int
       high edge frequency
    sr:int
       sampling rate
    N:int
      filter size

    returns
    --
    bpf:numpy.ndarray
        Band Pass Filter
    """

    pi = np.pi
    wl = 2 * pi * fl
    wln = wl / sr
    wh = 2 * pi * fh
    whn = wh / sr
    arange = np.arange(-N // 2, N // 2 + 1)
    h = (whn * np.sinc(whn * arange / pi) - wln * np.sinc(wln * arange / pi)) / pi
    hmm = np.hamming(N + 1)
    bpf = h * hmm

    return bpf


# フーリエ変換
def STFT(data, F_size, OVERLAP):
    """
    parameters
    --
    data:numpy.ndarray
         audio time series
    F_size:int
           frame size
    OVERLAP:float
           overlap size

    returns
    --
    spec:numpy.ndarray
         STFT of data
    """
    F_num = data.shape[0]
    S_num = int(F_num // (F_size - OVERLAP) - 1)
    window = np.hamming(F_size)
    spec = np.zeros([S_num, F_size], dtype=np.complex)
    pos = 0

    for fft_index in range(S_num):
        frame = data[int(pos) : int(pos + F_size)]
        if len(frame) == F_size:
            windowed = window * frame
            fft_result = np.fft.fft(windowed)
            for i in range(len(spec[fft_index])):
                spec[fft_index][i] = fft_result[i]

            pos += F_size - OVERLAP

    return spec


# スペクトログラム表示＆保存
def spec(data, filtered, sr, F_size, OVERLAP, Ts):
    """
    parameters
    --
    data:numpy.ndarray
         audio time series
    filtered:numply.ndarray
         filtered audio time series
    sr:int
       sampling rate
    F_size:int
           frame size
    OVERLAP:float
            overlap size
    Ts:float
    　 wave length
    """
    spec_ori = STFT(data, F_size, OVERLAP)
    spec_fil = STFT(filtered, F_size, OVERLAP)

    fig = plt.figure(figsize=(9, 6))

    # 元データのスペクトログラム
    ax1 = fig.add_subplot(2, 1, 1)
    im1 = ax1.imshow(
        np.log(abs(spec_ori[:, :512].T)),
        extent=[0, Ts, 0, sr / 2],
        cmap="rainbow",
        aspect="auto",
        origin="lower",
    )
    plt.colorbar(im1)
    plt.title("Original Spectrogram")
    plt.ylabel("Frequency[Hz]")

    # フィルタ後のスペクトログラム
    ax2 = fig.add_subplot(2, 1, 2)
    im2 = ax2.imshow(
        np.log(abs(spec_fil[:, :512].T)),
        extent=[0, Ts, 0, sr / 2],
        cmap="rainbow",
        aspect="auto",
        origin="lower",
    )
    plt.colorbar(im2)
    plt.title("Filtered Spectrogram")
    plt.ylabel("Frequency[Hz]")

    plt.xlabel("Times[s]")
    plt.tight_layout()
    # plt.savefig("ex2_Spectrogram.png")
    plt.show()
    plt.close()


# 処理後音声保存
def save(data, sr, name):
    """
    parameters
    --
    data:numpy.ndarray
         audio time series
    sr:int
       sampling rate
    name:str
         file name
    """
    wf = wave.open(name, "w")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    out_data = data * (2 ** 15)
    out_data = np.array(out_data, dtype="int16")
    wf.writeframe(out_data)
    wf.close()


def main():
    # 音声データよみこみ
    fname = "C:/Users/nobat/b4python/新規録音2.wav"
    data, samplerate = librosa.load(fname, sr=16000)

    # 条件設定
    pi = np.pi
    fl = 2000  # 周波数の下限
    fh = 5000  # 周波数の上限
    N = 100  # フィルタのサイズ
    F_size = 1024  # フレームのサイズ
    OVERLAP = F_size / 2
    F_num = data.shape[0]  # 音声ファイルのフレーム数
    Ts = float(F_num) / samplerate  # 波形の長さ

    # 音声データをフィルタにかける
    bpf = BPF(fl, fh, samplerate, N)
    filtered = convolution(data, bpf)

    # 元データとフィルタ後のデータのスペクトログラム
    spec(data, filtered, samplerate, F_size, OVERLAP, Ts)

    # save(filtered, samplerate, "filtered.wav")

    # BPFの解析
    bpf_fft = np.fft.fft(bpf)
    bpf_fft_abs = np.abs(bpf_fft)
    amp = 20 * np.log10(bpf_fft_abs)
    phase = np.unwrap(np.angle(bpf_fft)) * 180 / pi
    frequency_label = np.arange(0, samplerate / 2, (samplerate // 2) / (N // 2 + 1))

    # BPFの周波数特性図示＆保存
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(frequency_label, amp[0 : N // 2 + 1])
    plt.title("BPF Amplitude Characteristic")
    plt.ylabel("Amplitude[db]")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(frequency_label, phase[0 : N // 2 + 1])
    plt.title("BPF Phase Characteristic")
    plt.ylabel("Phase[deg]")
    plt.xlabel("Frequency[Hz]")
    plt.grid()
    plt.savefig("BPFFrequencyCharacteristic2.png")


# 実行
if __name__ == "__main__":
    main()