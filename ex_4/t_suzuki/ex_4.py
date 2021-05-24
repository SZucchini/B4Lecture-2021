import os
import librosa
import argparse
import numpy as np
import spectrogrum as sp
import scipy.signal as sig
import matplotlib.pyplot as plt


def auto_correlation(data):
    """
    add description
    """
    
    n = len(data)
    cor = np.zeros(n)
    for i in range(n):
        cor[i] = data[:n-i] @ data[i:]
    
    return cor


def cepstrum(data):
    """
    add description later
    """

    fft_log = np.log10(np.abs(np.fft.fft(data)))
    cep = np.fft.ifft(fft_log)
    
    return cep


def predict_base_freq(data, sr, method, frame_size, overlap, lifter):
    """
    add description
    """

    res = []
    n = len(data)
    
    for i in range(0, n, int(frame_size*overlap)):
        frame = data[i:i+frame_size]
        frame = frame * np.hamming(len(frame))
        
        if method == 'cor':
            cor = auto_correlation(frame)
            min_idx = np.argmin(cor)
            peak = np.argmax(cor[min_idx:len(cor)//2]) + min_idx
            
        else:
            cep = cepstrum(frame)
            peak = np.argmax(cep[lifter:len(cep)//2]) + lifter
        
        res.append(sr / peak)

    return np.array(res)


def cept_spectrum(frame_data, lifter):
    """
    add description
    """

    cep = cepstrum(frame_data)
    cep[lifter:len(cep)-lifter] = 0
    cep_env = 20 * np.fft.fft(cep).real
    
    return cep_env


def levinson_durbin(cor, lpc_order):
    """
    add description
    """

    a = np.zeros(lpc_order + 1)
    e = np.zeros(lpc_order + 1)

    a[0] = 1.0
    a[1] = - cor[1] / cor[0]
    e[1] = cor[0] + a[1] * cor[1]
    lam = - cor[1] / cor[0]

    for k in range(1, lpc_order):
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * cor[k+1-j]
        lam /= e[k]

        u = [1]
        # u += [a[i] for i in range(1, k+1)]
        u.extend(a[i] for i in range(1, k+1))
        u.append(0)
        v = [0]
        # v += [a[i] for i in range(k, 0, -1)]
        v.extend(a[i] for i in range(k, 0, -1))
        v.append(1)

        a = np.array(u) + lam * np.array(v)
        e[k+1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]


def lpc_spectrum(frame_data, lpc_order, frame_size):
    """
    add description
    """

    cor = auto_correlation(frame_data)
    cor = cor[:len(cor)//2]
    
    a, e = levinson_durbin(cor, lpc_order)
    h = sig.freqz(np.sqrt(e), a, frame_size, "whole")[1]
    lpc_env = 20 * np.log10(np.abs(h))
    
    return lpc_env


def main():
    # argment settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input file path")
    parser.add_argument("--method", type=str, default="cep", help="bese frequency method")
    parser.add_argument("--lifter", type=int, default=20, help="input coefficient of lifter")
    parser.add_argument("--lpc_order", type=int, default=32, help="input lpc order")
    parser.add_argument("--frame", type=int, default=1024, help="data frame size")
    parser.add_argument("--overlap", type=float, default=0.5, help="overlap rate")
    parser.add_argument("--start_idx", type=int, default=31000, help="start index of spectrum envelope")
    args = parser.parse_args()

    # check output dir
    if not os.path.exists('./out'):
        os.makedirs('./out')

    # load audio file
    data, sr = librosa.load(args.input)
    time = np.arange(0, len(data)) / sr

    # caluculate fundamental frequency and spectrogrum
    f0 = predict_base_freq(data, sr, args.method, args.frame, args.overlap, args.lifter)
    f0_label = np.arange(0, time[-1], time[-1] / len(f0))
    spec = sp.spec(data, args.frame, args.overlap)

    # caluculate spectrum by ceptrum and lpc
    f_label = np.fft.fftfreq(1024, d=1.0/sr)
    frame_data = data[args.start_idx:args.start_idx+args.frame] * np.hamming(args.frame)
    fft_log = 20 * np.log10(np.abs(np.fft.fft(frame_data)))
    cep_env = cept_spectrum(frame_data, args.lifter)
    lpc_env = lpc_spectrum(frame_data, args.lpc_order, args.frame)

    # plot data and save figure
    # fundamental frequency
    fig, ax = plt.subplots()
    im = ax.imshow(spec, extent=[0, time[-1], 0, sr/2], aspect="auto", cmap="rainbow")
    ax.plot(f0_label, f0)
    if args.method == 'cor':
        title = "Auto Correlation"
    else:
        title = "Cepstrum"
    ax.set(title=f"Fundamental Frequency of {title}",
           xlabel="Time [s]",
           ylabel="Frequency [Hz]")
    plt.show(block=True)
    fig.savefig(f'./out/f0_result_{args.method}.png')

    # spectrum envelope
    fig, ax = plt.subplots()
    ax.plot(f_label[:args.frame//2], fft_log[:len(fft_log)//2], label="Spectrum", color="yellowgreen")
    ax.plot(f_label[:args.frame//2], cep_env[:len(cep_env)//2], label="Cepstrum", color="royalblue")
    ax.plot(f_label[:args.frame//2], lpc_env[:len(lpc_env)//2], label="LPC", color="aqua")
    ax.legend()
    ax.set(title="Spectrum envelope",
           xlabel="Frequency [Hz]",
           ylabel="Amplitude [dB]")
    plt.show(block=True)
    fig.savefig('./out/spectrum_envelope.png')


if __name__ == '__main__':
    main()
