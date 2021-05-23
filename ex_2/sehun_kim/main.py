import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import soundfile
import argparse


def HPF(data_array, cf, sr):
    """
    Function for High Pass Filter.

    Parameters :
    cf : Cut Frequency
    sr : Sample rate

    Return :
    ir : Impulse response in time domain
    """
    # f_HPF : HPF in frequency domain
    f_HPF = np.ones(sr)
    f_HPF[:cf] = 0
    f_HPF[sr - cf:] = 0
    h = (np.fft.ifft(f_HPF)).real

    h_front = h[: int(sr / 2)]
    h_back = h[int(sr / 2):]
    ir = np.append(h_back, h_front)
    window = np.hamming(len(h))
    ir = ir * window
    return ir[len(ir): len(ir)]


def LPF(data_array, cf, sr):
    """
    Function for Low Pass Filter

    Parameters :
    cf : Cut Frequency
    sr : Sample rate

    Return :
    ir : Impulse response in time domain
    """
    # f_LPF : LPF in frequency domain
    f_LPF = np.zeros(sr)
    f_LPF[:cf] = 1
    f_LPF[sr - cf:] = 1
    h = (np.fft.ifft(f_LPF)).real

    h_front = h[: int(sr / 2)]
    h_back = h[int(sr / 2):]
    ir = np.append(h_back, h_front)
    window = np.hamming(len(h))
    ir = ir * window
    return ir


def BPF(data_array, cut_start, cut_end, sr):
    """
    Function for Band Pass Filter

    Parameters:
    cut_start : Start of Pass Frequency
    cut_end : End of Pass Frequency
    sr : Sample rate

    Return :
    ir : Impulse response in time domain
    """
    # f_BPF : BPF in frequency domain
    f_BPF = np.zeros(sr)
    f_BPF[cut_start:cut_end] = 1
    f_BPF[sr - cut_end: sr - cut_start] = 1
    h = (np.fft.ifft(f_BPF)).real

    h_front = h[: int(sr / 2)]
    h_back = h[int(sr / 2):]
    ir = np.append(h_back, h_front)
    window = np.hamming(len(h))
    ir = ir * window
    return ir


def BSF(data_array, cut_start, cut_end, sr):
    """
    Function for Band Stop Filter

    Parameters:
    cut_start : Start of Cut Frequency
    cut_end : End of Cut Frequency
    sr : Sample rate

    Return :
    ir : Impulse response in time domain
    """
    # f_BSF : BSF in frequency domain
    f_BSF = np.ones(sr)
    f_BSF[cut_start:cut_end] = 0
    f_BSF[sr - cut_end: sr - cut_start] = 0
    h = (np.fft.ifft(f_BSF)).real

    h_front = h[: int(sr / 2)]
    h_back = h[int(sr / 2):]
    ir = np.append(h_back, h_front)
    window = np.hamming(len(h))
    ir = ir * window
    return ir


def conv(x, h):
    N = len(x)
    result = np.zeros(N + len(h))
    for n in range(len(h)):
        result[n: n + N] = np.add(result[n: n + N], x * h[n])
    return result[:N]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program for applying digital filter.\nFile name, filter type, filtering frequency are required.')
    parser.add_argument("-f", dest="filename", help='Filename', required=True)
    parser.add_argument("-t", dest="filter_type", type=int,
                        help='Filter type : 1 for hpf, 2 for lpf, 3 for  bpf, 4 for bsf', required=True)
    parser.add_argument("-f1", type=int, dest="frequency_1",
                        help='Enter frequency #1', required=True)
    parser.add_argument("-f2", type=int, dest="frequency_2",
                        help='Enter frequency #2 (Optional)', required=False)
    args = parser.parse_args()

    wave_array, sr = librosa.load(args.filename, sr=44100)

    f1 = args.frequency_1
    f2 = args.frequency_2

    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(wspace=0.6, hspace=0.6)

    # Spectrogram for the original signal
    X = librosa.stft(wave_array)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar()
    plt.savefig("spectrogram_original.png")
    plt.figure(figsize=(14, 10))

    # Process for High Pass Filter
    if args.filter_type == 1:
        t_hpf = HPF(wave_array, f1, sr)
        f_hpf = np.real(np.fft.fft(t_hpf))
        f_hpf = f_hpf[0::2]
        f_hpf = f_hpf[:len(f_hpf)//2]
        t_hp_filtered = conv(wave_array, t_hpf)
        f_hp_filtered = np.fft.fft(t_hp_filtered)

        # Plot and save frequency response
        x = np.arange(0, len(f_hpf)*2, sr/(len(f_hpf)*2))
        plt.subplot(2, 1, 1)
        plt.title("Impulse response of High Pass Filter")
        plt.plot(t_hpf, "b")
        plt.xlabel("Time[samples]")
        plt.ylabel("Magnitude")
        plt.subplot(2, 1, 2)
        plt.title("Frequency response of High Pass Filter")
        plt.plot(x, f_hpf, "b")
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Magnitude")
        plt.savefig("irfr_hpf.png")

        # Plot and save spectrogram
        X = librosa.stft(t_hp_filtered)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar()
        plt.savefig("spectrogram_hpf.png")

        # Write as .wav file
        soundfile.write("result_hpf.wav", t_hp_filtered.real, samplerate=sr)

    # Process for Low Pass Filter
    elif args.filter_type == 2:
        t_lpf = LPF(wave_array, f1, sr)
        f_lpf = np.real(np.fft.fft(np.real(t_lpf)))
        f_lpf = f_lpf[0::2]
        f_lpf = f_lpf[:len(f_lpf)//2]
        t_lp_filtered = conv(wave_array, t_lpf)
        f_lp_filtered = np.fft.fft(t_lp_filtered)

        # Plot and save frequency response
        x = np.arange(0, len(f_lpf)*2, sr/(len(f_lpf)*2))
        plt.subplot(2, 1, 1)
        plt.title("Impulse response of Low Pass Filter")
        plt.plot(t_lpf, "b")
        plt.xlabel("Time[samples]")
        plt.ylabel("Magnitude")
        plt.subplot(2, 1, 2)
        plt.title("Frequency response of Low Pass Filter")
        plt.plot(x, f_lpf, "b")
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Magnitude")
        plt.savefig("irfr_lpf.png")

        # Plot and save spectrogram
        X = librosa.stft(t_lp_filtered)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar()
        plt.savefig("spectrogram_lpf.png")
        soundfile.write("result_lpf.wav", t_lp_filtered.real, samplerate=sr)

    # Process for Band Pass Filter
    elif args.filter_type == 3:
        t_bpf = BPF(wave_array, f1, f2, sr)
        f_bpf = np.real(np.fft.fft(t_bpf))
        f_bpf = f_bpf[0::2]
        f_bpf = f_bpf[:len(f_bpf)//2]
        t_bp_filtered = conv(wave_array, t_bpf)
        f_bp_filtered = np.fft.fft(t_bp_filtered)

        # Plot and save frequency response
        x = np.arange(0, len(f_bpf)*2, sr/(len(f_bpf)*2))
        plt.subplot(2, 1, 1)
        plt.title("Impulse response of Band Pass Filter")
        plt.plot(t_bpf, "b")
        plt.xlabel("Time[samples]")
        plt.ylabel("Magnitude")
        plt.subplot(2, 1, 2)
        plt.title("Frequency response of Band Pass Filter")
        plt.plot(x, f_bpf, "b")
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Magnitude")
        plt.savefig("irfr_bpf.png")

        # Plot and save spectrogram
        X = librosa.stft(t_bp_filtered)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar()
        plt.savefig("spectrogram_bpf.png")
        soundfile.write("result_bpf.wav", t_bp_filtered.real, samplerate=sr)

    # Process for Band Stop Filter
    elif args.filter_type == 4:
        t_bsf = BSF(wave_array, f1, f2, sr)
        f_bsf = np.real(np.fft.fft(np.real(t_bsf)))
        f_bsf = f_bsf[0::2]
        f_bsf = f_bsf[:len(f_bsf)//2]
        t_bs_filtered = conv(wave_array, t_bsf)
        f_bs_filtered = np.fft.fft(t_bs_filtered)

        # Plot and save frequency response
        x = np.arange(0, len(f_bsf)*2, sr/(len(f_bsf)*2))
        plt.subplot(2, 1, 1)
        plt.title("Impulse response of Band Stop Filter")
        plt.plot(t_bsf, "b")
        plt.xlabel("Time[samples]")
        plt.ylabel("Magnitude")
        plt.subplot(2, 1, 2)
        plt.title("Frequency response of Band Stop Filter")
        plt.plot(x, f_bsf, "b")
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Magnitude")
        plt.savefig("irfr_bsf.png")

        # Plot and save spectrogram
        X = librosa.stft(t_bs_filtered)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar()
        plt.savefig("spectrogram_bsf.png")
        soundfile.write("result_bsf.wav", t_bs_filtered.real, samplerate=sr)

    else:
        print("No matching filter type. \nFilter type : 1 for HPF, 2 for LPF, 3 for BPF, 4 for BSF \nExiting...")
        exit()
