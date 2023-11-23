import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import dct, idct


def DCT(signal):
    """Discrete Cosinus Transform"""
    return dct(signal, type=2)

def inverse_DCT(signal):
    """Inverse Discrete Cosinus Transform"""
    return idct(signal)

def FFT(signal):
    """Discrete Fourier Transform"""
    return np.fft.fft(signal)

def inverse_FFT(signal):
    """Inverse Discrete Fourier Transform"""
    return np.fft.ifft(signal)


def flatten_nested_array(nested_array):
    flat_list = []
    for item in nested_array:
        if isinstance(item, np.ndarray):
            flat_list.extend(item.tolist())
        else:
            flat_list.append(item)
    return flat_list


def DWT(signal, wavelet="db1"):
    signal_list = np.concatenate(pywt.dwt(signal, wavelet))
    if signal_list[-1] == 0:
        return signal_list[:-1]
    else:
        return signal_list
   
def reverse_DWT(signal, wavelet="db1"):
    n = len(signal) 
    if n % 2 == 0:
        cA = signal[:n//2]
        cD = signal[n//2:]
    if n % 2 == 1:
        cA = signal[:n//2+1]
        cD = np.concatenate((signal[n//2+1:],[0]))
    return pywt.idwt(cA, cD, wavelet)


if __name__=="__main__":
    signal=[1,2,3,4,5,6,7,8,9]
    print(signal)
    #print(DCT(signal))
    #print(inverse_DCT(signal))
    #print(FFT(signal))
    #print(inverse_FFT(FFT(signal)))
    print(DWT(signal))
    print(reverse_DWT(DWT(signal))) 



