import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev

def is_maxima(x1, x2, x3):
    """
    Check if the middle value is greater than its neighbors.

    Parameters:
    - x1 (float): Value to the left.
    - x2 (float): Middle value.
    - x3 (float): Value to the right.

    Returns:
    - bool: True if x2 is a local maxima, False otherwise.
    """
    return (x2 > x1 and x2 > x3)

def is_minima(x1, x2, x3):
    """
    Check if the middle value is smaller than its neighbors.

    Parameters:
    - x1 (float): Value to the left.
    - x2 (float): Middle value.
    - x3 (float): Value to the right.

    Returns:
    - bool: True if x2 is a local minima, False otherwise.
    """
    return (x2 < x1 and x2 < x3)


def get_local_extrema(signal, shorten_list=True):
    """
    Identify local maxima and minima in a signal.

    Parameters:
    - signal (numpy.ndarray): Input signal.

    Returns:
    - tuple: A tuple containing two numpy arrays - the first array represents
             the compressed signal with only local maxima and minima,
             and the second array represents the rest of the signal.
    """
    sig_len = len(signal)
    compressed_signal = np.zeros(sig_len)
    rest_signal = np.zeros(sig_len)

    for i in range(1, sig_len - 1):
        if is_maxima(signal[i - 1], signal[i], signal[i + 1]) or is_minima(signal[i - 1], signal[i], signal[i + 1]):
            compressed_signal[i] = signal[i]
        else:
            rest_signal[i] = signal[i]
    if shorten_list:
        compressed_signal = [x for x in compressed_signal if x != 0]
        rest_signal = [x for x in rest_signal if x!= 0]
    return compressed_signal, rest_signal


def signal_to_poly_coeff(signal, poly_type="Chebyshev"):
    # Get  Chebyshev coefficients
    degree = 5  # Choose the degree of the Chebyshev polynomial
    cheb_coefficients = chebyshev.chebfit(x, signal, degree)




if __name__=="__main__":
    # Example usage:
    signal_example = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1])
    compressed, rest = get_local_extrema(signal_example)
    print("example Signal:",signal_example)
    print("Compressed Signal:", compressed)
    print("Rest of the Signal:", rest)

