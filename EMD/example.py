import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

# Generate a sample signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)

# Initialize EMD
emd = EMD()

# Perform EMD on the signal
imfs = emd(signal)

# Plot the original signal and IMFs
plt.subplot(len(imfs)+1, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.legend()

for i, imf in enumerate(imfs):
    plt.subplot(len(imfs)+1, 1, i+2)
    plt.plot(t, imf, label=f'IMF {i+1}')
    plt.legend()

plt.savefig("example")