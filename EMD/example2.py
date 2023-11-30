import os
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from InputData.sin_cos_creator import sin_cos_creator7
from scipy.signal import hilbert


if __name__=="__main__":
    # Generate a sample signal
    signal = sin_cos_creator7()
    t = np.linspace(0, 1, len(signal))

    # Initialize EMD
    emd = EMD()

   # Perform EMD on the signal
    imfs = emd(signal)

    # Get the trend
    trend = emd.get_imfs_and_residue()[1]
    #print(trend)
    
# Plot the original signal, trend, and IMFs
plt.figure(figsize=(10, 8))

#Plot the original signal
plt.subplot(len(imfs) + 2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.legend()

# Plot the trend
plt.subplot(len(imfs) + 2, 1, 2)
plt.plot(t, trend, label='Trend')
plt.legend()


# Plot each IMF along with its Hilbert Transform
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs) + 2, 2, 2 * (i + 1) + 1)
    plt.plot(t, imf, label=f'IMF {i + 1}')
    plt.legend()

   # Compute the Hilbert Transform for each IMF
    analytic_signal = hilbert(imf)
    instantaneous_amplitude = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)

    # Plot instantaneous amplitude
    plt.subplot(len(imfs) + 2, 3, 3 * (i + 1) + 2)
    plt.plot(t, instantaneous_amplitude, label=f'Instantaneous Amplitude IMF {i + 1}')
    plt.legend()

    # Plot instantaneous phase
    plt.subplot(len(imfs) + 2, 3, 3 * (i + 1) + 3)
    plt.plot(t, instantaneous_phase, label=f'Instantaneous Phase IMF {i + 1}')
    plt.legend()

     # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save the figure in the same directory as the script
    plt.savefig(os.path.join(script_dir, "example8#.png"))