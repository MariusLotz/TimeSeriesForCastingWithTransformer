import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os

def sin_cos_creator(num_points=512, number_basis_functions=7):
    time = np.arange(num_points)
    cos_signal = np.zeros(num_points)
    for i in range(number_basis_functions):
        amplitude  = random.uniform(0.5, 1)
        frequence = random.uniform(3, 3.1)
        phase = random.uniform(-0.01, 0.01)
        cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequence)  # Cosine pattern
        cos_signal = cos_signal + cos_wave
    cos_signal[0] = 0
    #trend = [(0.0001 * x**2 - 0.000001*x**3) for x in range(num_points)]
    return (cos_signal) / np.sqrt(number_basis_functions)


def grapfical_print():
    # Generate synthetic time series data
    synthetic_data = sin_cos_creator()

    # Plot the synthetic time series data
    plt.figure(figsize=(10, 6))
    plt.plot(synthetic_data)
    plt.title('Synthetic Time Series Data with Sinusoidal and Cosine Patterns')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('sample_plot1.png')



def create_sample(n, name):
    """
    Create a sample of data using the sin_cos_creator function and save it to a file.

    Args:
        n (int): Number of data points to generate in the sample.
        name (str): Name of the file (without extension) to save the sample.

    Returns:
        None

    """
    sample = []
    for i in range(n):
        sample.append(sin_cos_creator())
        if i % 1000 == 0:
            print(i)

    # Get the current directory
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Construct the full path for the file
    file_path = os.path.join(current_directory, name + '.pkl')

    # Save data to a file using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(sample, file)

if __name__=="__main__":
    create_sample(9999, "9999_test_sample_sin_cos_creator")
    #grapfical_print()