import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def sin_cos_creator(num_points=25, number_basis_functions=10):
    time = np.arange(num_points)
    cos_signal = np.zeros(num_points)
    for i in range(number_basis_functions):
        amplitude  = random.uniform(0.1, 10.0)
        frequence = random.uniform(0.1, 10.0)
        phase = random.uniform(-0.25, 0.25)
        cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequence)  # Cosine pattern
        cos_signal = cos_signal + cos_wave
    return cos_signal


def sin_cos_creator2(num_points=25, number_basis_functions=100):
    time = np.arange(num_points)
    cos_signal = np.zeros(num_points)
    for i in range(number_basis_functions):
        amplitude  = random.uniform(0.66, 1)
        frequence = random.uniform(1, 2)
        phase = random.uniform(-0.25, 0.25)
        cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequence)  # Cosine pattern
        cos_signal = cos_signal + cos_wave
    return cos_signal / np.sqrt(number_basis_functions)


def sin_cos_creator3(num_points=100, number_basis_functions=10000):
    time = np.arange(num_points)
    cos_signal = np.zeros(num_points)
    for i in range(number_basis_functions):
        amplitude  = random.uniform(0.66, 1)
        frequence = random.uniform(0.5, 2)
        phase = random.uniform(-0.33, 0.33)
        cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequence)  # Cosine pattern
        cos_signal = cos_signal + cos_wave
    return cos_signal / np.sqrt(number_basis_functions)


def sin_cos_creator4(num_points=100, number_basis_functions=100):
    time = np.arange(num_points)
    cos_signal = np.zeros(num_points)
    for i in range(number_basis_functions):
        amplitude  = random.uniform(0.75, 1)
        frequence = random.uniform(0.75, 1.25)
        phase = random.uniform(-0.25, 0.25)
        cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequence)  # Cosine pattern
        cos_signal = cos_signal + cos_wave
    cos_signal[0] = 0
    return cos_signal / np.sqrt(number_basis_functions)


def sin_cos_creator5(num_points=100, number_basis_functions=23):
    time = np.arange(num_points)
    cos_signal = np.zeros(num_points)
    for i in range(number_basis_functions):
        amplitude  = random.uniform(0.33, 3)
        frequence = random.uniform(3, 3.3)
        phase = random.uniform(-0.25, 0.25)
        cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequence)  # Cosine pattern
        cos_signal = cos_signal + cos_wave
    cos_signal[0] = 0
    return cos_signal / np.sqrt(number_basis_functions)


def sin_cos_creator6(num_points=100, number_basis_functions=13):
    time = np.arange(num_points)
    cos_signal = np.zeros(num_points)
    for i in range(number_basis_functions):
        amplitude  = random.uniform(0.5, 1)
        frequence = random.uniform(3, 3.3)
        phase = random.uniform(-0.13, 0.13)
        cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequence)  # Cosine pattern
        cos_signal = cos_signal + cos_wave
    cos_signal[0] = 0
    return cos_signal / np.sqrt(number_basis_functions)


def sin_cos_creator7(num_points=133, number_basis_functions=13):
    time = np.arange(num_points)
    cos_signal = np.zeros(num_points)
    for i in range(number_basis_functions):
        amplitude  = random.uniform(0.5, 1)
        frequence = random.uniform(3, 3.3)
        phase = random.uniform(-0.13, 0.13)
        cos_wave = amplitude * np.cos(2 * np.pi * (time + phase) * frequence)  # Cosine pattern
        cos_signal = cos_signal + cos_wave
    cos_signal[0] = 0
    #trend = [(0.0001 * x**2 - 0.000001*x**3) for x in range(num_points)]
    return (cos_signal) / np.sqrt(number_basis_functions)


def grapfical_print():
    # Generate synthetic time series data
    synthetic_data = sin_cos_creator6()

    # Plot the synthetic time series data
    plt.figure(figsize=(10, 6))
    plt.plot(synthetic_data)
    plt.title('Synthetic Time Series Data with Sinusoidal and Cosine Patterns')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.savefig('sample_plot1.png')

def create_sample(n, name):
    sample = []
    for i in range(n):
        sample.append(sin_cos_creator6())
        print(i)

    # Save data to a file using pickle
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(sample, file)

def print_load_file():
    # Load data from the file
    with open('training_sample_sin_cos_creator6.pkl', 'rb') as file:
        loaded_data = pickle.load(file)

    #Display the loaded data
    print("Loaded Data:", loaded_data)


if __name__=="__main__":
    create_sample(10000, "10000_test_sample_sin_cos_creator6")
    #print_load_file()
    

    
