import numpy as np
import matplotlib.pyplot as plt


# Define the synthetic time series generator function
def generate_synthetic_data(num_points=100, noise_level=0.1):
    time = np.arange(num_points)
    noise = np.random.normal(0, noise_level, num_points)
    sine_wave = 2.1 * np.sin(2 * np.pi * time / 21)  # Sinusoidal pattern
    cosine_wave = 1.7 * np.cos(2 * np.pi * time / 13)  # Cosine pattern

    # Generate the synthetic time series data
    synthetic_data = sine_wave + cosine_wave + noise
    return synthetic_data

# Generate synthetic time series data
synthetic_data = generate_synthetic_data()

# Plot the synthetic time series data
plt.figure(figsize=(10, 6))
plt.plot(synthetic_data)
plt.title('Synthetic Time Series Data with Sinusoidal and Cosine Patterns')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('sample_plot.png')