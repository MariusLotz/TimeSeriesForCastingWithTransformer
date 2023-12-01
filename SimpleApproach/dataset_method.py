import pickle
import os
import matplotlib.pyplot as plt
from statistics import median

def dataset_prediction_method(dataset, input, pred_len):
    """
    Predicts a sequence of values for a given dataset, input, and prediction length.

    Parameters:
    - dataset (list of lists): The dataset containing sequences.
    - input (list): The input sequence used for prediction.
    - pred_len (int): The length of the predicted sequence.

    Returns:
    - list: The predicted sequence of values.
    """
    input_len = len(input)
    dataset_len = len(dataset[0])

    if input_len + pred_len > dataset_len:
        raise ValueError("Can't predict, because the datasets are not long enough!")

    # Initialize the output list to store predicted values
    output = []

    # Iterate over the prediction length
    for i in range(pred_len):
        # Predict the value at the next time point using dataset_prediction_for_k function
        prediction = dataset_prediction_for_k(dataset, input, input_len - 1 + i)
        
        # Append the predicted value to the output list
        output.append(prediction)

    return output


def dataset_prediction_for_k(dataset, input, k, fun=lambda x: x):
    """
    Predicts the value at time point k based on the given dataset and input.

    Parameters:
    - dataset (list of lists): The dataset containing sequences.
    - input (list): The input sequence.
    - k (int): The time point for prediction.

    Returns:
    - float: The predicted value at time point k.
    """

    input_len = len(input)
    dataset_len = len(dataset[0])

    if k > dataset_len:
        raise ValueError("Cant predict that time point k, because it is out of range!")
    
    # Calculating alpha's and beta:
    alpha_vec = [calculate_alpha(input, sequence[:input_len]) for sequence in dataset]
    beta = calculate_beta(dataset, input, alpha_vec)
    
    alpha_vec_adjusted = [fun(alpha_i) / beta for alpha_i in alpha_vec]
    
    prediction = sum(dataset[i][k] * alpha_i_adjusted for i, alpha_i_adjusted in enumerate(alpha_vec_adjusted))
    return prediction


def calculate_beta(dataset, input, alpha_vec):
    """
    Berechnet den Beta-Wert für gegebenes Dataset, Input und Alpha-Vektor.

    Parameters:
    - dataset (list of lists): Das Dataset mit Sequenzen.
    - input (list): Die Eingabesequenz.
    - alpha_vec (list): Der Alpha-Vektor.

    Returns:
    - float: Der Durchschnitt der berechneten Beta-Werte.
    """
    Y = []         # Vektor Y
    beta_vec = []  # Vektor der Beta-Werte

    for k in range(len(input)):
        sum_y = 0
        for i in range(len(dataset)):
            sum_y += dataset[i][k] * alpha_vec[i]

        Y.append(sum_y)

        # Berechne den Beta-Wert und füge ihn zum Beta-Vektor hinzu
        beta_value = sum_y / (input[k]+0.00001)
        beta_vec.append(beta_value)

    # Berechne den Durchschnitt der Beta-Werte
    median_beta = median(beta_vec)

    return median_beta


def calculate_alpha(seq1, seq2, filter_small_values=True, on_set=1e-9):
    """
    Calculates the scalar_product between two sequences of equal length.

    Parameters:
    - seq1 (list): The first sequence.
    - seq2 (list): The second sequence.

    Returns:
    - float: The calculated alpha value.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Vectors need the same length!")

    result_sum = 0
    energie1 = 0
    energie2 = 0
    for v1, v2 in zip(seq1, seq2):
        result_sum += v1 * v2
        energie1 += v1 * v1
        energie2 += v2 * v2

    if abs(result_sum) < 0.2*(abs(energie2) + abs(energie1)):
        return result_sum * on_set
    else:
        print(result_sum)
        return result_sum


def test_plot():
    """
    Loads datasets, performs predictions, and plots the predicted sequence against the true sequence.
    """
    # Load the training dataset
    with open("90000_training_sample_sin_cos_creator6.pkl", 'rb') as file:
        training_dataset = pickle.load(file)

    # Load the test dataset
    with open("10000_test_sample_sin_cos_creator6.pkl", 'rb') as file:
        test_inputset = pickle.load(file)

    # Extract the first sequence from the test dataset
    sample = 13
    starting_point = 60
    prediction_len = 100 - starting_point
    input_sequence = test_inputset[sample][:starting_point]
    true_sequence = test_inputset[sample][starting_point:]

    # Perform predictions using the dataset_prediction_method
    prediction = dataset_prediction_method(training_dataset[0:19999], input_sequence, pred_len=prediction_len)

    
    # Plot the predicted sequence against the true sequence
    plot_prediction_vs_true(prediction, true_sequence, "example_s13_starting60-2")


def plot_prediction_vs_true(prediction, true_sequence, name):
    """
    Plots the predicted sequence against the true sequence.

    Parameters:
    - prediction (list): The predicted sequence of values.
    - true_sequence (list): The true sequence of values.
    """
    plt.plot(range(len(true_sequence)), true_sequence, label='True Sequence', marker='o')
    plt.plot(range(len(prediction)), prediction, label='Predicted Sequence', marker='x')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    plt.title('Prediction vs True Sequence')
    plt.legend()
    # Save the plot in the specified folder
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, name)
    plt.savefig(save_path)


if __name__=="__main__":
    test_plot()



