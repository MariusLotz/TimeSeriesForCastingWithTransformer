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


def dataset_prediction_for_k(dataset, input, k):
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
    alpha_vec = []
    seq_sum_at_last_time_point = 0
    for sequence in dataset:
        alpha_i = calculate_alpha(input, sequence[:input_len])
        alpha_vec.append(alpha_i)
        seq_sum_at_last_time_point += sequence[-1] * alpha_i
    beta = seq_sum_at_last_time_point / input[-1]
    alpha_vec_adjusted = [alpha_i / beta for alpha_i in alpha_vec]
    
    prediction = sum(dataset[i][k] * alpha_i_adjusted for i, alpha_i_adjusted in enumerate(alpha_vec_adjusted))

    return prediction


def calculate_alpha(seq1, seq2):
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
    for v1, v2 in zip(seq1, seq2):
        result_sum += v1 * v2

    return result_sum


def test_plot():
    



