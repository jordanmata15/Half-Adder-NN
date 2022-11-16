import math
import numpy as np
import os
import pandas as pd
import random

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATA_DIR = os.path.join(os.path.join(SRC_DIR, ".."), "model_data")


def generate_n_random_inputs_equal_counts(n, skew):
    """Generates n sets of half adder inputs [x,y] with random skew.
    Ensures to create an even distribution based on each possible input.

    Args:
        n (int): Number of pairs of inputs to generate
        skew (float): The range of the random noise to add to each input

    Returns:
        List[List[float]]: The randomly generated half adder inputs.
    """
    values = []
    count_of_each = math.floor(n/4) # theres 4 possibilites [0,0] [1,0] [0,1] [1,1]
    for _ in range(count_of_each):
        values.append(np.array([0,0]) + np.array([random.uniform(-skew,skew), random.uniform(-skew,skew)]))
        values.append(np.array([0,1]) + np.array([random.uniform(-skew,skew), random.uniform(-skew,skew)]))
        values.append(np.array([1,0]) + np.array([random.uniform(-skew,skew), random.uniform(-skew,skew)]))
        values.append(np.array([1,1]) + np.array([random.uniform(-skew,skew), random.uniform(-skew,skew)]))
    return np.array(values)


def calculate_half_adder_outputs(input_pairs):
    """Calculates the expected half adder outputs.

    Args:
        input_pairs (List[List[float]]): List of half adder inputs 

    Returns:
        List[List[int]]: List of actual half adder outputs.
    """
    outputs = []
    for x1,x2 in input_pairs:
        x1_bin = round(x1)
        x2_bin = round(x2)
        sum = x1_bin ^ x2_bin
        carry_bit = x1_bin & x2_bin
        outputs.append([sum, carry_bit])
    return np.array(outputs)


def generate_n_random_input_outputs(n, skew=0.2):
    """Generates n random half adder inputs and outputs.

    Args:
        n (int): Number of pairs of inputs/outputs to generate
        skew (float): The range of the random noise to add to each input

    Returns:
        List[List[float], List[int]]: [Half adder inputs, half adder outputs]
    """
    inputs = generate_n_random_inputs_equal_counts(n, skew)
    outputs = calculate_half_adder_outputs(inputs)
    return [inputs, outputs]


def write_to_file(input_data, output_data, filename):
    """Writes the input/output data as a dataframe to a csv file

    Args:
        input_data (List[List[float]]): half adder inputs
        output_data (List[List[int]]): half adder outputs
        filename (str): file to write them to
    """
    df_inputs =  pd.DataFrame(input_data, columns=['x1','x2'])
    df_outputs = pd.DataFrame(output_data, columns=['y1','y2'])
    df_inputs_outputs = df_inputs
    df_inputs_outputs[['y1', 'y2']] = df_outputs
    df_inputs_outputs.to_csv(os.path.join(TRAINING_DATA_DIR, filename), index=False)


def read_from_file(filename):
    """Reads the input/output data from a csv file as a dataframe

    Args:
        filename (str): File to read from

    Returns:
        List[List[float], List[int]]: [Half adder inputs, half adder outputs]
    """
    df = pd.read_csv(os.path.join(TRAINING_DATA_DIR, filename), index_col=False)
    inputs = df[['x1','x2']].to_numpy()
    outputs = df[['y1','y2']].to_numpy()
    return [inputs, outputs]


if __name__ == "__main__":
    data_size = 100
    inputs, outputs = generate_n_random_input_outputs(100)
    write_to_file(inputs, outputs, "training_data.csv")
    inputs, outputs = read_from_file("training_data.csv")
