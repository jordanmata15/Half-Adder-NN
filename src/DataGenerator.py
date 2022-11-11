import numpy as np
import os
import pandas as pd
import random

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATA_DIR = os.path.join(os.path.join(SRC_DIR, ".."), "model_data")

def generate_n_random_inputs(n):
    values = []
    for _ in range(2*n):
        values.append(random.randint(0,1) + random.uniform(-0.2,0.2))
    x1_values = values[:n]
    x2_values = values[n:]
    return np.array(list(zip(x1_values, x2_values)))


def calculate_half_adder_outputs(input_pairs):
    outputs = []
    for x1,x2 in input_pairs:
        x1_bin = round(x1)
        x2_bin = round(x2)
        sum = x1_bin ^ x2_bin
        carry_bit = x1_bin & x2_bin
        outputs.append([sum, carry_bit])
    return np.array(outputs)


def generate_n_random_input_outputs(n):
    inputs = generate_n_random_inputs(n)
    outputs = calculate_half_adder_outputs(inputs)
    return [inputs, outputs]


def write_to_file(input_data, output_data, filename):
    df_inputs =  pd.DataFrame(input_data, columns=['x1','x2'])
    df_outputs = pd.DataFrame(output_data, columns=['y1','y2'])
    df_inputs_outputs = df_inputs
    df_inputs_outputs[['y1', 'y2']] = df_outputs
    df_inputs_outputs.to_csv(os.path.join(TRAINING_DATA_DIR, filename), index=False)


def read_from_file(filename):
    df = pd.read_csv(os.path.join(TRAINING_DATA_DIR, filename), index_col=False)
    print(df)
    inputs = df[['x1','x2']].to_numpy()
    outputs = df[['y1','y2']].to_numpy()
    return [inputs, outputs]


if __name__ == "__main__":
    data_size = 100
    inputs, outputs = generate_n_random_input_outputs(100)
    write_to_file(inputs, outputs, "training_data.csv")
    inputs, outputs = read_from_file("training_data.csv")
