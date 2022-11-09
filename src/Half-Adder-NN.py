import numpy as np
import random

from sklearn.model_selection import train_test_split

def generate_n_random_pairs(n):
    values = []
    for _ in range(2*n):
        values.append(random.randint(0,1) + random.uniform(-0.2,0.2))
    x1_values = values[:n]
    x2_values = values[n:]
    return np.array(list(zip(x1_values, x2_values)))


def generate_half_adder_outputs(input_pairs):
    outputs = []
    for x1,x2 in input_pairs:
        x1_bin = round(x1)
        x2_bin = round(x2)
        sum = x1_bin ^ x2_bin
        carry_bit = x1_bin & x2_bin
        outputs.append([sum, carry_bit])
    return np.array(outputs)
        

if __name__ == "__main__":
    data_size = 10
    half_adder_inputs = generate_n_random_pairs(data_size)
    half_adder_outputs = generate_half_adder_outputs(half_adder_inputs)
    x_train, x_test, y_train, y_test = train_test_split(half_adder_inputs, half_adder_outputs)