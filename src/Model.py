import keras
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

import DataGenerator

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(os.path.join(SRC_DIR, ".."), "models")
HISTORY_DIR = os.path.join(os.path.join(SRC_DIR, ".."), "training_history")

def build_model():
    """Builds and compiles our model

    Returns:
        kermas.model: The model we will train
    """
    learning_rate=0.1
    momentum=0.9
    input_size=2
    output_size=2
    model = Sequential()
    model.add(Dense(3, 
                    activation='relu', 
                    input_dim=input_size, 
                    bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dense(output_size, 
                    activation='sigmoid', 
                    bias_initializer=keras.initializers.Constant(0.1)))
    my_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=my_optimizer, loss='mse', metrics=['mse', 'mae'])
    return model


def train(input, output, model):
    """Trains the model with a given set of input and outputs

    Args:
        input (List[List[int]]): List of pairs of half adder inputs
        output (List[List[int]]): List of pairs of half adder outputs
        model (Keras.model): Model to train

    Returns:
        Keras.model: The trained model
    """
    return model.fit(input, output, epochs=100, batch_size=1)


def validate(predicted_labels, expected_labels):
    """Validates that all of the predicted labels match the 
    expected labels. Will print out any that don't match and return
    the percent correctly predicted.

    Args:
        predicted_labels (List[List[int]]): List of the predicted half adder outputs
        expected_labels (List[List[int]]): List of the actual half adder outputs

    Returns:
        float: percentage correctly identified
    """
    num_correct = 0
    num_total = 0
    for predicted, expected in zip(predicted_labels, expected_labels):
        num_total += 1
        if (predicted[0] == expected[0]) and (predicted[1] == expected[1]):
            num_correct += 1
        else:
            print(predicted, "!=", expected)
    return num_correct / num_total


def predict(model, test_input):
    """Predicts the output of a given list of half adder inputs

    Args:
        model (Keras.model): Model to predict with
        test_input (List[List[int]]): Half adder inputs to predict on

    Returns:
        List[List[int]]: Expected outputs (rounded to either 0 or 1)
    """
    return np.round(model.predict(test_input), 0).astype(int)


def save_model(model, model_name):
    """Saves the model to the "models" directory with a given name

    Args:
        model (keras.model): model to save
        model_name (str): name of the model directory
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path)


if __name__ == "__main__":
    skew = 0.2 # how much noise to add to the data
    x_test, y_test = DataGenerator.generate_n_random_input_outputs(200, skew)

    if True:
        # load an existing model
        model_name = "generated_model"
        model = keras.models.load_model(os.path.join(MODELS_DIR, model_name))
    else:
        # create a new model and train it
        model = build_model()
        x_train, y_train = DataGenerator.generate_n_random_input_outputs(200, skew)
        history = train(x_train, y_train, model)
        save_model(model, "generated_model")
        history_df = pd.DataFrame.from_dict(history.history)
        history_df.index.name = "epoch"
        history_df.to_csv(os.path.join(HISTORY_DIR, "history.csv"))


    predicted_labels = predict(model, x_test)
    correct_pct = validate(predicted_labels, y_test)
    print("Correctly predicted percentage: ", 100*correct_pct)