import keras
import numpy as np
import os
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import DataGenerator

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(os.path.join(SRC_DIR, ".."), "models")

def build_model():
    learning_rate=0.1
    momentum=0.9
    input_size=2
    output_size=2

    model = Sequential()
    # 3 neurons, 2 inputs, dense means fully connected 
    model.add(Dense(3, activation='relu', input_dim=input_size))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(output_size, activation='sigmoid'))

    #my_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    my_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=my_optimizer, loss='mse', metrics=['mse', 'mae'])

    return model


def train(input, output, model):
    model.fit(input, output, epochs=100, batch_size=1)


def validate(predicted_labels, expected_labels):
    num_correct = 0
    num_total = 0
    for predicted, expected in zip(predicted_labels, expected_labels):
        num_total += 1
        if (predicted[0] == expected[0]) and (predicted[1] == expected[1]):
            num_correct += 1
            print(predicted, "==", expected)
        else:
            print(predicted, "!=", expected)
            pass
    return num_correct / num_total

def predict(model, test_input):
    return np.round(model.predict(test_input), 0).astype(int)


def save_model(model, model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save(model_path)


if __name__ == "__main__":
    #x_train, y_train = DataGenerator.read_from_file("training_data.csv")
    skew = 0.2
    x_train, y_train = DataGenerator.generate_n_random_input_outputs(200, skew)
    x_test, y_test = DataGenerator.generate_n_random_input_outputs(200, skew)
    
    #x_train, x_test, y_train, y_test = train_test_split(inputs, outputs)

    if not True:
        model_name = "Initial_model"
        model = keras.models.load_model(os.path.join(MODELS_DIR, model_name))
    else:
        model = build_model()
        train(x_train, y_train, model)
        save_model(model, "Second_model")

    predicted_labels = predict(model, x_test)
    print(zip(x_test, predicted_labels))
    correct_pct = validate(predicted_labels, y_test)
    print(correct_pct)
    print(model.predict([[-0.08,0.1]]))
    
    DataGenerator.write_to_file(x_train, y_train, "training.csv")
    #print(y_test)
    #print(returns)