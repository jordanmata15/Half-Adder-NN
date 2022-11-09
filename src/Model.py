import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import DataGenerator

def build_model():
    learning_rate=0.001
    input_size = 2
    output_size = 2

    model = Sequential()
    # 3 neurons, 2 inputs, dense means fully connected 
    model.add(Dense(3, activation='relu', input_dim=input_size))
    model.add(Dense(3, activation='relu'))
    #model.add(Dense(3, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))

    my_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=my_optimizer, loss='mae', metrics=['mae'])

    return model

def train(input, output, model):
    model.fit(input, output, epochs=100, batch_size=1)

def validate():
    pass

if __name__ == "__main__":
    inputs, outputs = DataGenerator.generate_n_random_input_outputs(200)
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs)

    model = build_model()
    train(x_train, y_train, model)