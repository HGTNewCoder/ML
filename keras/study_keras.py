import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras import Model
from keras.models import load_model

class LinearRegression:
    def __init__(self):
        return None
    def build(self, input_dim):
        input_layer = keras.Input(shape = (input_dim, ))
        output_layer = keras.layers.Dense(units=1, use_bias=True, activation=None)(input_layer)
        self.model = Model(inputs = input_layer, outputs = output_layer)
        return self.model
    
    def train(self, x_train, y_train):
        opt = tf.keras.optimizers.SGD(learning_rate = 0.1)
        self.model.compile(optimizer=opt, loss='mse')
        return self.model.fit(x = x_train, y = y_train, epochs = 30)
    
    def save(self, model_file):
        return self.model.save(model_file)
    
    def load(self, model_file):
        self.model = load_model(model_file)

    def summary(self):
        self.model.summary()    

    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def get_weights(self):
        return self.model.layers[1].get_weights()

lr = LinearRegression()

x_train = np.arange(-5, 5, 0.5)
n = len(x_train)
std = 2
y_train = -3*x_train + 10 + np.random.normal(0, std, n)

lr.build(1)
lr.train(x_train, y_train)
print("The architecture of Linear Regression model: ")
lr.summary()

#plot
plt.plot(x_train, y_train, 'ro')
plt.show()
