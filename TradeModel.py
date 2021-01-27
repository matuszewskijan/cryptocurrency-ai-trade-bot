import math
import tensorflow as tf
from tensorflow.python.keras import Sequential, optimizers
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Input, Activation, BatchNormalization, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from sklearn.metrics import  f1_score, accuracy_score

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class TradeModel:

    def __init__(self, model_name, x_train, model_path = None):
        self.model_name = model_name
        self.model = tf.keras.models.load_model(model_path) if model_path else self.buildModel(x_train)
        print("> New model initialized: ", model_name)

    def buildModel(self, x_train):
        model = Sequential()
        x = len(x_train[0][0])

        model.add(LSTM(512, input_shape=((2,x)), return_sequences = True, activation = "relu"))
        model.add(Dropout(0.20))
        model.add(BatchNormalization())

        model.add(LSTM(256, input_shape=((2, x)), return_sequences=True, activation="relu"))
        model.add(Dropout(0.20))
        model.add(BatchNormalization())

        model.add(LSTM(128, input_shape=((2, x)), return_sequences=False, activation="relu"))
        model.add(Dropout(0.20))
        model.add(BatchNormalization())

        model.add(Dense(64, activation="softmax"))
        model.add(Dropout(0.15))

        model.add(Dense(32, activation="softmax"))
        model.add(Dropout(0.5))

        model.add(Dense(2, activation="softmax"))
        optimizer = tf.keras.optimizers.Adam(lr = 0.01)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='model.png', show_shapes=True)

        return model

    def train(self, x_train, y_train, x_test, y_test, batch_size, epochs):
        custom_early_stopping = EarlyStopping(
            monitor='accuracy', 
            patience=10, 
            min_delta=0.0001, 
            mode='max'
        )

        # x_train = x_train.reshape(-1,1,len(x_train[0]))
        # import pdb; pdb.set_trace()
        
        print("> Training model - ", self.model_name)
        
        return self.model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
                              validation_data=(x_test, y_test), validation_split=0.1, callbacks=[custom_early_stopping])

    def evaluate(self, x_test, y_test):
        # x_test = x_test.reshape(-1, 1, len(x_test[0]))
        print("> Evaluating model - ", self.model_name)
        
        predicated = self.model.predict(x_test)
        predictions = tf.argmax(predicated, 1).numpy().flatten()

        expected_stable = 0
        found_stable = 0
        expected_increase = 0
        found_increase = 0
        expected_decrease = 0
        found_decrease = 0
        # import pdb; pdb.set_trace()
        for i in range(0, len(predictions)):
            if y_test[i] == 2:
                expected_stable += 1
                if predictions[i] == 2:
                    found_stable += 1
            elif y_test[i] == 1:
                expected_increase += 1
                if predictions[i] == 1:
                    found_increase += 1
            elif y_test[i] == 0:
                expected_decrease += 1
                if predictions[i] == 0:
                    found_decrease += 1
        accuracy = accuracy_score(y_test, predictions)
        print(">> Accuracy: ",accuracy)
        print(">> Increase Acc: ", (found_increase / expected_increase), " Decrese Acc: ", found_decrease / expected_decrease)
        loss = self.model.evaluate(x_test, y_test)
        print("Loss: ", loss)
        return loss

    def predict(self,sample):
        sample = sample.reshape(-1, 1, len(sample[0]))
        predictions = self.model.predict(sample)
        prediction = tf.argmax(predictions, 2).numpy().flatten()[0]
        probability = predictions.flatten()[prediction]

        if probability > 0.54:
            return prediction
        else: 
            return 2

    # This model performed much worse in my tests, it's accuraccy were about 50% (so basically coin toss)
    def buildTwoInputModel(self, history_points, technical_indicators):
        lstm_input = Input(shape=(1, history_points.shape[1]), name='lstm_input')
        dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')
        
        # the first branch operates on the first input
        x = LSTM(512, name='lstm_0')(lstm_input)
        x = Dropout(0.2, name='lstm_dropout_0')(x)

        lstm_branch = Model(inputs=lstm_input, outputs=x)
        # the second branch opreates on the second input
        y = Dense(512, name='tech_dense_0', activation="relu")(dense_input)
        y = Dropout(0.2, name='tech_dropout_0')(y)
        technical_indicators_branch = Model(inputs=dense_input, outputs=y)
        
        # combine the output of the two branches
        combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
        
        z = Dense(256, activation="relu", name='dense_pooling')(combined)
        z = Dropout(0.2, name='combined_dropout_0')(z)
        z = Dense(128, activation="relu")(z)
        z = Dropout(0.2, name='combined_dropout_1')(z)
        z = Dense(64, activation="relu")(z)
        z = Dense(2, activation="softmax", name='dense_out')(z)
        
        # our model will accept the inputs of the two branches and then output a single value
        model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)

        adam = tf.keras.optimizers.Adam(lr = 0.01)

        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='model.png', show_shapes=True)

        return model