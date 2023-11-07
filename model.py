import logging
from random import seed

import numpy as np

import cv2

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras.engine.saving import model_from_json
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D
# Import other libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# Import tensorflow
import tensorflow as tf
from tensorflow import keras as tfk
seed = 42
np.random.seed(seed)
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
tf.config.list_physical_devices('GPU')
print(tf.__version__)



class CNNModel:
    def __init__(self):
        self.model = self.create_model()
        # if self.loadmodel() is None:
        #     self.model = self.createModel()
        #     #self.model = None
        # else:
        #     self.model = None
        #     #self.model = self.loadmodel()
        # # self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def load_dataset(self):
        data = np.load('public_data.npz', allow_pickle=True)
        data_arr = data['data']
        data_arr = data_arr.astype('uint8')
        labels_arr = data['labels']
        return data_arr, labels_arr

    def preprocess_data(self, data_arr, labels_arr):
        normalized_data = (data_arr / 255).astype('float32')
        normalized_labels = labels_arr
        criteria = normalized_labels == 'healthy'
        normalized_labels[criteria] = 1
        criteria_0 = normalized_labels == 'unhealthy'
        normalized_labels[criteria_0] = 0
        normalized_labels = normalized_labels.astype('float32')
        return normalized_data, normalized_labels

    def create_data_split(self, normalized_data, normalized_labels):
        X_train, X_val, y_train, y_val = train_test_split(normalized_data, normalized_labels, random_state=seed,
                                                          test_size=0.2,
                                                          stratify=normalized_labels)

        # Print the shapes of the resulting datasets
        print("Training Data Shape:", X_train.shape)
        print("Training Label Shape:", y_train.shape)
        print("Validation Data Shape:", X_val.shape)
        print("Validation Label Shape:", y_val.shape)
        return X_train, X_val, y_train, y_val

    # def loadmodel():
    #     try:
    #         json_file = open('model.json', 'r')
    #         loaded_model_json = json_file.read()
    #         json_file.close()
    #         loaded_model = model_from_json(loaded_model_json)
    #         loaded_model.load_weights('model.h5')
    #         return loaded_model
    #     except FileNotFoundError:
    #         return None

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(96, 96, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 2nd CNN layer
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd CNN layer
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4th CNN layer
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        # Fully connected 1st layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        # Fully connected layer 2nd layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(1, activation='sigmoid'))

        opt = Adam(lr=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self,  X_train, y_train, X_val, y_val, batch_size, epochs):
        early_stopping = tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max',
                                                     restore_best_weights=True)

        # Train the model and save its history
        history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val)
            # callbacks=[early_stopping]
        ).history

        return history

    def predict(self, X):

        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)  # Shape [BS]
        return out
