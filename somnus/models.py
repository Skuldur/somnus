import numpy as np
from tensorflow.keras.layers import Dense, GRU, Bidirectional, Input, Dropout, MaxPooling2D, Conv2D, Flatten, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import losses

class BaseModel():
    def __init__(self):
        self.model = None
        self.filepath = ''

    def compile(self, learning_rate):
        opt = Adam(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    def train(self, data, labels, val_data, val_labels, epochs, save_best, batch_size):
        callbacks = []
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)
        callbacks.append(reduce_lr)

        if save_best:
            checkpoint = ModelCheckpoint(
                self.filepath,
                monitor='loss',
                verbose=0,
                save_best_only=True,
                mode='min'
            )
            callbacks.append(checkpoint)

        self.model.fit(
            x=data,
            y=labels,
            validation_data=(val_data, val_labels),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            callbacks=callbacks
        )

    def predict(self, input):
        p = self.model.predict(input)

        return p.reshape(-1)

    def save(self, name):
        self.model.save_weights(name)

    def load(self, weights_path):
        self.model.load_weights(weights_path)


class CnnTradFPool(BaseModel):
    def __init__(self, input_shape):
        """
        Function creating the model's graph in Keras.
        
        Argument:
            input_shape: shape of the model's input data (using Keras conventions)
        """
        self.filepath = "cnn-trad-f-pool-{epoch:02d}-{loss:.4f}.hdf5"
        
        X_input = Input(shape = input_shape)

        conv1 = Conv2D(64, kernel_size=(66, 8), strides=1, padding='same', activation='relu')(X_input)
        drop1 = Dropout(0.2)(conv1)
        maxpool = MaxPooling2D(pool_size=[3, 3], strides=[3,3], padding='same')(drop1)

        conv2 = Conv2D(64, kernel_size=(32, 4), strides=1, padding='same', activation='relu')(maxpool)
        drop2 = Dropout(0.2)(conv2)
        flattened = Flatten()(drop2)

        dense = Dense(3, activation='softmax')(flattened)

        self.model = Model(inputs = X_input, outputs = dense)


class CnnOneFStride(BaseModel):
    def __init__(self, input_shape):
        """
        Function creating the model's graph in Keras.
        
        Argument:
        input_shape: shape of the model's input data (using Keras conventions)
        """
        self.filepath = "cnn-one-f-stride-{epoch:02d}-{loss:.4f}.hdf5"

        X_input = Input(shape = input_shape)

        conv1 = Conv2D(186, kernel_size=(101, 8), strides=(1,4), padding='valid', activation='relu')(X_input)
        drop1 = Dropout(0.2)(conv1)
        flattened = Flatten()(drop1)

        dense = Dense(128, activation='relu')(flattened)
        dense = Dense(128, activation='relu')(dense)
        dense = Dense(3, activation='softmax')(dense)

        self.model = Model(inputs = X_input, outputs = dense)


class CrnnTimeStride(BaseModel):
    def __init__(self, input_shape):
        """
        Function creating the model's graph in Keras.
        
        Argument:
            input_shape: shape of the model's input data (using Keras conventions)
        """
        self.filepath = "crnn-time-stride-{epoch:02d}-{loss:.4f}.hdf5"
        
        X_input = Input(shape = input_shape)

        conv1 = Conv2D(32, kernel_size=(20, 5), strides=(8,2), padding='same', activation='relu')(X_input)
        bigru1 = TimeDistributed(Bidirectional(GRU(units=32, return_sequences=True)))(conv1)
        bigru2 = TimeDistributed(Bidirectional(GRU(units=32)))(bigru1)
        flatten = Flatten()(bigru2)
        dense1 = Dense(64, activation='relu')(flatten)
        output = Dense(3, activation='softmax')(dense1)

        self.model = Model(inputs = X_input, outputs = output)


# Model utils
def get_model(model_name, shape):
    if model_name == 'cnn-one-stride':
        model = CnnOneFStride(input_shape=shape)
    elif model_name == 'cnn-trad-pool':
        model = CnnTradFPool(input_shape=shape)
    elif model_name == 'crnn-time-stride':
        model = CrnnTimeStride(input_shape=shape)
    else:
        raise ValueError("Model type %s not supported" % model)

    return model
