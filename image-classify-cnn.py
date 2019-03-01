import os
import pickle
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


class ConvNet:

    def __init__(self):

        self.data_dir = 'input'
        self.train_dir = os.path.join(self.data_dir, 'seg_train')
        self.test_dir = os.path.join(self.data_dir, 'seg_test')
        self.pred_dir = os.path.join(self.data_dir, 'seg_pred')
        self.model = models.Sequential()
        self.history = {}
        self.train_generator = None
        self.test_generator = None


    def instantiate_model(self):
        # Need to structure neural network properly
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dense(5, activation='sigmoid'))

        self.model.summary()

        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    def preprocess_data(self):

        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary'
        )

        self.test_generator = test_datagen.flow_from_directory(
            self.train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='binary'
        )

    def fit_model(self):

        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=self.test_generator,
            validation_steps=50
        )

        self.model.save('intel_images_1.h5')

        with open('intel_images_history_1', 'wb') as f:
            pickle.dump(self.history, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cnn = ConvNet()
    cnn.instantiate_model()
    cnn.preprocess_data()
    cnn.fit_model()
