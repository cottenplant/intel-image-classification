import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras import optimizers


class ConvNet:

    def __init__(self):
        self.model = Sequential()
        self.trained = None
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True
        )
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        self.train_generator = self.train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )
        self.validation_generator = self.test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )
        self.network()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.RMSprop(lr=1e-4),
                           metrics=['accuracy'])

    def __str__(self):
        return str(self.model.summary())

    def network(self):
        self.model.add(Conv2D(72, (3, 3), padding='same', input_shape=(150, 150, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(5, 5)))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(48, (3, 3), activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(Conv2D(24, (3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(5, 5)))

        self.model.add(Flatten())

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(96, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(6, activation='softmax'))

    def train(self):
        self.trained = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=439,
            epochs=20,
            validation_data=self.validation_generator,
            validation_steps=439
        )
        self.learning_curve()

    def learning_curve(self):
        plt.plot(self.trained.history['acc'])
        plt.plot(self.trained.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('figures/model_accuracy.png')
        plt.show()

        plt.plot(self.trained.history['loss'])
        plt.plot(self.trained.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('figures/model_loss.png')
        plt.show()


if __name__ == '__main__':
    net = ConvNet()
    net.train()
