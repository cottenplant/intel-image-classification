# intel-image-classification
training a convnet to predict various scenes from the Kaggel Intel Image dataset

Training on gcloud archtitexture

```
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras import optimizers
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from keras import callbacks
import argparse


def model():

    model = Sequential()

    model.add(Conv2D(72, (3, 3), padding='same', input_shape=(150, 150, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(5, 5)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(48, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(24, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(5, 5)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(6, activation='softmax'))

    return model


def main(job_dir, **args):

    logs_path = job_dir + 'logs/tensorboard'

    with tf.device('/device:GPU:0'):

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            job_dir + 'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )

        validation_generator = test_datagen.flow_from_directory(
            job_dir + 'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )

        Model = model()

        Model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.RMSprop(lr=1e-4),
                           metrics=['accuracy'])

        Model.summary()

        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        Model.fit_generator(
            train_generator,
            callbacks=[tensorboard],
            steps_per_epoch=439,
            epochs=4,
            validation_data=validation_generator,
            validation_steps=439
        )

        Model.save('model.h5')
        with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
```
