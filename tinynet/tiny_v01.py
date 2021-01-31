from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


def define_learning_model():
    model = Sequential()

    model.add(Rescaling(1./255, input_shape=(800, 600, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5, activation='softmax'))

    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=Adam(lr=0.001), metrics=['accuracy'])

    model.summary()
    return model


def summary(epochs, model, history, tig):
    score = model.evaluate_generator(tig)
    print('Score:', score)

    model_json = model.to_json()
    with open('./results/model.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights('./results/weights.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


class TinyNet:
    def __init__(self, project_path='/Users/alexdarie/Documents/cassava/'):
        self.project_path = project_path
        self.training_path = project_path + 'train_set/'
        self.test_path = project_path + 'test_set/'

    def train(self, epochs=5, batch_size=16):
        model_save = ModelCheckpoint('./results/best_model.h5', save_best_only=True, save_weights_only=True,
                                     monitor='val_loss', mode='min', verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='min', verbose=1,
                                   restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_delta=0.001, mode='min',
                                      verbose=1)

        img_height = 800
        img_width = 600
        train_set = image_dataset_from_directory(self.training_path, validation_split=0.2,
                                                       subset="training",
                                                       seed=123,
                                                       image_size=(img_height, img_width),
                                                       batch_size=batch_size)

        validation_set = image_dataset_from_directory(self.training_path, validation_split=0.2,
                                                      subset="validation",
                                                      seed=123,
                                                      image_size=(img_height, img_width),
                                                      batch_size=batch_size)

        """ Yey.. tf.data.AUTOTUNE isn't available in 2.3.1, or something is not right on dev machine. We are not yet 
        ready to make tiny_v01 crash and burn, so we will improve this in a later version. """

        # train_set = train_set.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
        # validation_set = validation_set.cache().prefetch(tf.data.AUTOTUNE)

        tiny_model = define_learning_model()
        history = tiny_model.fit_generator(train_set, validation_data=validation_set,
                                           epochs=epochs, callbacks=[model_save, early_stop, reduce_lr])

        summary(epochs, tiny_model, history, validation_set)


tiny = TinyNet()
tiny.train(epochs=1)
