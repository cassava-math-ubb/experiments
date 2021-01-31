from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras import models
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os

WORK_DIR = '/home/teofana/cassava leafs'
BATCH_SIZE = 2
EPOCHS = 10
TARGET_SIZE = 512


def create_model():
    _model = models.Sequential()

    _model.add(EfficientNetB0(include_top=False, weights='imagenet',
                              input_shape=(TARGET_SIZE, TARGET_SIZE, 3)))

    _model.add(layers.GlobalAveragePooling2D())
    _model.add(layers.Dense(5, activation="softmax"))

    _model.compile(optimizer=Adam(lr=0.001),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return _model


def save_model(_model):
    model_json = _model.to_json()
    with open('model.json', "w") as json_file:
        json_file.write(model_json)
    _model.save_weights('weights.h5')


def plot_history(_history):
    # summarize history for accuracy
    plt.plot(_history.history['accuracy'])
    plt.plot(_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.close()

    # summarize history for loss
    plt.plot(_history.history['loss'])
    plt.plot(_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss.png')


if __name__ == '__main__':
    train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    train_labels.head()
    train_labels.label = train_labels.label.astype('str')

    train_idg = ImageDataGenerator(validation_split=0.2,
                                   preprocessing_function=None,
                                   rotation_range=45,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest',
                                   shear_range=0.1,
                                   height_shift_range=0.1,
                                   width_shift_range=0.1)

    train_generator = train_idg.flow_from_dataframe(train_labels,
                                                    directory=os.path.join(WORK_DIR, "train_images"),
                                                    subset="training",
                                                    x_col="image_id",
                                                    y_col="label",
                                                    target_size=(TARGET_SIZE, TARGET_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="sparse")

    validation_idg = ImageDataGenerator(validation_split=0.2)

    validation_generator = validation_idg.flow_from_dataframe(train_labels,
                                                              directory=os.path.join(WORK_DIR, "train_images"),
                                                              subset="validation",
                                                              x_col="image_id",
                                                              y_col="label",
                                                              target_size=(TARGET_SIZE, TARGET_SIZE),
                                                              batch_size=BATCH_SIZE,
                                                              class_mode="sparse")

    model = create_model()
    model.summary()

    model_save = ModelCheckpoint('best_model.h5',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001,
                               patience=5, mode='min', verbose=1,
                               restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                  patience=2, min_delta=0.001,
                                  mode='min')
    tensorboard = TensorBoard(log_dir=f"logs/scalars/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_labels) * 0.8 / BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(train_labels) * 0.2 / BATCH_SIZE,
        callbacks=[model_save, early_stop, reduce_lr, tensorboard],
        verbose=0
    )

    save_model(model)
    plot_history(history)

    # evaluate model
    model.load_weights("best_model_.h5")
    score = model.evaluate_generator(validation_generator)
    print(f"Score: {score}")
