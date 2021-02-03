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
    plt.close()


if __name__ == '__main__':
    train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    train_labels.head()
    train_labels.label = train_labels.label.astype('str')

    train_idg = ImageDataGenerator(
        # Fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.2,
        # Degree range for random rotations
        rotation_range=45,
        # Range for random zoom.If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        zoom_range=0.2,
        # Randomly flip inputs horizontally
        horizontal_flip=True,
        # Randomly flip inputs vertically
        vertical_flip=True,
        # Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        shear_range=0.1,
        # Shift fraction of total height
        height_shift_range=0.1,
        # Shift fraction of total width
        width_shift_range=0.1)

    train_generator = train_idg.flow_from_dataframe(
        # Pandas dataframe
        train_labels,
        # Path to the directory to read images from
        directory=os.path.join(WORK_DIR, "train_images"),
        # Subset of data ("training" or "validation") if validation_split is set in ImageDataGenerator
        subset="training",
        # Column in dataframe that contains the filenames
        x_col="image_id",
        # Column in dataframe that has the target data
        y_col="label",
        # The dimensions to which all images found will be resized
        target_size=(TARGET_SIZE, TARGET_SIZE),
        # Size of the batches of data
        batch_size=BATCH_SIZE,
        # Mode for yielding the targets: here "sparse" creates a 1D numpy array of integer labels
        class_mode="sparse")

    validation_idg = ImageDataGenerator(
        # Fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.2)

    validation_generator = validation_idg.flow_from_dataframe(
        # Pandas dataframe
        train_labels,
        # Path to the directory to read images from
        directory=os.path.join(WORK_DIR, "train_images"),
        # Subset of data ("training" or "validation") if validation_split is set in ImageDataGenerator
        subset="validation",
        # Column in dataframe that contains the filenames
        x_col="image_id",
        # Column in dataframe that has the target data
        y_col="label",
        # The dimensions to which all images found will be resized
        target_size=(TARGET_SIZE, TARGET_SIZE),
        # Size of the batches of data
        batch_size=BATCH_SIZE,
        # Mode for yielding the targets: here "sparse" creates a 1D numpy array of integer labels
        class_mode="sparse")

    model = create_model()
    model.summary()

    model_save = ModelCheckpoint(
        # Path to save the model file
        'best_model.h5',
        # Only saves when the model is considered the "best"
        save_best_only=True,
        # Only the model's weights will be saved
        save_weights_only=True,
        # The metric name to monitor.
        monitor='val_loss',
        # If save_best_only=True, the decision to overwrite the current save file is made based
        # on either the maximization or the minimization of the monitored quantity.
        # The mode should be min for val_loss
        mode='min')
    early_stop = EarlyStopping(
        # Quantity to be monitored
        monitor='val_loss',
        # Minimum change in the monitored quantity to qualify as an improvement,
        # i.e. an absolute change of less than min_delta, will count as no improvement.
        min_delta=0.001,
        # Number of epochs with no improvement after which training will be stopped
        patience=5,
        # In min mode, training will stop when the quantity monitored has stopped decreasing
        mode='min',
        # Verbosity mode
        verbose=1,
        # Whether to restore model weights from the epoch with the best value of the monitored quantity
        restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        # Quantity to be monitored
        monitor='val_loss',
        # Factor by which the learning rate will be reduced. new_lr = lr * factor
        factor=0.3,
        # Number of epochs with no improvement after which learning rate will be reduced
        patience=2,
        # Threshold for measuring the new optimum, to only focus on significant changes
        min_delta=0.001,
        # The learning rate will be reduced when the quantity monitored has stopped decreasing
        mode='min')
    tensorboard = TensorBoard(log_dir=f"logs/scalars/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    history = model.fit_generator(
        # A generator
        train_generator,
        # Total number of steps (batches of samples) to yield from `generator` before
        # declaring one epoch finished and starting the next epoch. It should typically
        # be equal to the number of samples of your dataset divided by the batch size.
        steps_per_epoch=len(train_labels) * 0.8 / BATCH_SIZE,
        # Number of epochs to train the model
        epochs=EPOCHS,
        # A generator on which to evaluate the loss and any model metrics at the end of each epoch.
        validation_data=validation_generator,
        # Only relevant if `validation_data` is a generator. Total number of steps
        # (batches of samples) to yield from `validation_data` generator before stopping
        # at the end of every epoch. It should typically be equal to the number of samples
        # of your validation dataset divided by the batch size.
        validation_steps=len(train_labels) * 0.2 / BATCH_SIZE,
        # List of callbacks to apply during training
        callbacks=[model_save, early_stop, reduce_lr, tensorboard],
        # Verbosity mode
        verbose=0
    )

    save_model(model)
    plot_history(history)

    # evaluate model
    model.load_weights("best_model_.h5")
    score = model.evaluate_generator(validation_generator)
    print(f"Score: {score}")
