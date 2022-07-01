import keras
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Load MNIST digit dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
# Reshape dataset to have a single channel and normalize it
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) / 255.0
# One hot encode target values
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Split the train data to create train and validation data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)

# Create CNN with simple architecture since dataset isnt very complicated
# We apply dropout so model doesnt overfit very fast
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Define the optimizer
optimizer = RMSprop(learning_rate=0.001, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Most likely not needed, but I wanted to try it
# Halves the learning rate during training process if the validation accuracy doesnt improve
# for 3 epochs
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# Define generator with image augmentations to help model with unseen data
datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images by max of 10 degrees
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=False,  # flip wont be applied because most numbers are not symmetrical
    vertical_flip=False)  # flip wont be applied because most numbers are not symmetrical

# Fit the generator
datagen.fit(X_train)

# Fit the model
# Training takes about 7 minutes on Google Colaboratory's GPU and converges to
# training_loss: 0.0292 - training_accuracy: 0.9917 - val_loss: 0.0190 - val_accuracy: 0.9943
batch_size = 86
history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    epochs=30,
                    validation_data=(X_val, Y_val),
                    verbose=2,
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    callbacks=[learning_rate_reduction])

# Save the model
model.save("digit_classifier")

# Create model's predictions on the test dataset
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1)
Y_true = np.argmax(Y_test,axis = 1)

# Evaluate the model's accuracy, which was 99.67%
score = accuracy_score(Y_true, Y_pred_classes)
print(score)
