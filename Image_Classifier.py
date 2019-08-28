from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras import backend as K
import keras

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, input_shape=input_shape, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.2))
# end of first layer

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.2))
# end of second layer

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(rate=0.2))
# end of third layer

model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(units=10, activation='softmax'))
# end of neural network

sgd = SGD(lr=0.01, momentum=0.75, decay=0.15)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=50)
score = model.evaluate(X_test, y_test)
print(f"Test loss is: {score[0]};  Test accuracy is: {score[1]}")

model.save('saved_model.h5')
