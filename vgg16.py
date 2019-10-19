# Build the network based on pre-trained model VGG16

import numpy as np
from keras.layers import Input, Dense, Reshape, Activation
import keras

xTrain = np.load("data_greyScale/xtrainRandom100.npy")
yTrain = np.load('data_greyScale/ytrainRandom100.npy')
xTest = np.load('data_greyScale/xTestRandom100.npy')
yTest = np.load('data_greyScale/yTestRandom100.npy')

# Normalizes the data so it is floating point between 0 and 1.0
xTrain = xTrain.astype('float32') / 255
xTest = xTest.astype('float32') / 255

# Splits training into training and validation, first 5000 are validation
(xTrain, xValid) = xTrain[5000:], xTrain[:5000]
(yTrain, yValid) = yTrain[5000:], yTrain[:5000]

# Reshape input data from (100, 100) to (100, 100, 1)
w, h = 100, 100
xTrain = xTrain.reshape(xTrain.shape[0], w, h, 3)
xValid = xValid.reshape(xValid.shape[0], w, h, 3)
xTest = xTest.reshape(xTest.shape[0], w, h, 3)

# One-hot encode the labels (so exactly one is expected to be on at the output
yTrain = keras.utils.to_categorical(yTrain,29)

yValid = keras.utils.to_categorical(yValid,29)

yTest = keras.utils.to_categorical(yTest,29)



vgg_base = keras.applications.VGG16(weights = "imagenet", include_top = False, input_shape=(100,100,3))

for layer in vgg_base.layers:
        layer.trainable = False

model = keras.Sequential()

model.add(vgg_base)

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(8192,activation="relu"))
model.add(keras.layers.Dropout(0.8))

model.add(keras.layers.Dense(4096,activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(29, activation='softmax'))

sgd = keras.optimizers.SGD(lr=0.001)



model.compile(optimizer=sgd,
              loss= "categorical_crossentropy",
              metrics=["accuracy"])
checkpoint = keras.callbacks.ModelCheckpoint("checkpoint/checkpoint_VGG2/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

model.load_weights("checkpoint/checkpoint_VGG/weights.12-0.21.hdf5")
model.fit(xTrain, yTrain, epochs=10, batch_size=32,validation_data=(xValid,yValid), shuffle=True, callbacks=[checkpoint])



model.save("model/signlanguage_model_VGG16.h5")

# Evaluate the model on test set
score = model.evaluate(xTest, yTest, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score)
