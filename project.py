# Build the network on scratch

import tensorflow as tf
import numpy as np


checkpoint_path ="checkpoint"

xTrain = np.load("data_greyScale/xtrainRandom.npy")
yTrain = np.load('data_greyScale/ytrainRandom.npy')
xTest = np.load('data_greyScale/xTestRandom.npy')
yTest = np.load('data_greyScale/yTestRandom.npy')


print(yTrain)

# Normalizes the data so it is floating point between 0 and 1.0
xTrain = xTrain.astype('float32') / 255
xTest = xTest.astype('float32') / 255

# Splits training into training and validation, first 5000 are validation
(xTrain, xValid) = xTrain[5000:], xTrain[:5000]
(yTrain, yValid) = yTrain[5000:], yTrain[:5000]
print(yValid.shape)


w, h = 100, 100
xTrain = xTrain.reshape(xTrain.shape[0], w, h, 1)
xValid = xValid.reshape(xValid.shape[0], w, h, 1)
xTest = xTest.reshape(xTest.shape[0], w, h, 1)



print(yTrain.shape)
yTrain = tf.keras.utils.to_categorical(yTrain)

yValid = tf.keras.utils.to_categorical(yValid)

yTest = tf.keras.utils.to_categorical(yTest)

print("------------->", yTrain.shape, yValid.shape, yTest.shape)
# Build the model
model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=11, padding='same', activation='relu', input_shape=(100,100,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(29, activation='softmax'))

# Take a look at the model summary
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint/checkpoint_trail1/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                                             save_best_only=False, save_weights_only=False, mode='auto', period=1)


model.load_weights("checkpoint/checkpoint_trail5/weights.20-0.04.hdf5")
model.fit(xTrain,
         yTrain,
         batch_size=64,
         epochs=5,
         validation_data=(xValid, yValid),
         callbacks=[checkpoint])




# (5) Train
# model.load_weights("checkpoint/checkpoint_trail2/weights.25-5.89.hdf5")
# model.fit(xTrain, yTrain, batch_size=32, epochs=10, verbose=1,validation_data=(xValid,yValid), shuffle=True,callbacks=[checkpoint])


model.save("model/signlanguage_model_Basic.h5")


# Load the weights with the best validation accuracy
# model.load_weights('model.weights.best.hdf5')

# model.save("signlanguage_model.h5")

# Evaluate the model on test set
score = model.evaluate(xTest, yTest, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score)

