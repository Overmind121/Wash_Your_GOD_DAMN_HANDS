import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 0 = Not_Washing, 1= Washing

class_names = ["Not_Washing", "Washing"]

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
print(len(X_train))
def create_model():
    model = Sequential()

    model.add(Conv2D(64, (3,3), padding = 'same', input_shape=(50,50,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.5))

    model.add(Conv2D(128, (5,5), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.5))

    model.add(Conv2D(512, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.5))

    model.add(Conv2D(1024, (3,3), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.5))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=.3, verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(test_acc)

    model.save("Wash_Your_GOD_DAMN_HANDS.h5")

model_test = tf.keras.models.load_model('/Users/lukek/PycharmProjects/WashYourHands/Wash_Your_GOD_DAMN_HANDS.h5')
for i in range(20):
    test_img = X_test[i].reshape(-1, 50, 50, 1)
    prediction = model_test.predict(test_img)
    mod_img = X_test[i]
    plt.imshow(mod_img.reshape(50,50))
    plt.title("Prediction: "+class_names[np.argmax(prediction)])
    print(prediction)
    plt.xlabel("Actual: "+class_names[int(y_test[i])])
    plt.show()