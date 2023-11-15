import tensorflow as tf
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    train_new_model = False
    
    if train_new_model:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=3)

        loss, accuracy = model.evaluate(x_test, y_test)
        print(loss)
        print(accuracy)
    
        model.save('test-model.model')
    else:
        model = tf.keras.models.load_model("test-model.model")
    
    
    
    image_number = 1
    while os.path.isfile("digits/digit{}.png".format(image_number)):
        try:
            digit_image = cv2.imread("digits/digit{}.png".format(image_number))[:, :, 0]
            digit_image = np.invert(np.array([digit_image]))
            prediction = model.predict(digit_image)
            print("The digit is {}".format(np.argmax(prediction)))
            plt.imshow(digit_image[0], cmap=plt.cm.gray)
            plt.show()
            image_number += 1
        except:
            print("Error classifying the image")
            image_number += 1
            
            
if __name__ == "__main__":
    main()
