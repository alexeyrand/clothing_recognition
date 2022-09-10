import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coast', 'Sandal', 'Shirt', 'Sneaker', 'Bag', "Ankle boot"]


#plt.imshow(x_train[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()
x_train = x_train / 255
x_test = x_test / 255

print(class_name[y_train[0]])

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap = plt.cm.binary)
    plt.xlabel(class_name[y_train[i]])
#plt.show()

model = keras.Sequential([
                        keras.layers.Flatten(input_shape = (28, 28)),
                        keras.layers.Dense(128, activation = 'relu'),
                        keras.layers.Dense(10, activation = 'softmax')
                        ])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs = 5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test accuracy:', test_acc)

pred = model.predict(x_train)
item = int(input("Введите цифру: "))
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
print('Результат:', class_name[np.argmax(pred[item])],'\nОтвет:', class_name[y_train[item]])
