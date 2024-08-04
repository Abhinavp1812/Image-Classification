import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images , testing_images = training_images/255, testing_images/ 255
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


#The below code is to minimize the dataset to limited data (20000)
# training_images = training_images[:20000]
# training_labels = training_labels[:20000]
# testing_images  = testing_images[:4000]
# testing_labels  = testing_labels[:4000]

model_path = 'image_classifier.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded from file")
else:
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), activation ='relu', input_shape = (32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation ='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(10, activation ='softmax'))

    model.compile(optimizer = 'adam', loss ='sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(training_images, training_labels, epochs=10, validation_data = (testing_images, testing_labels))
    loss, accuracy = model.evaluate(testing_images, testing_labels)
    print(f"Loss : {loss}")
    print(f"Accuracy: {accuracy}")
    model.save(model_path)
    print("Model trained and saved to file.")

img = cv.imread('cat_2.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap = plt.cm.binary)
# img =cv.imread('car.jpg')
# img =cv.imread('deer.jpg')
# img =cv.imread('plane.jpg')
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
plt.show()


