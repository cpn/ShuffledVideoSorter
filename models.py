import tensorflow as tf
import numpy as np
import cv2 as cv
INPUT_SHAPE = (160,160, 3) #TF tutorial values

class CNNModel:
    def __init__(self, input_shape = INPUT_SHAPE):
        self.input_shape = input_shape
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       include_top=False,
                                                       weights='imagenet')

    def reshape(self, list_of_images):
        list_of_images = list(map(lambda x:cv.resize(x, self.input_shape[:2]), list_of_images))
        list_of_images = np.array(list_of_images)
        list_of_images = tf.cast(list_of_images, tf.float32)
        list_of_images = (list_of_images / 127.5) - 1
        return list_of_images

    def apply_model(self, list_of_images):
        list_of_images = self.reshape(list_of_images)
        prediction = self.base_model.predict(list_of_images)
        return list(prediction)
