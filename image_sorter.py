import numpy as np
import os
import glob
import random
import cv2 as cv
from scipy.spatial.distance import cosine
from models import CNNModel
from project_utils import set_seed
from matplotlib import pyplot

class ImageSorter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        set_seed()

        if not os.listdir(input_path):
            raise NameError("Input directory is empty")
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        # For speed optimization, all algorithms are defined here
        self.backSub = cv.createBackgroundSubtractorMOG2()
        self.CNN = CNNModel()

    def read_images(self):
        data_path = self.input_path + "/*"
        files = glob.glob(data_path)
        list_of_images = []
        for f in files:
            current_image = pyplot.imread(f)
            list_of_images.append(current_image)

        #add some randomness to the data (to remove any bias)
        random.shuffle(list_of_images)

        return list_of_images

    def compare_two_images_L2(self, image_1, image_2):
        distance = np.linalg.norm(image_1-image_2)
        return distance

    def compare_two_images_cosine(self, image_1, image_2):
        image_1 = image_1.flatten()
        image_2 = image_2.flatten()
        distance = cosine(image_1, image_2)
        return distance

    def apply_CNN(self, list_of_images):
        list_of_masks = self.CNN.apply_model(list_of_images)
        return list_of_masks

    def substract_background(self, list_of_images, background_sub_iterations = 5):
        list_of_edited_images = []
        for i in range(background_sub_iterations):
            for frame in list_of_images:
                _ = self.backSub.apply(frame)
        for frame in list_of_images:
            mask = self.backSub.apply(frame)
            new_frame = cv.bitwise_and(frame, frame, mask=mask)
            list_of_edited_images.append(new_frame)
        return list_of_edited_images

    def sort(self, type_of_metric = "L2"):
        if(type_of_metric == "L2"):
            score = self.compare_two_images_L2
        elif(type_of_metric == "cosine"):
            score = self.compare_two_images_cosine
        else:
            raise NameError("Uknown type of metric")
        list_of_images = self.read_images()
        images_to_save = list_of_images.copy()
        list_of_images = self.substract_background(list_of_images)
        list_of_images = self.apply_CNN(list_of_images)
        list_of_images = list(zip(images_to_save, list_of_images))
        main_image = list_of_images.pop()
        image_index = 0
        image_name = os.path.join(self.output_path, f"image_{image_index}.jpg")
        pyplot.imsave(image_name, main_image[0])
        while(list_of_images):
            current_image = list_of_images[0]
            smallest_distance = score(main_image[1], current_image[1])
            closest_image_index = 0

            for index, current_image in enumerate(list_of_images):
                distance = score(main_image[1], current_image[1])
                if(distance<smallest_distance):
                    smallest_distance = distance
                    closest_image_index = index
            main_image = list_of_images.pop(closest_image_index)
            image_index = image_index + 1
            image_name = os.path.join(self.output_path, f"image_{image_index}.jpg")
            pyplot.imsave(image_name, main_image[0])
