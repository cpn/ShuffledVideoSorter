import unittest
import numpy as np
import os
from matplotlib import pyplot

class TestImageSorter(unittest.TestCase):
    def create_ImageSorter(self):
        os.mkdir("tempInput")
        image = np.random.rand(50,50)
        pyplot.imsave("tempInput/image_1.jpg", image)
        os.mkdir("tempOutput")
        from image_sorter import ImageSorter
        return ImageSorter("tempInput", "tempOutput")

    def remove_temporary_directories(self):
        os.remove("tempInput/image_1.jpg")
        os.rmdir("tempInput")
        os.rmdir("tempOutput")

    def test_compare_two_images_L2(self):
        ImageSorter = self.create_ImageSorter()
        image_1 = np.random.rand(50,50)
        image_2 = np.random.rand(50,50)
        score_1 = ImageSorter.compare_two_images_L2(image_1, image_1)
        score_2 = ImageSorter.compare_two_images_L2(image_1, image_2)
        self.assertTrue(score_1<score_2)
        self.remove_temporary_directories()

    def test_compare_two_images_cosine(self):
        ImageSorter = self.create_ImageSorter()
        image_1 = np.random.rand(50,50)
        image_2 = np.random.rand(50,50)
        score_1 = ImageSorter.compare_two_images_cosine(image_1, image_1)
        score_2 = ImageSorter.compare_two_images_cosine(image_1, image_2)
        self.assertTrue(score_1<score_2)
        self.remove_temporary_directories()

    def test_substract_background(self):
        ImageSorter = self.create_ImageSorter()
        image_1 = np.random.rand(50,50)
        mask = ImageSorter.substract_background([image_1])
        self.assertTrue(not np.any(mask))
        self.remove_temporary_directories()

