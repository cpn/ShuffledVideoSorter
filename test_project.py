import os
import unittest
import numpy as np
from matplotlib import pyplot
from image_sorter import ImageSorter


class TestImageSorter(unittest.TestCase):
    def setUp(self):
        os.mkdir("tempInput")
        image = np.random.rand(50,50)
        pyplot.imsave("tempInput/image_1.jpg", image)
        os.mkdir("tempOutput")
        self.ImageSorter = ImageSorter("tempInput", "tempOutput")

    def tearDown(self):
        os.remove("tempInput/image_1.jpg")
        os.rmdir("tempInput")
        os.rmdir("tempOutput")
        del self.ImageSorter

    def test_compare_two_images_L2(self):
        image_1 = np.random.rand(50,50)
        image_2 = np.random.rand(50,50)
        score_1 = self.ImageSorter.compare_two_images_L2(image_1, image_1)
        score_2 = self.ImageSorter.compare_two_images_L2(image_1, image_2)
        self.assertTrue(score_1<score_2)

    def test_compare_two_images_cosine(self):
        image_1 = np.random.rand(50,50)
        image_2 = np.random.rand(50,50)
        score_1 = self.ImageSorter.compare_two_images_cosine(image_1, image_1)
        score_2 = self.ImageSorter.compare_two_images_cosine(image_1, image_2)
        self.assertTrue(score_1<score_2)

    def test_substract_background(self):
        image_1 = np.random.rand(50,50)
        mask = self.ImageSorter.substract_background([image_1])
        self.assertTrue(not np.any(mask))
