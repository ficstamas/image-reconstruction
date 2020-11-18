import numpy as np
import random
from skimage.transform.radon_transform import radon
from skimage.feature import local_binary_pattern
from skimage import draw
import cv2 as cv
import math
import copy
import tqdm
import matplotlib.pyplot as plt


# TODO test the code
# TODO \lambda * g(x) prior information to cost function
# TODO draw objects on self.updated_image (image) 💩
# TODO rest of SA (acceptance, loss)


class SA:
    def __init__(self, sinogram: np.ndarray, thetas, image_shape=[32, 32], N=5, l=0.1, t_0=500, t_n=50):
        self.sinogram = sinogram
        self.image_shape = image_shape

        # images
        self.optimal = np.zeros(image_shape)
        self.updated_image = np.zeros(image_shape)

        # objects on image
        self.objects = {}
        self.optimal_objects = {}

        # hyperparameters
        self.thetas = thetas
        self.N = N
        self.lamb = l
        self.t_0 = t_0
        self.t_n = t_n

        # other params
        self._i = 0
        self.n = 1
        self.c = math.inf
        self.t = t_0

    def add(self):
        self.objects[self._i] = [random.randint(0, self.image_shape[0]),         # x
                                 random.randint(0, self.image_shape[0]),         # y
                                 random.randint(1, self.image_shape[0] // 10)]         # r
        self._i += 1

    def remove(self, i):
        del self.objects[i]

    def resize(self, i, r):
        self.objects[i][2] += r

    def move(self, i, x, y):
        self.objects[i][1] += y
        self.objects[i][0] += x

    def temperature_change(self):
        self.t = self.t_0 - self.n * ((self.t_0 - self.t_n)/self.N)

    def generation(self):
        self.objects = copy.deepcopy(self.optimal_objects)
        self.updated_image = copy.deepcopy(self.optimal)
        if self.objects.__len__() == 0:
            self.add()
            self.draw()
            return
        r = random.randint(0, 100)
        if r <= 5:
            self.add()
        elif 5 < r <= 10:
            self.remove(random.choice([x for x in self.objects]))
        elif 10 < r <= 45:
            self.resize(random.choice([x for x in self.objects]),
                        random.randint(-5, 5))
        else:
            self.move(random.choice([x for x in self.objects]),
                      random.randint(-5, 5), random.randint(-5, 5))
        self.draw()

    def cost_function(self):
        # projections
        projections = radon(self.updated_image, theta=self.thetas, circle=False)
        regularization = np.power(np.linalg.norm(projections), 2) * self.lamb
        error = np.power(np.linalg.norm(projections - self.sinogram), 2) + regularization
        return error

    def iteration(self):
        for i in tqdm.trange(self.N):
            if self.t == self.t_n:
                break
            self.n = i
            self.generation()
            new_c = self.cost_function()
            if new_c < self.c:
                self.optimal_objects = copy.deepcopy(self.objects)
                self.optimal = copy.deepcopy(self.updated_image)
                self.c = new_c
                self.temperature_change()
                # print(self.c, self.t)
                continue
            # prob = np.exp(-np.abs(self.c-new_c)/self.t)
            # r = random.random()
            # if r < prob:
            #     print(self.c-new_c, prob, r)
            #     self.optimal_objects = copy.deepcopy(self.objects)
            #     self.optimal = copy.deepcopy(self.updated_image)
            #     self.c = new_c
            #     self.temperature_change()
            #     continue

    def draw(self):
        for i in self.objects.keys():
            rr, cc = draw.disk(center=(self.objects[i][0], self.objects[i][1]),
                               radius=self.objects[i][2],
                               shape=self.updated_image.shape)
            self.updated_image[rr, cc] = 255

    def lbp(self, image, radius=1, method="uniform"):
        """
        :param image: gray scale image
        :param radius: maximum distance from the kernel center point
        :param method: can be 'default', 'ror', 'uniform', 'nri_uniform', 'var'
        :return: local binary pattern matrix
        """
        n_points = 8*radius
        return local_binary_pattern(image=image, P=n_points, R=radius, method=method)


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    img = cv.imread(f"./images/cropped/5488.png", cv.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
    img[img == 1] = 255
    sinogram = radon(img, theta=[0, 30, 60, 90, 120, 150], circle=False)
    sa = SA(sinogram=sinogram, thetas=[0, 30, 60, 90, 120, 150], N=1000000)
    sa.iteration()

    cv.imshow("Output", cv.resize(sa.optimal, (32*10, 32*10)))
    cv.imshow("Original", cv.resize(img, (32*10, 32*10)))
    cv.waitKey(0)
