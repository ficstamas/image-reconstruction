import numpy as np
import random
from skimage.transform.radon_transform import radon


# TODO test the code
# TODO \lambda * g(x) prior information to cost function
# TODO draw objects on self.updated_image (image) 💩
# TODO rest of SA (acceptance, loss)

class SA:
    def __init__(self, sinogram: np.ndarray, thetas, N=5):
        self.image = sinogram
        self.updated_image = np.zeros([sinogram.shape[0], sinogram.shape[0]])
        self.objects = {}
        self.thetas = thetas
        self._i = 0
        self.t_0 = 0
        self.N = N
        self.n = 1

    def add(self):
        self.objects[self._i] = [random.randint(0, self.image.shape[0]),    # x
                                 random.randint(0, self.image.shape[0]),    # y
                                 random.randint(1, self.image.shape[0]//2)] # r
        self._i += 1

    def remove(self, i):
        del self.objects[i]

    def resize(self, i, r):
        self.objects[i][2] += r

    def move(self, i, x, y):
        self.objects[i][1] += y
        self.objects[i][0] += x

    def temperature_change(self, new_t):
        return self.t_0 - self.n * ((self.t_0 - new_t)/self.N)

    def generation(self):
        if self.objects.__len__() == 0:
            self.add()
            return
        r = random.randint(1, 100)
        if r <= 10:
            self.add()
        elif 10 < r <= 15:
            self.remove(random.choice([x for x in self.objects]))
        elif 15 < r <= 60:
            self.resize(random.choice([x for x in self.objects]),
                        random.randint(-5, 5))
        else:
            self.move(random.choice([x for x in self.objects]),
                      random.randint(-5, 5), random.randint(-5, 5))

    def cost_function(self):
        # projections
        projections = radon(self.updated_image, theta=self.thetas)
        error = np.linalg.norm(projections - self.image)
