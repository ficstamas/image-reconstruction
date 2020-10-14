from skimage.transform import radon, iradon_sart, iradon
import numpy as np


class Transform:
    def __init__(self, image: np.ndarray, verbose=True):
        self.image = image
        self._sinogram = None
        self._reconstructed = None
        self._theta = None
        self._circle = None
        self._iradon_iter = 0
        self._log = []
        self._verbose = verbose

    def sinogram(self, theta: np.ndarray, circle=False):
        self._theta = theta
        self._circle = circle
        self._sinogram = radon(self.image, theta, circle=circle)

    def _bin(self):
        self._reconstructed[self._reconstructed > 120] = 255
        self._reconstructed[self._reconstructed <= 120] = 0
        rme = self.rme()
        self._log.append(f"{self._iradon_iter}: RME {rme}\n")
        if self._verbose:
            print(f"{self._iradon_iter}: RME {rme}")

    def reconstruct(self):
        self._reconstructed = iradon(self._sinogram, theta=self._theta, circle=self._circle)
        self._bin()

    def reconstruct_sart(self):
        if self._iradon_iter == 0:
            self._reconstructed = iradon_sart(self._sinogram, theta=self._theta)
        else:
            self._reconstructed = iradon_sart(self._sinogram, theta=self._theta, image=self._reconstructed)
        self._bin()
        self._iradon_iter += 1

    def rme(self):
        return np.sum(np.abs(self._reconstructed - self.image)) / np.sum(self.image)

    def log_to_file(self, path: str):
        with open(path, mode="a", encoding="utf8") as f:
            f.write("\n")
            f.writelines(self._log)

    def log_to_file_gauss(self, path: str, sigma: float, mu: float):
        with open(path, mode="a", encoding="utf8") as f:
            # f.write("\n")
            f.writelines([f"sigma^2: {sigma}, mu: {mu}, {self._log[0]}"])

    def gauss(self, var, avg):
        row, col = self._sinogram.shape
        mean = avg
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        self._sinogram = self._sinogram + gauss

    def salt_papper(self, n):
        s_vs_p = 0.5
        amount = n
        out = np.copy(self._sinogram)
        # Salt mode
        num_salt = np.ceil(amount * self._sinogram.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in self._sinogram.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * self._sinogram.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in self._sinogram.shape]
        out[coords] = 0
        self._sinogram = out

    def poisson(self):
        vals = len(np.unique(self._sinogram))
        vals = 2 ** np.ceil(np.log2(vals))
        self._sinogram = np.random.poisson(self._sinogram * vals) / float(vals)

    def speckle(self):
        row, col = self._sinogram.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        self._sinogram = self._sinogram + self._sinogram * gauss