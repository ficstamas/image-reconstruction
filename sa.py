import numpy as np
import random
from skimage.transform.radon_transform import radon
from skimage.feature import local_binary_pattern
from skimage import draw
import cv2 as cv
import math
import os
import copy
import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from queue import Empty
import json
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


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
        self.cost_change = []

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
        x = random.randint(0, self.updated_image.shape[0]-1)
        y = random.randint(0, self.updated_image.shape[0]-1)
        self.updated_image[x, y] = 255 if self.updated_image[x, y] == 0 else 0

    def cost_function(self):
        # projections
        projections = radon(self.updated_image, theta=self.thetas, circle=False, preserve_range=True)
        error = np.sqrt(np.power(np.linalg.norm(projections - self.sinogram), 2)/(projections.shape[0] * projections.shape[1]))
        error = error + (np.sqrt(np.power(np.linalg.norm(projections - self.lbp(self.sinogram)), 2)/(projections.shape[0] * projections.shape[1]))*self.lamb)
        return error

    def iteration(self):
        for i in range(self.N):
            if self.t == self.t_n:
                break
            self.n = i
            self.generation()
            new_c = self.cost_function()
            delta = new_c-self.c
            if delta < 0:
                self.optimal_objects = copy.deepcopy(self.objects)
                self.optimal = copy.deepcopy(self.updated_image)
                self.c = new_c
                self.cost_change.append(self.c)
                self.temperature_change()
                # print(self.c, self.t)
                continue
            if delta == 0:
                continue
            prob = np.exp(-delta/self.t)
            r = random.random()
            if r < prob:
                # print(delta, prob, r)
                self.optimal_objects = copy.deepcopy(self.objects)
                self.optimal = copy.deepcopy(self.updated_image)
                self.c = new_c
                self.cost_change.append(self.c)
                self.temperature_change()
                continue
            self.temperature_change()

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


def rme(image, reconstructed):
    return np.sum(np.abs(reconstructed - image)) / np.sum(image)


def process(task_queue: mp.Queue, progress_queue: mp.Queue):
    while True:
        try:
            file: str
            file, seed, k, n, t_0, l = task_queue.get(block=True, timeout=0.5)
            # logging.info(f"File: {file}, seed: {seed}, k: {k}, n: {n}, t0: {t_0}")
        except Empty:
            break
        random.seed(seed)
        np.random.seed(seed)
        theta = np.arange(0, 180, k)
        img = cv.imread(file, cv.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
        img[img == 1] = 255
        sinogram = radon(img, theta=theta, circle=False, preserve_range=True)

        sa = SA(sinogram=sinogram, thetas=theta, N=n, t_0=t_0, t_n=0, l=l)
        sa.iteration()

        # images
        original_image = img
        original_sinogram = sinogram
        output_image = sa.optimal
        difference_image = np.abs(original_image-sa.optimal)

        # scores
        last_cost = sa.c
        cost_change = sa.cost_change
        rme_score = rme(img, sa.optimal)
        stats = {"cost_change": cost_change, "rme": rme_score}  # just json.dump it into "path.../<stats_filename>"
        # output files
        # TODO put them in a directory like "./data/sa/<filename>"
        path_stats = "sa/results/file_"+str(file[2:-4].replace('/', '-'))+"/stats"
        path_plots = "sa/results/file_"+str(file[2:-4].replace('/', '-'))+"/plots"

        if not os.path.exists(path_stats):
            os.makedirs(path_stats)
        if not os.path.exists(path_plots):
            os.makedirs(path_plots)

        stats_filename = f"stats_{file[2:-4].replace('/', '-')}_seed-{seed}_thetastep-{k}_iter-{n}_lamb-{l}_temp-{t_0}.json"
        plot_filename = f"plot_{file[2:-4].replace('/', '-')}_seed-{seed}_thetastep-{k}_iter-{n}_lamb-{l}_temp-{t_0}.pdf"

        json_object = json.dumps(stats, indent=2)
        with open(os.path.join(path_stats,stats_filename),"w") as outfile:
            outfile.write(json_object)
            outfile.close()
        # TODO plot original image, sinogram, output image, difference image
        # TODO Save everything in file
        #original_sinogram = cv.resize(original_sinogram,(original_sinogram.shape[0],original_sinogram.shape[0]))
        fig, axs = plt.subplots(1, 4)
        fig.set_size_inches(16, 4)
        axs.flatten()[0].set_title("Original")
        axs[0].imshow(original_image, cmap=plt.cm.Greys_r, aspect='auto')

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[1].imshow(original_sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, original_sinogram.shape[0]),aspect='auto')

        axs.flatten()[2].set_title("Output (SA)")
        axs[2].imshow(output_image, cmap=plt.cm.Greys_r, aspect='auto')

        axs.flatten()[3].set_title("Difference")
        axs[3].imshow(difference_image, cmap=plt.cm.Greys_r, aspect='auto')

        fig.tight_layout()
        fig.savefig(os.path.join(path_plots,plot_filename),bbox_inches="tight")
        # This handles the progress bar
        progress_queue.put(1)


def _progress_bar(queue: mp.Queue, total):
    progress = tqdm.tqdm(total=total, unit='dim', desc=f'Progress\t')
    while True:
        try:
            _ = queue.get(True, 0.5)
            progress.n += 1
            progress.update(0)
            if progress.n == total:
                break
        except Empty:
            continue


if __name__ == "__main__":
    mp.freeze_support()

    task_manager = mp.Manager()
    task_queue = task_manager.Queue()
    progress_queue = task_manager.Queue()

    # parameters
    files = ["./images/cropped/5494.png", "./images/cropped/5509.png", "./images/cropped/5490.png"]

    seeds = [0, 1, 2, 3, 4]
    thetas = [1, 10, 30]
    ns = [10, 100, 1000, 5000, 10000, 50000, 100000]
    t0s = [1, 5, 10, 50, 100, 250, 500]
    lamb = [0.1,0.5,1]


    inputs = []

    # preparing tasks
    for file in files:
        for seed in seeds:
            for k in thetas:
                for n in ns:
                    for t0 in t0s:
                        for l in lamb:
                            task_queue.put((file, seed, k, n, t0, l))
                            inputs.append([task_queue, progress_queue])

    progress = mp.Process(target=_progress_bar, args=(progress_queue, task_queue.qsize()))
    progress.start()

    pool = mp.Pool(processes=4)
    with pool as p:
        _ = p.starmap(process, inputs)

    progress.join()

