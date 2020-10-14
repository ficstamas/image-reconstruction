from skimage.transform import iradon
import matplotlib.pyplot as plt
from skimage.transform import radon
import numpy as np
import os
import cv2
import random

np.random.seed(0)
random.seed(0)


def noisy(noise_typ, image, var=0.1, n=0.05):
    if noise_typ == "gauss":
        row,col = image.shape
        mean = 0
        sigma = var ** 0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        # row,col = image.shape
        s_vs_p = 0.5
        amount = n
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
        return noisy


files = None

for i in os.walk('./images/resized/'):
    files = i[-1]
    break

im_res = {}
for file in files:
    img = cv2.imread(f"./images/resized/{file}", cv2.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
    img[img == 1] = 255
    im_res[file] = img

# ALL experiments

for noise in ["poisson", "speckle"]:
    selection = ["5494.png", "5509.png", "5490.png"]
    e = ""
    for k in [1]:
        e = e + "\n" + f"theta: {k}"
        print(f"theta: {k}")
        a = 4  # scale
        b = 4  # columns
        fig, axs = plt.subplots(len(selection), b)
        fig.set_size_inches(b * a, len(selection) * a)
        i = -1
        for key in selection:
            e = e + f"\nfile name: {key}"
            i += 1
            image = im_res[key].astype(np.float)
            # cv2_imshow(image)
            # image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

            # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
            theta = np.arange(0, 180, k)
            sinogram = radon(image, theta=theta, circle=False)

            noised_sinogram = noisy(noise, sinogram)

            reconstruction_fbp = iradon(noised_sinogram, theta=theta, circle=False)
            error = np.sum(np.abs(reconstruction_fbp - image)) / np.sum(image)
            print(f"FBP rme reconstruction error: {error}")
            e = e + "\n" + str(error)

            axs.flatten()[0].set_title("Original")
            axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

            axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
            axs[i][1].set_xlabel("Projection angle (deg)")
            axs[i][1].set_ylabel("Projection position (pixels)")
            axs[i][1].imshow(noised_sinogram, cmap=plt.cm.Greys_r,
                             extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

            axs[i][2].set_title("Reconstruction\nFiltered back projection")
            axs[i][2].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

            axs[i][3].set_title("Reconstruction error\nFiltered back projection")
            axs[i][3].imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

        fig.tight_layout()
        fig.savefig(f"results/plots/FBP_noise-{noise}_reconstruction_step_{k}.pdf", bbox_inches="tight")
    with open(f"results/errors/FBP_noise-{noise}.txt", mode="w", encoding="utf8") as f:
        f.write(e)


for noise in ["gauss"]:
    for var in [0.05, 0.1, 0.5, 1, 5, 10, 20, 40, 80, 100, 200, 2000, 5000, 50000]:
        # selection = ["5494.png", "5509.png", "5490.png"]
        selection = ["5494.png"]
        e = ""
        for k in [1]:
            e = e + "\n" + f"theta: {k}"
            # print(f"theta: {k}")
            a = 4  # scale
            b = 4  # columns
            fig, axs = plt.subplots(len(selection), b)
            fig.set_size_inches(b * a, len(selection) * a)
            i = -1
            for key in selection:
                e = e + f"\nfile name: {key}"
                i += 1
                image = im_res[key].astype(np.float)
                # cv2_imshow(image)
                # image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

                # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
                theta = np.arange(0, 180, k)
                sinogram = radon(image, theta=theta, circle=False)

                noised_sinogram = noisy(noise, sinogram, var=var)

                reconstruction_fbp = iradon(noised_sinogram, theta=theta, circle=False)
                error = np.sum(np.abs(reconstruction_fbp - image)) / np.sum(image)
                print(f"Var: {var} - FBP rme reconstruction error: {error}")
                e = e + "\n" + str(error)

                axs.flatten()[0].set_title("Original")
                axs[0].imshow(image, cmap=plt.cm.Greys_r)

                axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
                axs[1].set_xlabel("Projection angle (deg)")
                axs[1].set_ylabel("Projection position (pixels)")
                axs[1].imshow(noised_sinogram, cmap=plt.cm.Greys_r,
                                 extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

                axs[2].set_title("Reconstruction\nFiltered back projection")
                axs[2].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

                axs[3].set_title("Reconstruction error\nFiltered back projection")
                axs[3].imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

            fig.tight_layout()
            fig.savefig(f"results/plots/FBP_noise-{noise}-var-{var}_reconstruction_step_{k}.pdf", bbox_inches="tight")
        with open(f"results/errors/FBP_noise-{noise}-var-{var}.txt", mode="w", encoding="utf8") as f:
            f.write(e)

# for noise in ["s&p"]:
#     for n in [0.005, 0.01, 0.02]:
#         selection = ["5494.png", "5509.png", "5490.png"]
#         e = ""
#         for k in [1]:
#             e = e + "\n" + f"theta: {k}"
#             print(f"theta: {k}")
#             a = 4  # scale
#             b = 4  # columns
#             fig, axs = plt.subplots(len(selection), b)
#             fig.set_size_inches(b * a, len(selection) * a)
#             i = -1
#             for key in selection:
#                 e = e + f"\nfile name: {key}"
#                 i += 1
#                 image = im_res[key].astype(np.float)
#                 # cv2_imshow(image)
#                 # image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
#
#                 # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
#                 theta = np.arange(0, 180, k)
#                 sinogram = radon(image, theta=theta, circle=False)
#
#                 noised_sinogram = noisy(noise, sinogram, n=n)
#
#                 reconstruction_fbp = iradon(noised_sinogram, theta=theta, circle=False)
#                 error = np.sum(np.abs(reconstruction_fbp - image)) / np.sum(image)
#                 print(f"FBP rme reconstruction error: {error}")
#                 e = e + "\n" + str(error)
#
#                 axs.flatten()[0].set_title("Original")
#                 axs[i][0].imshow(image, cmap=plt.cm.Greys_r)
#
#                 axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
#                 axs[i][1].set_xlabel("Projection angle (deg)")
#                 axs[i][1].set_ylabel("Projection position (pixels)")
#                 axs[i][1].imshow(noised_sinogram, cmap=plt.cm.Greys_r,
#                                  extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
#
#                 axs[i][2].set_title("Reconstruction\nFiltered back projection")
#                 axs[i][2].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
#
#                 axs[i][3].set_title("Reconstruction error\nFiltered back projection")
#                 axs[i][3].imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)
#
#             fig.tight_layout()
#             fig.savefig(f"results/plots/FBP_noise-{noise}-n-{n}_reconstruction_step_{k}.pdf", bbox_inches="tight")
#         with open(f"results/errors/FBP_noise-{noise}-n-{n}.txt", mode="w", encoding="utf8") as f:
#             f.write(e)