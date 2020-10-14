from skimage.transform import iradon_sart
import matplotlib.pyplot as plt
from skimage.transform import radon
import numpy as np
import os
import cv2


files = None

for i in os.walk('./images/resized/'):
    files = i[-1]
    break

im_res = {}
for file in files:
    img = cv2.imread(f"./images/resized/{file}", cv2.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
    img[img == 1] = 255
    im_res[file] = img

e = ""
selection = ["5494.png", "5509.png", "5490.png"]
for k in [1, 10, 30]:
    e = e + "\n" + f"theta: {k}"
    print(f"theta: {k}")
    a = 4  # scale
    b = 6  # columns
    fig, axs = plt.subplots(len(im_res), b)
    fig.set_size_inches(b * a, len(im_res) * a)
    i = -1
    for key in im_res:
        e = e + f"\nfile name: {key}"
        i += 1
        image = im_res[key].astype(np.float)

        theta = np.arange(0, 180, k)
        sinogram = radon(image, theta=theta, circle=True)

        reconstruction_sart = iradon_sart(sinogram, theta=theta)
        print(reconstruction_sart.shape)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        e = e + "\n" + f"iter 1 {error}"
        print(f"SART iter 1 rme reconstruction error: {error}")
        axs[i][2].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 2 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 2 {error}"

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 3 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 3 {error}"
        axs[i][3].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 4 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 4 {error}"
        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 5 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 5 {error}"
        axs[i][4].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction SART\n iter 1")

        axs[i][3].set_title("Reconstruction SART\n iter 3")

        axs[i][4].set_title("Reconstruction SART\n iter 5")

        axs[i][5].set_title("Reconstruction error\nSART")
        axs[i][5].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

    fig.tight_layout()
    fig.savefig(f"results/plots/SART_reconstruction_step_{k}.pdf", bbox_inches="tight")

with open("results/errors/SART.txt", mode="w", encoding="utf8") as f:
    f.write(e)


e = ""
selection = ["5494.png", "5509.png", "5490.png"]
for k in [1, 10, 30]:
    e = e + "\n" + f"theta: {k}"
    print(f"theta: {k}")
    a = 4  # scale
    b = 6  # columns
    fig, axs = plt.subplots(len(selection), b)
    fig.set_size_inches(b * a, len(selection) * a)
    i = -1
    for key in selection:
        e = e + f"\nfile name: {key}"
        i += 1
        image = im_res[key].astype(np.float)

        theta = np.arange(0, 180, k)
        sinogram = radon(image, theta=theta, circle=True)

        reconstruction_sart = iradon_sart(sinogram, theta=theta)
        print(reconstruction_sart.shape)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        e = e + "\n" + f"iter 1 {error}"
        print(f"SART iter 1 rme reconstruction error: {error}")
        axs[i][2].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 2 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 2 {error}"

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 3 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 3 {error}"
        axs[i][3].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 4 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 4 {error}"
        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 5 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 5 {error}"
        axs[i][4].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction SART\n iter 1")

        axs[i][3].set_title("Reconstruction SART\n iter 3")

        axs[i][4].set_title("Reconstruction SART\n iter 5")

        axs[i][5].set_title("Reconstruction error\nSART")
        axs[i][5].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

    fig.tight_layout()
    fig.savefig(f"results/plots/SART_reconstruction_step_{k}_mini.pdf", bbox_inches="tight")

with open("results/errors/SART_mini.txt", mode="w", encoding="utf8") as f:
    f.write(e)


# CROPPED


files = None

for i in os.walk('./images/cropped/'):
    files = i[-1]
    break

im_res = {}
for file in files:
    img = cv2.imread(f"./images/cropped/{file}", cv2.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
    img[img == 1] = 255
    im_res[file] = img

e = ""
selection = ["5494.png", "5509.png", "5490.png"]
for k in [1, 10, 30]:
    e = e + "\n" + f"theta: {k}"
    print(f"theta: {k}")
    a = 4  # scale
    b = 6  # columns
    fig, axs = plt.subplots(len(im_res), b)
    fig.set_size_inches(b * a, len(im_res) * a)
    i = -1
    for key in im_res:
        e = e + f"\nfile name: {key}"
        i += 1
        image = im_res[key].astype(np.float)

        theta = np.arange(0, 180, k)
        sinogram = radon(image, theta=theta, circle=True)

        reconstruction_sart = iradon_sart(sinogram, theta=theta)
        print(reconstruction_sart.shape)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        e = e + "\n" + f"iter 1 {error}"
        print(f"SART iter 1 rme reconstruction error: {error}")
        axs[i][2].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 2 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 2 {error}"

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 3 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 3 {error}"
        axs[i][3].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 4 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 4 {error}"
        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 5 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 5 {error}"
        axs[i][4].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction SART\n iter 1")

        axs[i][3].set_title("Reconstruction SART\n iter 3")

        axs[i][4].set_title("Reconstruction SART\n iter 5")

        axs[i][5].set_title("Reconstruction error\nSART")
        axs[i][5].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

    fig.tight_layout()
    fig.savefig(f"results/plots/SART_cropped_reconstruction_step_{k}.pdf", bbox_inches="tight")

with open("results/errors/SART_cropped.txt", mode="w", encoding="utf8") as f:
    f.write(e)

e = ""
selection = ["5494.png", "5509.png", "5490.png"]
for k in [1, 10, 30]:
    e = e + "\n" + f"theta: {k}"
    print(f"theta: {k}")
    a = 4  # scale
    b = 6  # columns
    fig, axs = plt.subplots(len(selection), b)
    fig.set_size_inches(b * a, len(selection) * a)
    i = -1
    for key in selection:
        e = e + f"\nfile name: {key}"
        i += 1
        image = im_res[key].astype(np.float)

        theta = np.arange(0, 180, k)
        sinogram = radon(image, theta=theta, circle=True)

        reconstruction_sart = iradon_sart(sinogram, theta=theta)
        print(reconstruction_sart.shape)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        e = e + "\n" + f"iter 1 {error}"
        print(f"SART iter 1 rme reconstruction error: {error}")
        axs[i][2].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 2 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 2 {error}"

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 3 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 3 {error}"
        axs[i][3].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 4 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 4 {error}"
        reconstruction_sart = iradon_sart(sinogram, theta=theta, image=reconstruction_sart)
        error = np.sum(np.abs(reconstruction_sart - image)) / np.sum(image)
        print(f"SART iter 5 rme reconstruction error: {error}")
        e = e + "\n" + f"iter 5 {error}"
        axs[i][4].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r)

        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction SART\n iter 1")

        axs[i][3].set_title("Reconstruction SART\n iter 3")

        axs[i][4].set_title("Reconstruction SART\n iter 5")

        axs[i][5].set_title("Reconstruction error\nSART")
        axs[i][5].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

    fig.tight_layout()
    fig.savefig(f"results/plots/SART_cropped_reconstruction_step_{k}_mini.pdf", bbox_inches="tight")

with open("results/errors/SART_cropped_mini.txt", mode="w", encoding="utf8") as f:
    f.write(e)