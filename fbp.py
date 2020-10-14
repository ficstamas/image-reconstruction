from skimage.transform import iradon
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

# ALL experiments

selection = ["5494.png", "5509.png", "5490.png"]
e = ""
for k in [1, 10, 30]:
    e = e + "\n" + f"theta: {k}"
    print(f"theta: {k}")
    a = 4  # scale
    b = 4  # columns
    fig, axs = plt.subplots(len(im_res), b)
    fig.set_size_inches(b * a, len(im_res) * a)
    i = -1
    for key in im_res:
        e = e + f"\nfile name: {key}"
        i += 1
        image = im_res[key].astype(np.float)
        # cv2_imshow(image)
        # image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

        # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        theta = np.arange(0, 180, k)
        sinogram = radon(image, theta=theta, circle=False)

        reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
        error = np.sum(np.abs(reconstruction_fbp - image)) / np.sum(image)
        print(f"FBP rme reconstruction error: {error}")
        e = e + "\n" + str(error)

        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction\nFiltered back projection")
        axs[i][2].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

        axs[i][3].set_title("Reconstruction error\nFiltered back projection")
        axs[i][3].imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

    fig.tight_layout()
    fig.savefig(f"results/plots/FBP_reconstruction_step_{k}.pdf", bbox_inches="tight")
with open("results/errors/FBP.txt", mode="w", encoding="utf8") as f:
    f.write(e)

# selected experiments

selection = ["5494.png", "5509.png", "5490.png"]
e = ""
for k in [1, 10, 30]:
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

        reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
        error = np.sum(np.abs(reconstruction_fbp - image)) / np.sum(image)
        print(f"FBP rme reconstruction error: {error}")
        e = e + "\n" + str(error)

        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction\nFiltered back projection")
        axs[i][2].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

        axs[i][3].set_title("Reconstruction error\nFiltered back projection")
        axs[i][3].imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

    fig.tight_layout()
    fig.savefig(f"results/plots/FBP_reconstruction_step_{k}_mini.pdf", bbox_inches="tight")

with open("results/errors/FBP_mini.txt", mode="w", encoding="utf8") as f:
    f.write(e)


# =============================================
# cropped images
# ==============================================

for i in os.walk('./images/cropped/'):
    files = i[-1]
    break

im_res = {}
for file in files:
    img = cv2.imread(f"./images/cropped/{file}", cv2.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
    img[img == 1] = 255
    im_res[file] = img

# ALL experiments

selection = ["5494.png", "5509.png", "5490.png"]
e = ""
for k in [1, 10, 30]:
    e = e + "\n" + f"theta: {k}"
    print(f"theta: {k}")
    a = 4  # scale
    b = 4  # columns
    fig, axs = plt.subplots(len(im_res), b)
    fig.set_size_inches(b * a, len(im_res) * a)
    i = -1
    for key in im_res:
        e = e + f"\nfile name: {key}"
        i += 1
        image = im_res[key].astype(np.float)
        # cv2_imshow(image)
        # image = rescale(image, scale=0.4, mode='reflect', multichannel=False)

        # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        theta = np.arange(0, 180, k)
        sinogram = radon(image, theta=theta, circle=False)

        reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
        error = np.sum(np.abs(reconstruction_fbp - image)) / np.sum(image)
        print(f"FBP rme reconstruction error: {error}")
        e = e + "\n" + str(error)

        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction\nFiltered back projection")
        axs[i][2].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

        axs[i][3].set_title("Reconstruction error\nFiltered back projection")
        axs[i][3].imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

    fig.tight_layout()
    fig.savefig(f"results/plots/FBP_cropped_reconstruction_step_{k}.pdf", bbox_inches="tight")
with open("results/errors/FBP_cropped.txt", mode="w", encoding="utf8") as f:
    f.write(e)

# selected experiments

selection = ["5494.png", "5509.png", "5490.png"]
e = ""
for k in [1, 10, 30]:
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

        reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
        error = np.sum(np.abs(reconstruction_fbp - image)) / np.sum(image)
        print(f"FBP rme reconstruction error: {error}")
        e = e + "\n" + str(error)

        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction\nFiltered back projection")
        axs[i][2].imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

        axs[i][3].set_title("Reconstruction error\nFiltered back projection")
        axs[i][3].imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)

    fig.tight_layout()
    fig.savefig(f"results/plots/FBP_cropped_reconstruction_step_{k}_mini.pdf", bbox_inches="tight")

with open("results/errors/FBP_cropped_mini.txt", mode="w", encoding="utf8") as f:
    f.write(e)