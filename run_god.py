from god import Transform
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tqdm
files = None

for i in os.walk('./images/resized/'):
    files = i[-1]
    break

im_res = {}
for file in files:
    img = cv2.imread(f"./images/resized/{file}", cv2.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
    img[img == 1] = 255
    im_res[file] = img


selection = ["5494.png", "5509.png", "5490.png"]
thetas = [np.arange(0, 180, i) for i in [1, 10, 30]]

for theta in thetas:
    a = 4  # scale
    b = 4  # columns
    fig, axs = plt.subplots(len(selection), b)
    fig.set_size_inches(b * a, len(selection) * a)
    for i, image in enumerate(selection):
        transform = Transform(im_res[image])
        transform.sinogram(theta=theta, circle=False)
        transform.reconstruct()

        # plotting
        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(transform.image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(transform._sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, transform._sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction\nFiltered back projection")
        axs[i][2].imshow(transform._reconstructed, cmap=plt.cm.Greys_r)

        axs[i][3].set_title("Reconstruction error\nFiltered back projection")
        axs[i][3].imshow(transform._reconstructed - transform.image, cmap=plt.cm.Greys_r)

        # save data
        fig.tight_layout()
        path = "results/plots/"
        name = f"FBP_reconstruction_step_{theta[1]}"
        fig.savefig(f"{path}{name}.pdf", bbox_inches="tight")
        transform.log_to_file(f"{path}{name}.txt")


for m in [0, 1, 5, 10, 25, 50, 100]:
    for v in [0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100]:
        for i, image in enumerate(selection):
            transform = Transform(im_res[image])
            transform.sinogram(theta=thetas[0], circle=False)
            transform.gauss(v, m)
            transform.reconstruct()

            # save data
            path = "results/plots/"
            name = f"FBP_gauss-mean-{m}-var-{v}_reconstruction_step_{thetas[0][1]}"
            transform.log_to_file_gauss(f"{path}{name}.txt", sigma=v, mu=m)

for n in [0.005, 0.01, 0.02, 0.1, 0.5]:
    for i, image in enumerate(selection):
        transform = Transform(im_res[image])
        transform.sinogram(theta=thetas[0], circle=False)
        transform.salt_papper(n)
        transform.reconstruct()

        # save data
        path = "results/plots/"
        name = f"FBP_sap-{n}_reconstruction_step_{thetas[0][1]}"
        transform.log_to_file(f"{path}{name}.txt")

for i, image in enumerate(selection):
    transform = Transform(im_res[image])
    transform.sinogram(theta=thetas[0], circle=False)
    transform.speckle()
    transform.reconstruct()

    # save data
    path = "results/plots/"
    name = f"FBP_speckle_reconstruction_step_{thetas[0][1]}"
    transform.log_to_file(f"{path}{name}.txt")


for i, image in enumerate(selection):
    transform = Transform(im_res[image])
    transform.sinogram(theta=thetas[0], circle=False)
    transform.poisson()
    transform.reconstruct()

    # save data
    path = "results/plots/"
    name = f"FBP_poisson_reconstruction_step_{thetas[0][1]}"
    transform.log_to_file(f"{path}{name}.txt")

for theta in thetas:
    for i, image in enumerate([selection[0]]):
        transform = Transform(im_res[image], verbose=False)
        transform.sinogram(theta=theta, circle=True)
        for _ in tqdm.trange(100):
            transform.reconstruct_sart()

        # save data
        path = "results/plots/"
        name = f"SART_reconstruction_step_{theta[1]}"
        transform.log_to_file(f"{path}{name}.txt")


files = None

for i in os.walk('./images/cropped/'):
    files = i[-1]
    break

im_res = {}
for file in files:
    img = cv2.imread(f"./images/cropped/{file}", cv2.IMREAD_GRAYSCALE).astype(np.bool).astype(np.uint8)
    img[img == 1] = 255
    im_res[file] = img

for theta in thetas:
    for i, image in enumerate([selection[0]]):
        transform = Transform(im_res[image], verbose=False)
        transform.sinogram(theta=theta, circle=True)
        for _ in tqdm.trange(100):
            transform.reconstruct_sart()

        # save data
        path = "results/plots/"
        name = f"SART_cropped_reconstruction_step_{theta[1]}"
        transform.log_to_file(f"{path}{name}.txt")


for theta in thetas:
    a = 4  # scale
    b = 4  # columns
    fig, axs = plt.subplots(len(selection), b)
    fig.set_size_inches(b * a, len(selection) * a)
    for i, image in enumerate(selection):
        transform = Transform(im_res[image])
        transform.sinogram(theta=theta, circle=False)
        transform.reconstruct()

        # plotting
        axs.flatten()[0].set_title("Original")
        axs[i][0].imshow(transform.image, cmap=plt.cm.Greys_r)

        axs.flatten()[1].set_title("Radon transform\n(Sinogram)")
        axs[i][1].set_xlabel("Projection angle (deg)")
        axs[i][1].set_ylabel("Projection position (pixels)")
        axs[i][1].imshow(transform._sinogram, cmap=plt.cm.Greys_r,
                         extent=(0, 180, 0, transform._sinogram.shape[0]), aspect='auto')

        axs[i][2].set_title("Reconstruction\nFiltered back projection")
        axs[i][2].imshow(transform._reconstructed, cmap=plt.cm.Greys_r)

        axs[i][3].set_title("Reconstruction error\nFiltered back projection")
        axs[i][3].imshow(transform._reconstructed - transform.image, cmap=plt.cm.Greys_r)

        # save data
        fig.tight_layout()
        path = "results/plots/"
        name = f"FBP_cropped_reconstruction_step_{theta[1]}"
        fig.savefig(f"{path}{name}.pdf", bbox_inches="tight")
        transform.log_to_file(f"{path}{name}.txt")