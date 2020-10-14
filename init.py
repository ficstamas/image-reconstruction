import cv2
import numpy as np
import os

files = None

for i in os.walk('./data/mask/'):
    files = i[-1]
    break

im_res = {}
im_crop = {}

for file in files:
    img = cv2.imread(f"./data/mask/{file}", cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (256, 256)).astype(np.bool).astype(np.uint8)
    resized[resized == 1] = 255

    im_res[file] = resized
    im_crop[file] = resized[:32, :32]

    cv2.imwrite("images/resized/" + file, resized)
    cv2.imwrite("images/cropped/" + file, resized[32:64, 64:96])