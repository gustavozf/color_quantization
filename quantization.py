'''
PDI - Prof Franklin Cesar Flores
Trabalho 1 - Quantization


Gustavo Zanoni Felipe
'''
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

MAX = 256

def qtz_median_cut(img, n):
    hei, wid = img.shape[:2]

    if n == 1:
        return n
    else:
        n /= 2
        return 

def qtz_cube_cut(img, n):
    global MAX
    
    hei, wid = img.shape[:2]
    img = img.reshape((hei * wid, 3))

    steps = MAX//n
    print(img.shape)

    for i in range(1, n+1):
        for j in range(3):
            args = np.argwhere((img[:,j] >=steps*(i-1)) & (img[:,j] <= steps*i))
            img[args[:, 0], j] = img[args[:, 0], j].max()
            print(args[:, 0].shape)

    return img.reshape((hei, wid, 3))

def qtz_k_means(img, n):
    hei, wid = img.shape[:2]

    img = img.reshape((hei * wid, 3))
    cluster = MiniBatchKMeans(n_clusters = n)
    labels = cluster.fit_predict(img)
    qtz = cluster.cluster_centers_.astype("uint8")[labels]    
    
    return qtz.reshape((hei, wid, 3))

input_img = cv2.imread('./rgb_cube.png', 1)
print(input_img.shape)
cv2.imshow('x', input_img)
cv2.imshow('cube-cut', qtz_cube_cut(input_img.copy(), 4))
cv2.imshow('k-means', qtz_k_means(input_img.copy(), 4))
cv2.waitKey(0)
