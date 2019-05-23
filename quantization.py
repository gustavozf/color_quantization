'''
PDI - Prof Franklin Cesar Flores
Trabalho 1 - Quantization


Gustavo Zanoni Felipe
'''
import numpy as np
import cv2, os
from sklearn.cluster import MiniBatchKMeans

class ColorQuantization():
    def __init__(self):
        self.quatizators = {
            #'median_cut' : self.median_cut,
            'uniform_cut': self.uniform_cut,
            'k_means'    : self.k_means
        }


    # ----------------------------------------------------- MEDIAN CUT
    def median_cut(self, img, n):
        img = img.copy()

        V = self.__get_V(img)
        max_channel = self.__get_max(V)

        while n > 1:
            
            n //= 2
            
        
        return img

    def __get_V(self, img):
        b = img[:,:, 0]
        g = img[:,:, 1]
        r = img[:,:, 2]

        return [
            [b.min(), b.max()],
            [g.min(), g.max()],
            [r.min(), r.max()]
        ]

    def __get_max(self, V):

        return np.argmax([
            V[0, 1] - V[0, 0],
            V[1, 1] - V[1, 0],
            V[2, 1] - V[2, 0]
        ])

    # ----------------------------------------------------- CUBE CUT

    def uniform_cut(self, img, n, MAX=256):
        img = img.copy()

        hei, wid = img.shape[:2]
        img = img.reshape((hei * wid, 3))
        buckets = self.__get_buckets_cube(n)

        # for B, G and R
        for channel in range(3):
            steps = MAX//buckets[channel]

            # for each bucket
            for i in range(1, buckets[channel]+1):
                args = np.argwhere((img[:,channel] >=steps*(i-1)) & (img[:,channel] < steps*i))[:, 0] #<= steps*i))
                img[args, channel] = int(np.mean(img[args, channel]))

        return img.reshape((hei, wid, 3))


    def __get_buckets_cube(self, n):
        x = [0,0,0]
        p = len(bin(n)) - 3

        for i in range(p):
            j = i%3
            x[j] += 1
        
        return [2**x[0], 2**x[1], 2**x[2]]

    # ----------------------------------------------------- K-MEDIANAS

    def k_means(self, img, n):
        img = img.copy()
        hei, wid = img.shape[:2]
        img = img.reshape((hei * wid, 3))

        cluster = MiniBatchKMeans(n_clusters = n)
        labels = cluster.fit_predict(img)
        qtz = cluster.cluster_centers_.astype("uint8")[labels]    
        
        return qtz.reshape((hei, wid, 3))

    def PSNR(self, img_orig, img_quant, MAX=256.0):
        MAX -= 1

        mse = np.mean((img_orig - img_quant) ** 2)
        if mse == 0:
            return 100

        return 20 * np.log10(MAX / np.sqrt(mse))

    def every_quantization(self, img, name):
        output = "./outputs/{}/".format(name)

        if not os.path.isdir("./outputs/"):
            os.mkdir("./outputs/")

        if not os.path.isdir(output):
            os.mkdir(output)

        print("Input name: "+ name)
        for quantizator in self.quatizators.keys():
            print("Quantizator: " + quantizator)
            current_output = output + quantizator +"/"

            if not os.path.isdir(current_output):
                os.mkdir(current_output)

            for i in [2**j for j in range(9)]:
                cv2.imwrite(
                    current_output + str(i) + ".png",
                    self.quatizators[quantizator](img, i)
                )


        
        
qtz = ColorQuantization()
input_img = cv2.resize(cv2.imread('./inputs/rgb_cube.png', 1), (512,512))

'''
qtz.every_quantization(input_img, 'rgb_cube')

input_img = cv2.imread('./inputs/Lenna_2019.jpg', 1)
qtz.every_quantization(input_img, 'Lenna_2019')

input_img = cv2.imread('./inputs/Lenna.png', 1)
qtz.every_quantization(input_img, 'Lenna')

'''
n = 16
print("Input shape: {} / MAX: {}".format(input_img.shape, input_img.max()))
cv2.imshow('x', input_img)
cv2.imshow('cube-cut', qtz.cube_cut(input_img, n))
cv2.imshow('k-means', qtz.k_means(input_img, n))
cv2.imshow('median-cut', qtz.median_cut(input_img, n))
cv2.waitKey(0)
