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
            'cube_cut'   : self.cube_cut,
            'k_means'    : self.k_means
        }


    # ----------------------------------------------------- MEDIAN CUT
    def median_cut(self, img, n):
        hei, wid = img.shape[:2]
        img = img.reshape((hei * wid, 3))

        buckets = [img]

        while n > 1:
            for _ in range(len(buckets)):
                bucket = buckets.pop(0)
                index_heav = self.__get_heaviest(bucket)

                heav_buck = bucket[:, index_heav]
                heav_buck.sort()
                
                index_mid = len(heav_buck)//2

                bucket[:, index_heav] = heav_buck[ : index_mid]
                buckets.append(bucket)

                bucket[:, index_heav] = heav_buck[index_mid : ]
                buckets.append(bucket)

            n //= 2

        #for 

    def __get_heaviest(self, img):
        heaviest = 0
        heav_value = np.amax(img[:, 0]) - np.amin(img[:, 0])

        for i in [1, 2]:
            current_value =  np.amax(img[:, i]) - np.amin(img[:, i])

            if current_value > heav_value:
                heaviest = i
                heav_value = current_value

        return heaviest


    # ----------------------------------------------------- CUBE CUT

    def cube_cut(self, img, n, MAX=256):
        hei, wid = img.shape[:2]
        img = img.reshape((hei * wid, 3))
        buckets = self.__get_buckets_cube(n)

        # for B, G and R
        for channel in range(3):
            steps = self.MAX//buckets[channel]

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
        hei, wid = img.shape[:2]

        img = img.reshape((hei * wid, 3))
        cluster = MiniBatchKMeans(n_clusters = n)
        labels = cluster.fit_predict(img)
        qtz = cluster.cluster_centers_.astype("uint8")[labels]    
        
        return qtz.reshape((hei, wid, 3))

    def check_number_colors(self, img):
        hei, wid = img.shape[:2]
        return np.unique(img)

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
                    self.quatizators[quantizator](img.copy(), i)
                )


        
        

qtz = ColorQuantization()

input_img = cv2.resize(cv2.imread('./inputs/rgb_cube.png', 1), (512,512))
qtz.every_quantization(input_img, 'rgb_cube')

input_img = cv2.imread('./inputs/Lenna_2019.jpg', 1)
qtz.every_quantization(input_img, 'Lenna_2019')

input_img = cv2.imread('./inputs/Lenna.png', 1)
qtz.every_quantization(input_img, 'Lenna')

'''
n = 8
print("Input shape: {} / MAX: {}".format(input_img.shape, input_img.max()))
cv2.imshow('x', input_img)
cv2.imshow('cube-cut', qtz.cube_cut(input_img.copy(), n))
cv2.imshow('k-means', qtz.k_means(input_img.copy(), n))
cv2.waitKey(0)
print(qtz.check_number_colors(qtz.cube_cut(input_img.copy(), n)))
'''

