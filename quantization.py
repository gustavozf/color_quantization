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
            'median_cut' : self.median_cut,
            'uniform_cut': self.uniform_cut,
            'k_means'    : self.k_means
        }


    # ----------------------------------------------------- MEDIAN CUT
    def median_cut(self, img, n):
        img = img.copy()
        hei, wid = img.shape[:2]
        img = img.reshape((hei * wid, 3))

        args = list(range(hei*wid))
        buckets = [args]

        while n > 1:
            print('Iteration: ', n)
            i = 0
            for _ in range(len(buckets)):
                current_args = buckets.pop(0)

                # se caso caia em uma balde vazio, pula a iteracao
                # isso pode acontecer por dividir um balde de 1 elemento
                # logo, um novo balde fica vazio e o outro com o elemento
                if not len(current_args):
                    continue

                # pega o mais pesado entre o BGR
                current_v = self.__get_V(img, current_args)
                heaviest_c = self.__get_heaviest(current_v)

                # pega a mediana do eixo mais pesado
                median = np.median(np.sort(img[current_args, heaviest_c]))

                # serapa em dois vetores, um "<= mediana" e outro "> mediana"
                inf_cut = set(current_args) & set(np.argwhere(img[:, heaviest_c] <= median)[:,0])
                sup_cut = set(current_args) & set(np.argwhere(img[:, heaviest_c] > median)[:,0])

                # atualiza os baldes
                buckets.extend([list(inf_cut), list(sup_cut)])

                i += 1

            n //= 2
            
        # para todos os buckets
        for args in buckets:
            for i in range(3):
                img[args, i] = np.mean(img[args, i])

        return img.reshape((hei, wid, 3))

    def __get_V(self, img, args):
        b = img[args, 0]
        g = img[args, 1]
        r = img[args, 2]

        #print('B: {} / G: {} / R : {}\n'.format(len(b), len(g), len(r)))

        return [
            [b.min(), b.max()],
            [g.min(), g.max()],
            [r.min(), r.max()]
        ]

    def __get_heaviest(self, V):

        return np.argmax([
            V[0][1] - V[0][0],
            V[1][1] - V[1][0],
            V[2][1] - V[2][0]
        ])

    def __get_indexes(self, img, channel, min_v, max_v):
        return set(np.argwhere(img[:, channel] >= min_v)[:,0]) & set(np.argwhere(img[:, channel] <= max_v)[:,0])

    # ----------------------------------------------------- CUBE CUT
    def __tri_factors(self, n):
        mmc = [] #vetor contendo os mmc's de n
        divisor = 2 
        while n > 1:
            if n%divisor == 0:
                mmc.append(divisor)
                n = n/divisor
            else:
                divisor += 1

        if len(mmc) < 3:
            mmc.extend([1 for i in range(3 - len(mmc))])
        elif len(mmc) > 3:
            step = len(mmc)//3
            mmc = [np.prod(mmc[:step]), np.prod(mmc[step:step*2+1]), np.prod(mmc[step*2+1:])]
        return mmc    

    def uniform_cut(self, img, n, MAX=256):
        img = img.copy()

        a, b, c = self.__tri_factors(n)

        a = np.linspace(0, MAX-1, a, dtype=int)
        b = np.linspace(0, MAX-1, b, dtype=int)
        c = np.linspace(0, MAX-1, c, dtype=int)

        print(a, b, c)

        colors = np.array([[A, B, C] for A in a for B in b for C in c])
        '''
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
        '''
    # ----------------------------------------------------- K-MEDIANAS

    def k_means(self, img, n):
        img = img.copy()
        hei, wid = img.shape[:2]
        img = img.reshape((hei * wid, 3))

        cluster = MiniBatchKMeans(n_clusters = n)
        labels = cluster.fit_predict(img)
        qtz = cluster.cluster_centers_.astype("uint8")[labels]    
        
        return qtz.reshape((hei, wid, 3))

    # ---------------------------------------------------- CPSNR

    def CPSNR(self, img_orig, img_quant, MAX=256.0):
        MAX -= 1

        mse = np.mean((img_orig - img_quant) ** 2)
        if mse == 0:
            return 100

        return 20 * np.log10(MAX / np.sqrt(mse))


        
qtz = ColorQuantization()
input_img = cv2.resize(cv2.imread('./inputs/Lenna.png', 1), (512,512))

for i in [1,2,4,8,16,32,64,128,256]:
    print()
    print(i)
    cv2.imwrite('./outputs/Lenna/median_cut/' + str(i) + '.png', qtz.median_cut(input_img, i))
