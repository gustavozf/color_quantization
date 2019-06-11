'''
PDI - Prof Franklin Cesar Flores
Trabalho 1 - Quantization


Gustavo Zanoni Felipe
'''
import numpy as np
import cv2, os, time
from sklearn.cluster import MiniBatchKMeans

class ColorQuantization():
    # ----------------------------------------------------- MEDIAN CUT
    def median_cut(self, img, n):
        img = img.copy()
        hei, wid = img.shape[:2]
        img = img.reshape((hei * wid, 3))

        args = list(range(hei*wid))
        buckets = [args]

        while n > 1:
            print('Iteration: ', n)
            for _ in range(len(buckets)):
                current_args = buckets.pop(0)

                # se caso caia em uma balde vazio, pula a iteracao
                # isso pode acontecer por dividir um balde de 1 elemento
                # logo, um novo balde fica vazio e o outro com o elemento
                if not len(current_args):
                    continue

                # pega o mais pesado entre o BGR
                heaviest_c = self.__get_heaviest(img, current_args)

                # pega a mediana do eixo mais pesado
                median = np.median(np.sort(img[current_args, heaviest_c]))

                # serapa em dois vetores, um "<= mediana" e outro "> mediana"
                inf_cut = set(current_args) & set(np.argwhere(img[:, heaviest_c] <= median)[:,0])
                sup_cut = set(current_args) & set(np.argwhere(img[:, heaviest_c] > median)[:,0])

                # atualiza os baldes
                buckets.extend([list(inf_cut), list(sup_cut)])

            n //= 2
            
        # para todos os buckets
        for args in buckets:
            for i in range(3):
                img[args, i] = np.mean(img[args, i])

        return img.reshape((hei, wid, 3))

    def __get_heaviest(self, img, args):
        b = img[args, 0]
        g = img[args, 1]
        r = img[args, 2]

        return np.argmax([
            b.max() - b.min(),
            g.max() - g.min(),
            r.max() - r.min()
        ])
    # ----------------------------------------------------- UNIFORM CUT
    def uniform_cut(self, img, n, MAX=256, mode=1):
        img = img.copy()
        hei, wid = img.shape[:2]
        img = img.reshape((hei * wid, 3))

        a, b, c = self.__tri_factors(n)

        a = np.linspace(0, MAX-1, a, dtype=int)
        b = np.linspace(0, MAX-1, b, dtype=int)
        c = np.linspace(0, MAX-1, c, dtype=int)
        #print(a, b, c)

        colors = np.array(np.meshgrid(a, b, c)).T.reshape(-1, 3)

        distances = []
        # olha para todos os pontos calculados
        for color in colors:
            # adiciona as distancias para cada ponto em uma lista
            distances.append(list(map(lambda i: np.linalg.norm(i-color), img)))

        # pega o argumento minimo dentre todos os pontos vistos
        # ou seja, olha o mais proximo e pega o index
        # utilizar o "colors" como um LUT para gerar a nova image
        img = colors[np.argmin(distances, axis=0)]

        return img.reshape((hei, wid, 3))

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

# --------------------------------------------------------------------------------------------- Main
        
qtz = ColorQuantization()
input_img = cv2.imread('./inputs/Lenna.png', 1)

begin = time.time()
for i in [1,2,4,8,16,32,64,128,256]:
    print()
    print(i)
    cv2.imwrite('./outputs/Lenna/median_cut/' + str(i) + '.png', qtz.median_cut(input_img, i))

print("Execution time= ", time.time() - begin)

begin = time.time()
for i in [1,2,4,8,16,32,64,128,256]:
    print()
    print(i)
    cv2.imwrite('./outputs/Lenna/uniform_cut/' + str(i) + '.png', qtz.uniform_cut(input_img, i))

print("Execution time= ", time.time() - begin)
