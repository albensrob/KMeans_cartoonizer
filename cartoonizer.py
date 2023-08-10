import cv2
import numpy as np
from scipy.spatial.distance import cdist

#membaca file gambar, ganti ukuran, dan mengambil nilai RGB
def read(filename):
    img1 = cv2.imread(filename)
    (h, w) = img1.shape[:2]
    r = 300 / float(h)
    dim = (int(w * r), 300)
    img_resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    res = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    res = res[..., ::-1]
    return res

#membuat edgemask dengan canny
def edgemask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray,100,150)
    edges1 = cv2.bitwise_not(edges1)
    return edges1

#Mendapatkan Centroid(digunakan untuk elbow method)
def centroid(data, crit, k):
    ret, label, center = cv2.kmeans(data, k, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
    return center

#Menemukan nilai K optimal dengan elbow method
def elbow(img):
    distorsi = []
    emaks = 0
    k = range(3,7)
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    for x in k:
        center = centroid(data,criteria,x)
        distorsi.append(sum(np.min(cdist(data,center,'euclidean'),axis=1))/ data.shape[0])
        emaks = max(emaks, (distorsi[x - 4] - distorsi[x - 3]))
        print(abs(distorsi[x - 4] - distorsi[x - 3]))

    for x in k:
        if emaks == (abs(distorsi[x-4]-distorsi[x-3])): hasilk = x
    print(distorsi)
    return hasilk

#K-Means
def kmeans(img, k, criteria):
    data = np.float32(img).reshape((-1, 3))
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

#K-Means dan Segmentasi
def cluster(img, k):
    print("main elbow: "+str(k))
    kmeans_img = []
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    for y in range(k):
        kmeans_img.append([])
        result = center[label.flatten()]
        temp = result
        for x in range(len(result)):
            if result[x][0] == center[y][0] and result[x][1] == center[y][1] and result[x][2] == center[y][2]:
                temp[x][0]=data[x][0]
                temp[x][1]=data[x][1]
                temp[x][2]=data[x][2]
            else:
                temp[x][0] = 255
                temp[x][1] = 255
                temp[x][2] = 255

        temp = temp.reshape(img.shape)
        innerk = elbow(temp)
        print("inner elbow ("+ str(y+1) +"): "+ str(innerk))
        kmeans_img[y] = kmeans(temp,innerk,criteria)
        kmeans_img[y] = cv2.bitwise_and(kmeans_img[y],kmeans_img[y-1])
    return kmeans_img[k-1]


def cartoon(img,edges):
    c = cv2.bitwise_and(img, img, mask=edges)
    return c

