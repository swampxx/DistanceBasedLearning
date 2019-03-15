import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import math
from copy import deepcopy
import scipy.spatial.distance as distance
import seaborn as sn



def to_origin(image):
    """ 
       It moves the image towards the origin,
                returns new_image and black pixels: (new_image,blacks)
    """

    x,y = image.shape
    xshift = x
    yshift = y

    blacks = []

    for i in range(x):
        for j in range(y):
            if image[i][j] == False:
                blacks.append((i,j))
                if i<xshift:
                    xshift = i
                if j<yshift:
                    yshift = j

    new_image = np.ones(shape=(x,y))

    for i in range(len(blacks)):
        (a,b) = blacks[i]
        blacks[i] = (a-xshift, b-yshift)
        new_image[a-xshift][b-yshift] = 0
    
    blacks = np.array(blacks)

    return (new_image, blacks)

def score(image):
    """
        Trying to get unique score for each image class
    """
    new_image, blacks = to_origin(image)
    dist = distance.cdist(blacks,[[0,0]], 'euclidean')
    
            


def distf(v1,v2):
    x,y = v1.shape

    v1_new, blacks_v1 = to_origin(v1)
    v2_new, blacks_v2 = to_origin(v2)

    print("black_pixel_count_v1: ", len(blacks_v1))
    print("black_pixel_count_v2: ", len(blacks_v2))
    
    randoms = np.random.permutation(min(blacks_v1.shape[0], blacks_v2.shape[0]))

    if blacks_v1.shape[0] < blacks_v2.shape[0]:
        blacks_v2 = blacks_v2[randoms[:]]
    else:
        blacks_v1 = blacks_v1[randoms[:]]

    denom = math.sqrt(x**2+y**2)
    distances_euclidean = distance.cdist(blacks_v1,blacks_v2,'euclidean')
    mins = np.amin(distances_euclidean,axis=1)

    diff = np.sum(mins)
    print("euclidean_difference :", diff)

    distances_cityblock = distance.cdist(blacks_v1,blacks_v2,'cityblock')
    mins = np.amin(distances_cityblock,axis=1)

    diff = np.sum(mins)
    print("manhattan_difference :", diff)



    distances_cosine = distance.cdist(blacks_v1,blacks_v2,'cosine')
    mins = np.amin(distances_cosine,axis=1)

    diff = np.sum(mins)

    print("cosine_difference: ", diff)


    denom = (x**3+y**3) ** (1./3)
    distances_minkowski = distance.cdist(blacks_v1,blacks_v2,'minkowski',p=3)
    mins = np.amin(distances_minkowski,axis=1)

    diff = np.sum(mins)

    print("minkowski_difference: ", diff/denom)


    distances_canberra = distance.cdist(blacks_v1,blacks_v2,'canberra')
    mins = np.amin(distances_canberra,axis=1)
    diff = np.sum(mins)

    print("canberra_difference: ", diff)


    

    #TODO: mahalanobis distance

    """#jaccard distance: dissimilarity between 2 boolean vector

    distances_jaccard = distance.jaccard(v1.flatten(),v2.flatten())

    print("distances_jaccard: ", distances_jaccard)"""

    fig = plt.figure(figsize=(2,1))



    fig.add_subplot(2,1,1)
    plt.imshow(v1_new)

    fig.add_subplot(2,1,2)
    plt.imshow(v2_new)
    plt.show()

image1 = Image.open("1.png") #e
#image1.show()

image2 = Image.open("4.png")
#image2.show()

"""image3 = Image.open("3.png")
#image3.show()

image4 = Image.open("4.png")
#image4.show()

image5 = Image.open("5.png")
#image5.show()

image6 = Image.open("6.png")
#image6.show()

image7 = Image.open("7.png")
#image7.show()

image8 = Image.open("8.png")
#image8.show()

image9 = Image.open("9.png")
#image9.show()"""

v1 = np.array(image1)
v2 = np.array(image2)
#v3 = np.array(image3)

print("shape of image", v1.shape)


distf(v1,v2)

#score(v1)




"""
img1 = mpimg.imread("1.png")
plt.imshow(img1)
plt.show()
"""
