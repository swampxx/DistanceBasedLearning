<a
href="https://colab.research.google.com/github/swampxx/DistanceBasedLearning/blob/master/DistanceBasedLearning.ipynb"
target="_parent"><img src="https://colab.research.google.com/assets/colab-
badge.svg" alt="Open In Colab"/></a>

# Dataset

```python
#!rm -r CMANN
#!git clone https://github.com/Orkun-tanik/CMANN.git
```

#Omniglot

*The Omniglot* dataset is a collection of 1623 hand drawn characters from 50
alphabets. For every character there are just 20 examples, each drawn by a
different person at resolution 105x105.

```python
#!unzip CMANN/images_evaluation.zip
#!unzip CMANN/images_background.zip

"""  Do not execute again!!  """
```

```python
# Imports

import os
import numpy as np
import matplotlib.image as mpimg
```

```python
def to_origin(image):
    """
       Image: Black pixels labeled as True
        
       It moves the image towards the origin,
                returns new_image and black pixels coordinates array: (new_image,blacks)  
                
    """

    x,y = image.shape
    xshift = x
    yshift = y

    blacks = []

    for i in range(x):
        for j in range(y):
            if image[i][j] == True:
                blacks.append((i,j))
                if i<xshift:
                    xshift = i
                if j<yshift:
                    yshift = j

    new_image = np.zeros(shape=(x,y))

    for i in range(len(blacks)):
        (a,b) = blacks[i]
        blacks[i] = (a-xshift, b-yshift)
        new_image[a-xshift][b-yshift] = 1
    
    blacks = np.array(blacks)

    return (new_image, blacks)
```

```python

data_path = ""
train_path = os.path.join(data_path,'images_background')
validation_path = os.path.join(data_path,'images_evaluation')



def load_images_from_directory(path):
    X=[]

    ## We load every alphabet seperately and append that to one tensor
    for alphabet in os.listdir(path):
        #print("loading alphabet: " + alphabet)
        alphabet_path = os.path.join(path,alphabet)
        
        ## Each character in alphabet is in a separate folder
        for letter in os.listdir(alphabet_path):
            #print(" + letter: " + letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
        
            
            if not os.path.isdir(letter_path):
                continue

            ## Read every image in this directory
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = mpimg.imread(image_path)
                #print(image)
                
                
                #TODO: recreate images with black pixels coordinate values
                
                
                ### Image preprocessing!
                #image = image/255
                #image = 1-image
                
                image = np.logical_not(image)
                #print("prepocessing image...")
                
                new_image, _ = to_origin(image)
                
                
                #print("done.")
                #print(image)           
                
                category_images.append(new_image)
            
            try:
                X.append(np.array(category_images))
            #edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
    
    X = np.array(X)

    return X



```

```python
print("Loading training set")
#Xtrain = load_images_from_directory(train_path)
#print(Xtrain.shape)

print("Now loading evaluation set")
#Xval = load_images_from_directory(validation_path)
#print(Xval.shape)
```

```python
#np.save('Xtrain.npy', Xtrain)
#np.save('Xval.npy', Xval)
```

```python
#Xtrain = np.load("Xtrain.npy")
#Xval = np.load("Xval.npy")
```

```python
#@title Example Image to be displayed { run: "auto" }
import matplotlib.pyplot as plt

character = 270 #@param {type:"integer"}
drawing = 16 #@param {type:"slider", min:0, max:19, step:1}
image_set = 'Xtrain' #@param ["Xval", "Xtrain"]

if (image_set == 'Xval'):
    imgplot = plt.imshow(Xval[character,drawing])
else:
    imgplot = plt.imshow(Xtrain[character,drawing])
```

# Latin Images

```python

latin_path = "Latin"
#latin_path_evaluation = "images_evaluation"

print("Loading Latin alphabet..")
latin = load_images_from_directory(latin_path)
print(latin.shape)
```

```python
#@title Latin Images { run: "auto" }
import matplotlib.pyplot as plt

character = 0 #@param {type:"slider", min:0, max:25, step:1}
drawing = 0 #@param {type:"slider", min:0, max:19, step:1}
#image_set = 'Xtrain' #@param ["Xval", "Xtrain"]
plt.imshow(latin[character,drawing])

```

```python

def black_pixels(image):
    """
       returns black pixel coordinates of image, array-like
    """
    
    x,y = image.shape
    blacks = []
    for i in range(x):
        for j in range(y):
            if image[i][j] == True:
                blacks.append((i,j)) 



    return np.array(blacks)

# Extract random samples from each character of given alphabet

def get_sample(alphabet):
    
    """
        alphabet numpy array [size, drawing, 105, 105]
    
        returns samples and altered alphabet
    """
    character_count, drawing_count, _, _ = alphabet.shape
    samples = []
    new_alphabet = []    
    for i in range(character_count):
        rand = np.random.randint(0,drawing_count)
        samples.append(alphabet[i,rand])
        new_alphabet.append(np.delete(alphabet[i], rand, 0))  
    
    
    samples = np.array(samples)
    new_alphabet = np.array(new_alphabet)
    
    return samples, new_alphabet
```

```python
# Extract samples from latin alphabet
samples, new_latin = get_sample(latin)
```

```python
#@title Sample Latin Images { run: "auto" }
import matplotlib.pyplot as plt

character = 24 #@param {type:"slider", min:0, max:25, step:1}
#image_set = 'Xtrain' #@param ["Xval", "Xtrain"]
plt.imshow(samples[character])
```

***These sample latin images will be used for testing distance functions. ***

# Experiments


---


*In this section,  there are experiments of distance
functions for one-shot learning . Follow the headers for distance functions
properties.*



---

## Elementary Distance Functions





---
* Minkowski Family
> *   Euclidean 
>
*   Manhattan
> *  LP Norm Distance Function

* Angular Distance Functions
>*
Cosine Similarity

>### Euclidean
---
Euclidean distance is a special case of Minkowski distance
with $\lambda=2$

$Euclidean(\vec{x}, \vec{y}):= \sqrt{\sum_i(x_i-y_i)^2}$

```python
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import math

def mins_helper(arr):
    
    indexed_used = []
    for i in range(len(arr)):

        row_min = np.argmin(arr[i])
        if row_min in indexed_used:
            new_row = arr[i]
            new_row[row_min] = 999
            row_min = np.argmin(new_row)     
        
        indexed_used.append(row_min)
    
    return indexed_used

def euclidean_distance_std(img1,img2):
    """
        params: x and y images
        
        To apply eucl. distance It transforms images to 1D vectors by flatten()
        
        return: euclidean distance between two images
       
    """
    
    
    v1 = img1.flatten()
    v2 = img2.flatten()
    
    return distance.euclidean(v1, v2)


    
    
def euclidean_distance_modified(img1,img2):
    """
        params: img1 and img2 images
        
        It extracts each black pixels from images, find their coordinates on
        x-y plane by taking origin as a reference.
        
        Then, computes distance between all pixel pairs and sums minimum
        distance along axis=1 (row based).
        
        can be normalized, divides with biggest distance on the plane
        diagonal sqrt(max_x**2+max_y**2)
        
        returns: modified euclidean distance
        
    """
    max_x, max_y = img1.shape
    
    v1 = black_pixels(img1)
    v2 = black_pixels(img2)
    
    print("black pixel count of img1: ", v1.shape )
    
    print("black pixel count of img1: ", v2.shape )
    
    # Number of black pixels is not same for every image pair.
    
    # Randomly select black pixels from larger one
    randoms = np.random.permutation(min(len(v1), len(v2)))
    
    if len(v1) < len(v2) :
        v2 = v2[randoms[:]]
    else:
        v1 = v1[randoms[:]]
    
    
    dist = distance.cdist(v1,v2,'euclidean')
    
    d = mins_helper(dist)
    sum = 0
    for i in range(len(dist)):
        sum += dist[i,d[i]]
    return sum
    difference_sum = np.sum(np.amin(dist,axis=1))
    
    return difference_sum

def euclidean_distance_modified_v2(img1,img2):
    """
        params: img1 and img2 images
        
        It extracts each black pixels from images, find their coordinates on
        x-y plane by taking origin as a reference.
        
        Then, computes distance between all pixel pairs and sums minimum
        distance along axis=1 (row based).
        
        can be normalized, divides with biggest distance on the plane
        diagonal sqrt(max_x**2+max_y**2)
        
        returns: modified euclidean distance
        
    """
    max_x, max_y = img1.shape
    
    v1 = black_pixels(img1)
    v2 = black_pixels(img2)
    
    
    # Number of black pixels is not same for every image pair.
    
    # Randomly select black pixels from larger one
    randoms = np.random.permutation(min(len(v1), len(v2)))
    
    if len(v1) < len(v2) :
        v2 = v2[randoms[:]]
    else:
        v1 = v1[randoms[:]]
    
    
    dist = distance.cdist(v1,v2,'euclidean')
    
    d = mins_helper(dist)
    sum = 0
    for i in range(len(dist)):
        sum += dist[i,d[i]]
    return sum
```

### Manhattan
---
Manhattan distance is a special case of Minkowski distance
with $\lambda=1$

$Manhattan(\vec{x}, \vec{y}):= \sqrt{\sum\limits_i |x_i-y_i|}$
Since flatten function is used, standard manhattan distance can be deceptive.

A
modified Manhattan distance will be more informative}.

```python
def manhattan_distance_std(img1,img2):
    """
        params: img1 and img2 images
        
        To apply eucl. distance It transforms images to 1D vectors by flatten()
        
        return: euclidean distance between two images
       
    """
    
    
    v1 = img1.flatten()
    v2 = img2.flatten()
    
    return distance.cityblock(v1, v2)

def manhattan_distance_modified(img1,img2):
    """
        params: images
        
        Like modified euclidean, finds coordinates of black pixels, then
        computes distance.
        
        
        return manhattan distance between two images
    """
    max_x, max_y = img1.shape
    
    v1 = black_pixels(img1)
    v2 = black_pixels(img2)
    
    # Number of black pixels is not same for every image pair.
    
    # Randomly select black pixels from larger one
    randoms = np.random.permutation(min(len(v1), len(v2)))
    
    if len(v1) < len(v2) :
        v2 = v2[randoms[:]]
    else:
        v1 = v1[randoms[:]]
    
    
    dist = distance.cdist(v1,v2,'cityblock')
    
    difference_sum = np.sum(np.amin(dist,axis=1))
    
    return difference_sum
```

### Cosine Similarity

$CosineSimilarity(\vec{x} , \vec{y})
=\dfrac{\vec{x}.\vec{y}}{||a||.||b||} $

```python
def cosine_distance_std(img1, img2):
    
    v1 = img1.flatten()
    v2 = img2.flatten()
    
    return distance.cosine(v1, v2)


def arccos_distance(img1,img2):
    
    cos_sim = 1 - cosine_distance_std(img1,img2)
    
    return(math.acos(cos_sim))

```

## Distance Functions Experiment Suite

```python
#@title  { run: "auto" }
#@markdown Select characters and drawings to compute distances.



character1 = 17 #@param {type:"slider", min:0, max:25, step:1}
drawing1 = 3 #@param {type:"slider", min:0, max:19, step:1}

character2 = 3 #@param {type:"slider", min:0, max:25, step:1}
drawing2 = 13 #@param {type:"slider", min:0, max:18, step:1}

img1 = new_latin[character1,drawing1]
img2 = new_latin[character2,drawing2]

#print(latin.shape)

fig=plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(img1)
fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img2)


print("standard euclidean distance: ", euclidean_distance_std(img1,img2))
print("modified euclidean distance: ", euclidean_distance_modified(img1,img2))
print("modified euclidean distance_v2: ", euclidean_distance_modified_v2(img1,img2))
print("standard manhattan distance: ",manhattan_distance_std(img1,img2))
print("modified manhattan distance: ", manhattan_distance_modified(img1,img2))
print("cosine similarity: ",cosine_distance_std(img1,img2))
print("arccos similarity: ", arccos_distance(img1,img2))
```

```python

```
