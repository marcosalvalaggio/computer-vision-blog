# Single pixel operations

Considering a grayscale image, single pixel operations/transforms (AKA intensity transforms) are functions that change the gray levels (values of pixels from 0 to 255) of the image. Elements involved:

* Function $I(x,y)$ representing the original image $(r=I(x_i,y_i)$ is the $i$-th pixel)
* Function $T(\cdot)$ representing the gray level change ($s=T(r)$ output level of $i$-th pixel)

The following are commonly used intensity transformations:

* Negative
* Logarithm
* Gamma
* Contrast stretching
* Intensity slicing
* Histogram equalization

<figure markdown>
<center>
  ![Image title](img/spo/Spo_1.png){ width="500" }
  <figcaption></figcaption>
</center>
</figure>


```python
# Libraries 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline
```


```python
img = cv2.imread('ditto_gray.jpeg',0)
print(f'Image dimenions: {img.shape}')
plt.imshow(img, cmap = 'gray')
plt.axis('off')
```

    Image dimenions: (432, 512)

    (-0.5, 511.5, 431.5, -0.5)

<figure markdown>
<center>
  ![Image title](img/spo/Spo_2.png){ width="300" }
  <figcaption></figcaption>
</center>
</figure>
    


## Negative transformations
Switch dark and light. The negative transformation can be described by the expression

$$s = (L-1)-r, \;\; L = 256$$

where $r$ is the initial intensity level and $s$ is the final intensity level of a pixel.


```python
# apply the negative transformation
L = 256
neg_img = (L-1)-img
plt.imshow(neg_img, cmap = 'gray')
plt.axis('off')
```

    (-0.5, 511.5, 431.5, -0.5)

<figure markdown>
<center>
  ![Image title](img/spo/Spo_3.png){ width="300" }
  <figcaption></figcaption>
</center>
</figure>
    


## Log transform
Practically, log transformation maps a narrow range of low-intensity input values to a wide range of output values. Mathematically, log transformations can be expressed as

$$s=c\cdot log(1+r),\;\;c=\frac{L-1}{logL}$$

where $c$ is a scaling constant. The scaling operation is done to ensure that the final pixel value does not exceed (L-1), or 255.


```python
fig, ax = plt.subplots()
x = list(range(1,255))
y1 = x
y2 = np.log(x)
# all possible value of original image [0,255]
ax.plot(x, y1, color="blue")
ax.set_xlabel('r')
ax.set_ylabel('s', color="blue")       
ax.tick_params(axis='y', colors="blue")  
# log transformation 
ax2 = ax.twinx()
ax2.plot(x, y2, color="red") 
plt.axis('off')
# show 
plt.show()
```
<figure markdown>
<center>
  ![Image title](img/spo/Spo_4.png){ width="500" }
  <figcaption></figcaption>
</center>
</figure>
    



```python
# original image
img = cv2.imread('img/log.png',0)
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Original image')
plt.axis('off')
# Apply log transform.
L = 256
c = (L-1)/(np.log(L))
img_log = c * np.log(1 + img)
plt.subplot(122)
plt.imshow(img_log, cmap = 'gray')
plt.title('Log transformed')
plt.axis('off')
```

    (-0.5, 829.5, 835.5, -0.5)

<figure markdown>
<center>
  ![Image title](img/spo/Spo_5.png){ width="500" }
  <figcaption></figcaption>
</center>
</figure>
    


## Gamma transform 

Gamma correction is important for displaying images on a screen correctly, to prevent bleaching or darkening of images when viewed from different types of monitors with different display settings. This is done because our eyes perceive images in a gamma-shaped curve, whereas cameras capture images in a linear fashion.

<figure markdown>
<center>
  ![Image title](img/spo/Spo_6.png){ width="500" }
  <figcaption></figcaption>
</center>
</figure>

Gamma transformations can be mathematically expressed as:

$$s = cr^\gamma $$

$$c = (L-1)^{1-\gamma} $$

<figure markdown>
<center>
  ![Image title](img/spo/Spo_7.png){ width="500" }
  <figcaption></figcaption>
</center>
</figure>


```python
# original image 
img = cv2.imread('ditto_gray.jpeg',0)
# gamma transformation 
L = 256
gamma = 0.10
c = (L-1) ** (1-gamma)
img_gamma = c * (img ** gamma)
# plot
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Original image')
plt.axis('off')
plt.subplot(122)
plt.imshow(img_gamma, cmap = 'gray')
plt.title('Gamma transformed')
plt.axis('off')
```

    (-0.5, 511.5, 431.5, -0.5)

<figure markdown>
<center>
  ![Image title](img/spo/Spo_8.png){ width="500" }
  <figcaption></figcaption>
</center>
</figure>


## Histogram Equalization

**What is an Image Histogram?**

* It is a graphical representation of the intensity distribution of an image
* It quantifies the number of pixels for each intensity value considered.

<figure markdown>
<center>
  ![Image title](img/spo/Spo_9.png){ width="400" }
  <figcaption></figcaption>
</center>
</figure>

### Histogram of a grayscale image
OpenCV provides us with the ***cv2.calcHist*** function to calculate the image histograms.
Let's familiarize with the function and its parameters:

` cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])` 

* **images**: it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
* **channels**: it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
* **mask**: mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
* **histSize**: this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
* **ranges**: this is our RANGE. Normally, it is [0,256].


```python
# original image 
img = cv2.imread('img/landscape.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.figure(figsize=(9,4))
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Original image')
plt.subplot(122)
plt.plot(hist, color='black')
plt.title('Image Histogram')
plt.show()
```
<figure markdown>
<center>
  ![Image title](img/spo/Spo_10.png){ width="600" }
  <figcaption></figcaption>
</center>
</figure>
    


### Histogram Calculation in Numpy


```python
# original image 
img = cv2.imread('img/landscape.jpg',0)
# using numpy ravel 
plt.figure(figsize=(9,4))
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Original image')
plt.subplot(122)
plt.hist(img.ravel(),256,[0,256])
plt.title('Image Histogram')
plt.show()
```
<figure markdown>
<center>
  ![Image title](img/spo/Spo_11.png){ width="600" }
  <figcaption></figcaption>
</center>
</figure>
    


### Application of Mask


```python
img = cv2.imread('img/landscape.jpg',0)
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:500, 100:600] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)
# Calculate histogram with mask and without mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
# plot
plt.figure(figsize=(12,8))
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])
```
    (0.0, 256.0)

<figure markdown>
<center>
  ![Image title](img/spo/Spo_12.png){ width="700" }
  <figcaption></figcaption>
</center>
</figure>
    
    


In the histogram plot, blue line shows histogram of full image while orange line shows histogram of masked region.


## References

* [Computer Vision Course by Prof. Ghidoni Stefano (Unipd)](http://www.dei.unipd.it/~ghidoni/)
* [OpeCV Histograms](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)
* [OpenCV Histogram Equalization
](https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html)
