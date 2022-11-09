# 1) Opencv helloworld 
OpenCV (Open Source Computer Vision Library: [opencv.org](http://opencv.org)) is an open-source library that includes several hundreds of computer vision algorithms.


```python
# library 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline
```

Reading, displaying, and writing images are basic to image processing and computer vision.

* imread() - helps us read an image
* imshow() - displays an image in a window
* imwrite() - writes an image into the file directory


```python
# The function cv2.imread() is used to read an image.
img_bgr = cv2.imread('ditto.jpeg',1)
# image features
print(img_bgr.shape)
print(type(img_bgr))
```

    (432, 512, 3)
    <class 'numpy.ndarray'>


Beware that cv2.imread() returns a numpy array in BGR not RGB

### Plot a color image 
We plot the image directly into the notebook with matplotlib for simplicity without using the imshow function. Remember that OpenCV upload an image with BGR codification of colour, so if you want to plot with matplotlib you have to convert it into the RGB codification The method imread, imwrite and imshow indeed all work with the BGR order, so there is no need to change the order when you read an image with cv2.imread and then want to show it with cv2.imshow. Finally, BGR and RGB are not color spaces, they are just conventions for the order of the different color channels.


```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
```




    (-0.5, 511.5, 431.5, -0.5)




    
<figure markdown>
  ![Image title](img/output_6_1.png){ width="300" }
  <figcaption>Ditto</figcaption>
</figure>
    


### More details on the imread function 
The first argument is the image name, which requires a fully qualified pathname to the file.
The second argument is an optional flag that lets you specify how the image should be represented. OpenCV offers several options for this flag, but those that are most common include:

* cv2.IMREAD_UNCHANGED  or -1
* cv2.IMREAD_GRAYSCALE  or 0
* cv2.IMREAD_COLOR  or 1


```python
# The function cv2.imread() is used to read an image.
img_grayscale = cv2.imread('ditto.jpeg',0)
# plot the image 
plt.imshow(img_grayscale, cmap='gray')
plt.axis('off')
```




    (-0.5, 511.5, 431.5, -0.5)




    
<figure markdown>
  ![Image title](img/output_8_1.png){ width="300" }
  <figcaption>Ditto</figcaption>
</figure>
    


### Save the grayscale image


```python
cv2.imwrite('ditto_gray.jpeg', img_grayscale)
```




    True


