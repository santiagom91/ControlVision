# Control Vision Methods
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt  

########################################################################################################
######################################### Rescale an immage ############################################
########################################################################################################
def rescaleFrame(Frame, scale=0.2):
    width = int(Frame.shape[1] * scale)
    height = int(Frame.shape[0] * scale)

    dimension = (width, height)
    return cv.resize(Frame, dimension, interpolation = cv.INTER_AREA)

img = cv.imread('SpaceMountain.JPG') #carico un immagine
resized_image = rescaleFrame(img) #ridimensiono l'immagine
cv.imshow('Stars', resized_image)

########################################################################################################
############################################ Shapes on an image ########################################
########################################################################################################
blank = np.zeros((500,500,3), dtype = 'uint8')
cv.imshow('Blank', blank)

# blank[200:300, 300:400] = 0,0,255
# cv.imshow('Red', blank)

cv.rectangle(blank, (0,0), (250, 250), (0,255,0), thickness=cv.FILLED)
cv.imshow('Rectangle', blank)

cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness =-1)
cv.imshow('Circle', blank)

cv.line(blank, (100,250), (blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness =3)
cv.imshow('Line', blank)

cv.putText(blank, 'Hello, my name is Santiago!', (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,0,0), 2)
cv.imshow('Text', blank)

## cv.waitKey(0)

############################################################################################################
######################################## Color of an image #################################################
############################################################################################################

img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#canny = cv.Canny(img, 125, 175) # see all edges
canny = cv.Canny(blur, 125, 175) # to see only external edges
cv.imshow('Canny Edges', canny)

# Dilating the image
dilated = cv.dilate(canny, (7,7), iterations = 3)
cv.imshow('Dilated', dilated)

# Eroding 
eroded = cv.erode(dilated, (3,3), iterations = 3)
cv.imshow('Eroded', eroded)

#Resize
resized = cv.resize(img, (500, 500))
cv.imshow('Resized', resized)

#Cropping 
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

## cv.waitKey(0)

############################################################################################################
######################################## Image trasformations ##############################################
############################################################################################################

img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

# Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

## -x = left; 
## -y = up
##  x = right
##  y = down 

translated = translate(img, -100,100)
cv.imshow('Translated', translated)

# Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimension = (width, height)

    return cv.warpAffine(img, rotMat, dimension)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

rotated_rotated = rotate(img, -90)
cv.imshow('Rotated Rotated', rotated_rotated)

#Resizing 
resized = cv.resize(img, (500, 500), interpolation = cv.INTER_CUBIC)
cv.imshow('Resized', resized)

#Flipping 
flip = cv.flip(img, 0)
cv.imshow('Flip', flip)

#Cropping
cropped = img[200:400, 300: 400]
cv.imshow('Cropped', cropped)

## cv.waitKey(0)

############################################################################################################
######################################## Image edges #######################################################
############################################################################################################

img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT) # for external edges
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175) # see all edges
cv.imshow('Canny Edges', canny)

# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0,0,255), 2)
cv.imshow('Contrours Draw', blank)

## cv.waitKey(0)

############################################################################################################
######################################## Image edges #######################################################
############################################################################################################
img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

#BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# Matplot in RGB problem
plt.imshow(img) #matplot display RGB image and not BGR
plt.show()

#BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

plt.imshow(rgb)
plt.show()

#HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV --> BGR', hsv_bgr)

#lab to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_HSV2BGR)
cv.imshow('lab --> BGR', lab_bgr)

##cv.waitKey(0)

############################################################################################################
######################################## Image color channels ##############################################
############################################################################################################
img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, r, blank])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged) #reconstruct the image

## cv.waitKey(0)

############################################################################################################
######################################## Image smooth noise ################################################
############################################################################################################

img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
#cv.imshow('Stars', img)

#Avaraging
average = cv.blur(img, (7,7))
cv.imshow('Average Blur', average)

#Gaussian Blur
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow('Gaussian Blur', gauss)

#Median Blur
median = cv.medianBlur(img, 3) #introduce some noise
cv.imshow('Median Blur', median)

#Bilateral 
bilateral = cv.bilateralFilter(img, 5, 15, 25) #introduce some noise
cv.imshow('Bilateral Blur', bilateral)

#cv.waitKey(0)

############################################################################################################
######################################## Image Bitwise #####################################################
############################################################################################################

##img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
##cv.imshow('Stars', img)

blank = np.zeros((400, 400), dtype = 'uint8')

rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Cricle', circle)

#Bitwise AND (intersection of 2 images)
bitwise_and =cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND',bitwise_and)

cv.waitKey(0)


###########################################################################################################
####################################### Image Bitwise mask ################################################
###########################################################################################################
img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

blank = np.zeros(img.shape[:2], dtype = 'uint8')
#cv.imshow('Blank Image', blank)

# mask = cv.rectangle(blank, (img.shape[1]//2+45, img.shape[0]//2), (img.shape[1]//2 +100, img.shape[0]//2 +100), 100, 255, -1)
# cv.imshow('Mask', mask)

circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1)
# cv.imshow('Mask', mask)

rectangle = cv.rectangle(blank.copy(), (30,30), (370, 370), 255, -1)

weird_shape = cv.bitwise_and(circle, rectangle)
cv.imshow('Weird Shape',weird_shape)

masked = cv.bitwise_and(img, img, mask= weird_shape)
cv.imshow('Weird Masked Image', masked)

cv.waitKey(0)

###########################################################################################################
####################################### Image Histogram ###################################################
###########################################################################################################
img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

#Grayscale histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

#Colour Histogram
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b','g','r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

cv.waitKey(0)

###########################################################################################################
####################################### Image Thresholding ################################################
###########################################################################################################
img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Simple Thresholding 
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Threshold', thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Threshold Inverse', thresh_inv)

#Adaptive Thresholding 
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Threshold', adaptive_thresh)

#Gaussian Thresholding 
gaussian_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Gaussian Threshold', gaussian_thresh)

cv.waitKey(0)
###########################################################################################################
####################################### Image Edge Detection ##############################################
###########################################################################################################
img = cv.imread('SpaceMountain.JPG') #upload an image from your pc
cv.imshow('Stars', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_and(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)
