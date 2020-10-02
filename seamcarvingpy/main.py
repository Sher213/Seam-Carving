from skimage.exposure import rescale_intensity
from matplotlib import pyplot as plt
import cv2
import numpy as np
import sys

def gaussianKernel(size):
    array = cv2.getGaussianKernel(size, 1)
    array = array.reshape(1,size)
    return np.dot(array.T, array)

def convolve(kernel, img):
    pad = (len(kernel) - 1) // 2
    height, width = img.shape[:2]
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((height, width, 3), dtype="float32")

    try:
        for y in np.arange(pad, height + pad):
            for x in np.arange(pad, width + pad):
                roi = img[y - pad:y + pad + 1, x - pad:x + pad + 1]
                r = 0
                g = 0
                b = 0
                for i in np.arange(0, len(roi)):
                    for j in np.arange(0, len(roi)):
                        b += roi[i,j,0] * kernel[i,j] 
                        g += roi[i,j,1] * kernel[i,j] 
                        r += roi[i,j,2] * kernel[i,j]
                output[y - pad, x - pad, 0] = b
                output[y - pad, x - pad, 1] = g
                output[y - pad, x - pad, 2] = r
    except Exception as e:
        print(e)

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

def imageEdges(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1)
    edges = np.sqrt(np.square(gx) + np.square(gy))
    return edges

def findEnergy(edges):
    height, width = edges.shape[:2]
    energy = np.zeros((height, width), dtype="float32")
    path = np.zeros((height, width), dtype="float32")
    for i in np.arange(0, width):
        energy[height-1, i] = edges[height - 1, i]


    for y in np.arange(height - 2, -1, -1):
        for x in np.arange(0, width):
            p1 = 10000
            p2 = 10000
            p3 = 10000
            if not(x == 0):
                p1 = energy[y+1, x - 1] + edges[y,x]
            p2 = energy[y+1, x] + edges[y,x]
            if not (x == width-1):
                p3 = energy[y+1, x+1] + edges[y,x]
            
            if p3 > p1 < p2:
                path[y,x] = -1
                energy[y,x] = p1
            elif p1 > p3 < p2:
                path[y,x] = 1
                energy[y,x] = p3
            else:
                path[y,x] = 0
                energy[y,x] = p2
    return energy, path
            
def carveSeam(topRowE, path, img, count, c):
    height, width = img.shape[:2]
    minE = 9999
    minECol = -1
    if c > count:
        return img
    else:
        for i in np.arange(len(topRowE)):
            if topRowE[i] < minE:
                minE = topRowE[i]
                minECol = i
        seam = [minECol]
        del topRowE[minECol]
        pathRow = 0
        pathCol = minECol
        while pathRow < height - 1:
            pathDirection = path[pathRow][pathCol]
            del path[pathRow][pathCol]
            if pathDirection == -1:
                if not(pathCol <= 0):
                    pathCol-=1
                seam.append(pathCol)
            elif pathDirection == 0:
                seam.append(pathCol)
            elif pathDirection == 1:
                if not(pathCol >= width-1):  
                    pathCol+=1
                seam.append(pathCol)
            pathRow+=1
        
        newimg = np.zeros((height,width-1,3), np.uint8)
        for y in np.arange(height):
            nX = 0
            for x in np.arange(width):
                if x == seam[y]:
                    continue
                else:
                    newimg[y][nX] = img[y][x]
                    nX+=1
        return carveSeam(topRowE, path, newimg, count, c+1)
    
        
    


def main():
    # Load an color image in grayscale
    src = cv2.imread(sys.argv[1])
    grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #kSize = int(sys.argv[2])
    #new = convolve(gaussianKernel(kSize), img)

    #Blur plus Sobel Edge Detection
    edgy = imageEdges(src)
    weightedImage = findEnergy(edgy)
    final = carveSeam(weightedImage[0][0].tolist(), weightedImage[1].tolist(), src, 25, c=0)
    #plot and show
    plt.subplot(3,3,1),plt.imshow(src)
    plt.subplot(3,3,2),plt.imshow(grey, cmap="gray")
    plt.subplot(3,3,3),plt.imshow(edgy)
    plt.subplot(3,3,4),plt.imshow(weightedImage[0])
    plt.subplot(3,3,5),plt.imshow(weightedImage[1])
    plt.subplot(3,3,6),plt.imshow(final)
    plt.show()
if __name__ == "__main__":
    main()

