import sys
import os
import cv2
from skimage.color import rgb2gray
import numpy as np
import math
from joblib import Parallel, delayed
import multiprocessing
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA

# Zheng, Y., Jiao, L., Liu, H., Zhang, X., Hou, B., & Wang, S. (2017).
# Unsupervised saliency-guided SAR image change detection. Pattern Recognition, 61, 309–326.
# https://doi.org/10.1016/j.patcog.2016.07.040

def appearance_distance(image, blockSize, r1, c1, r2, c2):
    height, width = image.shape
    step = math.floor(blockSize/2)

    rMinLimit1 = max([1, r1 - step])
    rMinLimit2 = max([1, r2 - step])

    cMinLimit1 = max([1, c1 - step])
    cMinLimit2 = max([1, c2 - step])

    rMaxLimit1 = min([height, r1 + step])
    rMaxLimit2 = min([height, r2 + step])

    cMaxLimit1 = min([width, c1 + step])
    cMaxLimit2 = min([width, c2 + step])

    rMinMargin = min([r1 - rMinLimit1, r2 - rMinLimit2])
    rMaxMargin = min([rMaxLimit1 - r1, rMaxLimit2 - r2])

    cMinMargin = min([c1 - cMinLimit1, c2 - cMinLimit2])
    cMaxMargin = min([cMaxLimit1 - c1, cMaxLimit2 - c2])

    diffImage = np.absolute(image[r1 - 1 - rMinMargin: r1 + rMaxMargin, c1 - 1 - cMinMargin: c1 + cMaxMargin] - image[r2 - 1 - rMinMargin: r2 + rMaxMargin, c2 - 1 - cMinMargin: c2 + cMaxMargin])
    return np.sum(diffImage[:]) / ((rMinMargin + rMaxMargin + 1) * (cMinMargin + cMaxMargin + 1))


def distance(image, blockSize, r1, c1, r2, c2):
    appDistance = appearance_distance(image, blockSize, r1, c1, r2, c2)
    height, width = image.shape
    dRow = (r1-r2) / height
    dCol = (c1-c2) / width
    dist = math.sqrt(dRow ** 2 + dCol ** 2)
    return appDistance / (1 + 3 * dist)

def salient(image, height, width, blockSize, K, i):
    blockCenterR = (i%height) + 1
    blockCenterC = math.ceil((i + 1) / height)
    numberOfPixels = height * width
    distanceVector = np.zeros(numberOfPixels)
    for i in range(numberOfPixels):
        c = math.ceil((i + 1) / height)
        r = (i % height) + 1
        distanceVector[i] = distance(image, blockSize, blockCenterR, blockCenterC, r, c)

    sortedVector = np.sort(distanceVector)
    return 1 - math.exp(-np.sum(sortedVector[0:K]) / K)

def compute_single_scale_saliency(image, blockSize, K):
    height, width = image.shape
    numberOfPixels = height * width
    numCores = multiprocessing.cpu_count()
    temp = Parallel(n_jobs=numCores)(delayed(salient)(image, height, width, blockSize, K, i) for i in range(numberOfPixels))

    return np.transpose(np.resize(temp, (width, height)))

def prepare_output_image(image, changeMap):
    if len(image.shape) == 2:
        image = np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w, c = image.shape

    nSize = (w, h)
    changeMap = cv2.resize(changeMap, nSize)
    image[:, :, 2] = changeMap
    return image

def find_vectorSet(diffImage, newSize, h):
    i = 0
    j = 0
    vectorSet = np.zeros((int(newSize[0] * newSize[1] / (h * h)), (h * h)))

    while i < vectorSet.shape[0]:
        while j < newSize[0]:
            k = 0
            while k < newSize[1]:
                block = diffImage[j:j + h, k:k + h]
                feature = block.ravel()
                vectorSet[i, :] = feature
                k = k + h
            j = j + h
        i = i + 1

    meanVec = np.mean(vectorSet, axis=0)
    vectorSet = vectorSet - meanVec

    return vectorSet, meanVec


def find_FVS(EVS, diffImage, meanVec, new, h):
    i = math.floor(h / 2)
    featureVectorSet = []

    while i < new[0] - math.floor(h / 2):
        j = math.floor(h / 2)
        while j < new[1] - math.floor(h / 2):
            block = diffImage[i - math.floor(h / 2):i + math.ceil(h / 2), j - math.floor(h / 2):j + math.ceil(h / 2)]
            feature = block.flatten()
            featureVectorSet.append(feature)
            j = j + 1
        i = i + 1

    FVS = np.dot(featureVectorSet, EVS)
    FVS = FVS - meanVec
    return FVS


def clustering(FVS, components, new, h):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)

    leastIndex = min(count, key=count.get)
    if h % 2 == 0:
        changeMap = np.reshape(output, (new[0] - h, new[1] - h))
    else:
        changeMap = np.reshape(output, (new[0] - (h - 1), new[1] - (h - 1)))

    return leastIndex, changeMap

def change_detector_main(imageFilePath1, imageFilePath2, outputLoc, scaleFactor, algorithmType, blockSize, K, threshold, h, components):
    image1 = cv2.imread(imageFilePath1)
    image2 = cv2.imread(imageFilePath2)

    grayImage1 = rgb2gray(image1)
    grayImage2 = rgb2gray(image2)

    grayImage1[grayImage1 == 0] = 1
    grayImage2[grayImage2 == 0] = 1

    h1, w1 = grayImage1.shape
    h2, w2 = grayImage2.shape
    grayImage1 = cv2.resize(grayImage1, (int(w1 * scaleFactor), int(h1 * scaleFactor)), interpolation=cv2.INTER_AREA)
    grayImage2 = cv2.resize(grayImage2, (int(w2 * scaleFactor), int(h2 * scaleFactor)), interpolation=cv2.INTER_AREA)

    ratioImage = np.divide(grayImage1, grayImage2)
    logRatioImage = np.absolute(np.log10(ratioImage))
    saliencyMap = compute_single_scale_saliency(logRatioImage, blockSize, K)

    filteredSaliency = np.zeros(saliencyMap.shape).astype(np.uint8)
    filteredSaliency[saliencyMap > threshold] = 1

    salientImage1 = filteredSaliency * grayImage1
    salientImage2 = filteredSaliency * grayImage2

    salientImage1[salientImage1 == 0] = 1
    salientImage2[salientImage2 == 0] = 1

    salientRatioImage = np.divide(salientImage1, salientImage2)
    salientLogRatioImage = np.absolute(np.log10(salientRatioImage))
    
    if algorithmType == 'saliencyWithPcaKmean':
         salientLogRatioImage[salientLogRatioImage == 0] = 1
         
         dim = (image1.shape[0], image1.shape[1])
         newSize = np.asarray(dim) / h
         newSize = newSize.astype(int) * h
         upscaledSalientLogRatioImage = cv2.resize(salientLogRatioImage, (newSize[1], newSize[0]), interpolation=cv2.INTER_AREA)
     
         vectorSet, meanVec = find_vectorSet(upscaledSalientLogRatioImage, newSize, h)
     
         pca = PCA()
         pca.fit(vectorSet)
         EVS = pca.components_
         
         FVS = find_FVS(EVS, upscaledSalientLogRatioImage, meanVec, newSize, h)
         leastIndex, changeMap = clustering(FVS, components, newSize, h)
         
         changeMap[changeMap == leastIndex] = 255
         changeMap[changeMap != 255] = 0
         changeMap = changeMap.astype(np.uint8)
         
    if algorithmType == 'saliency':
         changeMap = np.zeros(salientRatioImage.shape)
         changeMap[salientLogRatioImage > 0] = 1
     
         dim = (image1.shape[1], image1.shape[0])
         changeMap = cv2.resize(changeMap, dim, interpolation=cv2.INTER_AREA)
         changeMap[changeMap == 1] = 255
         changeMap = changeMap.astype(np.uint8)
         
    if displayType == 'first':
        image1 = prepare_output_image(image1, changeMap)
        cv2.imwrite(outputLoc, image1)

    if displayType == 'second':
        image2 = prepare_output_image(image2, changeMap)
        cv2.imwrite(outputLoc, image2)

    if displayType == 'none' or displayType == '':
        cv2.imwrite(outputLoc, changeMap)


if __name__ == '__main__':

    displayType = ''
    input_check = True
    args = sys.argv[1:]
    if len(args) != 4 and len(args) != 5:
        print('Usage: python changeDetector.py <imageFilePath1> <imageFilePath2> <outputLoc> <scaleFactor> <algorithmType> <displayType(optional)>')
        print('You did give wrong number of parameters')
        input_check = False
    else:
        imageFilePath1 = args[0]
        imageFilePath2 = args[1]
        if not (os.path.isfile(imageFilePath1) and os.path.isfile(imageFilePath2)):
            print('Usage: python changeDetector.py <imageFilePath1> <imageFilePath2> <outputLoc> <scaleFactor> <algorithmType> <displayType(optional)>')
            print(imageFilePath1 + ' and/or ' + imageFilePath2 + ' is/are not (a) existing file(s).')
            input_check = False

        outputLoc = args[2]
        output_dir = os.path.dirname(outputLoc)
        if not os.path.isdir(output_dir):
            print('Usage: python changeDetector.py <imageFilePath1> <imageFilePath2> <outputLoc> <scaleFactor> <algorithmType> <displayType(optional)>')
            print(output_dir + ' is not a existing directory.')
            input_check = False

        scaleFactor = args[3]
        try:
            scaleFactor = float(scaleFactor)
            if scaleFactor < 0:
                print('Usage: python changeDetector.py <imageFilePath1> <imageFilePath2> <outputLoc> <scaleFactor> <algorithmType> <displayType(optional)>')
                print(str(scaleFactor) + ' is not a positive number')
                input_check = False
        except ValueError:
            print('Usage: python changeDetector.py <imageFilePath1> <imageFilePath2> <outputLoc> <scaleFactor> <algorithmType> <displayType(optional)>')
            print(scaleFactor + ' is not a positive number')
            input_check = False

        algorithmType = args[4]
        if not (algorithmType == 'saliencyWithPcaKmean' or algorithmType == 'saliency'):
            print('Usage: python changeDetector.py <imageFilePath1> <imageFilePath2> <outputLoc> <scaleFactor> <algorithmType> <displayType(optional)>')
            print('Undefined flag type. Valid flag types for using PCA-Kmean algorithm: \'saliencyWithPcaKmean\', \'saliency\'')
            input_check = False

        if len(args) == 6:
            displayType = args[5]
            types = ['first', 'second', 'none']
            if not (displayType in types):
                print('Usage: python changeDetector.py <imageFilePath1> <imageFilePath2> <outputLoc> <scaleFactor> <algorithmType> <displayType(optional)>')
                print('Undefined visualization type. Valid visualization types: \'first\', \'second\' ,\'none\'')
                input_check = False

    if input_check:
        blockSize = 7
        K = 64
        threshold = 0.05
        h = 5
        components = 2

        change_detector_main(imageFilePath1, imageFilePath2, outputLoc, scaleFactor, algorithmType, blockSize, K, threshold, h, components)