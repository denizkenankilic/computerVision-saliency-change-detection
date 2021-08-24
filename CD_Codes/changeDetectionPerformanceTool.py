# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:21:01 2021

@author: deniz.kilic
"""

import cv2
import os
from skimage import measure
import pandas as pd
import sys
import math
import numpy

def convert_to_binary_ground_truth(groundTruthImg, BINARY_THRESH_VAL, BINARY_MAX_VAL, groundTruthFileName):
    """
    The function to convert a graylevel images to binary images.
    """
    minPixelValue = groundTruthImg.min()
    maxPixelValue = groundTruthImg.max()

    if minPixelValue == 0 and numpy.unique(groundTruthImg).shape[0] == 2 and not maxPixelValue == 255:
         groundTruthImg[groundTruthImg == maxPixelValue] = 255
    elif minPixelValue == 0 and numpy.unique(groundTruthImg).shape[0] == 2 and maxPixelValue == 255:
         return groundTruthImg
    elif minPixelValue == 0 and maxPixelValue == 255 and numpy.unique(groundTruthImg).shape[0] > 2:
         retvalGroundTruth, groundTruthImg = cv2.threshold(groundTruthImg, BINARY_THRESH_VAL, BINARY_MAX_VAL, cv2.THRESH_BINARY)
    else:
         print('The type of the ground truth is not valid:' + ' ' + groundTruthFileName)
    return groundTruthImg

def prepare_data(groundTruthImagePath, outputMaskImagePath, BINARY_THRESH_VAL, BINARY_MAX_VAL):
    """
    A function to check whether sizes of images are equal, if not
    resize the outputImg image with respect to the ground truth image.
    It is assumed that performance is calculated by black-and-white images.
    """

    groundTruthImg = cv2.imread(groundTruthImagePath, cv2.IMREAD_GRAYSCALE)
    groundTruthFileName = os.path.basename(groundTruthImagePath)
    outputImg = cv2.imread(outputMaskImagePath, cv2.IMREAD_GRAYSCALE)
    if groundTruthImg.shape[0] != outputImg.shape[0] or groundTruthImg.shape[1] != outputImg.shape[1]:
         outputImg = cv2.resize(outputImg, (groundTruthImg.shape[1], groundTruthImg.shape[0]), interpolation=cv2.INTER_AREA)

    blackWhiteGroundTruth = convert_to_binary_ground_truth(groundTruthImg, BINARY_THRESH_VAL, BINARY_MAX_VAL, groundTruthFileName)
    retvalOutput, blackWhiteOutput = cv2.threshold(outputImg, BINARY_THRESH_VAL, BINARY_MAX_VAL, cv2.THRESH_BINARY)

    return blackWhiteGroundTruth, blackWhiteOutput

def get_object_properties(groundTruthImage):
     """
     A function that measures intended properties of labeled image regions.
     """

     labeledRegions = measure.label(groundTruthImage)
     regionProps = measure.regionprops(labeledRegions)

     areaList = []
     coordinateList = []

     for i in range(len(regionProps)):
          areaList.append(regionProps[i].area)
          coordinateList.append(regionProps[i].coords)

     return areaList, coordinateList

def perf_measure(groundTruthImage, outputMaskImage, areaList, coordinateList, RATIO_VALUE, BINARY_MAX_VAL):
    """
    The performance function for a given ground truth and change map data.
    It calculates the number of true positive, false positive, true negative, false negative
    in the image.
    """

    totalPixel = groundTruthImage.shape[0] * groundTruthImage.shape[1]
    thresholdToDefineSize = totalPixel * RATIO_VALUE
    objectCountInImage = len(areaList)
    noneFalseNegative = 0

    # noneFalseNegative's of the small objects are discarded to calculate falseNegative
    for k in range(objectCountInImage):
         if areaList[k] < thresholdToDefineSize:
                for m in range(coordinateList[k].shape[0]):
                     if outputMaskImage[coordinateList[k][m][0], coordinateList[k][m][1]] == 0:
                        noneFalseNegative += 1
    truePositive = cv2.sumElems(groundTruthImage & outputMaskImage)[0] / BINARY_MAX_VAL
    falsePositive = cv2.sumElems(~groundTruthImage & outputMaskImage)[0] / BINARY_MAX_VAL
    trueNegative = cv2.sumElems(~groundTruthImage & ~outputMaskImage)[0] / BINARY_MAX_VAL
    falseNegative = totalPixel - truePositive - falsePositive - trueNegative - noneFalseNegative

    return truePositive, falsePositive, trueNegative, falseNegative

def calculate_detected_object_ratio(groundTruthImage, outputMaskImage, areaList, coordinateList, RATIO_VALUE, DETECTED_OBJECT_RATIO_THR):
    """
    The performance function for a given ground truth and change map data.
    It calculates the ratio of detected objects in image with respect to the
    "DETECTED_OBJECT_RATIO_THR" threshold value.
    """

    totalPixel = groundTruthImage.shape[0] * groundTruthImage.shape[1]
    thresholdToDefineSize = totalPixel * RATIO_VALUE
    objectCountInImage = len(areaList)

    bigObjectsCount = 0
    detectedObjectsCount = 0
    detectedSmallObjectsCount = 0
    for k in range(objectCountInImage):
         singleObjectTruePositiveCountTreshold = math.ceil(areaList[k] * DETECTED_OBJECT_RATIO_THR)
         if areaList[k] >= thresholdToDefineSize:
              bigObjectsCount += 1
         singleObjectTruePositiveCount  = 0
         for m in range(coordinateList[k].shape[0]):
              if outputMaskImage[coordinateList[k][m][0], coordinateList[k][m][1]] == 255:
                   singleObjectTruePositiveCount += 1
                   if singleObjectTruePositiveCount >= singleObjectTruePositiveCountTreshold:
                        detectedObjectsCount += 1
                        if areaList[k] < thresholdToDefineSize:
                             detectedSmallObjectsCount += 1
                        break
         else:
              continue
    concernedObjectCount = bigObjectsCount + detectedSmallObjectsCount
    if concernedObjectCount == 0:
        detectedObjectRatio = None
    else:
        detectedObjectRatio =  detectedObjectsCount / concernedObjectCount

    return detectedObjectRatio

def perf_metrics(truePositive, falsePositive, trueNegative, falseNegative, performanceMetricType, groundTruthImage, outputMaskImage, areaList, coordinateList):
     """
     A function that calculates the performance metrics.
     """

     totalValidPixel = truePositive + falsePositive + trueNegative + falseNegative
     pre = ((truePositive + falsePositive) * (truePositive + falseNegative) + (falseNegative + trueNegative) * (trueNegative + falsePositive)) / (totalValidPixel * totalValidPixel)
     pcc = (truePositive + trueNegative) / totalValidPixel
     if truePositive == 0 and falsePositive == 0:
          prec = None
     else:
          prec = truePositive / (truePositive + falsePositive)
     if truePositive == 0 and falseNegative == 0:
          recall = None
     else:
          recall = truePositive / (truePositive + falseNegative)

     if performanceMetricType == 'pcc':
          return pcc
     elif performanceMetricType == 'oe':
          oe = falsePositive + falseNegative
          return oe
     elif performanceMetricType == 'kc':
          if pre == 1:
               kc = 1
          else:
               kc = (pcc - pre) / (1 - pre)
          return kc
     elif performanceMetricType == 'jc':
          if truePositive + falsePositive + falseNegative == 0:
              jc = None
          else:
              jc = truePositive / (truePositive + falsePositive + falseNegative)
          return jc
     elif performanceMetricType == 'yc':
          if truePositive + falsePositive == 0 or trueNegative + falseNegative ==0:
               yc = None
          else:
               yc = truePositive / (truePositive + falsePositive) + trueNegative / (trueNegative + falseNegative) - 1
          return yc
     elif performanceMetricType == 'prec':
          return prec
     elif performanceMetricType == 'recall':
          return recall
     elif performanceMetricType == 'fmeas':
          if prec == 0 and recall == 0:
               fmeas = 0
          elif prec == None or recall == None:
               fmeas = None
          else:
               fmeas = 2 * prec * recall / (prec + recall)
          return fmeas
     elif performanceMetricType == 'sp':
          if trueNegative == 0 and falsePositive == 0:
               sp = None
          else:
               sp = trueNegative / (trueNegative + falsePositive)
          return sp
     elif performanceMetricType == 'fpr':
          if falsePositive == 0 and trueNegative == 0:
               fpr = None
          else:
               fpr = falsePositive / (falsePositive + trueNegative)
          return fpr
     elif performanceMetricType == 'fnr':
          if truePositive == 0 and falseNegative == 0:
               fnr = None
          else:
               fnr = falseNegative / (falseNegative + truePositive)
          return fnr
     elif performanceMetricType == 'pwc':
          pwc = 100 * (falseNegative + falsePositive) / (truePositive + falseNegative + falsePositive + trueNegative)
          return pwc
     elif performanceMetricType == 'odr':
          detectedObjectRatio = calculate_detected_object_ratio(groundTruthImage, outputMaskImage, areaList, coordinateList, RATIO_VALUE, DETECTED_OBJECT_RATIO_THR)
          return detectedObjectRatio
     else:
          print("Warning: Supported metrics are 'pcc', 'oe', 'kc', 'jc', 'yc', 'prec', 'recall', 'fmeas', 'sp', 'fpr', 'fnr', 'pwc', 'odr'.")
          return None

# Main Function
def perf_eval(groundTruthImagePath, outputMaskImagePath, RATIO_VALUE, performanceMetricTypeList):
    """
    groundTruthImagePath: Ground truth image path to test the accuracy of image analysis processes.
    outputMaskImagePath: Output image path of the change detection algorithm.
    RATIO_VALUE: It is multiplied with the size of the ground truth image to
                 define area threshold of small objects.
    performanceMetricTypeList (i.e. ['recall', 'prec']):
       It can be:
         'pcc' -> percentage correct classification (accuracy)
         'oe' -> overall error
         'kc' -> Kappa coefficient
         'jc' -> Jaccard coefficient
         'yc' -> Yule coefficient
         'prec' -> precision
         'recall' -> recall
         'fmeas' -> F-measure
         'sp' -> specificity
         'fpr' -> false positive rate
         'fnr' -> false negative rate
         'pwc' -> percentage of wrong classifications
         'odr' -> detected object ratio
    """
    scoreList = []

    [blackWhiteGroundTruth, blackWhiteOutput] = prepare_data(groundTruthImagePath, outputMaskImagePath, BINARY_THRESH_VAL, BINARY_MAX_VAL)
    areaList, coordinateList = get_object_properties(blackWhiteGroundTruth)
    [truePositive, falsePositive, trueNegative, falseNegative] = perf_measure(blackWhiteGroundTruth, blackWhiteOutput, areaList, coordinateList, RATIO_VALUE, BINARY_MAX_VAL)
    for i in range(len(performanceMetricTypeList)):
         score = perf_metrics(truePositive, falsePositive, trueNegative, falseNegative, performanceMetricTypeList[i], blackWhiteGroundTruth, blackWhiteOutput, areaList, coordinateList)
         scoreList.append(score)

    return scoreList

if __name__ == '__main__':

    args = sys.argv[1:]
    args_len = len(args)
    inputCheck = True

    if not args_len == 2:
        print('Usage: changeDetectionPerformanceTool <excelTypeInputPath> <excelTypeOutputPath>')
        print('You gave insufficient number of input arguments.')
        inputCheck = False
    else:
        excelTypeInputPath = args[0]
        excelTypeOutputPath = args[1]
        if inputCheck and not os.path.isfile(excelTypeInputPath):
            print('Usage: changeDetectionPerformanceTool <excelTypeInputPath> <excelTypeOutputPath>')
            print(excelTypeInputPath + ' is not an existing file.')
            inputCheck = False
        if inputCheck and not os.path.splitext(excelTypeOutputPath)[1] == '.xlsx':
            print('Usage: changeDetectionPerformanceTool <excelTypeInputPath> <excelTypeOutputPath>')
            print('Extension of the output file need to be ".xlsx".')
            inputCheck = False

    if inputCheck:
         inputs = pd.read_excel(excelTypeInputPath)
         for i in range(2, inputs.shape[1]):
             if inputCheck and not (inputs.columns[i] == 'pcc' or
                              inputs.columns[i] == 'oe' or
                              inputs.columns[i] == 'kc' or
                              inputs.columns[i] == 'jc' or
                              inputs.columns[i] == 'yc' or
                              inputs.columns[i] == 'prec' or
                              inputs.columns[i] == 'recall' or
                              inputs.columns[i] == 'fmeas' or
                              inputs.columns[i] == 'sp' or
                              inputs.columns[i] == 'fpr' or
                              inputs.columns[i] == 'fnr' or
                              inputs.columns[i] == 'pwc' or
                              inputs.columns[i] == 'odr'):
                  print('performanceMetricType can be \'pcc\' or \'oe\' or \'kc\' or \'jc\' or \'yc\' or \'prec\' or \'recall\' or \'fmeas\' or \'sp\' or \'fpr\' or \'fnr\' or \'pwc\' or \'odr\'')
                  inputCheck = False

         if inputCheck:
             for j in range(inputs.shape[0]):
                    groundTruthImagePath = inputs.iloc[j][0]
                    outputMaskImagePath = inputs.iloc[j][1]
                    if inputCheck and not (os.path.isfile(groundTruthImagePath) and os.path.isfile(outputMaskImagePath)):
                       print(groundTruthImagePath + ' and/or ' + outputMaskImagePath + ' is/are not (a) existing file(s) in line:' + str(j + 2))
                       inputCheck = False

         if inputCheck:
              for j in range(inputs.shape[0]):
                  if inputCheck:
                       groundTruthOutputMaskPathsAndMetricsToBeCalculated = []
                       groundTruthOutputMaskPathsAndMetricsToBeCalculated.append(inputs.iloc[j][0])
                       groundTruthOutputMaskPathsAndMetricsToBeCalculated.append(inputs.iloc[j][1])
                       for k in range(2,inputs.shape[1]):
                            if inputs.iloc[j][k] == True:
                                 groundTruthOutputMaskPathsAndMetricsToBeCalculated.append(inputs.columns[k])
                       imageInputsCount = len(groundTruthOutputMaskPathsAndMetricsToBeCalculated)
                       groundTruthImagePath = groundTruthOutputMaskPathsAndMetricsToBeCalculated[0]
                       outputMaskImagePath = groundTruthOutputMaskPathsAndMetricsToBeCalculated[1]

                       performanceMetricTypeList = []
                       if imageInputsCount > 2:
                            for i in range(2, imageInputsCount):
                                 performanceMetricTypeList.append(groundTruthOutputMaskPathsAndMetricsToBeCalculated[i])
                       else:
                            print('Warning: No metric information is entered in line:' + str(j + 2))

                       #groundTruthImagePath = r'C:\Users\deniz.kilic\SAR_change_detection\gt19.tif'
                       #outputMaskImagePath = r'C:\Users\deniz.kilic\SAR_change_detection\changemap19bs6.jpg'
                       RATIO_VALUE = 0.001
                       #performanceMetricTypeList = ['kc', 'fmeas', 'odr']
                       BINARY_THRESH_VAL = 127
                       BINARY_MAX_VAL = 255
                       DETECTED_OBJECT_RATIO_THR = 0.3

                       scoreList = perf_eval(groundTruthImagePath,
                                             outputMaskImagePath,
                                             RATIO_VALUE,
                                             performanceMetricTypeList)
                       for l in range(len(scoreList)):
                            inputs.loc[j, performanceMetricTypeList[l]] = scoreList[l]
                  inputs.to_excel(excelTypeOutputPath)
