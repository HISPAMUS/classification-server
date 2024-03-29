#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================

from __future__ import print_function
import sys, os, warnings
from keras import backend as K
import argparse
import cv2
import numpy as np

TYPE_NORMALIZATION_255 = "255"
TYPE_NORMALIZATION_INVERSE_255 = "inv255"
TYPE_NORMALIZATION_STANDARD = "standard"
TYPE_NORMALIZATION_MEAN = "mean"
TYPE_NORMALIZATION_ORIGINAL = "original"

EQUALIZATION_TYPE_NONE = "none"
EQUALIZATION_TYPE_WHITE_BALANCE = 'wb'
EQUALIZATION_TYPES = [EQUALIZATION_TYPE_NONE, EQUALIZATION_TYPE_WHITE_BALANCE]


#****************************
#Functions for normalization
#****************************
# 255 normalization
def normalization255(imgs):
    return imgs.astype(np.float32) / 255.

#inverse 255 normalization
def normalizationInverse255(imgs):
    return (255. - imgs.astype(np.float32)) / 255.

#Standard normalization
def normalizationStandardWithParams(imgs, mean, std):
    return (imgs.astype(np.float32) - mean) / (std + 0.00001)

#Mean normalization
def normalizationMeanWithParams(imgs, mean):
    return imgs.astype(np.float32) - mean

#Apply image normalization
def applyNormalizationDataset(type_normalization, imgs, mean = None, std = None):
    if type_normalization == TYPE_NORMALIZATION_255:
        return normalization255(imgs)
    if (type_normalization == TYPE_NORMALIZATION_INVERSE_255):
        return normalizationInverse255(imgs)
    elif type_normalization == TYPE_NORMALIZATION_STANDARD:
        assert(mean is not None)
        assert(std is not None)
        return normalizationStandardWithParams(imgs, mean, std)
    elif type_normalization == TYPE_NORMALIZATION_MEAN:
        assert(mean is not None)
        return normalizationMeanWithParams(imgs, mean)
    elif type_normalization == TYPE_NORMALIZATION_ORIGINAL:
        return imgs
    else:
        assert(False)


#Apply white balance for a RGB image
def whiteBalanceForRGBImage(image, perc = 0.05):
    return np.dstack([whiteBalanceForChannelImage(channel, 0.05) for channel in cv2.split(image)] )

#Apply white balance for a channel from a RGB image
def whiteBalanceForChannelImage(channel, perc = 0.05):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel
        
#Apply equalization if it is selected in the program arguments.
def apply_equalization(img_x, type):
    if type == EQUALIZATION_TYPE_NONE:
        return img_x
    elif type == EQUALIZATION_TYPE_WHITE_BALANCE:
        img_x = whiteBalanceForRGBImage(img_x)
    else:
        raise Exception('Undefined equalization type: ' + type)

    return img_x

#Resize an image into a selected size
def redimImage(img, height, width, interpolation = cv2.INTER_LINEAR):
    img2 = img.copy()
    return cv2.resize(img2,(width,height), interpolation=interpolation)

def predictFullDocument(src_im, block_size, batch_size, model):
    list_samples = [src_im]
    list_samples = np.asarray(list_samples).reshape(len(list_samples), block_size[0], block_size[1], block_size[2])
    prediction_src_sample = model.predict(x=list_samples, batch_size=batch_size, verbose=0)
    binarized_prediction = np.argmax(prediction_src_sample, axis=3)
    assert(len(binarized_prediction) == 1)
    prediction_full_src = binarized_prediction[0]
    prediction_full_src = np.array(prediction_full_src, dtype='uint8')
    return prediction_full_src

# Return the list of bounding boxes of a image calculating the contours.
def getBoundingBoxes(image, val=100):
    threshold = val
    minContourSize= int(image.shape[0]*image.shape[1]*0.0025)
    
    img = np.copy(image)
    ROWS = img.shape[0]
    COLS = img.shape[1]

    for j in range(COLS):
        img[0, j] = 0
        img[1, j] = 0
        img[2, j] = 0

        img[ROWS-1, j] = 0
        img[ROWS-2, j] = 0
        img[ROWS-3, j] = 0
    
    for i in range(ROWS):
        img[i, 0] = 0
        img[i, 1] = 0
        img[i, 2] = 0

        img[i, COLS-1] = 0
        img[i, COLS-2] = 0
        img[i, COLS-3] = 0
        
    #minContourSize = 500
    im = np.uint8(img)
    canny_output = cv2.Canny(im, threshold, threshold * 2)
    
    contours, herarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []
    boundRect = []

    # calculate points for each contour
    for i in range(len(contours)):

        contour_i = contours[i]
        contour_poly = cv2.approxPolyDP(contour_i, 3, True)
        hull_i = cv2.convexHull(contour_poly, False)

        area = cv2.contourArea(hull_i)

        if (area > minContourSize):
            bbox_i = cv2.boundingRect(hull_i)

            rect_by_corners = (bbox_i[1], bbox_i[0], bbox_i[1]+bbox_i[3], bbox_i[0]+bbox_i[2])

            if (bbox_i[3] > bbox_i[2]):# If tt is a vertical region, we ignore it.
                continue

            if (bbox_i[3] / bbox_i[2] < 0.05):# If width and height are so much different, we ignore the region
                continue 
             
            boundRect.append(rect_by_corners)

    return boundRect

#Return the list of bounding boxes detected in the prediction image.
def getBoundingBoxesFromPrediction(prediction_image, vertical_reduction_regions=0., margin=0):
    val_bbox = 127
    bboxes = getBoundingBoxes(prediction_image, val=val_bbox)

    pred_shape = prediction_image.shape

    for idx in range(len(bboxes)):
        
        x_start = bboxes[idx][0]
        y_start = bboxes[idx][1]
        x_end = bboxes[idx][2]
        y_end = bboxes[idx][3]
        if vertical_reduction_regions > 0.:
            vertical_region_size = x_end - x_start
            gt_prediction_vertical_size = int(vertical_region_size // (1-vertical_reduction_regions))
            vertical_reduction_region_side = int((gt_prediction_vertical_size - vertical_region_size) // 2)
            bboxes[idx] = (
                        max(0,x_start-int(vertical_reduction_region_side)), 
                        max(0,y_start-margin), 
                        min(pred_shape[0], x_end + int(vertical_reduction_region_side)), 
                        min(y_end + margin, pred_shape[1]))
            
    return bboxes
    
#Predict the regions by a SAE model.
def predict_regions(
                        model,
                        src_image,
                        equalization_mode='wb',
                        normalization_mode='inv255',
                        block_size=(524, 524, 3), 
                        gt_height_reduction = 0.2,
                        batch_size = 4
                        ):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    original_src_shape = src_image.shape

    equalized_im = apply_equalization(src_image, equalization_mode)

    rescaled_equalized_im = redimImage(equalized_im, block_size[0], block_size[1])
    normalized_im = applyNormalizationDataset(type_normalization=normalization_mode, imgs=rescaled_equalized_im, mean = None, std = None)
    
    prediction = predictFullDocument(normalized_im, block_size, batch_size, model)

    prediction_image = (1-prediction) * 255
    prediction_image = redimImage(prediction_image, original_src_shape[0], original_src_shape[1])
    prediction_image = (prediction_image==255)*255
    prediction_image = np.uint8(prediction_image)

    #Clean the borders of the image
    margin = 10
    prediction_image[0:original_src_shape[0], 0:margin] = 0
    prediction_image[0:original_src_shape[0], original_src_shape[1]-margin:original_src_shape[1]] = 0
    prediction_image[0:margin, 0:original_src_shape[1]] = 0
    prediction_image[original_src_shape[0]-margin:original_src_shape[0], 0:original_src_shape[1]] = 0

    bboxes_prediction_image = getBoundingBoxesFromPrediction(prediction_image, gt_height_reduction, margin)

    return bboxes_prediction_image
