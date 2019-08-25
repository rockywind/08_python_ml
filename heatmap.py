# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:28:12 2019

@author: rockywin.wang
"""

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    gaussian_map = np.where(gaussian_map < 1e-8, 0, gaussian_map) 
    return gaussian_map

def gaussian_limit(heatmap, c_x, c_y):
    
    gaussian_stride = np.array([[-1,-1],[0,-1], [1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]])
    #gaussian_stride = np.array([[-1,-1],[0,-1], [1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]])
    search_num = gaussian_stride.shape[0]
    heatmap_sum = 0
    heatmap_new = np.zeros(heatmap.shape)
    x_index = []
    y_index = []
    ep = 1e-8
    
    for i in range(search_num):
        x_stride = c_x + gaussian_stride[i, 0]
        y_stride = c_y + gaussian_stride[i, 1]
        
        if x_stride <0:
            x_stride = 0
            continue
            
        if x_stride > heatmap.shape[1] - 1:
            x_stride = heatmap.shape[1] - 1
            continue
            
        if y_stride <0:
            y_stride = 0
            continue
            
        if y_stride > heatmap.shape[0] - 1:
            y_stride = heatmap.shape[0] - 1
            continue
            
        heatmap_sum = heatmap_sum + heatmap[y_stride, x_stride]
        x_index.append(x_stride)
        y_index.append(y_stride)
    
    for i in range(np.shape(x_index)[0]):
        heatmap_new[y_index[i], x_index[i]] = heatmap[y_index[i], x_index[i]] / (ep + heatmap_sum)
        
    return heatmap_new
        
        
# normalize 
#def normalize_gaussian(heatmap, c_x, c_y):
def normalize_gaussian(heatmap):
    gaussian_len = 1
    ep = 1e-7
    row, col = np.shape(heatmap) 
    print '---np.shape(heatmap)=', np.shape(heatmap)
    point_sum = np.sum(heatmap)
    
    for y_p in range(row):
        for x_p in range(col):
            heatmap[y_p, x_p] = heatmap[y_p, x_p] / (point_sum + ep)
    
          
    return heatmap
 
#def truncation(heatmap):
    #low_thresh
def gaussian_softmax(heatmap):
    #shiftx = heatmap - np.max(heatmap)
    shiftx = heatmap
    ep = 1e-8
    exps = np.exp(shiftx)
    heatmap = exps / (np.sum(exps) + ep)
    
    return heatmap
    
def get_gaussian_map(center_x, center_y):
    
    feature_map_size = 28
    
    cx = center_x 
    cy = center_y 
    
 
    heatmap2 = CenterGaussianHeatMap(feature_map_size, feature_map_size, cx, cy, 1)
    #heatmap3 = normalize_gaussian(heatmap2)
    #heatmap3 = gaussian_softmax(heatmap2)
    plt.imshow(heatmap2)
    plt.show()
    
    return heatmap2

if __name__ == "__main__":
    center_x = 14
    center_y = 14
    #center_x = 0
    #center_y = 1
    a = get_gaussian_map(center_x,center_y)
    norm = normalize_gaussian(a)
    #b = gaussian_limit(a, center_x, center_y)
    softmax = gaussian_softmax(norm)
    #np.mu
    
'''
image_file = 'test.jpg'
img = cv2.imread(image_file)
img = cv2.resize(img, (100,100))

img = img[:,:,::-1]
img = np.zeros([28,28])
#height, width,_ = np.shape(img)
height, width = np.shape(img)
cy, cx = height/2.0 - 14, width/2.0 - 10
print 'cy=', cy
print 'cx=', cx
#start = time.time()
#heatmap1 = CenterLabelHeatMap(width, height, cx, cy, 21)
#t1 = time.time() - start

#start = time.time()
heatmap2 = CenterGaussianHeatMap(height, width, cx, cy, 1)
#t2 = time.time() - start

#print(t1, t2)

#plt.subplot(1,2,1)
#plt.imshow(heatmap1)
#plt.subplot(1,2,2)
plt.imshow(heatmap2)
plt.show()

print('End.')
'''