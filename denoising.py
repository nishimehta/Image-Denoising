#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:16:41 2018

@author: nishimehta
"""

import numpy as np
import cv2 as cv2

def extract_matrix(i,j,mat,n):
    x = int(n/2)
    extract = np.zeros((n,n)) 
    for k in range(n):
        for l in range(n):
            extract[k][l] = mat[i-x+k][j-x+l]
    return extract


def erosion(kernel,im):
    eroded = np.zeros(im.shape)
    s = int(len(kernel)/2)
    for i in range(s,len(im)-s):
        for j in range(s,len(im[0])-s):
            #extract neighbouring pixel values to calculate erosion
            p = extract_matrix(i,j,im,len(kernel))
            eroded[i][j] = 255 if np.array_equal(p,kernel) else 0
    return eroded

def dilation(kernel,im):
    dilated = np.zeros(im.shape)
    s = int(len(kernel)/2)
    for i in range(s,len(im)-s):
        for j in range(s,len(im[0])-s):
            #extract neighbouring pixel values to calculate dilation
            p = extract_matrix(i,j,im,len(kernel))
            dilated[i][j] = 255 if (p == kernel).any() else 0
    return dilated

def opening(kernel,im):
    return dilation(kernel,erosion(kernel,im))
def closing(kernel,im):
    return erosion(kernel,dilation(kernel,im))

#%%
image = cv2.imread('noise.jpg',0)
kernel = np.array([[255,255,255],[255,255,255],[255,255,255]])
cv2.imwrite('kernel.jpg',kernel)
#morphological algorithm 1 to remove noise 
morph1 = closing(kernel,opening(kernel,image))
#morphological algorithm 2 to remove noise 
morph2 = opening(kernel,closing(kernel,image))

cv2.imwrite('res_noise1.jpg',morph1)
cv2.imwrite('res_noise2.jpg',morph2)
#%% Boundary extraction image - erosion
eroded1 = erosion(kernel,morph1)
eroded2 = erosion(kernel,morph2)

boundary1 = morph1 - eroded1
boundary2 = morph2 - eroded2

cv2.imwrite('res_bound1.jpg',boundary1)
cv2.imwrite('res_bound2.jpg',boundary2)
#%%


