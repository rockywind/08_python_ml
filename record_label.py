# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:39:44 2019

@author: rockywin.wang
"""

import caffe
import numpy as np
from google.protobuf import text_format
from caffe.proto import caffe_pb2



#DEBUG = False


class GenerateRecordLabelLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    #def _init_(self):
        #self._label_record = 0
        
    def setup(self, bottom, top):
        N = 1
        C = 8
        H = 28
        W = 28
        self._label_record = np.zeros([N,C,H,W])
        #self._label_record = np.reshape(self._label_record, [N,C,H,W])
        top[0].reshape(N, C, H, W)

    def forward(self, bottom, top):
        
        
        bottom_flag = bottom[0].data
        bottom_roi_pool = bottom[1].data
         
        if bottom_flag:
            self._roi_align_record = bottom_roi_pool
        else:
            bottom_roi_pool = self._roi_align_record
        
        bottom_roi_pool = self._roi_align_record
        top[0].reshape(*bottom_roi_pool.shape)
        top[0].data[...] = bottom_roi_pool

    def backward(self, top, propagate_down, bottom):
        pass 
    
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

        
