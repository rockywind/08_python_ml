import caffe
import numpy as np
from google.protobuf import text_format
from caffe.proto import caffe_pb2



#DEBUG = False


class MinMaxNorm(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """


    def setup(self, bottom, top):
        N = 1
        C = 8
        H = 28
        W = 28
        top[0].reshape(N, H*W, C, 1)

    def forward(self, bottom, top):
        ep = 1e-9
        scale = 10
        bottom_data = bottom[0].data
        bottom_data_shape = bottom_data.shape
        top_data_zeros = np.zeros_like(bottom_data)
        
        for n in range(bottom_data_shape[0]):
            for c in range(bottom_data_shape[2]):
                min_max_data = (bottom_data[n,:, c, :] - np.min(bottom_data[n,:, c, :])) / (np.max(bottom_data[n,:, c, :]) - np.min(bottom_data[n,:, c, :]) + ep)
                min_max_data = min_max_data *scale
                top_data_zeros[n,:,c,:] = min_max_data
                #one_channel_image = top_data_zeros[n,:,c,:]
                #top_data_zeros_tru[n,:,c,:] = tru(one_channel_image, 1e-4, 0.9217)

        min_max_data = top_data_zeros
        top[0].reshape(*min_max_data.shape)
        top[0].data[...] = min_max_data

    def backward(self, top, propagate_down, bottom):
        scale = 10
        ep = 1e-9
        top_diff = top[0].diff[...] # [N,H*W,C,1]
        forward_top_data = top[0].data[...]   # [N,H*W,C,1]
        #print '-------forward_top_data.shape----', forward_top_data.shape
        for n in range(forward_top_data.shape[0]):
            for c in range(forward_top_data.shape[2]):
                diff = 1 / (np.max(forward_top_data[n,:, c, :]) - np.min(forward_top_data[n,:, c, :]) + ep)
                top_diff[n,:,c,:] = top_diff[n,:,c,:] * diff * scale 
                bottom[0].diff[n,:,c,:] = top_diff[n,:,c,:]
        #print '-------------top_diff.shape-----', type(top_diff), top_diff.shape
        #print '-------------bottom[0].diff.shape-----', type(bottom[0].diff), bottom[0].diff.shape
        #bottom[0].diff = top_diff 
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

        
