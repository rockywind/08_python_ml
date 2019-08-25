import caffe
import numpy as np
from google.protobuf import text_format
from caffe.proto import caffe_pb2



#DEBUG = False


class GenerateMaskLayer(caffe.Layer):
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
        
        bottom__mask_flag = bottom[0].data
        bottom__feature = bottom[1].data
        feature_map_size = np.shape(bottom__feature)[0]

        if bottom__mask_flag == False:
            mask = np.zeros_like(bottom__feature)

        else:
            mask = np.ones_like(bottom__feature)

        feature_mul_mask = np.multiply(mask, bottom__feature)

        top[0].reshape(*feature_mul_mask.shape)
        top[0].data[...] = feature_mul_mask

        '''
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
        '''
    def backward(self, top, propagate_down, bottom):
        #pass
        bottom__mask_flag = bottom[0].data
        bottom[1].diff[...] = top[1].diff[...] * bottom__mask_flag


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

        
