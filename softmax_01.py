import caffe
import numpy as np
from google.protobuf import text_format
from caffe.proto import caffe_pb2



#DEBUG = False

def tru(arr, small_thersh, big_thresh):
    arr_cp = np.copy(arr) 
    smallest_x = np.zeros_like(arr) + small_thersh
    biddest_x = np.zeros_like(arr) + big_thresh 
    #print('smallest_x = ',smallest_x)
    #print('biddest_x = ',biddest_x)
    bigger_then_small = np.where(arr_cp<small_thersh,smallest_x,arr_cp)
    arr_ok = np.where(bigger_then_small>big_thresh,biddest_x,bigger_then_small)
    return arr_ok



class MySoftmax(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """


    def setup(self, bottom, top):
        pass

    def forward(self, bottom, top):
        ep = 0.000001
        bottom_data = bottom[0].data
        print "------Befor softmax , min is --------------",np.min(bottom_data)
        print "------Befor softmax , max is --------------",np.max(bottom_data)
        bottom_data_shape = np.shape(bottom_data) # [N, H*W, C, 1]
        
        bottom_data = bottom_data - np.max(bottom_data)

        bottom_data_exp = np.exp(bottom_data)   # exp(bottom), [N, H*W, C, 1]
        bottom_data_exp_sum = np.sum(bottom_data_exp,1)    # sum(exp(bottom)), [N,C,1]
        weights = np.divide(1,bottom_data_exp_sum + ep)

        top_data_zeros = np.zeros_like(bottom_data)
        top_data_zeros_tru = np.zeros_like(bottom_data)

        for n in range(bottom_data_shape[0]):
            top_data_zeros[n,:,:,:] = np.multiply(bottom_data_exp[n,:,:,:],weights[n,:,:])
        for n in range(bottom_data_shape[0]):
            for c in range(bottom_data_shape[2]):
                one_channel_image = top_data_zeros[n,:,c,:]
                top_data_zeros_tru[n,:,c,:] = tru(one_channel_image, 1e-4, 0.9217)
        #print "------After softmax , Min is --------------",np.min(top_data_zeros)
        #print "------After softmax , Max is --------------",np.max(top_data_zeros)
        #top_data_zeros_tru = tru(top_data_zeros, 1e-4, 0.9217)
        top[0].reshape(*top_data_zeros.shape)
        top[0].data[...] = top_data_zeros_tru

    def backward(self, top, propagate_down, bottom):
        ep = 0.000001
        top_diff = top[0].diff[...] # [N,H*W,C,1]
        forward_top_data = top[0].data[...]   # [N,H*W,C,1]
        
        bottom_data_exp = np.exp(forward_top_data)
        exp_arr = np.sum(bottom_data_exp,1)
        weights = np.divide(1,exp_arr + ep)
        top_data_zeros = np.zeros_like(forward_top_data)
        for n in range(forward_top_data.shape[0]):
            top_data_zeros[n,:,:,:] = np.multiply(bottom_data_exp[n,:,:,:],weights[n,:,:])
        
        for n in range(forward_top_data.shape[0]):
            for c in range(forward_top_data.shape[2]):
                one_hang = np.ones((1,np.shape(top_data_zeros)[1]))
                y_arr = np.matmul(top_data_zeros[n,:,c,:],one_hang)
                eyes = np.eye(np.shape(y_arr)[0])
                delta = np.multiply(np.subtract(y_arr,eyes),np.transpose(np.multiply(top_data_zeros[n,:,c,:],-1),axes=[1,0]))
                diff_trans = np.matmul(np.transpose(top_diff[n,:,c,:],[1,0]),delta)
                diff = np.transpose(diff_trans,[1,0])
                bottom[0].diff[n,:,c,:] = diff        
#         arr_sum = np.sum(forward_top_data,1)   # N, C, 1
#         arr_sum_ep = arr_sum + ep             # N, C, 1
#         weights = np.divide(1,arr_sum_ep)     # N, C, 1
#         for n in range(top_diff.shape[0]):
#             bottom_diff_zeros[n,:,:,:] = np.multiply(top_diff[n,:,:,:],weights[n,:,:])
#         bottom[0].diff[...] = bottom_diff_zeros

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
