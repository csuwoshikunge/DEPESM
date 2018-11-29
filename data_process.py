import caffe
import numpy as np 
import cv2

SCALE = 6000.
class DataTypeConvertLayer(caffe.Layer):
    def setup(self, bottom, top):
        b,c,h,w = bottom[0].shape
        top[0].reshape(b,c,h,w)
    
    def forward(self, bottom, top):
        data = bottom[0].data / SCALE
        top[0].reshape(*(data.shape))
        top[0].data[...] = data
    
    def backward(self, top, propagate_down, bottom):
        '''
        data process need not propagate gradients.
        '''
        pass 
    
    def reshape(self, bottom, top):
        '''
        reshape when call forward.
        '''
        pass
    
    '''
    python layer for calling this layer, for example,
    #############
    layer{
        name:'xxxx'
        type:'Python'
        bottom:'A'
        top:'B'
        python_param{
            module:'data_process' # the name of data process .py file
            layer:'DataTypeConvertLayer'
        }
    }

    '''