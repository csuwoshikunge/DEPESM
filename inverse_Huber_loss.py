import numpy as np 
import cv2 
import caffe

THRESHOLD = 0.5
class inverseHuberlayer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom)!=2:
            raise exception("require two inputs to compute loss!")
        if bottom[0].count!=bottom[1].count:
            raise exception("two inputs must have equal dimension!")
        top[0].reshape(1) # just a scalar
        self.diff=np.zeros_like(bottom[0].data, np.float32)
    
    def forward(self,bottom,top):
        
        prediction  = bottom[0].data
        groundtruth = bottom[1].data
        b,c,h,w = groundtruth.shape
        valid_mask = groundtruth > THRESHOLD

        # inverse Huber loss
        pred_valid = prediction*valid_mask 
        gt_valid = groundtruth*valid_mask
        pw_f1 = np.abs(pred_valid - gt_valid)
        inverse_Huber_const = 0.2 * np.max(pw_f1)
        pw_f2 = 0.5 * pw_f1**2 / inverse_Huber_const + 0.5 * inverse_Huber_const
        pw_f = np.where(pw_f1<=inverse_Huber_const, pw_f1, pw_f2)
        top[0].data[...] = np.sum(pw_f) / np.sum(valid_mask)

        pw_f1_diff_ind = pw_f1 <= inverse_Huber_const
        pw_f2_diff_ind = pw_f2 >  inverse_Huber_const

        pw_f1_diff = np.sign(pred_valid - gt_valid) * pw_f1_diff_ind / np.sum(valid_mask)
        pw_f2_diff = (pred_valid - gt_valid) * pw_f2_diff_ind / np.sum(valid_mask)
        self.diff = np.where(pw_f1<=inverse_Huber_const, pw_f1_diff, pw_f2_diff)

    def backward(self,top,propagate_down,bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i==0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff  #????????




