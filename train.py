import numpy as np 
import os 
from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format # new version
import cv2
import matplotlib.pyplot as plt 
import caffe 

GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
SOLVER = 'path/to/solver.prototxt'
MAX_ITERS = 10000
WEIGHTS = 'path/to/pretrained_caffemodel'
RANDOM_SEED = 256

class SolverWrapper(object):
    """
    A simple wrapper for caffe's solver,
    which controls the snapshotting process flexibly
    """

    def __init__(self, solver_prototxt, output_dir, pretrained_model=None):
        self.output_dir = output_dir
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print(('loading model from {:s}').format(pretrained_model))
            self.solver.net.copy_from(pretrained_model)
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            text_format.Merge(f.read(), self.solver_param)

    def snapshot(self):
        net = self.solver.net
        filename = (self.solver_param.snapshot_prefix + '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)
        net.save(str(filename))

    def train_model(self, max_iters):
        last_snapshot_iter = -1
        model_paths = []
        while self.solver.iter < max_iters:
            self.solver.step(1)
            if self.solver.iter %100 ==0:
                out = self.solver.net.blobs['some_layer_top_data'].data[...] # with shape b*c*h*w
                save_as_png(out[0,0,:,:], 'temp.png')
            if self.solver.iter % 10000 ==0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())  # save model
            if last_snapshot_iter != self.solver.iter:
                model_paths.append(self.snapshot())  # save the last trained step model
        return model_paths

def save_as_png(img, save_name):
    fig = plt.figure()
    ii = plt.imshow(img, interpolation='nearest')
    plt.set_cmap('viridis')
    fig.colorbar(ii)
    plt.savefig(save_name)

def train_net(solver_prototxt, output_dir, pretrained_model=None, max_iters=10000):
    sw = SolverWrapper(solver_prototxt, output_dir, pretrained_model=pretrained_model)
    model_paths = sw.train_model(max0_iters)
    return model_paths

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_net(SOLVER, output_dir, WEIGHTS, MAX_ITERS)




