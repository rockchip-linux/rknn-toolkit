#!/usr/bin/env python3
import numpy as np

import re
import math
import random
import cv2
import os

from rknn.api import RKNN

INPUT_SIZE = 300

NUM_RESULTS = 1917
NUM_CLASSES = 91

Y_SCALE = 10.0
X_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0


def expit(x):
    return 1. / (1. + math.exp(-x))

def unexpit(y):
    return -1.0 * math.log((1.0 / y) - 1.0);

def CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    i = w * h
    u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i

    if u <= 0.0:
        return 0.0

    return i / u


def load_box_priors():
    box_priors_ = []
    fp = open('./box_priors.txt', 'r')
    ls = fp.readlines()
    for s in ls:
        aList = re.findall('([-+]?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', s)
        for ss in aList:
            aNum = float((ss[0]+ss[2]))
            box_priors_.append(aNum)
    fp.close()

    box_priors = np.array(box_priors_)
    box_priors = box_priors.reshape(4, NUM_RESULTS)

    return box_priors



if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # Config for Model Input PreProcess
    rknn.config(channel_mean_value='128 128 128 128', reorder_channel='0 1 2')

    if os.access('./ssd_mobilenet_v1_coco.rknn', os.F_OK) :
        # Direct Load RKNN Model
        print('./ssd_mobilenet_v1_coco.rknn exist, load it')
        rknn.load_rknn('./ssd_mobilenet_v1_coco.rknn')
    else :
        # Load TensorFlow Model
        print('--> Loading model')
        rknn.load_tensorflow(tf_pb='./ssd_mobilenet_v1_coco_2017_11_17.pb',
                         inputs=['FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/mul_1'],
                         outputs=['concat', 'concat_1'],
                         input_size_list=[[INPUT_SIZE, INPUT_SIZE, 3]])
        print('done')

        # Build Model
        print('--> Building model')
        rknn.build(do_quantization=True, dataset='./dataset.txt')
        print('done')

        # Export RKNN Model
        rknn.export_rknn('./ssd_mobilenet_v1_coco.rknn')

    # Set inputs
    orig_img = cv2.imread('./road.bmp')
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)

    # init runtime environment
    print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    ret = rknn.init_runtime(target='rk1808')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    times = 0
    first_outputs = None
    while True :
        # Inference
        times += 1
        print('--> Running model: %d' % (times))
        outputs = rknn.inference(inputs=[img])
        # print('', type(outputs[0]), type(outputs[1]))
        print('inference result: ', outputs)
        if not first_outputs:
            first_outputs = outputs
            continue
        if not ((outputs[0] == first_outputs[0]).all() and (outputs[1] == first_outputs[1]).all()):
            print('!! mismatch: ')
            print('the first_outputs --> ', first_outputs)
            print('the outputs --> ', outputs)
            break

    # Release RKNN Context
    rknn.release()
