import platform
import sys
import numpy as np
import cv2
from rknn.api import RKNN

import gluoncv
import mxnet as mx


def export_mxnet_model():

    shape = [1,3,224,224]
    mxnet_model_name = 'resnext50_32x4d'

    net = gluoncv.model_zoo.get_model(mxnet_model_name, pretrained=True)

    net.hybridize()
    net(mx.nd.ones(shape=shape))
    net.export('./' + str(mxnet_model_name))
    print('export mxnet model done')


def show_top5(outputs):
    output = softmax(outputs[0][0])
    reverse_sort_index = np.argsort(output)[::-1]
    print('-----TOP 5-----')
    for i in range(5):
        print(reverse_sort_index[i], ':', output[reverse_sort_index[i]])


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':
    # Export mxnet resnext50 model
    export_mxnet_model()

    # Default target and device_id
    target = 'rv1126'
    device_id = None

    # Parameters check
    if len(sys.argv) == 1:
        print("Using default target rv1126")
    elif len(sys.argv) == 2:
        target = sys.argv[1]
        print('Set target: {}'.format(target))
    elif len(sys.argv) == 3:
        target = sys.argv[1]
        device_id = sys.argv[2]
        print('Set target: {}, device_id: {}'.format(target, device_id))
    elif len(sys.argv) > 3:
        print('Too much arguments')
        print('Usage: python {} [target] [device_id]'.format(sys.argv[0]))
        print('Such as: python {} rv1126 c3d9b8674f4b94f6'.format(
            sys.argv[0]))
        exit(-1)

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]],
                std_values=[[57.63, 57.63, 57.63]],
                reorder_channel='0 1 2',
                target_platform=[target])
    print('done')

    # Load mxnet model
    symbol = './resnext50_32x4d-symbol.json'
    params = './resnext50_32x4d-0000.params'
    input_size_list = [[3, 224, 224]]
    print('--> Loading model')
    ret = rknn.load_mxnet(symbol, params, input_size_list)
    if ret != 0:
        print('Load mxnet model failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./resnext50_32x4d.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./space_shuttle_224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Init runtime environment
    print('--> Init runtime environment')
    if target.lower() == 'rk3399pro' and platform.machine() == 'aarch64':
        print('Run demo on RK3399Pro, using default NPU.')
        target = None
        device_id = None
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print('Init runtime environment failed')
        rknn.release()
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    # Show the top5 predictions
    show_top5(outputs)
    print('done')

    rknn.release()

