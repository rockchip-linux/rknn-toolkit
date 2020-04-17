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

    export_mxnet_model()

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='123.675 116.28 103.53 57.63', reorder_channel='0 1 2')
    print('done')

    # Load mxnet model
    symbol = './resnext50_32x4d-symbol.json'
    params = './resnext50_32x4d-0000.params'
    input_size_list = [[3,224,224]]
    print('--> Loading model')
    ret = rknn.load_mxnet(symbol, params, input_size_list)
    if ret != 0:
        print('Load mxnet model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build mxnet model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./resnext50_32x4d.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./space_shuttle_224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    #ret = rknn.init_runtime(target='rk1808')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    show_top5(outputs)
    print('done')

    # # perf
    # print('--> Begin evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    # print('done')

    rknn.release()

