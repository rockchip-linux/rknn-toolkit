import numpy as np
import cv2
from rknn.api import RKNN

import gluoncv
import mxnet as mx


def export_mxnet_model():

    shape = [1,3,480,480]
    mxnet_model_name = 'fcn_resnet101_voc'

    net = gluoncv.model_zoo.get_model(mxnet_model_name, pretrained=True)

    net.hybridize()
    net(mx.nd.ones(shape=shape))
    net.export('./' + str(mxnet_model_name))
    print('export mxnet model done')


# Define the helper function
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb



if __name__ == '__main__':

    export_mxnet_model()

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[57.63, 57.63, 57.63]], reorder_channel='0 1 2')
    print('done')

    # Load mxnet model
    symbol = './fcn_resnet101_voc-symbol.json'
    params = './fcn_resnet101_voc-0000.params'
    input_size_list = [[3,480,480]]
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
    ret = rknn.export_rknn('./fcn_resnet101_voc.rknn')
    if ret != 0:
        print('Export mxnet model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./test_image.jpeg')
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
    print('done')

    segmap = np.squeeze(np.argmax(outputs[0], axis=1))
    seg_img = decode_segmap(segmap)

    overlapping = cv2.addWeighted(img, 1, seg_img, 0.9, 0)
    overlapping = cv2.cvtColor(overlapping, cv2.COLOR_RGB2BGR)
    cv2.imwrite('seg_image.jpg', overlapping)

    print('please open seg_image.jpg to see segmentation result.')

    # # perf
    # print('--> Begin evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    # print('done')

    rknn.release()

