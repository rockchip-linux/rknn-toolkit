import platform
import sys
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
    # Export mxnet fcn_resnet101 model
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
    symbol = './fcn_resnet101_voc-symbol.json'
    params = './fcn_resnet101_voc-0000.params'
    input_size_list = [[3, 480, 480]]
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
    ret = rknn.export_rknn('./fcn_resnet101_voc.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./test_image.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
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
    print('done')

    # Post process: get segmap and write to orignal image
    segmap = np.squeeze(np.argmax(outputs[0], axis=1))
    seg_img = decode_segmap(segmap)

    overlapping = cv2.addWeighted(img, 1, seg_img, 0.9, 0)
    overlapping = cv2.cvtColor(overlapping, cv2.COLOR_RGB2BGR)

    # Save segmentation result
    print('Save segmentation result to seg_image.jpg')
    cv2.imwrite('seg_image.jpg', overlapping)

    rknn.release()
