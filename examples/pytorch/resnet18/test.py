import platform
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch


def export_pytorch_model():
    net = models.resnet18(pretrained=True)
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1,3,224,224))
    trace_model.save('./resnet18.pt')


def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':
    # Export resnet18 torchscript model
    export_pytorch_model()

    model = './resnet18.pt'
    input_size_list = [[3, 224, 224]]

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
                std_values=[[58.395, 58.395, 58.395]],
                reorder_channel='0 1 2',
                target_platform=[target])
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load Pytorch model failed!')
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
    ret = rknn.export_rknn('./resnet_18.rknn')
    if ret != 0:
        print('Export resnet_18.rknn failed!')
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
    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()
