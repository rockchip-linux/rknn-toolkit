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

    export_pytorch_model()

    model = './resnet18.pt'
    input_size_list = [[3,224,224]]

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2')
    print('done')

    # Load pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build pytorch failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./resnet_18.rknn')
    if ret != 0:
        print('Export resnet_18.rknn failed!')
        exit(ret)
    print('done')

    ret = rknn.load_rknn('./resnet_18.rknn')

    # Set inputs
    img = cv2.imread('./space_shuttle_224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    # # perf
    # print('--> Begin evaluate model performance')
    # perf_results = rknn.eval_perf(inputs=[img])
    # print('done')

    rknn.release()
