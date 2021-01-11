import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch


def export_pytorch_model():
    net = models.mnasnet0_5(pretrained=True)
    net.eval()
    trace_model = torch.jit.trace(net, torch.Tensor(1,3,224,224))
    trace_model.save('./mnasnet0_5.pt')


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

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn.load_rknn('./mnasnet0_5.rknn')
    if ret != 0:
        print('Load mnasnet0_5.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./space_shuttle_224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    #ret = rknn.init_runtime()
    ret = rknn.init_runtime(target='rk1808', device_id='1808s1')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()
