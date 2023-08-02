import platform
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import torch

model = 'mnist_cnn.pt'
rknn_model = 'mnist_cnn.rknn'
input_size_list = [[1, 28, 28]]

jpg_data_path = './test.jpg'
npy_data_path = './test.npy'
dataset_path = 'dataset.txt'

def postprocess(input_data):
    index = input_data.argmax()
    print('  The digit number is {}, with predicted confidence as {}'.format(index, input_data[0,index]))

def prepare_data(jpg_data_path, npy_data_path, dataset_path):
    img = cv2.imread(jpg_data_path) 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.reshape(28, 28, 1) # hw --> hwc, this is important, don't miss it
    np.save(npy_data_path, gray_img)
    with open(dataset_path, 'w') as F:
        F.write(npy_data_path)

if __name__ == '__main__':
    # Prepare input data
    prepare_data(jpg_data_path, npy_data_path, dataset_path)

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
    rknn.config(mean_values=[[0.1307*255]],
                std_values=[[0.3081*255]],
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
    ret = rknn.build(do_quantization=True, dataset=dataset_path)
    if ret != 0:
        print('Build model failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        print('Export RKNN model failed!')
        rknn.release()
        exit(ret)
    print('done')

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

    # Load data
    input_data = np.load(npy_data_path)

    # Inference with RKNN
    print('--> Running model')
    outputs = rknn.inference(inputs=[input_data])
    print('done')

    print('--> RKNN result')
    postprocess(torch.tensor(outputs[0]))
    rknn.release()

    # Inference with PyTorch
    pt_model = torch.jit.load(model)
    pt_input_data = input_data.transpose(2,0,1) # hwc -> chw
    pt_input_data = pt_input_data.reshape(1, *pt_input_data.shape) # chw -> nchw
    pt_input_data = (pt_input_data/255.0 - 0.1307)/0.3081
    pt_input_data = torch.tensor(pt_input_data).float()
    pt_result = pt_model(pt_input_data)
    print('--> PT result')
    postprocess(pt_result)
