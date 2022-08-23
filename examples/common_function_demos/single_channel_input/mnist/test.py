import numpy as np
import cv2
from rknn.api import RKNN
import torch

model = 'fc_net.pt'
rknn_model = 'fc_net.rknn'
input_size_list = [[1, 28, 28]]

jpg_data_path = './test.jpg'
npy_data_path = './test.npy'
dataset_path = 'dataset.txt'

def postprocess(input_data):
    index = input_data.argmax()
    print('  The digit number is {}, with predict confident as {}'.format(index, input_data[0,index]))

def prepace_data(jpg_data_path, npy_data_path, dataset_path):
    img = cv2.imread(jpg_data_path) 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.reshape(28, 28, 1) # hw --> hwc, this is important, don't miss it
    np.save(npy_data_path, gray_img)
    with open(dataset_path, 'w') as F:
        F.write(npy_data_path)

if __name__ == '__main__':
    prepace_data(jpg_data_path, npy_data_path, dataset_path)
    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    # rknn.config(target_platform='rv1109')
    rknn.config(mean_values=[[0]],
                std_values=[[255]])
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_model)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Load data
    input_data = np.load(npy_data_path)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[input_data])
    print('done')

    print('--> RKNN result')
    postprocess(torch.tensor(outputs[0]))
    rknn.release()

    pt_model = torch.jit.load(model)
    pt_input_data = input_data.transpose(2,0,1) # hwc -> chw
    pt_input_data = pt_input_data.reshape(1, *pt_input_data.shape) # chw -> nchw
    pt_input_data = pt_input_data/255.0
    pt_input_data = torch.tensor(pt_input_data).float()
    pt_result = pt_model(pt_input_data)
    print('--> PT result')
    postprocess(pt_result)
