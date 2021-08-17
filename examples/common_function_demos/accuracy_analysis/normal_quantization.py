import numpy as np
from rknn.api import RKNN

ONNX_MODEL = 'shufflenetv2_x1.onnx'
RKNN_MODEL = 'shufflenetv2_x1.rknn'


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'shufflnetv2_x1\n-----TOP 5-----\n'
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


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123.68, 116.28, 103.53]], std_values=[[57.38, 57.38, 57.38]], reorder_channel='0 1 2')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load shufflnetv2_x1 failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build shufflnetv2_x1 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export shufflnetv2_x1.rknn failed!')
        exit(ret)
    print('done')

    # Accuracy analysis
    print('--> Accuracy analysis')
    rknn.accuracy_analysis(inputs='./dataset.txt', output_dir='./normal_quantization_analysis')
    print('done')

    rknn.release()

