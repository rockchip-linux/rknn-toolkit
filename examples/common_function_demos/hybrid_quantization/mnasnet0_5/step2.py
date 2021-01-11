import numpy as np
import cv2
from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2')
    print('done')

    # Build model
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./mnasnet0_5.json', data_input='./mnasnet0_5.data', model_quantization_cfg='./mnasnet0_5.quantization.cfg', dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./mnasnet0_5.rknn')
    if ret != 0:
        print('Export model failed!')
        exit(ret)
    print('done')

    rknn.release()

