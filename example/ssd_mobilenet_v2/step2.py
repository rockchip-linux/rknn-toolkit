import numpy as np
import cv2
from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='127.5 127.5 127.5 128', reorder_channel='0 1 2', quantized_dtype='asymmetric_quantized-u8')
    print('done')

    # Build model
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./ssd_mobilenet_v2.json', data_input='./ssd_mobilenet_v2.data', model_quantization_cfg='./ssd_mobilenet_v2.quantization.cfg', dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./ssd_mobilenet_v2.rknn')
    if ret != 0:
        print('Export model failed!')
        exit(ret)
    print('done')

    rknn.release()

