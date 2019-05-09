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

    # Load tflite model
    print('--> Loading model')
    ret = rknn.load_tensorflow(tf_pb='./ssd_mobilenet_v2.pb',
                               inputs=['FeatureExtractor/MobilenetV2/MobilenetV2/input'],
                               outputs=['concat_1', 'concat'],
                               input_size_list=[[300,300,3]],
                               predef_file=None)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    # Tips
    print('Please modify ssd_mobilenet_v2.quantization.cfg!')
    print('==================================================================================================')
    print('Modify method:')
    print('1. delete @FeatureExtractor/MobilenetV2/expanded_conv/depthwise/depthwise_227:weight and its value')
    print('2. delete @FeatureExtractor/MobilenetV2/expanded_conv/depthwise/depthwise_227:bias and its value')
    print('3. delete @FeatureExtractor/MobilenetV2/Conv/Relu6_228:out0 and its value')
    print('==================================================================================================')
    print('Original quantization profile and modified quantization profile diff shows in file quantization_profile.diff')

    rknn.release()

