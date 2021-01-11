import numpy as np
import cv2
from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], reorder_channel='0 1 2', quantized_dtype='asymmetric_quantized-u8', batch_size=16)
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

    # Build model
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    rknn.export_rknn("ssd_mobilenet_v2.rknn")

    # Tips
    print('Please modify ssd_mobilenet_v2.quantization.cfg!')
    print('==================================================================================================')
    print('Modify method:')
    print('Add {layer_name}: {quantized_dtype} to dict of customized_quantize_layers')
    print('==================================================================================================')
    print('Notes:')
    print('1. The layer_name comes from quantize_parameters, please strip \'@\' and \':xxx\';')
    print('   If layer_name contains special characters, please quote the layer name.')
    print('2. Support quantized_type: asymmetric_affine-u8, dynamic_fixed_point-i8, dynamic_fixed_point-i16, float32.')
    print('3. Please fill in according to the grammatical rules of yaml.')
    print('4. For this model, RKNN Toolkit has provided the corresponding configuration, please directly proceed to step2.')
    print('==================================================================================================')

    rknn.release()

