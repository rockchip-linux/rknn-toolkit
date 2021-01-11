import numpy as np
import cv2
from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # model config
    print('--> config model')
    rknn.config(channel_mean_value='123.675 116.28 103.53 58.395', reorder_channel='0 1 2')
    print('done')

    # Load pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model='./mnasnet0_5.pt', input_size_list=[[3, 224, 224]])
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
    print('Please modify mnasnet0_5.quantization.cfg!')
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

