import sys
from rknn.api import RKNN

if __name__ == '__main__':
    # Default target platform
    target = 'rv1126'

    # Parameters check
    if len(sys.argv) == 1:
        print("Using default target rv1126")
    elif len(sys.argv) == 2:
        target = sys.argv[1]
        print('Set target: {}'.format(target))
    elif len(sys.argv) > 2:
        print('Too much arguments')
        print('Usage: python {} [target]'.format(sys.argv[0]))
        print('Such as: python {} rv1126'.format(
            sys.argv[0]))
        exit(-1)

    # Create RKNN object
    rknn = RKNN()
    
    # Set model config
    print('--> config model')
    rknn.config(mean_values=[[123.68, 116.28, 103.53]],
                std_values=[[57.38, 57.38, 57.38]],
                reorder_channel='0 1 2',
                target_platform=[target])
    print('done')

    # Hybrid quantization step2
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./torchjitexport.json',
                                         data_input='./torchjitexport.data',
                                         model_quantization_cfg='./torchjitexport.quantization.cfg',
                                         dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./shufflenet_hybrid_quant.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    rknn.release()

