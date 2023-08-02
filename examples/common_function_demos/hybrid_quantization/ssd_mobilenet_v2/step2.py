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
    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]],
                std_values=[[127.5, 127.5, 127.5]],
                reorder_channel='0 1 2',
                batch_size=16,
                target_platform=[target])
    print('done')

    # Hybrid quantization step2
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./ssd_mobilenet_v2.json',
                                         data_input='./ssd_mobilenet_v2.data',
                                         model_quantization_cfg='./ssd_mobilenet_v2.quantization.cfg',
                                         dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./ssd_mobilenet_v2.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        rknn.release()
        exit(ret)
    print('done')

    rknn.release()

