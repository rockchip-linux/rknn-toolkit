from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # Set model config
    print('--> config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 58.395, 58.395]], reorder_channel='0 1 2')
    print('done')

    # Hybrid quantization step2
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./mnasnet0_5.json',
                                         data_input='./mnasnet0_5.data',
                                         model_quantization_cfg='./mnasnet0_5.quantization.cfg',
                                         dataset='./dataset.txt')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./mnasnet0_5.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')

    rknn.release()

