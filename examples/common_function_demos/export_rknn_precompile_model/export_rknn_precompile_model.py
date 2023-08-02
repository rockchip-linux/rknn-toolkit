import platform
import sys
from rknn.api import RKNN

if __name__ == '__main__':

    if len(sys.argv) not in [3, 4, 5]:
        print('Usage: python {} xxx.rknn xxx.hw.rknn [target] [device_id]'.format(sys.argv[0]))
        print('Such as: python {} mobilenet_v1.rknn mobilenet_v1.hw.rknn rv1126 c3d9b8674f4b94f6'.format(
            sys.argv[0]))
        exit(1)

    orig_rknn = sys.argv[1]
    hw_rknn = sys.argv[2]

    # default target and device_id
    target = 'rv1126'
    device_id = None

    if len(sys.argv) == 3:
        print('Use default target: {}'.format(target))
    elif len(sys.argv) == 4:
        target = sys.argv[3]
        print('Set target: {}'.format(target))
    elif len(sys.argv) == 5:
        target = sys.argv[3]
        device_id = sys.argv[4]
        print('Set target: {}, device id: {}'.format(target, device_id))

    # Create RKNN object
    rknn = RKNN()
    
    # Load rknn model
    print('--> Loading RKNN model')
    ret = rknn.load_rknn(orig_rknn)
    if ret != 0:
        print('Load RKNN model failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')

    # Note: you must set rknn2precompile=True when call rknn.init_runtime()
    #       RK3399Pro with android system does not support this function.
    ret = rknn.init_runtime(target=target, device_id=device_id, rknn2precompile=True)
    if ret != 0:
        print('Init runtime environment failed')
        rknn.release()
        exit(ret)
    print('done')

    ret = rknn.export_rknn_precompile_model(hw_rknn)

    rknn.release()

