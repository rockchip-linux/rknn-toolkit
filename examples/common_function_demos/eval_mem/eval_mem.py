import platform
import sys
from rknn.api import RKNN

if __name__ == '__main__':
    if len(sys.argv) not in [2, 3, 4]:
        print('Usage: python {} xxx.rknn [target] [device_id]'.format(sys.argv[0]))
        print('Such as: python {} mobilenet_v1.rknn rv1126 c3d9b8674f4b94f6'.format(
            sys.argv[0]))
        exit(-1)

    # default target and device_id
    target = 'rv1126'
    device_id = None
    if len(sys.argv) == 2:
        print('Use default target: {}'.format(target))
    elif len(sys.argv) == 3:
        target = sys.argv[2]
        print('Set target: {}'.format(target))
    elif len(sys.argv) == 4:
        target = sys.argv[2]
        device_id = sys.argv[3]
        print('Set target: {}, device id: {}'.format(target, device_id))

    rknn = RKNN()

    model_path = sys.argv[1]

    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print('load rknn model failed')
        rknn.release()
        exit(ret)

    # init runtime environment
    # Note: you must connect an NPU target, or this function will fail.
    print('--> Init runtime environment')
    if target.lower() == 'rk3399pro' and platform.machine() == 'aarch64':
        print('Run demo on RK3399Pro, using default NPU.')
        target = None
        device_id = None
    ret = rknn.init_runtime(target=target, device_id=device_id, eval_mem=True)
    if ret != 0:
        print('Init runtime environment failed')
        rknn.release()
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    rknn.eval_memory()

    rknn.release()
