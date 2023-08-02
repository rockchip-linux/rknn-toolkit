import platform
import sys
from rknn.api import RKNN

if __name__ == '__main__':
    if len(sys.argv) not in [2, 3, 4, 5]:
        print('Usage: python {} xxx.rknn [perf_debug] [target] [device_id]'.format(sys.argv[0]))
        print('       if perf_debug set 0, only show total inference time;')
        print('       if perf_debug set 1, show the time spent on each layer.')
        print('Such as: python {} mobilenet_v1.rknn 1 rv1126 c3d9b8674f4b94f6'.format(
            sys.argv[0]))
        exit(-1)

    rknn = RKNN()

    model_path = sys.argv[1]
    perf_debug = False

    # default target and device_id
    target = 'rv1126'
    device_id = None

    if len(sys.argv) == 3:
        perf_debug = True if sys.argv[2] == '1' else False
        print('Use default target: {}'.format(target))
    elif len(sys.argv) == 4:
        perf_debug = True if sys.argv[2] == '1' else False
        target = sys.argv[3]
        print('Set target: {}'.format(target))
    elif len(sys.argv) == 5:
        perf_debug = True if sys.argv[2] == '1' else False
        target = sys.argv[3]
        device_id = sys.argv[4]
        print('Set target: {}, device id: {}'.format(target, device_id))

    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print('load rknn model failed')
        rknn.release()
        exit(ret)

    # init runtime environment
    # Note: you must set perf_debug=True if you want to analysis time spent on each layer.
    print('--> Init runtime environment')
    if target.lower() == 'rk3399pro' and platform.machine() == 'aarch64':
        print('Run demo on RK3399Pro, using default NPU.')
        target = None
        device_id = None
    ret = rknn.init_runtime(target=target, device_id=device_id, perf_debug=perf_debug)
    if ret != 0:
        print('Init runtime environment failed')
        rknn.release()
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    rknn.eval_perf(loop_cnt=100)

    rknn.release()
