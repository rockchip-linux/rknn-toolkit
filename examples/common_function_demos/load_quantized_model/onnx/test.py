import platform
import sys
import os
import numpy as np
import cv2
from rknn.api import RKNN


ONNX_MODEL = 'shufflenet-v2_quant.onnx'
IMG_PATH = './test_images/dog_224x224.jpg'
RKNN_MODEL = './shufflenet-v2_quant.rknn'


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'shufflenet\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)


if __name__ == '__main__':
    # Default target and device_id
    target = 'rv1126'
    device_id = None

    # Parameters check
    if len(sys.argv) == 1:
        print("Using default target rv1126")
    elif len(sys.argv) == 2:
        target = sys.argv[1]
        print('Set target: {}'.format(target))
    elif len(sys.argv) == 3:
        target = sys.argv[1]
        device_id = sys.argv[2]
        print('Set target: {}, device_id: {}'.format(target, device_id))
    elif len(sys.argv) > 3:
        print('Too much arguments')
        print('Usage: python {} [target] [device_id]'.format(sys.argv[0]))
        print('Such as: python {} rv1126 c3d9b8674f4b94f6'.format(
            sys.argv[0]))
        exit(-1)

    # Create RKNN object
    rknn = RKNN(verbose=False)

    if not os.path.exists(ONNX_MODEL):
        print('no model exist')
        exit()
    
    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123.675, 116.28, 103.53]],
                std_values=[[58.82, 58.82, 58.82]], 
                reorder_channel='0 1 2',
                optimization_level=3,
                quantize_input_node=True,
                merge_dequant_layer_and_output_node=True,
                target_platform=[target]
                )
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load shufflenet failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='dataset.txt')
    if ret != 0:
        print('Build shufflenet failed!')
        rknn.release()
        exit(ret)
    print('done')

    # exit()

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export shufflenet.rknn failed!')
        rknn.release()
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    if target.lower() == 'rk3399pro' and platform.machine() == 'aarch64':
        print('Run demo on RK3399Pro, using default NPU.')
        target = None
        device_id = None
    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print('Init runtime environment failed')
        rknn.release()
        exit(ret)
    print('done')

    # exit()

    # Inference
    print('--> Running rknn model')
    outputs = rknn.inference(inputs=[img])
    x = outputs[-1]

    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]
    show_outputs(outputs)

    print('--> Running onnx model')
    import onnxruntime as rt
    sess = rt.InferenceSession(ONNX_MODEL)
    
    input_name = sess.get_inputs()[0].name
    output_name= sess.get_outputs()[0].name
    output_shape = sess.get_outputs()[0].shape
    img = img.astype(np.float32)
    mean_values=[123.675, 116.28, 103.53]
    std_values=[58.82, 58.82, 58.82]
    img[:,:,0] = (img[:,:,0] - mean_values[0])/std_values[0]
    img[:,:,1] = (img[:,:,1] - mean_values[1])/std_values[1]
    img[:,:,2] = (img[:,:,2] - mean_values[2])/std_values[2]
    img = img.reshape(1,*img.shape)
    img = img.transpose(0,3,1,2)
    #forward model
    res = sess.run([output_name], {input_name: img})
    outputs = np.array(res)
    x = outputs[0]
    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]
    show_outputs(outputs)
    print('done')
