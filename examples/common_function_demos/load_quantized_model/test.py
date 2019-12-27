import numpy as np
import cv2
import os
import urllib
import tarfile
import shutil
import traceback
import time
import sys
from rknn.api import RKNN

PB_FILE = './inception_v3_quant_frozen.pb'
RKNN_MODEL_PATH = './inception_v3_quant_frozen.rknn'
INPUTS = ['input']
OUTPUTS = ['InceptionV3/Logits/SpatialSqueeze']
IMG_PATH = './goldfish_299x299.jpg'
INPUT_SIZE = 299

def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'inception_v3\n-----TOP 5-----\n'
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


def show_progress(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: {}".format(readable_speed(speed))
    recv_size = blocknum * blocksize

    f = sys.stdout
    progress = (recv_size / totalsize)
    progress_str = "{:.2f}%".format(progress * 100)
    n = round(progress * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(progress_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r\n')

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # If inception_v3_quant_frozen.pb does not exist, download it.
    # Download address:
    # https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz
    if not os.path.exists(PB_FILE):
        print('--> Download {}'.format(PB_FILE))
        url = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz'
        download_file = 'inception_v3_quant.tgz'
        try:
            start_time = time.time()
            urllib.request.urlretrieve(url, download_file, show_progress)
        except:
            print('Download {} failed.'.format(download_file))
            print(traceback.format_exc())
            exit(-1)
        try:
            tar = tarfile.open(download_file)
            target_dir = os.path.splitext(download_file)[0]
            if os.path.isdir(target_dir):
                pass
            else:
                os.mkdir(target_dir)
            tar.extractall(target_dir)
            tar.close()
        except:
            print('Extract {} failed.'.format(download_file))
            exit(-1)
        pb_file = os.path.join(target_dir, PB_FILE)
        if os.path.exists(pb_file):
            shutil.copyfile(pb_file, './inception_v3_quant_frozen.pb')
            shutil.rmtree(target_dir)
            os.remove(download_file)
        print('done')
    # pre-process config
    print('--> Config model')
    rknn.config(reorder_channel='0 1 2')
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_tensorflow(tf_pb=PB_FILE,
                               inputs=INPUTS,
                               outputs=OUTPUTS,
                               input_size_list=[[INPUT_SIZE, INPUT_SIZE, 3]],
                               predef_file=None,
                               mean_values='127.5',
                               std_values='128')
    if ret != 0:
        print('Load inception_v3_quant_frozen failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build inception_v3_quant_frozen.rknn failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Export inception_v3_quant_frozen.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    # ret = rknn.init_runtime(target='rk1808')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    x = outputs[0]
    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]
    show_outputs(outputs)
    print('done')

    # perf
    print('--> Begin evaluate model performance')
    perf_results = rknn.eval_perf(inputs=[img])
    print('done')

    rknn.release()

