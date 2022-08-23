from PIL import Image
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
import cv2
import sys
import argparse
import os
import time
import urllib
import traceback

from rknn.api import RKNN

# Needed to show segmentation colormap labels
import get_dataset_colormap

INPUT_SIZE = 513
TEST_IMAGE = './bike_boy.jpg'
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)

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

def run(image, inference_result):
    """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    b = inference_result
    b.shape = 65 * 65, 21    # ResizeBilinear_2
    b = np.transpose(b)
    seg_img = np.argmax(b, axis=-2)
    seg_img = np.reshape(seg_img, (65, 65))    # ResizeBilinear_2

    return resized_image, seg_img

def vis_segmentation(image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0)

    plt.show()

def deeplabv3_post_process(img, inference_result):
    origin_im = Image.open(img)
    print('running deeplab on image %s...' % img)
    resized_im, seg_map = run(origin_im, inference_result)

    vis_segmentation(resized_im, seg_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-through",
                        dest='pass_through',
                        action='store_true',
                        help='Whether pass through input data to RKNN model.')
    parser.add_argument('--load-rknn',
                        dest='load_rknn',
                        action='store_true',
                        help='Whether load RKNN model directly.')
    args = parser.parse_args()

    pass_through = args.pass_through

    LOAD_RKNN = args.load_rknn
    PB_MODEL = './deeplab-v3-plus-mobilenet-v2.pb'

    # Create RKNN object
    rknn = RKNN()

    if not os.path.exists(PB_MODEL):
        print('--> Download {}'.format(PB_MODEL))
        url = 'https://cnbj1.fds.api.xiaomi.com/mace/miai-models/deeplab-v3-plus/deeplab-v3-plus-mobilenet-v2.pb'
        download_file = PB_MODEL
        try:
            start_time = time.time()
            urllib.request.urlretrieve(url, download_file, show_progress)
        except:
            print('Download {} failed.'.format(download_file))
            print(traceback.format_exc())
        print('done')

    if not LOAD_RKNN:
        # Load tensorflow model
        print('--> Loading model')
        ret = rknn.load_tensorflow(tf_pb=PB_MODEL,
                                   inputs=['sub_7'],
                                   outputs=['ResizeBilinear_2'],
                                   input_size_list=[[513, 513, 3]])
        if ret != 0:
            print('load_tensorflow failed')
            exit(ret)
        print('done')

        # set config refer to pass_througn value.
        rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], reorder_channel='0 1 2')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset='./dataset.txt', pre_compile=False)
        if ret != 0:
            print('build rknn model failed')
            exit(ret)
        print('done')

        # Export rknn model
        ret = rknn.export_rknn('./deeplab-v3-plus-mobilenet-v2.rknn')
        if ret != 0:
            print('export rknn model failed')
            exit(ret)
        print('done')
    else:
        print('--> Load model')
        ret = rknn.load_rknn(path='./deeplab-v3-plus-mobilenet-v2.rknn')
        if ret < 0:
            print('load model failed.')
        print('done')

    # init runtime 
    print('--> init runtime')
    ret = rknn.init_runtime(target='rk1808', device_id='1808')
    if ret < 0:
        print('init runtime failed')
        exit(ret)
    print('done')

    img = cv2.imread(TEST_IMAGE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # preprocess if pass_through is True
    if pass_through == True:
        print('Pass through input data to model directly.')
        img = (img.astype(np.float32)-127.5)/127.5
        # quantize input
        scale = 0.007843137718737125
        zp = 127
        img = ((img / scale) + zp).astype(np.uint8)
        inputs_pass_through=[1]
    else:
        print('Pass normal data to runtime, runtime will do preprocess and quantize.')
        inputs_pass_through=[0]

    # inference
    print('--> inference')
    outputs = rknn.inference(inputs=[img], inputs_pass_through=inputs_pass_through)
    for idx, out in enumerate(outputs):
        if pass_through:
            np.save('out{}_pass.npy'.format(idx), out)
        else:
            np.save('out{}_norm.npy'.format(idx), out)
    print('done')

    rknn.release()

    deeplabv3_post_process(img=TEST_IMAGE, inference_result=outputs[0])

    exit(0)
