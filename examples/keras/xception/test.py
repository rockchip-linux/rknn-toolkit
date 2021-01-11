import os
import numpy as np
import cv2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from rknn.api import RKNN


KERAS_MODEL_PATH = './xception.h5'
IMG_PATH = 'goldfish_299x299.jpg'

def export_keras_model():
    if not os.path.exists(KERAS_MODEL_PATH):
        model = Xception(weights='imagenet')

        img = image.load_img(IMG_PATH, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        #preds.tofile('out_0.txt', '\n')
        print('Keras Predicted:', decode_predictions(preds, top=5)[0])

        model.save(KERAS_MODEL_PATH)

def show_outputs(outputs):
    output = outputs[0].reshape(-1)
    output_sorted = sorted(output, reverse=True)
    top5_str = 'mobilenet_v2\n-----TOP 5-----\n'
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


if __name__ == '__main__':

    export_keras_model()

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]], reorder_channel='0 1 2')
    print('done')

    # Load keras model
    print('--> Loading model')
    ret = rknn.load_keras(model=KERAS_MODEL_PATH)
    if ret != 0:
        print('Load keras model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build pytorch failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./xception.rknn')
    if ret != 0:
        print('Export xception.rknn failed!')
        exit(ret)
    print('done')

    # ret = rknn.load_rknn('./xception.rknn')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    #ret = rknn.init_runtime(target='rk1808')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    #outputs[0].tofile('out.txt', '\n')

    show_outputs(outputs)
    print('done')

    rknn.release()

