import numpy as np
import cv2
from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], reorder_channel='2 1 0')
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_caffe(model='./deploy.prototxt',
                          proto='caffe',
                          blobs='./solver_iter_45.caffemodel')
    if ret != 0:
        print('Load interp_test failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build interp_test failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./interp_test.rknn')
    if ret != 0:
        print('Export interp_test.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    old_img = cv2.imread('./cat.jpg')
    img = cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB)

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    result = outputs[0].astype(np.uint8)

    new_img = np.reshape(np.transpose(np.reshape(result, (3, 480* 640))), (480, 640,3))

    cv2.imshow("image", old_img)
    cv2.imshow("new image",new_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    rknn.release()

