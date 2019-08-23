import cv2
import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':

    rknn = RKNN(verbose=False)

    rknn.register_op('./resize_area/ResizeArea.rknnop')

    rknn.load_tensorflow(tf_pb='./resize_area_test.pb',
                         inputs=['input'],
                         outputs=['resize_area_0'],
                         input_size_list=[[32, 32, 3]])
    rknn.build(do_quantization=False)
    # rknn.export_rknn('./resize_area.rknn')

    # rknn.load_rknn('./resize_area.rknn')

    rknn.init_runtime()

    img = cv2.imread('./dog_32x32.jpg')

    outs = rknn.inference(inputs=[img])

    out_img = outs[0].astype('uint8')
    out_img = np.reshape(out_img, (64, 64, 3))
    cv2.imwrite('./out.jpg', out_img)

