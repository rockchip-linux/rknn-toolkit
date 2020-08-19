import cv2
import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':

    rknn = RKNN(verbose=False)

    rknn.register_op('./truncatediv/TruncateDiv.rknnop')
    rknn.register_op('./exp/Exp.rknnop')

    rknn.load_tensorflow(tf_pb='./custom_op_math.pb',
                         inputs=['input'],
                         outputs=['exp_0'],
                         input_size_list=[[1, 512]])
    rknn.build(do_quantization=False)
    # rknn.export_rknn('./rknn_test.rknn')

    # rknn.load_rknn('./rknn_test.rknn')

    rknn.init_runtime()

    print("init runtime done")

    in_data = np.full((1, 512), 50.0)
    in_data = in_data.astype(dtype='float32')

    output = rknn.inference(inputs=[in_data])

    print(output)

    rknn.release()
    pass

