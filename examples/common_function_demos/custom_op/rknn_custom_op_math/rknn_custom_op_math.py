import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':

    rknn = RKNN(verbose=False)

    # RKNN model config.
    print("--> Model config")
    rknn.config(target_platform=['rv1126'])
    print("done")

    # Register customer op.
    rknn.register_op('./truncatediv/TruncateDiv.rknnop')
    rknn.register_op('./exp/Exp.rknnop')

    # Load TensorFlow model with customer op.
    print("--> Load tensorflow model.")
    ret = rknn.load_tensorflow(tf_pb='./custom_op_math.pb',
                               inputs=['input'],
                               outputs=['exp_0'],
                               input_size_list=[[1, 512]])
    if ret != 0:
        print("Load TF model failed.")
        rknn.release()
        exit(ret)
    print("done")

    # Build RKNN model.
    print("--> Build RKNN model")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Build RKNN model failed.")
        rknn.release()
        exit(ret)
    print('done')

    # rknn.export_rknn('./rknn_test.rknn')

    # rknn.load_rknn('./rknn_test.rknn')

    # Inference with RKNN
    print("--> Init RKNN runtime.")
    ret = rknn.init_runtime(target="rv1126")
    if ret != 0:
        print("Init runtime failed.")
        rknn.release()
        exit(ret)
    print("done")

    in_data = np.full((1, 512), 50.0)
    in_data = in_data.astype(dtype='float32')

    output = rknn.inference(inputs=[in_data])

    # Show output
    print(output)

    rknn.release()
