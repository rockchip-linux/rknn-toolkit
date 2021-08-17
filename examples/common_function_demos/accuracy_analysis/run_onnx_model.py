import cv2
import numpy as np
import onnxruntime as rt

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def show_outputs(outputs):
    output = outputs
    output_sorted = sorted(output, reverse=True)
    top5_str = 'shufflnetv2_x1\n-----TOP 5-----\n'
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

def run_onnx_part(inputs, model_path):
    sess = rt.InferenceSession(model_path)
    
    img = inputs[0]
    img = img.transpose((2,0,1))
    img = img.reshape((1,*img.shape))

    input_name_0 = sess.get_inputs()[0].name
    output_name= sess.get_outputs()[0].name
    
    #forward model
    res = sess.run([output_name], {input_name_0: img})
    output = np.array(res[0])
    return output

if __name__ == '__main__':
    model_path = './shufflenetv2_x1.onnx'
    img = cv2.imread('./dog_224x224.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img[:,:,0] -= 123.68
    img[:,:,1] -= 116.28
    img[:,:,2] -= 103.53
    img /= 57.38

    result = run_onnx_part([img], model_path)
    show_outputs(softmax(result[0]))
    print('done')
