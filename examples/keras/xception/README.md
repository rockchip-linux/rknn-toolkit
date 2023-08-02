# Example for Keras Xception


## Model Source
The model used in this example come from the tensorflow keras project, download link:
https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5


## Usage for the script

*Usage:*
```
python test.py [target] [device_id]
```
*Parameter Description:*
- target: target platform. Optional parameter, the default value is `rv1126`, you can fill in `rk1806`, `rk1808`, `rk3399pro`, `rv1109`, `rv1126`.
- device_id: Device ID, when multiple devices are connected, this parameter is used to distinguish them. Optional parameter, default value is None.

If the target device is `RV1109` or `RV1126`, you can directly execute the following command to run the example:
```
python test.py
```
If the target device is RK1806, RK1808 or RK3399Pro, you can execute the following command to run the example:
```
python test.py rk1808
```
If you connect multiple devices, you need to specify the device ID, please refer to the following command to run the example:
```
python test.py rv1126 c3d9b8674f4b94f6
```


## Expected results

This example will print the TOP5 labels and corresponding scores of the test image classification results. For example, the inference results of this example are as follows:
```
[1]: 0.9990234375
[0]: 0.0001596212387084961
-1: 0.0
-1: 0.0
-1: 0.0
```

1. The label index with the highest score is 1, the corresponding label is `goldfish, Carassius auratus`.
2. The download link for labels file: https://s3.amazonaws.com/onnx-model-zoo/synset.txt
3. Different platforms, different versions of tools and drivers may have slightly different results.
4. If the label index is -1, it means that the scores of other categories are all 0.


## Notes
- If the following error occurs when loading keras, please check the h5py version and downgrade it to a version below 3.0.0, such as 2.6.0
```
E Catch exception when loading keras model: ./xception.h5!
E Traceback (most recent call last):
E   File "rknn/base/RKNNlib/converter/convert_keras.py", line 24, in rknn.base.RKNNlib.converter.convert_keras.convert_keras.__init__
E   File "/home/rk/ envs/rknn-test/lib/python3.6/site-packages/tensorflow/python/keras/saving/save.py", line 146, in load_model
E     return hdf5_format.load_model_from_hdf5(filepath, custom_objects, compile)
E   File "/home/rk/ envs/rknn-test/lib/python3.6/site-packages/tensorflow/python/keras/saving/hdf5_format.py", line 210, in load_model_from_hdf5
E     model_config = json.loads(model_config.decode('utf-8'))
E AttributeError: 'str' object has no attribute 'decode'
```
Note: This error occurs because the current version of TensorFlow does not support h5py 3.0.0 and above.
