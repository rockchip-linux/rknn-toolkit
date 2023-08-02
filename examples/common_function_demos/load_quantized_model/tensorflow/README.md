# Example for quantized TensorFlow inception_v3_quant


## Model Source
The model used in this example come from the TensorFlow model zoo, the download link:
https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz


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
If you connect multiple devices, you need to specify the device ID, please refer to the following command to run the example::
```
python test.py rv1126 c3d9b8674f4b94f6
```


## Expected results

This example will print the TOP5 labels and corresponding scores of the test image classification results. For example, the inference results of this example are as follows:
```
[2]: 0.9897317290306091
[408]: 0.0009118635789491236
[795]: 0.0006728958687745035
[974]: 0.00015886672190390527
[352]: 0.00014724396169185638
```

1. The label index with the highest score is 2, the corresponding label is `goldfish, Carassius auratus`.
2. The labels used in this model contains background, download link of labels file: https://github.com/leferrad/tensorflow-mobilenet/blob/master/imagenet/labels.txt
4. Different platforms, different versions of tools and drivers may have slightly different results.
