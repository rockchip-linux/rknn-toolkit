# Example for TFLite mobilenet_v1


## Model Source
The model used in this example come from the TensorFlow Lite offical model zoo:
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md


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
[156]: 0.8515625
[155]: 0.091796875
[205]: 0.0135955810546875
[284]: 0.0064697265625
[194 260]: 0.002239227294921875
```

1. The label index with the highest score is 156, the corresponding label is `Pekinese, Pekingese, Peke`.
2. The labels used in this model contains background, download link of labels file: https://github.com/leferrad/tensorflow-mobilenet/blob/master/imagenet/labels.txt
3. Different platforms, different versions of tools and drivers may have slightly different results.
