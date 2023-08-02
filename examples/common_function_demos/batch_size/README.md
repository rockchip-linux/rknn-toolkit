# Example for multi-batch



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
If you connect multiple devices, you need to specify the device ID, please refer to the following command to run the example::
```
python test.py rv1126 c3d9b8674f4b94f6
```


## Expected results

This example will print the TOP5 labels and corresponding scores of the test images classification results. For example, the inference results of this example are as follows:
```
mobilenet_v1 input[0]
-----TOP 5-----
[156]: 0.85400390625
[155]: 0.092041015625
[205]: 0.01226806640625
[284]: 0.006488800048828125
[260]: 0.002246856689453125

mobilenet_v1 input[1]
-----TOP 5-----
[156]: 0.85400390625
[155]: 0.092041015625
[205]: 0.01226806640625
[284]: 0.006488800048828125
[260]: 0.002246856689453125

mobilenet_v1 input[2]
-----TOP 5-----
[156]: 0.85400390625
[155]: 0.092041015625
[205]: 0.01226806640625
[284]: 0.006488800048828125
[260]: 0.002246856689453125

mobilenet_v1 input[3]
-----TOP 5-----
[156]: 0.85400390625
[155]: 0.092041015625
[205]: 0.01226806640625
[284]: 0.006488800048828125
[260]: 0.002246856689453125
```

1. The label index with the highest score is 156, the corresponding label is `Pekinese, Pekingese, Peke`.
2. The label used in this model contains background, so its index will be 1 more than other examples.
3. The top-5 perdictions of 4 inputs should be exactly the same.
4. Different platforms, different versions of tools and drivers may have slightly different results.


## Notes

1. the inputs data need to be merged together using np.concatenate.
```
e.g:
    img = np.concatenate((img, img, img, img), axis=0)
```
