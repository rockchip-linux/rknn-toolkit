# Example for MXNet resnext50_32x4d


## Model Source
The model used in this example come from the GluonCV project:
https://cv.gluon.ai/model_zoo/classification.html


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
812 : 0.99996233
460 : 6.33277e-07
525 : 5.4181595e-07
61 : 5.0116495e-07
628 : 2.2975814e-07
```

1. The label index with the highest score is 812, the corresponding label is space shuttle.
2. The download link for labels file: https://s3.amazonaws.com/onnx-model-zoo/synset.txt
3. Different platforms, different versions of tools and drivers may have slightly different results.
