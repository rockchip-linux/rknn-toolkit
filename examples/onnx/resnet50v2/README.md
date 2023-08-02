# Example for ONNX resnext50_32x4d


## Model Source
The models used in this example come from the ONNX model zoo project:
https://github.com/onnx/models/tree/main/vision/classification/resnet


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
[155]: 0.6183234453201294
[154]: 0.3416009843349457
[262]: 0.021962042897939682
[152]: 0.003988372161984444
[204]: 0.002964471932500601
```

1. The label index with the highest score is 155, the corresponding label is `Pekinese, Pekingese, Peke`.
2. The download link for labels file: https://s3.amazonaws.com/onnx-model-zoo/synset.txt
3. Different platforms, different versions of tools and drivers may have slightly different results.
