# Example for PyTorch resnet18


## Model Source
The models used in this example come from the torchvision project:
https://github.com/pytorch/vision/tree/main/torchvision/models


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
[812]: 0.9993903636932373
[404]: 0.0004593881603796035
[657 833]: 2.9284470656421036e-05
[657 833]: 2.9284470656421036e-05
[895]: 1.8508895664126612e-05
```

1. The label index with the highest score is 812, the corresponding label is `space shuttle`.
2. The download link for labels file: https://s3.amazonaws.com/onnx-model-zoo/synset.txt
3. Different platforms, different versions of tools and drivers may have slightly different results.
