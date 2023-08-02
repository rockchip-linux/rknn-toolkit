# Example for hybrid quantization with PyTorch mnasnet0_5


## Model Source
The models used in this example come from the torchvision project:
https://github.com/pytorch/vision/tree/main/torchvision/models


## Usage for the script

*Usage:*
```
# The step1 of hybrid quantization: get model structure, weight and quantization configure files
python step1.py [target]
# Modify quantization configure file
# The step2 of hybrid quantization: requantize and generate RKNN model
pyton step2.py [target]
# Inference with hybrid quantization model
python step3.py [target] [device_id]
```
*Parameter Description:*
- target: target platform. Optional parameter, the default value is `rv1126`, you can fill in `rk1806`, `rk1808`, `rk3399pro`, `rv1109`, `rv1126`.
- device_id: Device ID, when multiple devices are connected, this parameter is used to distinguish them. Optional parameter, default value is None.

If the target device is `RV1109` or `RV1126`, you can directly execute the following command to run the example:
```
python step1.py
python step2.py
python step3.py
```
If the target device is RK1806, RK1808 or RK3399Pro, you can execute the following command to run the example:
```
python step1.py rk1808
python step2.py rk1808
python step3.py rk1808
```
If you connect multiple devices, you need to specify the device ID, please refer to the following command to run the example::
```
python step1.py
python step2.py
python step3.py rv1126 c3d9b8674f4b94f6
```


## The method to modify quantization configuration file
Add {layer_name}: {quantized_dtype} to dict of customized_quantize_layers.
If no layer changed, please set {} as empty directory for customized_quantize_layers

*Notes:*
1. The layer_name comes from quantize_parameters, please strip '@' and ':xxx';
   If layer_name contains special characters, please quote the layer name.
2. Support quantized_type: asymmetric_affine-u8, dynamic_fixed_point-i8, dynamic_fixed_point-i16, float32.
3. Please fill in according to the grammatical rules of yaml.
4. For this model, RKNN Toolkit has provided the corresponding configuration, please directly proceed to step2.


## Expected results

This step3.py will print the TOP5 labels and corresponding scores of the test image classification results. For example, the inference results of this example are as follows:
```
[812]: 0.7788181304931641
[484]: 0.029811235144734383
[404]: 0.01403999887406826
[833]: 0.011485863476991653
[403]: 0.010388719849288464
```

1. The label index with the highest score is 812, the corresponding label is `space shuttle`.
2. The download link for labels file: https://s3.amazonaws.com/onnx-model-zoo/synset.txt.
3. If you manually modify the quantization configuration file, the top5 obtained may be different.
4. Different platforms, different versions of tools and drivers may have slightly different results.
