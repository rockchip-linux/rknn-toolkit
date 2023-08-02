# Example for hybrid quantization with TensorFlow ssd_mobilenet_v2_coco_2018_03_29


## Model Source
The model comes from TensorFlow's official model zoo:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md


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

The test result should be similar to picutre `ref_detect_result.jpg`.

- Different platforms, different versions of tools and drivers may have slightly different results.
