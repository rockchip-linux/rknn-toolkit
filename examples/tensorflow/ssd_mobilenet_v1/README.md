# Example for TensorFlow ssd_mobilenet_v1_coco


## Model Source
The model comes from TensorFlow's official model zoo:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md


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

The test result should be similar to picutre `ref_detect_results.jpg`.

- Different platforms, different versions of tools and drivers may have slightly different results.
- The color of the detected object box is random.
