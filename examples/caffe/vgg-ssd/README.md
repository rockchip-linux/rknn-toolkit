# Example for Caffe VGG-SSD


## Model Source
The model used in this example come from the following open source projects:
https://github.com/weiliu89/caffe/tree/ssd
The download link provided by this project has expired, please download the model weight from the following network disk link(fetch code is `rknn`):
https://eyun.baidu.com/s/3jJhPRzo


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

The test result should be similar to picutre `ref_detect_result.jpg`.

*Note: Different platforms, different versions of tools and drivers may have slightly different results.*


## Notes

- The prototxt in the open source model uses the old version of Caffe, which has been modified in the actual example. The modified content is as follows
```
# Comment or remove the following content:
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 300
  dim: 300
}

# Add the following content:
layer {
  name: "input"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 300
      dim: 300
    }
  }
}
```
- The DetectionOutput layer is also been removed.
