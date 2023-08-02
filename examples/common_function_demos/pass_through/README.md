# Example of passing through input data


## Model Source
The model used in this example come from the miai-models project, download link:
https://cnbj1.fds.api.xiaomi.com/mace/miai-models/deeplab-v3-plus/deeplab-v3-plus-mobilenet-v2.pb


## Usage for the script

*Usage:*
```
python test.py [pass-through] [use-rknn] [target] [device-id]
```
*Parameter Description:*
- pass-through: whether to pass through input data. Optional parameter, the default value is `True`. If set to `1`, pass through input data, other value, pass-through will set to `False`.
- use-rknn: whether to use RKNN model directly. Optional paramter, the default value is `False`. If set to `1`, the script will load RKNN model directly, If the RKNN model does not exist, the script will throw an error and exit.
- target: target platform. Optional parameter, the default value is `rv1126`, you can fill in `rk1806`, `rk1808`, `rk3399pro`, `rv1109`, `rv1126`.
- device_id: Device ID, when multiple devices are connected, this parameter is used to distinguish them. Optional parameter, default value is None.

If the target device is `RV1109` or `RV1126`, you can directly execute the following command to run the example:
```
python test.py
```
If the target device is RK1806, RK1808 or RK3399Pro, you can execute the following command to run the example:
```
python test.py 1 0 rk1808
```
If you connect multiple devices, you need to specify the device ID, please refer to the following command to run the example:
```
python test.py 1 0 rv1126 c3d9b8674f4b94f6
```


## Expected results

The test result should be similar to picutre `ref_seg_results.jpg`.

*Note: Different platforms, different versions of tools and drivers may have slightly different results.*
