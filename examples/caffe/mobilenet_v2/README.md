# Example for Caffe mobilenet_v2


## Model Source
The model used in this example come from the following open source projects:
https://github.com/shicai/MobileNet-Caffe


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
[1]: 0.73388671875
[115 996]: 0.033447265625
[115 996]: 0.033447265625
[927]: 0.026214599609375
[794]: 0.024169921875
```

1. The label index with the highest score is 1, the corresponding label is `goldfish`.
2. The download link for labels file: https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt.
3. Different platforms, different versions of tools and drivers may have slightly different results.


## Notes

- The prototxt in the open source model uses the old version of Caffe, which has been modified in the actual example. The modified content is as follows
```
# Comment or remove the following content:
input: "data"
input_dim: 1
input_dim: 3
input_dim: 224
input_dim: 224

# Add the following content
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 224
      dim: 224
    }
  }
}
```

