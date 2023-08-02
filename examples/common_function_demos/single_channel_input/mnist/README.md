# Single channel input demo

## Model Source

The mnist_cnn.pt model is trained based on pytorch example https://github.com/pytorch/examples/blob/main/mnist/main.py . Notice that the inference result could be different if training on different pytorch versions. 



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

The example would print the predicted digit and corresponding scores as follows:

```
--> RKNN result
  The digit number is 1, with predicted confidence as 1.0
(Due to different RKNN versions, confidence may be close but not the same, such as 0.9999)
--> PT result
  The digit number is 1, with predicted confidence as 1.0
```



## Notice

If npy file is used for quantization, the npy data shape should be like **hwc** instead of **hw**. In test.py, it's done as follows code:

```
def prepare_data(jpg_data_path, npy_data_path, dataset_path):
    img = cv2.imread(jpg_data_path) 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.reshape(28, 28, 1) # hw --> hwc, this is important, don't miss it
    np.save(npy_data_path, gray_img)
    with open(dataset_path, 'w') as F:
        F.write(npy_data_path)
```