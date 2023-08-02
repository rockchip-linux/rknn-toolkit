# Example for loading quantized Pytorch model - Mobilenet_v2_quant

## Require

Pytorch == 1.9.x or 1.10.x



## Model Source

The model comes from https://pytorch.org/vision/stable/models/generated/torchvision.models.quantization.mobilenet_v2.html#torchvision.models.quantization.MobileNet_V2_QuantizedWeights. 



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

This example will print the TOP5 labels and corresponding scores of the test image classification results. For example, the inference results of this example are as follows:

```
rk_result:
-----TOP 5-----
[155]: 0.6007367372512817
[154]: 0.17650733888149261
[204]: 0.08209411054849625
[283]: 0.028111280873417854
[187]: 0.009626070968806744

pt_result:
-----TOP 5-----
[155]: 0.5076675415039062
[154]: 0.32070720195770264
[204]: 0.03760503977537155
[283]: 0.027686361223459244
[284]: 0.015007374808192253
```

1. The label index with the highest score is 155, the corresponding label is `Pekinese, Pekingese, Peke`.
2. It's obvious that RKNN and Pytorch inference results are not close. We had test this RKNN model and got 70.0 acc@1(on ImagedNet-1K), which is close to 71.6 refer to torchvision document. 
3. Different platforms, different versions of tools and drivers may have slightly different results.


## NOTICE

1. The optimization strategy can affect the inference results of the RKNN model, and this effect will be more obvious when loading the QAT model. If the inference results of the RKNN model and the QAT model are inconsistent, you can try to set the configuration parameter **optimization_level** of **rknn.config** to 2. But lower optimization_level may slower inference speed.

2. Attention! Due to the hardware limited, RKNN-Toolkit1 currently only supports **per_tensor** quantization, corresponding to PyTorch's **qnnpack** quantization mode. If you choose **per_channel** quantization, that is, **PyTorch's fbgemm quantization mode, it will not be able to convert to RKNN model in the end.**

3. Compared with loading the floating-point model, rknn.config additionally sets the **quantize_input_node** and **merge_dequant_layer_and_output_node** parameters to **True**. After the former is enabled, the input node and quantize_layer will be merged, and the latter will be merged with the output node and dequantize_layer. This will make the performance and characteristics of the model when deployed consistent with the normal RKNN quantization model. For details, please refer to chapter 7.2 in the user guide.

4. **While using QAT model to convert RKNN model, it's recommended to test on dataset instead of single data.**

5. This demo may error on **macos**. The reason for the error is that the native function of pytorch is abnormal and the parsing model is wrong. Abnormal reproduction code:

   ```
   list(torch.jit.load('QAT_model_path.pt').graph.nodes())
   ```

   

