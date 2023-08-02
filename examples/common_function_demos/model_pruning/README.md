# Example for model pruning

1. Currently RKNN-Toolkit1 supports **structured pruning**. Assuming that there is a 4D tensor in the model calculation process, and its shape information is NCHW, if there are some values on the C (channel) that are all zero, this part can be eliminated to avoid invalid operations. In most cases, convolutions and their context-dependent network layers are relatively easy to trigger this pruning effect. If you want to achieve the effect of structured pruning on the real model, you need to make some adjustments to the training code. For more pruning-related introductions, you can refer to Pytorch tutorial https://pytorch.org/tutorials/intermediate/pruning_tutorial.html .
2. This demo builds a simple network layer and uses **torch.nn.utils.prune.ln_structured** for structured pruning. When converting the model, set **model_pruning=True** in **rknn.config** to enable it. 
3. Please note that the pruning function is currently in the experimental stage. If you encounter any problems, you can contact the Rockchips-AI-Teams for feedback through https://github.com/rockchip-linux/rknn-toolkit. Thank you for your support.



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

The pruning log will show as follow(order may changed)
```
I Pruning: const change detail
const name                              src_size    pruned_size diff        
@convolution_at_input.3_1_1:bias        256         192         -64         
@convolution_at_input.7_4_4:weight      16384       9024        -7360       
@convolution_at_input.2_7_7:weight      65536       48128       -17408      
@convolution_at_input.7_4_4:bias        256         188         -68         
@convolution_at_input.3_1_1:weight      6912        5184        -1728
```

And the cosine similarity between RKNN and Pytorch model inference result will show like:

```
cos sim: 0.9999999
```

