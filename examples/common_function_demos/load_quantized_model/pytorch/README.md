注意：

1. 使用该demo需要将依赖的PyTorch升级到1.9.0版本。

2. Demo的结果看起来RKNN与PyTorch的计算结果有不小的偏差。经测试，在imagenet的子集1000张图上RKNN模型的得分更高，具体数值如下：

| 模型                                               | Top1精度（imagenet-1000） |
| -------------------------------------------------- | ------------------------- |
| RKNN_mobilenet_v2_i8(from_quantized_pytorch_model) | 70.0%                     |
| PyTorch_mobilenet_v2_i8                            | 61.5%                     |

​	造成这个的原因可能是PyTorch本身的量化推理实现有误差，目前他们也在持续完善模型量化功能。	

3. 由于底层驱动原因，RKNN-Toolkit1 目前只支持per_tensor量化，对应 PyTorch的 qnnpack 量化模式。如果选择 per_channel量化，也即 PyTorch的 fbgemm 量化模式，最终将无法转为RKNN模型，请用户多加留意！

4. 相比载入浮点模型，rknn.config里面额外将 quantize_input_node 和 merge_dequant_layer_and_output_node 参数均设为 True，前者开启后会合并输入节点与quantize_layer，后者开启后会合并输出节点与dequantize_layer，这样会使得模型在部署的时候性能及特性可以与普通RKNN量化模型保持一致。具体请参考用户手册里面的7.2章节。

5. 该demo在 macos 上存在异常，异常原因为pytorch依赖库异常，解析模型出错。异常复现代码：

   ```
   list(torch.jit.load('QAT_model_path.pt').graph.nodes())
   ```

   

