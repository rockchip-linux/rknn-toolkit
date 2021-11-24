## RKNN-Toolkit accuracy_analysis 例程

#### 一.使用说明

​		本例程以 shufflenetv2.onnx 模型作为示范，展示 RKNN-Toolkit 精度分析 accuracy_analysis 接口的使用流程，在常规量化模式效果有限的情况下，使用混合量化 hybrid_quantization 接口，提高 RKNN 模型的最终推理精度。实际使用时，请根据具体需求，在推理精度与推理速度的之间作一定的取舍。



#### 二.使用步骤

1. 生成常规量化模型，并生成精度分析报告。

   `python normal_quantizition.py`

   - 执行后在该工程目录下可得到 normal_quantization_analysis 的文件夹，各文件/目录的含义参考 《Rockchip_User_Guide_RKNN_Toolkit_CN.docx》的 4.3章节说明。

   - 这里我们生成文件夹中 individual_qnt_error_analysis.txt 文件，可以看到Conv_418_152、Conv_434_142、Conv_530_69 网络层的 cosine_norm 值分别为0.972651、0.989210、0.963591，我们就判断这几个网络层对量化不友好，需要我们使用混合量化去处理这些网络层。而其他网络层的 cosine_norm 值都在0.99以上，通常可以认为是量化友好的。我们先记录下这些网络层的名称，供后续步骤使用。

   

2. 执行混合量化步骤一，生成配置文件，具体资料可参考《Rockchip_User_Guide_RKNN_Toolkit_CN.docx》的 4.5 章节 。

   `python hybrid_quantization_step1.py`

   - 打开生成的 torchjitexport.quantization.cfg 配置文件，参考 《Rockchip_User_Guide_RKNN_Toolkit_CN.docx》4.5 节 的说明，我们就可以在 customized_quantize_layers 信息里面添加我们想要使用混合量化处理的网络层（从步骤1中得到）。

   - 配置文件中的 customized_quantize_layers 会给初始混合量化层的建议，这个建议的效果不一定是最优的，实际使用过程中请根据具体模型灵活设置。示例模型生成的 quantization.cfg 混合量化建议为：

     ```
     customized_quantize_layers:
         Reshape_614_2: dynamic_fixed_point-i16
         Gemm_615_1: dynamic_fixed_point-i16
         AveragePool_612_3: dynamic_fixed_point-i16
         Reshape_614_2_acuity_mark_perm_213: dynamic_fixed_point-i16
         Conv_434_142: dynamic_fixed_point-i16
         Conv_436_133: dynamic_fixed_point-i16
         Slice_363_204: dynamic_fixed_point-i16
         Conv_364_201: dynamic_fixed_point-i16
         Conv_530_69: dynamic_fixed_point-i16
         Conv_466_118: dynamic_fixed_point-i16
         Conv_418_152: dynamic_fixed_point-i16
         Conv_383_181: dynamic_fixed_point-i16
         Conv_367_193: dynamic_fixed_point-i16
         Conv_345_195: dynamic_fixed_point-i16
         Conv_353_196: dynamic_fixed_point-i16
         Conv_343_202: dynamic_fixed_point-i16
         Conv_348_209: dynamic_fixed_point-i16
     ```

     这里我们按照步骤一的观察结果，修改为：

     ```
     customized_quantize_layers:
         Conv_434_142: dynamic_fixed_point-i16
         Conv_530_69: dynamic_fixed_point-i16
         Conv_418_152: dynamic_fixed_point-i16
     ```

   

3. 执行混合量化步骤二，生成混合量化模型。

   `python hybrid_quantization_step2.py`

   

4. 对比原模型、常规量化模型和混合量化模型的分类结果得分。

   `python run_onnx_model.py`	

   ```
   -----TOP 5-----
   [155]: 0.9758387804031372
   [154]: 0.02226063795387745
   [364]: 0.00038293670513667166
   [960]: 0.00022784945031162351
   [879]: 0.0001287872582906857
   ```

   

   `python run_normal_quantization_model.py`

   ```
   -----TOP 5-----
   [155]: 0.8933969140052795
   [154]: 0.08192264288663864
   [364]: 0.0026254409458488226
   [193]: 0.00216865842230618
   [879]: 0.0014796829782426357
   ```

   

   `python run_hybrid_quantization_model.py`

   ```
   -----TOP 5-----
   [155]: 0.9505037665367126
   [154]: 0.04464513808488846
   [364]: 0.0006053716060705483
   [194]: 0.0005000471719540656
   [879]: 0.0003411838551983237
   ```

   可以看到混合量化模型的分类得分得到了不小的提升，更接近于原始模型的分类得分。这里用户也可以自己尝试使用默认推荐配置进行混合量化时，混合量化模型的得分效果。

