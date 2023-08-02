# RKNN-Toolkit accuracy_analysis 例程

## 使用说明

​    本例程以 shufflenetv2.onnx 模型作为示范，展示 RKNN-Toolkit 精度分析 accuracy_analysis 接口的使用流程。在常规量化模式效果有限的情况下，通过使用混合量化，提高 RKNN 模型的推理精度。实际使用时，请根据具体需求，在推理精度与推理速度的之间作一定的取舍。


## 使用步骤

1. 生成常规量化模型，并生成精度分析报告。

   `python normal_quantizition.py`

   - 执行后在该工程目录下可得到 normal_quantization_analysis 文件夹，各文件/目录的含义参考 《Rockchip_User_Guide_RKNN_Toolkit_CN.docx》4.3章节说明。

   - 观察生成的 individual_qnt_error_analysis.txt 文件，可以看到 Conv_418、Conv_434、Conv_530 这三层归一化后的余弦值分别为 0.97、0.98、0.96，比其他层的归一化余弦值低一些，初步判断这几层对量化不友好，需要使用混合量化去处理这些网络层。而其他网络层归一化后的余弦值都在0.99以上，通常可以认为是量化友好的。先记录下量化不友好网络层的名称，供后续步骤使用。

   - 这个脚本默认的 target 是 rv1126，如果要在 rk1808/rk3399pro 上运行导出的 RKNN 模型，需要在调用该脚本时指定target，如： `python normal_quantizition.py rk1808`。


2. 执行混合量化步骤一，生成配置文件，具体资料可参考《Rockchip_User_Guide_RKNN_Toolkit_CN.docx》的 4.5 章节 。

   `python hybrid_quantization_step1.py`

   - 打开生成的 torchjitexport.quantization.cfg 配置文件，参考 《Rockchip_User_Guide_RKNN_Toolkit_CN.docx》4.5 节 的说明，在 customized_quantize_layers 信息里面添加想要使用混合量化处理的网络层（从步骤1中得到）。

   - 配置文件中的 customized_quantize_layers 会给初始混合量化层的建议，这个建议的效果不一定是最优的，实际使用过程中请根据具体模型灵活设置。示例模型生成的 quantization.cfg 混合量化建议为：

     ```
     customized_quantize_layers:
         Reshape_Reshape_614_2: dynamic_fixed_point-i16
         Gemm_Gemm_615_1: dynamic_fixed_point-i16
         AveragePool_AveragePool_612_3: dynamic_fixed_point-i16
         Reshape_Reshape_614_2_acuity_mark_perm_157: dynamic_fixed_point-i16
         Conv_Conv_434_104: dynamic_fixed_point-i16
         Conv_Conv_436_101: dynamic_fixed_point-i16
         Slice_Slice_363_148: dynamic_fixed_point-i16
         Conv_Conv_364_145_acuity_mark_perm_209: dynamic_fixed_point-i16
         Conv_Conv_530_49: dynamic_fixed_point-i16
         Conv_Conv_466_86: dynamic_fixed_point-i16
         Conv_Conv_418_113: dynamic_fixed_point-i16
         Conv_Conv_383_132: dynamic_fixed_point-i16
         Conv_Conv_367_141: dynamic_fixed_point-i16
         Conv_Conv_345_146: dynamic_fixed_point-i16
         Conv_Conv_353_147: dynamic_fixed_point-i16
         Conv_Conv_343_149: dynamic_fixed_point-i16
         Conv_Conv_348_153: dynamic_fixed_point-i16
     ```

     按照步骤一的观察结果，修改为：

     ```
     customized_quantize_layers:
         Conv_Conv_434_104: dynamic_fixed_point-i16
         Conv_Conv_530_49: dynamic_fixed_point-i16
         Conv_Conv_418_113: dynamic_fixed_point-i16
     ```

   - 这个脚本默认的 target 是 rv1126，如果要在 rk1808/rk3399pro 上运行导出的 RKNN 模型，需要在调用该脚本时指定target，如： `python hybrid_quantization_step1.py rk1808`


3. 执行混合量化步骤二，生成混合量化模型。

   `python hybrid_quantization_step2.py`

   *Note: 这个脚本默认的 target 是 rv1126，如果要在 rk1808/rk3399pro 上运行导出的 RKNN 模型，需要在调用该脚本时指定target，如： `python hybrid_quantization_step1.py rk1808`*


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
   [155]: 0.9094294309616089
   [154]: 0.06887859851121902
   [364]: 0.0020060015376657248
   [193]: 0.0018231587018817663
   [879]: 0.0012439332203939557
   ```

   `python run_hybrid_quantization_model.py`

   ```
   -----TOP 5-----
   [155]: 0.9504370093345642
   [154]: 0.04463796317577362
   [194 364]: 0.0005500342231243849
   [194 364]: 0.0005500342231243849
   [879]: 0.00041292302194051445
   ```

   可以看到混合量化模型的分类得分得到了不小的提升，更接近于原始模型的分类得分。这里用户也可以自己尝试使用默认推荐配置进行混合量化。

   Note: 不同平台、不同版本的工具得到的结果可能有细微差异。


## 各脚本参数说明


- normal_quantizition
   ```
   python normal_quantizition.py [target]
   ```
   - target: 目标设备平台，可选参数，默认值是 rv1126。可填如下值: rk1806, rk1808, rk3399pro, rv1109, rv1126。


- hybrid_quantization_step1
   ```
   python hybrid_quantization_step1.py [target]
   ```
   - target: 目标设备平台，可选参数，默认值是 rv1126。可填如下值: rk1806, rk1808, rk3399pro, rv1109, rv1126。


- hybrid_quantization_step2
   ```
   python hybrid_quantization_step2.py [target]
   ```
   - target: 目标设备平台，可选参数，默认值是 rv1126。可填如下值: rk1806, rk1808, rk3399pro, rv1109, rv1126。


- run_normal_quantization_model
   ```
   python run_normal_quantization_model.py [target] [device_id]
   ```
   - target: 目标设备平台，可选参数，默认值是 rv1126。可填如下值: rk1806, rk1808, rk3399pro, rv1109, rv1126。
   - device_id: 目标设备 ID，如果连有多个设备，需要填写设备 ID 进行区分。该参数是一个可选参数，如果不填写，默认的设备 ID 为 None。


- run_hybrid_quantization_model
   ```
   python run_hybrid_quantization_model.py [target] [device_id]
   ```
   - target: 目标设备平台，可选参数，默认值是 rv1126。可填如下值: rk1806, rk1808, rk3399pro, rv1109, rv1126。
   - device_id: 目标设备 ID，如果连有多个设备，需要填写设备 ID 进行区分。该参数是一个可选参数，如果不填写，默认的设备 ID 为 None。
