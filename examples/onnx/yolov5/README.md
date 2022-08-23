### 关于模型来源：

1. 权重来源于https://github.com/ultralytics/yolov5
2. 导出模型时，由于硬件限制，移除了后处理网络层
3. 关于如何导出 .onnx 文件，请参考 https://github.com/airockchip/rknn_model_zoo/tree/main/models/vision/object_detection/yolov5-pytorch



### Demo 运行步骤：

1. 将导出的onnx模型复制到该demo目录下，执行命令:

   ```
   python test.py
   ```

   

### 注意事项：

1. 切换成自己训练的模型时，请注意对齐anchor,BOX_THESH,NMS_THRESH等后处理参数，否则会导致后处理解析出错。


