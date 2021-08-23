### Demo 运行步骤：

1. 使用yolov5官方仓库导出模型，链接：https://github.com/ultralytics/yolov5。该demo创建时yolov5的最新节点sha码为 8acb5734c7f0d1b7baf62b5c5dab6107a37896c6。

2. 在yolov5工程的根目录下导出已训练好的yolov5模型，如yolov5s/m/l.pt，可参考以下指令导出。

   `python detect.py --weight yolov5s.pt`

   `python export.py --weight yolov5s.pt`

   注：yolov5工程需要使用pytorch 1.8.0 或 1.9.0 版本才能正常导出，导出的opset version使用默认的即可。

3. 将导出的onnx模型复制到该demo目录下，执行命令会绘出两个检测结果窗口。

   `python test.py`

   

### 注意事项：

1. 切换成自己训练的模型时，请注意对齐anchor等后处理参数，否则会导致后处理解析出错。

2. 测试代码导出模型的时候指定了输出节点['396', '458', '520']，分别为原模型的第2、3、4输出节点的去掉 transpose 算子部分，此举是为了兼容 rknpu/rknn/rknn_api/examples 的示例。实际部署代码请灵活处理，该demo只是提供了其中一种方式，并不是唯一的。对于其他模型，如yolov5m, yolov5l 等，请使用工具自行查找对应输出节点的名称，这里推荐使用可视化工具 netron。 

   

