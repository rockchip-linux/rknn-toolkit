### Demo 运行步骤：

1. 使用yolov5官方仓库导出模型，链接：https://github.com/ultralytics/yolov5。该demo创建时yolov5的最新节点sha码为 8acb5734c7f0d1b7baf62b5c5dab6107a37896c6。

2. 在yolov5工程的根目录下导出已训练好的yolov5模型，如yolov5s/m/l.pt，可参考以下指令导出。

   `python detect.py --weight yolov5s.pt`

   `python export.py --weight yolov5s.pt`

3. 将导出的onnx模型复制到该demo目录下，执行命令会绘出两个检测结果窗口。

   `python test.py`

   

### 注意事项：

1. 切换成自己训练的模型时，请注意对齐anchor等后处理参数，否则会导致后处理解析出错。

2. 最新版本的yolov5模型得到的结果包含两部分：

   A部分：经模型后处理完成的结果。对应 ‘direct result’ 的绘图窗口。

   B部分：未经模型后处理的结果。对应 ‘full post process result’ 的绘图窗口。

3. 在不进行量化的情况下，使用任意一种结果都可以得到正确的结果。

4. 在进行**量化**的情况下，在A部分结果中，坐标的数值范围为[0,img_size]，而置信度的数值范围为[0,1]，量化过程中置信度的值会由于尺度太小，与坐标 concat 到同一个 tensor 时造成置信度值的精度丢失，所以模型量化后不可直接使用A部分结果，**只能自己根据B部分结果进行后处理得到正确的值**。<u>**这个特性是量化本身的性质导致的，用户在使用过程中也应当注意这种不同尺度数据在同个tensor里面时，量化操作会导致严重的精度丢失问题。**</u>

   

