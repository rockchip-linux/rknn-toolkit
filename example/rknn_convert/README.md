## 环境准备

安装rknn-toolkit

## 模型转换

```
python rknn_convert models/face_detection out_rknn False
```

第1个参数是要转换的原始模型的路径（可以直接填目录，但该目录需要包含`model_config.yml`文件）

第2个参数是转换后模型输出目录

第3个参数是是否开启预编译（加速模型加载时间）


## 添加模型

参考models下的目录，其中

- tensorflow可以参考tensorflow/mobilenet-ssd
- caffe模型可以参考caffe/mobilenet_v2
- onnx模型可以参考onnx/mobilenet_v2
- tflite模型可以参考tflite/mobilenet_v1

需要包括以下文件

- 模型原始文件
- model_config.yml模型的配置文件
- 量化的dataset.txt和量化图片
