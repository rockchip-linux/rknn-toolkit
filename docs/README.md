# 文档说明

## 文档结构

RKNN Toolkit 文档目录结构如下：
```
docs
├── 01_Rockchip_Quick_Start_RKNN_SDK_V1.7.5_CN.pdf
├── 01_Rockchip_Quick_Start_RKNN_SDK_V1.7.5_EN.pdf
├── 02_Rockchip_User_Guide_RKNN_Toolkit_V1.7.5_CN.pdf
├── 02_Rockchip_User_Guide_RKNN_Toolkit_V1.7.5_EN.pdf
├── 03_Rockchip_User_Guide_RKNN_Toolkit_Lite_V1.7.5_CN.pdf
├── 03_Rockchip_User_Guide_RKNN_Toolkit_Lite_V1.7.5_EN.pdf
├── 04_Rockchip_Trouble_Shooting_RKNN_Toolkit_V1.7.5_CN.pdf
├── 04_Rockchip_Trouble_Shooting_RKNN_Toolkit_V1.7.5_EN.pdf
├── 05_Rockchip_User_Guide_RKNN_Toolkit_Visualization_V1.7.5_CN.pdf
├── 05_Rockchip_User_Guide_RKNN_Toolkit_Visualization_V1.7.5_EN.pdf
├── 06_Rockchip_Developer_Guide_RKNN_Toolkit_Custom_OP_V1.7.5_CN.pdf
├── 06_Rockchip_Developer_Guide_RKNN_Toolkit_Custom_OP_V1.7.5_EN.pdf
├── changelog.txt
├── README.md
├── RKNN_OP_Support_And_Limit_CN.xlsx
├── RKNN_OP_Support_And_Limit.xlsx
└── RKNN_OP_Support_V1.7.5.md
```

- 01_Rockchip_Quick_Start_RKNN_SDK_V1.7.5_CN.pdf: 中文版快速上手指南
- 01_Rockchip_Quick_Start_RKNN_SDK_V1.7.5_EN.pdf: 英文版快速上手指南
- 02_Rockchip_User_Guide_RKNN_Toolkit_V1.7.5_CN.pdf: 中文版RKNN Toolkit使用说明文档
- 02_Rockchip_User_Guide_RKNN_Toolkit_V1.7.5_EN.pdf: 英文版RKNN Toolkit使用说明文档
- 03_Rockchip_User_Guide_RKNN_Toolkit_Lite_V1.7.5_CN.pdf: 中文版RKNN Toolkit Lite使用说明文档
- 03_Rockchip_User_Guide_RKNN_Toolkit_Lite_V1.7.5_EN.pdf: 英文版RKNN Toolkit Lite使用说明文档
- 04_Rockchip_Trouble_Shooting_RKNN_Toolkit_V1.7.5_CN.pdf: 中文版常见问题解答
- 04_Rockchip_Trouble_Shooting_RKNN_Toolkit_V1.7.5_EN.pdf: 英文版常见问题解答
- 05_Rockchip_User_Guide_RKNN_Toolkit_Visualization_V1.7.5_CN.pdf: 中文版 RKNN Toolkit 可视化功能使用说明
- 05_Rockchip_User_Guide_RKNN_Toolkit_Visualization_V1.7.5_EN.pdf: 英文版 RKNN Toolkit 可视化功能使用说明
- 06_Rockchip_Developer_Guide_RKNN_Toolkit_Custom_OP_V1.7.5_CN.pdf: 中文版自定义算子功能使用说明
- 06_Rockchip_Developer_Guide_RKNN_Toolkit_Custom_OP_V1.7.5_EN.pdf: 英文版自定义算子功能使用说明
- changelog.txt: 各版本功能更新说明
- README.md: 本文档
- RKNN_OP_Support_And_Limit_CN.xlsx: 中文版RKNN算子支持和限制说明
- RKNN_OP_Support_And_Limit.xlsx: 英文版RKNN算子支持和限制说明
- RKNN_OP_Support_V1.7.5.md: RKNN Toolkit 1.7.5版本支持的各平台算子列表

## 建议阅读顺序

- 对于初次使用 RKNPU 的用户，建议先阅读[快速上手指南文档](https://github.com/rockchip-linux/rknn-toolkit/blob/master/doc/01_Rockchip_Quick_Start_RKNN_SDK_V1.7.5_CN.pdf)，将RKNN Toolkit工程中的示例跑通；
- 在对RKNPU有初步了解的基础上，根据自身的任务，阅读[使用说明文档](https://github.com/rockchip-linux/rknn-toolkit/blob/master/doc/02_Rockchip_User_Guide_RKNN_Toolkit_V1.7.5_CN.pdf)，该文档提供了详细的功能说明，模型转换、评估、部署说明和工具各接口的详细使用说明；
- 如果部署阶段使用 Python 作为整个工程的开发语言，则可以参考[RKNN Toolkit Lite 工具的使用说明文档](https://github.com/rockchip-linux/rknn-toolkit/blob/master/doc/03_Rockchip_User_Guide_RKNN_Toolkit_Lite_V1.7.5_CN.pdf)，该文档提供了 Python 推理接口的详细使用说明；
- 如果在模型转换、评估、部署过程中遇到任何问题，可以先查阅[Trouble Shooting 文档](https://github.com/rockchip-linux/rknn-toolkit/blob/master/doc/04_Rockchip_Trouble_Shooting_RKNN_Toolkit_V1.7.5_CN.pdf)，按照文档提供的方法进行排查；
- 在模型转换、评估阶段可以参考[可视化功能使用说明文档](https://github.com/rockchip-linux/rknn-toolkit/blob/master/doc/05_Rockchip_User_Guide_RKNN_Toolkit_Visualization_V1.7.5_CN.pdf)直接使用RKNN Toolkit提供的可视化功能完成模型转换、评估等任务；
- 如果模型转换过程中，提示算子不支持，可以先查阅[RKNN算子支持和限制说明文档](https://github.com/rockchip-linux/rknn-toolkit/blob/master/doc/RKNN_OP_Support_And_Limit_CN.xlsx)或者各平台[算子支持列表文档](https://github.com/rockchip-linux/rknn-toolkit/blob/master/doc/RKNN_OP_Support_V1.7.5.md)，确认算子是否支持；
- 对于 C API 的详细使用说明文档，请参考[rknpu](https://github.com/rockchip-linux/rknpu)工程rknpu/rknn/doc目录下的使用说明文档。

