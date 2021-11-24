### Demo 运行步骤：

---

1. 使用yolov5官方仓库导出模型，链接：https://github.com/ultralytics/yolov5。该demo创建时yolov5的节点的commit id为:c5360f6e7009eb4d05f14d1cc9dae0963e949213

2. 在yolov5工程的根目录下导出已训练好的yolov5模型，如yolov5s.torchscript.pt模型

   ```
   python export.py --weights yolov5s.pt --img 640 --batch 1 --include torchscript
   ```

   **注意**：模型导出前需要对yolov5/models/yolo.py进行修改，具体修改步骤在后面。

3. 将导出的pt模型复制到该demo目录下，执行命令:

   ```
   python test.py
   ```

4. 加载yolov5 pytorch模型需要将 rknn_toolkit版本升级至 1.7.1

5. 开启量化时，rknn.config中的quantize_input_node 被设置为True。若不启用，yolov5模型由于模型头部是slice算子而非常规算子，这种情况有可能导致yolov5模型的输入层没有被转为量化算子，导致RKNN模型在板端使用 RKNN C api 部署时，input_set耗时异常。该参数具体作用请参考用户手册7.2节。

6. 模型转换后，c代码部署可参考https://github.com/rockchip-linux/rknpu/tree/master/rknn/rknn_api/examples/rknn_yolov5_demo



## yolov5模型导出需要注意的地方

1. yolov5官方仓库地址为 https://github.com/ultralytics/yolov5

3. 直接使用pt模型转为rknn模型时，需要修改 yolov5/models/yolo.py文件的后处理部分，将class Detect(nn.Module) 类的子函数forward由

   ```python
   def forward(self, x):
           z = []  # inference output
           for i in range(self.nl):
               x[i] = self.m[i](x[i])  # conv
               bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
               x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
   
               if not self.training:  # inference
                   if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                       self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
   
                   y = x[i].sigmoid()
                   if self.inplace:
                       y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                       y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                   else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                       xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                       wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                       y = torch.cat((xy, wh, y[..., 4:]), -1)
                   z.append(y.view(bs, -1, self.no))
   
           return x if self.training else (torch.cat(z, 1), x)
   ```

   修改为：

   ```python
   def forward(self, x):
           z = []  # inference output
           for i in range(self.nl):
               x[i] = self.m[i](x[i])  # conv
   
           return x
   ```

