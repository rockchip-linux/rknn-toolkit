## 单通道输入Demo

1. 当模型的输入是单通道，即channel维度为1时，建议用户将所有量化数据集预先处理成npy格式保存。这样可以避免在读取灰度图时，因为代码实现上的差异而引起量化误差。

   Demo中的处理方式如下：

   ```python
   img = cv2.imread(jpg_data_path) 
   gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   gray_img = gray_img.reshape(28, 28, 1) # hw --> hwc, this is important, don't miss it
   np.save(npy_data_path, gray_img)
   ```

2. 请注意，如果模型的原始输入是nchw格式的，而灰度图的没有channel维度、尺寸为 hw时，我们需要额外添加channel维度，如上文的代码中第三句代码的处理方式。

3. 当模型的原始输入是3维度或2维的情况时，RKNN模型在量化、推理时候则不会对输入进行 hwc -> chw 的转换操作，此时无需对灰度图补充 channel 维度，数据依照原格式保存成 npy 文件即可。