# Example of model inference with RKNN Toolkit Lite


## Model Source

The models used in this example come from the torchvision project:
https://github.com/pytorch/vision/tree/main/torchvision/models

For model conversion, please refer to the following example:
https://github.com/rockchip-linux/rknn-toolkit/tree/master/examples/pytorch/resnet18


## Usage for the script

*Usage:*
```
python test.py
```

- If run this example on a PC, please connect a RK1808 development board.
- If there are multiple devices, please modify the script to specify `device_id` in the `init_runtime` interface.
- If run the example on(or with) rv1109/1126, please adjust the `model` and `target` in script.


## Expected results

This example will print the TOP5 labels and corresponding scores of the test image classification results. For example, the inference results of this example are as follows:
```
-----TOP 5-----
[812]: 0.9994382262229919
[404]: 0.00040962465573102236
[657]: 3.284523336333223e-05
[833]: 2.928587309725117e-05
[895]: 1.850978151196614e-05
```

1. The label index with the highest score is 812, the corresponding label is `space shuttle`.
2. The download link for labels file: https://s3.amazonaws.com/onnx-model-zoo/synset.txt
3. Different platforms, different versions of tools and drivers may have slightly different results.
