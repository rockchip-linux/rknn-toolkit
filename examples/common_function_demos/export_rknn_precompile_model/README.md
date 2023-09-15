# Online pre-compilation

## Usage
Execute command below to export pre-compilation RKNN model with specified target.
```
python export_rknn_precompile_model.py xxx.rknn xxx_precompile.rknn [target] [device_id]
```
- xxx.rknn: the RKNN model path.
- xxx_precompile.rknn: the pre-compiled RKNN model path.
- target: target device, like rv1109, rv1126, rk1808 or rk3399pro. Optional, default target is `rv1126`.
- device_id: target device id. Optional.

