# The RKNN model performance evaluation
## Usage
Execute command below to evaluate the performance of RKNN model on specified target.
```
python eval_perf.py xxx.rknn [perf_debug] [target] [device_id]
```
- xxx.rknn: the RKNN model.
- perf_debug: if perf_debug set 0, only show total inference time; if perf_debug set 1, show the time spent on each layer. Optional.
- target: target device, like rv1109, rv1126, rk1808 or rk3399pro. Optional.
- device_id: target device id. Optional.
