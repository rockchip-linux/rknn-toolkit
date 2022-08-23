# Usage of pass_through demo
## Default usage
execute comand below to run this demo, pass original data to NPU, NPU will do pre-process for input.
```
python3 test.py
``` 
## Pass through input
execute command below to run this demo, pass input data to NPU directly, NPU will not do pre-process again.  
```
python3 test.py --pass-through
```
If this demo has been run, you can add the --load-rknn parameter to avoid model conversion again.
```
python3 test.py --pass-through --load-rknn
```
