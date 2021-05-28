# Export encrypted RKNN model
## Usage
Execute command below to export an encrypted RKNN model. 
```
python export_encrypt_rknn_model.py xxx.model xxx.encrypt.rknn encrypt_level
```
- xxx.model: Unencrypted RKNN model.
- xxx.encrypt.rknn: Save name of encrypted RKNN model.
- encrypt_level: Encryption level, valid value: 1, 2 or 3.
## Use encrypted RKNN model
1. The encrypted RKNN model is used in the same way as the ordinary RKNN model.
2. If the encrypted RKNN model is deployed with the Python interface, the target cannot be a simulator; if the C API is used for deployment, only the RKNN model needs to be replaced, and other codes do not need to be modified.
