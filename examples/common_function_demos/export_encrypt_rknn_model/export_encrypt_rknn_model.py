import sys

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Usage: python {} xxx.rknn xxx.encrypt.rknn encrypt_level'.format(sys.argv[0]))
        print('Such as: python {} mobilenet_v1.rknn mobilenet_v1.encrypt.rknn 1'.format(sys.argv[0]))
        exit(1)

    from rknn.api import RKNN

    orig_rknn = sys.argv[1]
    encrypt_rknn = sys.argv[2]
    encrypt_level = int(sys.argv[3])

    if encrypt_level < 1 or encrypt_level > 3:
        print("Invalid encryption level, the value should be 1, 2 or 3.")

    # Create RKNN object
    rknn = RKNN()
    
    # Export encrypted RKNN model
    print('--> Export encrypted RKNN model')
    ret = rknn.export_encrypted_rknn_model(orig_rknn, encrypt_rknn, encrypt_level)
    if ret != 0:
        print('Encrypt RKNN model failed!')
        exit(ret)
    print('done')

    rknn.release()

