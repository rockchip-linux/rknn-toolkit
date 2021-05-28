#!/usr/bin/env python3

import sys
import os
import ruamel.yaml
from rknn.api import RKNN


def parse_model_config(config_file):
    config_text = ""
    with open(config_file) as f:
        config_text = f.read()
    if config_text:
        yaml = ruamel.yaml.YAML(typ='rt')
        return yaml.load(config_text)


def convert_model(config_path, out_path, pre_compile):
    if os.path.isfile(config_path):
        config_file = os.path.abspath(config_path)
        config_path = os.path.dirname(config_file)
    else:
        config_file = os.path.join(config_path, 'model_config.yml')
    if not os.path.exists(config_file):
        print('Model config {:} not exist!'.format(config_file))
        exit(-1)

    exported_rknn_model_paths = []
    config = parse_model_config(config_file)
    if config is None:
        print('Invalid configuration.')
        return exported_rknn_model_paths

    for model_name in config['models']:
        model = config['models'][model_name]

        rknn = RKNN()

        rknn.config(**model['configs'])

        print('--> Load model...')
        if model['platform'] == 'tensorflow':
            model_file_path = os.path.join(config_path, model['model_file_path'])
            input_size_list = []
            for input_size_str in model['subgraphs']['input-size-list']:
                input_size = list(map(int, input_size_str.split(',')))
                input_size_list.append(input_size)
            pass
            rknn.load_tensorflow(tf_pb=model_file_path,
                                 inputs=model['subgraphs']['inputs'],
                                 outputs=model['subgraphs']['outputs'],
                                 input_size_list=input_size_list)
        elif model['platform'] == 'tflite':
            model_file_path = os.path.join(config_path, model['model_file_path'])
            rknn.load_tflite(model=model_file_path)
        elif model['platform'] == 'caffe':
            prototxt_file_path = os.path.join(config_path,model['prototxt_file_path'])
            caffemodel_file_path = os.path.join(config_path,model['caffemodel_file_path'])
            rknn.load_caffe(model=prototxt_file_path, proto='caffe', blobs=caffemodel_file_path)
        elif model['platform'] == 'onnx':
            model_file_path = os.path.join(config_path, model['model_file_path'])
            rknn.load_onnx(model=model_file_path)
        else:
            print("Platform {:} is not supported! Moving on.".format(model['platform']))
            continue
        print('Done')

        if model['quantize']:
            dataset_path = os.path.join(config_path, model['dataset'])
        else:
            dataset_path = './dataset'

        print('--> Build RKNN model...')
        rknn.build(do_quantization=model['quantize'], dataset=dataset_path, pre_compile=pre_compile)
        print('Done')

        export_rknn_model_path = "{:}.rknn".format(os.path.join(out_path, model_name))
        print('--> Export RKNN model to: {:}'.format(export_rknn_model_path))
        rknn.export_rknn(export_path=export_rknn_model_path)
        exported_rknn_model_paths.append(export_rknn_model_path)
        print('Done')

    return exported_rknn_model_paths


if __name__ == '__main__':
    config_path = sys.argv[1]
    out_path = sys.argv[2]
    pre_compile = sys.argv[3] in ['true', '1', 'True']

    convert_model(config_path, out_path, pre_compile)
