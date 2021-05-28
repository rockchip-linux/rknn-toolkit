#!/usr/bin/env python3

import sys
import os
import argparse
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
    exported_rknn_model_paths = []

    if os.path.isfile(config_path):
        config_file = os.path.abspath(config_path)
        config_path = os.path.dirname(config_file)
    else:
        config_file = os.path.join(config_path, 'model_config.yml')
    if not os.path.exists(config_file):
        print('Model config {:} not exist!'.format(config_file))
        return exported_rknn_model_paths

    config = parse_model_config(config_file)
    if config is None:
        print('Invalid configuration.')
        return exported_rknn_model_paths

    for model_name in config['models']:
        model = config['models'][model_name]

        rknn = RKNN()

        rknn.config(**model['configs'])

        print('--> Load model...')
        model_file_path = os.path.join(config_path, model['model_file_path'])
        if model['platform'] == 'tensorflow':
            subgraphs = model['subgraphs']
            rknn.load_tensorflow(tf_pb=model_file_path,
                                 inputs=subgraphs['inputs'],
                                 outputs=subgraphs['outputs'],
                                 input_size_list=subgraphs['input_tensor_shapes'])
        elif model['platform'] == 'tflite':
            rknn.load_tflite(model=model_file_path)
        elif model['platform'] == 'onnx':
            rknn.load_onnx(model=model_file_path)
        elif model['platform'] == 'caffe':
            prototxt_file_path = os.path.join(config_path, model['prototxt_file_path'])
            caffemodel_file_path = os.path.join(config_path, model['caffemodel_file_path'])
            rknn.load_caffe(model=prototxt_file_path, proto='caffe', blobs=caffemodel_file_path)
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


def parse_args(*argv):
    parser = argparse.ArgumentParser(description="Build RKNN models")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--out_dir", required=True)
    parser.add_argument("-p", "--precompile", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.config):
        print("Enter an existing config file.")
        sys.exit(-1)
    return args.config, args.out_dir, args.precompile


if __name__ == '__main__':
    config_path, out_path, pre_compile = parse_args(*sys.argv[1:])
    #print(config_path, out_path, pre_compile)

    if out_path:
        os.makedirs(out_path, exist_ok=True)

    convert_model(config_path, out_path, pre_compile)
