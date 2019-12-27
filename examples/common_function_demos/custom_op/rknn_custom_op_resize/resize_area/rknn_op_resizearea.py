import tensorflow as tf
import numpy as np

from rknn.api import RKNNCustomOP, IoMap, Shape


class ResizeArea(RKNNCustomOP):

    op = 'ResizeArea'

    def_input = [
        IoMap('in0', 'in', 'input port'),
    ]
    def_output = [
        IoMap('out0', 'out', 'output port'),
    ]

    def load_params_from_tf(self, node_def, tensor_data_map):
        """
        Get params from tensorflow NodeDef and config tensors map
        :param node_def: tf.NodeRef object (reference https://www.tensorflow.org/api_docs/python/tf/NodeDef)
        :param tensor_data_map: Dict of Input Const Tensor
        :return: Dict of parameters (The key of each parameter must be consistent with the definition in the yaml config file)
        """
        p = dict()
        # set params dict
        p['size'] = tensor_data_map['C:out0'].tolist()
        p['align_corners'] = node_def.attr['align_corners'].b
        return p

    def compute_output_shape(self, inputs_shape, params):
        """
        Compute outputs shape
        :param inputs_shape: Input shape list
        :param params: Parameters dict
        :return: Output shape list (all output must be set values by calling set_shape())
        """
        outputs_shape = [Shape() for i in range(len(self.def_output))]
        # set outputs shape by set_shape()
        in_shape = inputs_shape[0].format('nhwc')
        in_channel = in_shape[-1]
        out_shape = []
        out_shape.extend(params['size'])
        out_shape.append(in_channel)
        outputs_shape[0].set_shape(shape=out_shape, fmt='nhwc')
        return outputs_shape

    def compute_output_tensor(self, const_tensor, inputs_tensor, params):
        """
        Compute outputs tensor
        :param const_tensor: Const tensor dict
        :param inputs_tensor: Input tensor list
        :param params: OP parameters dict
        :return: Output tensor list
        """
        outputs_tensor = list()
        # compute outputs tensor
        out = tf.image.resize_area(inputs_tensor[0], size=params['size'], align_corners=params['align_corners'])
        outputs_tensor.append(out)
        return outputs_tensor
