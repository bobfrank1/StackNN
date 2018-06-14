from __future__ import division

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

from model import Controller as AbstractController


class Controller(AbstractController):
    def __init__(self, input_size, read_size, output_size, **args):
        self._input_size = input_size
        self._output_size = output_size
        super(Controller, self).__init__(read_size, **args)

    def _init_network(self):
        network_input_size = self._input_size + self.get_read_size()
        network_output_size = 2 + self.get_read_size() + self._output_size
        self._network = nn.Linear(network_input_size, network_output_size)

        AbstractController.init_normal(self._network.weight)
        self._network.bias.data.fill_(0)

    def _get_network_inputs(self, x):
        return torch.cat([x, self._read], 1),

    def _get_output(self, network_output):
        return network_output[:, 2 + self.get_read_size():], None

    def _get_stack_instructions(self, network_output):
        read_params = sigmoid(network_output[:, :2 + self.get_read_size()])
        v = read_params[:, 2].contiguous()
        u = read_params[:, 0].contiguous()
        d = read_params[:, 1].contiguous()
        return v, u, d
