from __future__ import division

from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from structs.simple import Stack


class Controller(nn.Module):
    """
    Abstract class for creating policy networks (controllers) that
    operate a neural data structure, such as a neural stack or a neural
    queue. To create a custom controller, create a class inheriting from
    this one that overrides self.__init__ and self.forward.
    """
    __metaclass__ = ABCMeta

    def __init__(self, read_size, struct_type=Stack, k=None):
        """
        Constructor for the Controller object.

        :type read_size: int
        :param read_size: The size of the vectors that will be placed on
            the neural data structure

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Controller will operate. Please pass the *class* for the
            data structure type to this parameter, not a specific
            instance of that class
        """
        super(Controller, self).__init__()
        self._read_size = read_size
        self._struct_type = struct_type
        self._k = k

        self._network = None  # The controller network
        self._state = None  # The hidden state of the network, if any

        self._stack = None  # The data structure
        self._read = None  # The last item read from the structure

        self._init_network()

    @abstractmethod
    def _init_network(self):
        """
        Sets up the neural network that determines the action of the
        Controller.

        :return: None
        """
        raise NotImplementedError("Missing implementation for _init_network")

    def init_stack(self, batch_size):
        """
        Resets self._read to be zero and self._stack to be an empty data
        structure. This function assumes that this Controller is being
        used to train a network in mini-batches.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch

        :return: None
        """
        self._read = Variable(torch.zeros([batch_size, self._read_size]))
        self._stack = self._struct_type(batch_size, self._read_size, k=self._k)

    @staticmethod
    def init_normal(tensor):
        n = tensor.data.shape[0]
        tensor.data.normal_(0, 1. / np.sqrt(n))

    """ Controller Operation """

    def forward(self, x):
        """
        Based on its input, hidden state, and previous item read from
        the data structure, this function needs to do the following:
            - produce an output and return it
            - save the hidden state to self._state
            - send instructions to the data structure
            - read from the data structure and save it to self._read.

        :param x: The input to this Controller

        :return: The output of this Controller
        """
        network_inputs = self._get_network_inputs(x)
        network_output = self._network(*network_inputs)
        output, state = self._get_output(network_output)
        stack_instructions = self._get_stack_instructions(network_output)

        self._read = self.read_stack(*stack_instructions)
        self._state = state
        return output

    @abstractmethod
    def _get_network_inputs(self, x):
        """
        Combines the controller input, hidden state, and previously read
        vector from the data structure to produce the inputs to the
        neural network module for this Controller.

        :param x: The input to this Controller

        :rtype: tuple
        :return: The arguments to be passed to self._network, formatted
            as a tuple
        """
        raise NotImplementedError("Missing implementation for " +
                                  "_get_network_inputs")

    @abstractmethod
    def _get_output(self, network_output):
        """
        Extracts the output of the Controller from the output of
        self._network.

        :param network_output: The output of self._network

        :return: The output of this Controller
        """
        raise NotImplementedError("Missing implementation for _get_output")

    @abstractmethod
    def _get_stack_instructions(self, network_output):
        """
        Extracts instructions for operating the neural data structure
        from the output of self._network.

        :param network_output: The output of self._network

        :rtype: tuple
        :return: Instructions for the data structure, formatted as a
            tuple
        """
        raise NotImplementedError("Missing implementation for " +
                                  "_get_stack_instructions")

    def read_stack(self, v, u, d):
        if self._k is None:
            return self._stack.forward(v, u, d)
        else:
            return torch.cat(torch.unbind(self._read, dim=1), dim=1)
            # TODO verify this

    """ Analytical Tools """

    def get_read_size(self):
        """
        Return the effective read size, taking k into account.
        """
        k = 1 if self._k is None else self._k
        return self._read_size * k

    def trace(self, trace_X):
        """
        Visualize stack activations for a single training sample.
        Draws a graphic representation of these stack activations.
        @param trace_X [1, max_length, input_size] tensor
        """
        self.eval()
        self.init_stack(1)
        max_length = trace_X.data.shape[1]
        data = np.zeros([2 + self._read_size, max_length])  # 2 + len(v)
        for j in xrange(1, max_length):
            self.forward(trace_X[:, j - 1, :])
            data[0, j] = self.u.data.numpy()
            data[1, j] = self.d.data.numpy()
            data[2:, j] = self.v.data.numpy()
        plt.imshow(data, cmap="hot", interpolation="nearest")
        plt.show()
