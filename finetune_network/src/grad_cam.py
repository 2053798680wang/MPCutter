# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""GradCAM."""

from mindspore.ops import operations as op
from mindspore import get_context, PYNATIVE_MODE
from mindspore import nn
from mindspore.ops.composite import GradOperation
from mindspore_xai.common.utils import ForwardProbe, retrieve_layer
from mindspore_xai.common.utils import resize as resize_fn
from copy import deepcopy
from mindspore import Tensor
from mindspore_xai.common.utils import abs_max, unify_inputs, unify_targets
from mindspore_xai.common.attribution import Attribution
from mindspore_xai.common.utils import generate_one_hot
from mindspore.ops import operations as P
import mindspore

def get_bp_weights(num_classes, targets, weights=None):

    if weights is None:
        weights = generate_one_hot(targets[0], num_classes)
    return weights


class Gradient(Attribution):
    def __init__(self, network):
        super(Gradient, self).__init__(network)
        self._backward_model = deepcopy(network)
        self._backward_model.set_train(False)
        self._backward_model.set_grad(True)
        self._grad_net = GradNet(self._backward_model)
        self._aggregation_fn = abs_max
        self._num_classes = None

    def __call__(self, inputs, targets, ret='tensor', show=None):
        """Call function for `Gradient`."""
        self._verify_data(inputs, targets)
        self._verify_other_args(ret, show)

        inputs = unify_inputs(inputs)
        targets = unify_targets(inputs[0].shape[0], targets)

        weights = self._get_bp_weights(inputs, targets)
        gradient = self._grad_net(*inputs, weights)
        saliency = self._aggregation_fn(gradient)

        return self._postproc_saliency(saliency, ret, show)

    def _get_bp_weights(self,inputs, input_mask, token_type_id, unified_targets):
        if self._num_classes is None:
            output = self._backward_model(inputs, input_mask, token_type_id)
            self._num_classes = output.shape[-1]
        return get_bp_weights(self._num_classes, unified_targets)


class IntermediateLayerAttribution(Gradient):
    """
    Base class for generating attribution map at intermediate layer.

    Args:
        network (nn.Cell): DNN model to be explained.
        layer (str, optional): string that specifies the layer to generate
            intermediate attribution. When using default value, the input layer
            will be specified. Default: ''.

    Raises:
        TypeError: Be raised for any argument type problem.
    """

    def __init__(self, network, layer=''):
        super(IntermediateLayerAttribution, self).__init__(network)

        # Whether resize the attribution layer to the input size.
        self._resize = True
        # string that specifies the resize mode. Default: 'nearest_neighbor'.
        self._resize_mode = 'nearest_neighbor'

        self._layer = layer

    @staticmethod
    def _resize_fn(attributions, inputs, mode):
        """Resize the intermediate layer attribution to the same size as inputs."""
        height, width = inputs.shape[2], inputs.shape[3]
        return resize_fn(attributions, (height, width), mode)


class GradNet(nn.Cell):
    """
    Network for gradient calculation.

    Args:
        network (Cell): The network to generate backpropagated gradients.
        sens_param (bool): Enable GradOperation sens_params.
    """

    def __init__(self, network, sens_param=True):
        super(GradNet, self).__init__()
        self.network = network
        self.grad = GradOperation(get_all=True, sens_param=False)(network)

    def construct(self, inputs, input_mask, token_type_id,weights):
        """
        Get backpropgated gradients.

        Returns:
            Tensor, output gradients.
        """
        inputs=P.Cast()(inputs, mindspore.float32)

        gout = self.grad(input_ids=inputs, input_mask=input_mask, token_type_id=token_type_id)
        return gout

def _gradcam_aggregation(attributions):
    """
    Aggregate the gradient and activation to get the final attribution.

    Args:
        attributions (Tensor): the attribution with channel dimension.

    Returns:
        Tensor: the attribution with channel dimension aggregated.
    """
    sum_ = op.ReduceSum(keep_dims=False)
    relu_ = op.ReLU()
    attributions = relu_(sum_(attributions, 1))
    attributions = op.Reshape()(attributions, (attributions.shape[0], 1, *attributions.shape[1:]))
    return attributions

class GradCAM(IntermediateLayerAttribution):

    def __init__(self, network, layer=""):
        super(GradCAM, self).__init__(network, layer)

        self._saliency_cell = retrieve_layer(self._backward_model, target_layer=layer)
        self._avgpool = op.ReduceMean(keep_dims=True)
        self._intermediate_grad = None
        self._aggregation_fn = _gradcam_aggregation
        self._resize_mode = 'bilinear'

        self._grad_net = GradNet(self._backward_model)
        self._hook_cell()

    def _hook_cell(self):
        if get_context("mode") != PYNATIVE_MODE:
            raise TypeError(f"Hook is not supported in graph mode currently, you can use"
                            f"'set_context(mode=PYNATIVE_MODE)'to set pynative mode.")
        if self._saliency_cell:
            self._saliency_cell.register_backward_hook(self._cell_hook_fn)
            self._saliency_cell.enable_hook = True
        self._intermediate_grad = None

    def _cell_hook_fn(self, _, grad_input, grad_output):
        """
        Hook function to deal with the backward gradient.

        The arguments are set as required by `Cell.register_backward_hook`.
        """
        del grad_output
        self._intermediate_grad = grad_input

    def __call__(self,inputs, input_mask, token_type_id, targets, ret='tensor', show=None):

        with ForwardProbe(self._saliency_cell) as probe:

            weights = self._get_bp_weights(inputs, input_mask, token_type_id,targets)
            gradients = self._grad_net(inputs, input_mask, token_type_id, weights)
            # get intermediate activation
            activation = (probe.value,)

            if self._layer == "":
                activation = inputs
                self._intermediate_grad = unify_inputs(gradients)
            if self._intermediate_grad is not None:
                # average pooling on gradients
                intermediate_grad = unify_inputs(
                    self._avgpool(self._intermediate_grad[0], (1)))
            else:
                raise ValueError("Gradient for intermediate layer is not "
                                 "obtained")
            mul = op.Mul()

            if len(intermediate_grad) != 1 or len(activation) != 1:
                raise ValueError("Length of `intermediate_grad` and `activation` must be 1.")

            # manually braodcast
            intermediate_grad = op.Tile()(intermediate_grad[0], (1, 1, *activation[0].shape[2:]))
            attribution = self._aggregation_fn(mul(intermediate_grad, activation[0]))


            if self._resize:
                attribution = self._resize_fn(attribution, *inputs,
                                              mode=self._resize_mode)
            self._intermediate_grad = None

        return self._postproc_saliency(attribution, ret, show)
