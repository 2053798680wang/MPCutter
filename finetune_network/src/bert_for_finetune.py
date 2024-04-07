# Copyright 2020 Huawei Technologies Co., Ltd
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

'''
Bert for finetune script.
'''

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import MaskedSelect as Mask
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from .bert_for_pre_training import clip_grad
from .finetune_eval_model import BertCLSModel, BertSEQModel, BertSEQModelEval,BertResidualModel,BertCLSModelEval,BertBertSecondaryStructureModel,BertRegModel,BertConcatModel
from .utils import CrossEntropyCalculation,FocalLossCalculation

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


class BertFinetuneCell(nn.TrainOneStepWithLossScaleCell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Different from the builtin loss_scale wrapper cell, we apply grad_clip before the optimization.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertFinetuneCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)

class BertFinetuneStep3Cell(nn.TrainOneStepWithLossScaleCell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Different from the builtin loss_scale wrapper cell, we apply grad_clip before the optimization.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertFinetuneStep3Cell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  all_pssms,
                  all_dssps,
                  pssms,
                  dssps,
                  window_seq,
                  label_ids,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            all_pssms,
                            all_dssps,
                            pssms,
                            dssps,
                            window_seq,
                            label_ids)
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 all_pssms,
                                                 all_dssps,
                                                 pssms,
                                                 dssps,
                                                 window_seq,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class BertFinetunePDCell(nn.TrainOneStepWithLossScaleCell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Different from the builtin loss_scale wrapper cell, we apply grad_clip before the optimization.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertFinetunePDCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  pssms,
                  dssps,
                  label_ids,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            pssms,
                            dssps,
                            label_ids)
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 pssms,
                                                 dssps,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)



class BertFinetuneCellConcat(nn.TrainOneStepWithLossScaleCell):

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertFinetuneCellConcat, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

    def construct(self,
                  input_ids_0,
                  input_mask,
                  token_type_id,
                  input_ids_1,
                  label_ids,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        loss = self.network(input_ids_0,
                            input_mask,
                            token_type_id,
                            input_ids_1,
                            label_ids)
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids_0,
                                                 input_mask,
                                                 token_type_id,
                                                 input_ids_1,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)




class BertSquadCell(nn.TrainOneStepWithLossScaleCell):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertSquadCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  start_position,
                  end_position,
                  unique_id,
                  is_impossible,
                  sens=None):
        """BertSquad"""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            start_position,
                            end_position,
                            unique_id,
                            is_impossible)
        if sens is None:
            scaling_sens = self.scale_sense
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 start_position,
                                                 end_position,
                                                 unique_id,
                                                 is_impossible,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class BertCLS(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLS, self).__init__()
        self.bert = BertCLSModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.is_training = is_training

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.assessment_method == "spearman_correlation":
            if self.is_training:
                loss = self.loss(logits, label_ids)
            else:
                loss = logits
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss



class BertReg(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=1, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertReg, self).__init__()
        self.bert = BertRegModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.is_training = is_training
        self.mse=nn.MSELoss()

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.is_training:
            loss = self.mse(logits, label_ids)
        else:
            loss = logits * 1.0
        return loss

class BertSecondaryStructure(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, unique_seq_label, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSecondaryStructure, self).__init__()
        self.bert = BertBertSecondaryStructureModel(config, is_training, unique_seq_label, dropout_prob, use_one_hot_embeddings)
        self.loss = CrossEntropyCalculation(is_training)
        self.is_training = is_training
        self.seq_length=unique_seq_label

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        loss = self.loss(logits, label_ids, self.seq_length)
        return logits

class BertCLSEval(nn.Cell):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSEval, self).__init__()
        self.bert = BertCLSModelEval(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.is_training = is_training

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits,pooled_output,sequence_output,all_polled_output,all_sequence_output = self.bert(input_ids, input_mask, token_type_id)
        if self.assessment_method == "spearman_correlation":
            if self.is_training:
                loss = self.loss(logits, label_ids)
            else:
                loss = logits
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss,pooled_output,sequence_output,all_polled_output,all_sequence_output

class BertSeq(nn.Cell):
    """
    Train interface for sequence labeling finetuning task.
    """

    def __init__(self, config,  is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False,loss_func="CrossEntropy",label_percent=None,loss_gama=2.0):
        super(BertSeq, self).__init__()
        self.bert = BertSEQModel(config, is_training, num_labels,  with_lstm, dropout_prob,
                                 use_one_hot_embeddings)
        if loss_func=="CrossEntropy":
            self.loss = CrossEntropyCalculation(is_training)
        elif loss_func=="Focal":
            self.loss = FocalLossCalculation(weight=Tensor(label_percent), gamma=loss_gama, reduction='mean',is_training=is_training)
        else:
            raise "Error Loss Name"
        self.loss_name=loss_func
        self.num_labels = num_labels

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        loss = self.loss(logits, label_ids, self.num_labels)

        return loss



class BertSeqEval(nn.Cell):
    """
    Train interface for sequence labeling finetuning task.
    """

    def __init__(self, config,  is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False,loss_func="CrossEntropy",label_percent=None,loss_gama=2.0):
        super(BertSeqEval, self).__init__()
        self.bert = BertSEQModelEval(config, is_training, num_labels,  with_lstm, dropout_prob,
                                 use_one_hot_embeddings)
        if loss_func=="CrossEntropy":
            self.loss = CrossEntropyCalculation(is_training)
        elif loss_func=="Focal":
            self.loss = FocalLossCalculation(weight=Tensor(label_percent), gamma=loss_gama, reduction='mean',is_training=is_training)
        else:
            raise "Error Loss Name"
        self.loss_name=loss_func
        self.num_labels = num_labels

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits,sequence_output = self.bert(input_ids, input_mask, token_type_id)
        loss = self.loss(logits, label_ids, self.num_labels)

        return loss,sequence_output