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
Bert finetune and evaluation model script.
'''
import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import Concat as C
from mindspore import context
from .bert_model import BertModel,BertModelEval,BertModelAllSeqs
import mindspore.numpy as mnp
from mindspore.ops import MaskedSelect as Mask
from .bert_model import CreateAttentionMaskFromInputMask,BertSelfAttention
import mindspore.ops as ops


class BertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        if not is_training:
            self.log_softmax = P.Softmax(axis=-1)
        else:
            self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1.0 - dropout_prob)
        self.assessment_method = assessment_method

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        if self.assessment_method != "spearman_correlation":
            logits = self.log_softmax(logits)
        return logits

class BertConcatModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertConcatModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size*2, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.assessment_method = assessment_method

        self.lstm_onehot = nn.OneHot(depth=config.vocab_size)
        self.rnn = nn.LSTM(input_size=config.vocab_size,
                               hidden_size=int(config.hidden_size/2),
                               num_layers=2,
                               has_bias=True,
                               bidirectional=True,
                               dropout=0.0,
                               batch_first=True
                                )

    def construct(self, input_ids_0, input_mask, token_type_id,input_ids_1):
        sequence_output, pooled_output_1, _ = self.bert(input_ids_0, token_type_id, input_mask)
        cls = self.cast(pooled_output_1, self.dtype)
        cls = self.dropout(cls)

        embedded = self.dropout(self.lstm_onehot(input_ids_1))
        _, (hidden, _) = self.rnn(embedded)
        hidden = self.cast(hidden, self.dtype)
        hidden = self.dropout(mnp.concatenate((hidden[-2, :, :], hidden[-1, :, :]), axis=1))

        cls = mnp.concatenate((cls,hidden),axis=1)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        if self.assessment_method != "spearman_correlation":
            logits = self.log_softmax(logits)
        return logits

class BertRegModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=1, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertRegModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, 1, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.assessment_method = assessment_method

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        cls = self.dropout(pooled_output)
        logits=self.dense_1(cls)

        return logits

class BertCLSModelEval(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSModelEval, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModelEval(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.Softmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.assessment_method = assessment_method

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _,all_sequence_output,all_polled_output = self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        if self.assessment_method != "spearman_correlation":
            logits = self.log_softmax(logits)
        return logits,pooled_output,sequence_output,all_polled_output,all_sequence_output

class BertBertSecondaryStructureModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, unique_seq_label, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertBertSecondaryStructureModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.dense_1 = nn.Dense(config.hidden_size, unique_seq_label, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(sequence_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        return logits

class BertSquadModel(nn.Cell):
    '''
    This class is responsible for SQuAD
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSquadModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense1 = nn.Dense(config.hidden_size, num_labels, weight_init=self.weight_init,
                               has_bias=True).to_float(config.compute_type)
        self.num_labels = num_labels
        self.dtype = config.dtype
        self.log_softmax = P.LogSoftmax(axis=1)
        self.is_training = is_training
        self.gpu_target = context.get_context("device_target") == "GPU"
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.shape = (-1, config.hidden_size)
        self.origin_shape = (-1, config.seq_length, self.num_labels)
        self.transpose_shape = (-1, self.num_labels, config.seq_length)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        sequence = self.reshape(sequence_output, self.shape)
        logits = self.dense1(sequence)
        logits = self.cast(logits, self.dtype)
        logits = self.reshape(logits, self.origin_shape)
        if self.gpu_target:
            logits = self.transpose(logits, (0, 2, 1))
            logits = self.log_softmax(self.reshape(logits, (-1, self.transpose_shape[-1])))
            logits = self.transpose(self.reshape(logits, self.transpose_shape), (0, 2, 1))
        else:
            logits = self.log_softmax(logits)
        return logits

class BertSEQModel(nn.Cell):
    """
    This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
    The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSEQModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
            self.log_softmax=nn.Softmax(axis=-1)
        else:
            self.log_softmax = P.LogSoftmax(axis=-1)
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        if with_lstm:
            self.lstm_hidden_size = config.hidden_size // 2
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.with_lstm = with_lstm
        self.origin_shape = (-1, config.seq_length, self.num_labels)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)
        if self.with_lstm:
            batch_size = input_ids.shape[0]
            data_type = self.dtype
            hidden_size = self.lstm_hidden_size
            h0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            c0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            seq, _ = self.lstm(seq, (h0, c0))

        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        return_value = self.log_softmax(logits)
        return return_value

class BertSEQModelEval(nn.Cell):
    """
    This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
    The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSEQModelEval, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
            self.log_softmax=nn.Softmax(axis=-1)
        else:
            self.log_softmax = P.LogSoftmax(axis=-1)
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        if with_lstm:
            self.lstm_hidden_size = config.hidden_size // 2
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.with_lstm = with_lstm
        self.origin_shape = (-1, config.seq_length, self.num_labels)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)
        if self.with_lstm:
            batch_size = input_ids.shape[0]
            data_type = self.dtype
            hidden_size = self.lstm_hidden_size
            h0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            c0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            seq, _ = self.lstm(seq, (h0, c0))

        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        return_value = self.log_softmax(logits)
        return return_value,sequence_output

class ResidualBlock(nn.Cell):
    def __init__(self, in_channels, out_channels,dim,config,weight_init):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, has_bias=False,pad_mode="pad")
        self.bn1 = nn.BatchNorm1d(dim*out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, has_bias=False,pad_mode="pad")
        self.bn2 = nn.BatchNorm1d(dim*out_channels)
        self.relu2 = nn.ReLU()
        self.dense1=nn.Dense(out_channels,in_channels, weight_init=weight_init,
                                has_bias=True).to_float(config.compute_type)

    def construct(self, x):
        identity = x
        identity = identity
        out = self.conv1(x)
        out_1=P.Reshape()(out, (out.shape[0], -1))
        out_1 = self.bn1(out_1)
        out = P.Reshape()(out_1, (out.shape[0], out.shape[1],out.shape[2]))
        out = self.relu1(out)
        out = self.conv2(out)
        out_1 = P.Reshape()(out, (out.shape[0], -1))
        out_1 = self.bn2(out_1)
        out = P.Reshape()(out_1, (out.shape[0], out.shape[1],out.shape[2]))
        out=P.Transpose()(self.dense1(P.Transpose()(out, (0, 2, 1))), (0, 2, 1))
        out = out + identity
        out = self.relu2(out)
        return out

class BertResidualModel(nn.Cell):
    def __init__(self, config, is_training, num_labels=11, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertResidualModel, self).__init__()
        self.weight_init = TruncatedNormal(config.initializer_range)
        #Bert Block
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
            self.log_softmax = nn.Softmax(axis=-1)
        else:
            self.log_softmax = P.LogSoftmax(axis=-1)
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dropout = nn.Dropout(1.0 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.origin_shape = (-1, config.seq_length, self.num_labels)
        self.bert_dense1 = nn.Dense(512, 256, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.bert_dense2 = nn.Dense(256, 1, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.bert_dense3 = nn.Dense(2048, 1024, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.bert_dense4 = nn.Dense(1024, 512, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)

        #ResidualBlock
        self.one_hot = nn.OneHot(depth=20, axis=-1)
        self.conv1 = nn.Conv1d(20, 32, kernel_size=3, stride=1, padding=1, has_bias=False, pad_mode="pad")
        self.bn1 = nn.BatchNorm1d(32 * 7 * 3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.residual_block1 = ResidualBlock(32, 64,10,config,self.weight_init)
        self.flatten = nn.Flatten()
        self.res_dense1 = nn.Dense(256, 128, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.relu2 = nn.ReLU()
        self.res_dense2 = nn.Dense(128, 1, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.sigmoid = nn.Sigmoid()

        #ConcatBlock
        self.concat_dense5 = nn.Dense(832, 128, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.concat_dense6 = nn.Dense(128, 2, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)

    def construct(self, input_ids, input_mask, token_type_id,pssms,dssps,window_seq):
        #BertBlock
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        bert_seq = self.dropout(sequence_output)
        bert_seq = self.bert_dense1(bert_seq)
        bert_seq = P.Flatten()(self.bert_dense2(bert_seq))
        bert_seq = self.bert_dense3(bert_seq)
        bert_seq = self.dropout(bert_seq)
        bert_seq = self.bert_dense4(bert_seq)
        bert_seq = self.dropout(bert_seq)

        #ResidualBlock
        dssps = P.Pad(((0, 0), (0, 0), (0, 10)))(dssps)
        window_seq=self.one_hot(window_seq)
        feature = P.Concat(axis=1)([window_seq, pssms, dssps])
        feature = P.Transpose()(feature, (0, 2, 1))
        feature_1 = self.conv1(feature)
        feature_1 = self.dropout(feature_1)
        feature_2 = P.Reshape()(feature_1, (feature_1.shape[0], -1))
        feature_2 = self.bn1(feature_2)
        feature_3 = P.Reshape()(feature_2, (feature_1.shape[0], feature_1.shape[1], feature_1.shape[2]))
        feature_3 = self.relu1(feature_3)
        feature_4 = self.maxpool1(feature_3)
        feature_4 = self.dropout(feature_4)
        feature_5 = self.residual_block1(feature_4)
        feature_5 = self.dropout(feature_5)
        feature_6 = P.Reshape()(feature_5, (feature_1.shape[0], -1))

        #ConcatBlock
        concat_feature = P.Concat(axis=-1)([feature_6, P.Cast()(bert_seq,mindspore.float32)])
        concat_feature = self.concat_dense5(concat_feature)
        concat_feature = self.concat_dense6(concat_feature)
        logits = self.cast(concat_feature, self.dtype)
        logits=P.Softmax()(logits)
        return logits

# class BertSEQPDModel(nn.Cell):
#     """
#     This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
#     The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
#     """
#
#     def __init__(self, config, is_training, num_labels=11, with_lstm=False,
#                  dropout_prob=0.0, use_one_hot_embeddings=False,lstm_cfg=None):
#         super(BertSEQPDModel, self).__init__()
#         if not is_training:
#             config.hidden_dropout_prob = 0.0
#             config.hidden_probs_dropout_prob = 0.0
#             self.log_softmax=nn.Softmax(axis=-1)
#         else:
#             self.log_softmax = P.LogSoftmax(axis=-1)
#         self.bert = BertModel(config, is_training, use_one_hot_embeddings)
#         self.cast = P.Cast()
#         self.weight_init = TruncatedNormal(config.initializer_range)
#         self.dtype = config.dtype
#         self.num_labels = num_labels
#         self.dense_concat = nn.Dense(2*lstm_cfg.concat_hidden_size, self.num_labels, weight_init=self.weight_init,
#                                 has_bias=True).to_float(config.compute_type)
#
#         self.lstm_PSSM = nn.LSTM(
#             input_size=20,
#             hidden_size=lstm_cfg.pssm_hidden_size,
#             num_layers=lstm_cfg.pssm_layer_num,
#             batch_first=True,
#             bidirectional=True)
#
#         self.lstm_DSSP = nn.LSTM(
#             input_size=10,
#             hidden_size=lstm_cfg.pssm_hidden_size,
#             num_layers=lstm_cfg.dssp_layer_num,
#             batch_first=True,
#             bidirectional=True)
#
#         self.lstm_concat = nn.LSTM(
#             input_size=config.hidden_size+2*lstm_cfg.pssm_hidden_size+2*lstm_cfg.dssp_hidden_size,
#             hidden_size=lstm_cfg.concat_hidden_size,
#             num_layers=lstm_cfg.concat_layer_num,
#             batch_first=True,
#             bidirectional=True)
#
#         self.dropout = nn.Dropout(1 - dropout_prob)
#         self.reshape = P.Reshape()
#         self.shape = (-1, 2*lstm_cfg.concat_hidden_size)
#         self.origin_shape = (-1, config.seq_length, self.num_labels)
#         self.concat=C(axis=-1)
#
#     def construct(self, input_ids, input_mask, token_type_id,pssms,dssps):
#         """Return the final logits as the results of log_softmax."""
#         sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
#         aa_seq = self.dropout(sequence_output)
#
#         pssm_seq, _ = self.lstm_PSSM(pssms)
#         dssp_seq, _ = self.lstm_DSSP(dssps)
#
#         pssm_seq = self.dropout(pssm_seq)
#         dssp_seq = self.dropout(dssp_seq)
#
#         concat_feature=self.concat([aa_seq,pssm_seq,dssp_seq])
#         concat_seq,_=self.lstm_concat(concat_feature)
#         concat_seq = self.dropout(concat_seq)
#
#         concat_feature = self.reshape(concat_seq, self.shape)
#
#         logits = self.dense_concat(concat_feature)
#         logits = self.cast(logits, self.dtype)
#         return_value = self.log_softmax(logits)
#         return return_value
#
# class BertSEQAtModel(nn.Cell):
#     """
#     This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
#     The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
#     """
#
#     def __init__(self, config, is_training, num_labels=11, with_lstm=False,
#                  dropout_prob=0.0, use_one_hot_embeddings=False,lstm_cfg=None):
#         super(BertSEQAtModel, self).__init__()
#         if not is_training:
#             config.hidden_dropout_prob = 0.0
#             config.hidden_probs_dropout_prob = 0.0
#             self.log_softmax=nn.Softmax(axis=-1)
#         else:
#             self.log_softmax = P.LogSoftmax(axis=-1)
#         self.bert = BertModel(config, is_training, use_one_hot_embeddings)
#         self.cast = P.Cast()
#         self.weight_init = TruncatedNormal(config.initializer_range)
#         self.dtype = config.dtype
#         self.num_labels = num_labels
#
#         self.lstm_PSSM = nn.LSTM(
#             input_size=20,
#             hidden_size=lstm_cfg.pssm_hidden_size,
#             num_layers=lstm_cfg.pssm_layer_num,
#             batch_first=True,
#             bidirectional=True)
#
#         self.lstm_DSSP = nn.LSTM(
#             input_size=10,
#             hidden_size=lstm_cfg.pssm_hidden_size,
#             num_layers=lstm_cfg.dssp_layer_num,
#             batch_first=True,
#             bidirectional=True)
#
#         self.attention_hidden_size=config.hidden_size+lstm_cfg.pssm_hidden_size*2+lstm_cfg.pssm_hidden_size*2
#
#         self._attention_mask = CreateAttentionMaskFromInputMask(config)
#
#         self.attention_concat = BertSelfAttention(
#             hidden_size=self.attention_hidden_size,
#             num_attention_heads=4)
#
#         self.dense_concat = nn.Dense(self.attention_hidden_size, self.num_labels, weight_init=self.weight_init,
#                                 has_bias=True).to_float(config.compute_type)
#
#         self.dropout = nn.Dropout(1 - dropout_prob)
#         self.reshape = P.Reshape()
#         self.shape = (-1, self.attention_hidden_size)
#         self.origin_shape = (-1, config.seq_length, self.num_labels)
#         self.concat=C(axis=-1)
#
#     def construct(self, input_ids, input_mask, token_type_id,pssms,dssps):
#         """Return the final logits as the results of log_softmax."""
#         sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
#         aa_seq = self.dropout(sequence_output)
#
#         pssm_seq, _ = self.lstm_PSSM(pssms)
#         dssp_seq, _ = self.lstm_DSSP(dssps)
#
#         pssm_seq = self.dropout(pssm_seq)
#         dssp_seq = self.dropout(dssp_seq)
#
#         attention_mask = self._attention_mask(input_mask)
#
#         concat_feature=self.concat([aa_seq,pssm_seq,dssp_seq])
#
#         concat_seq=self.attention(concat_feature,attention_mask)
#         concat_seq = self.dropout(concat_seq)
#
#         concat_seq = self.reshape(concat_seq, self.shape)
#
#         logits = self.dense_concat(concat_seq)
#         logits = self.cast(logits, self.dtype)
#         return_value = self.log_softmax(logits)
#         return return_value
#
# class CNNTagger(nn.Cell):
#     def __init__(self, input_dim,  n_filters, filter_sizes, dropout):
#         super(CNNTagger, self).__init__()
#         self.convs = nn.CellList([
#             nn.Conv1d(in_channels=input_dim,
#                       out_channels=n_filters,
#                       kernel_size=fs)
#             for fs in filter_sizes
#         ])
#         self.dropout = nn.Dropout(dropout)
#         self.relu = ops.ReLU()
#         self.max_pool = nn.MaxPool1d(kernel_size=5)
#
#     def construct(self, inputs):
#         embedded =  P.Transpose()(inputs, (0, 2, 1))
#         all_encoder_layers=()
#         for layer_module in self.convs:
#             layer_output=self.relu(layer_module(embedded))
#             embedded=layer_output
#             all_encoder_layers = all_encoder_layers + (layer_output,)
#
#         pooled = [self.max_pool(conv) for conv in all_encoder_layers]
#         cat = self.dropout(ops.Concat()(pooled, 1))
#         return cat
#
# class BertSEQCnnAtModel(nn.Cell):
#     """
#     This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
#     The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
#     """
#
#     def __init__(self, config, is_training, num_labels=11, with_lstm=False,
#                  dropout_prob=0.0, use_one_hot_embeddings=False,lstm_cfg=None):
#         super(BertSEQCnnAtModel, self).__init__()
#         if not is_training:
#             config.hidden_dropout_prob = 0.0
#             config.hidden_probs_dropout_prob = 0.0
#             self.log_softmax=nn.Softmax(axis=-1)
#         else:
#             self.log_softmax = P.LogSoftmax(axis=-1)
#         self.bert = BertModel(config, is_training, use_one_hot_embeddings)
#         self.cast = P.Cast()
#         self.weight_init = TruncatedNormal(config.initializer_range)
#         self.dtype = config.dtype
#         self.num_labels = num_labels
#         self.dense_concat = nn.Dense(config.hidden_size+lstm_cfg.pssm_hidden_size*2+lstm_cfg.pssm_hidden_size*2, self.num_labels, weight_init=self.weight_init,
#                                 has_bias=True).to_float(config.compute_type)
#
#
#
#         self.dropout = nn.Dropout(1 - dropout_prob)
#         self.reshape = P.Reshape()
#         self.shape = (-1, 512)
#         self.origin_shape = (-1, config.seq_length, self.num_labels)
#         self.concat=C(axis=-1)
#
#         self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)
#
#         self.attention = BertSelfAttention(
#             hidden_size=config.hidden_size+lstm_cfg.pssm_hidden_size*2+lstm_cfg.pssm_hidden_size*2,
#             num_attention_heads=4)
#
#         self.pssm_cnn=CNNTagger(input_dim=20,n_filters=20,filter_sizes=[3,4,5],dropout=1 - dropout_prob)
#         self.dssp_cnn=CNNTagger(10,10,[3,4,5],1 - dropout_prob)
#
#
#
#     def construct(self, input_ids, input_mask, token_type_id,pssms,dssps):
#         """Return the final logits as the results of log_softmax."""
#         sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
#         aa_seq = self.dropout(sequence_output)
#
#         attention_mask = self._create_attention_mask_from_input_mask(input_mask)
#
#         pssm_seq = self.pssm_cnn(pssms)
#         dssp_seq = self.dssp_cnn(dssps)
#         #
#         # concat_feature=self.concat([aa_seq,pssm_seq,dssp_seq])
#         # concat_seq=self.attention(concat_feature,attention_mask)
#         # concat_seq = self.dropout(concat_seq)
#         # #
#         # concat_feature = self.reshape(concat_seq, self.shape)
#         # #
#         # logits = self.dense_concat(concat_feature)
#         # logits = self.cast(logits, self.dtype)
#         # return_value = self.log_softmax(logits)
#         return pssm_seq,dssp_seq
#
# class BertLSTMModel(nn.Cell):
#     """
#     This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
#     LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
#     logits as the results of log_softmax is proportional to that of softmax.
#     """
#
#     def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
#                  assessment_method=""):
#         super(BertLSTMModel, self).__init__()
#         if not is_training:
#             config.hidden_dropout_prob = 0.0
#             config.hidden_probs_dropout_prob = 0.0
#         self.bert = BertModel(config, is_training, use_one_hot_embeddings)
#         self.cast = P.Cast()
#         self.weight_init = TruncatedNormal(config.initializer_range)
#         self.log_softmax = P.LogSoftmax(axis=-1)
#         self.dtype = config.dtype
#         self.num_labels = num_labels
#         self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
#                                 has_bias=True).to_float(config.compute_type)
#         self.dropout = nn.Dropout(1 - dropout_prob)
#
#         self.encoder = nn.LSTM(input_size=512,
#                                hidden_size=128,
#                                num_layers=2,
#                                has_bias=True,
#                                bidirectional=True,
#                                dropout=0.0)
#         self.concat = P.Concat(1)
#         self.squeeze = P.Squeeze(axis=0)
#         self.decoder = nn.Dense(128 *4, 2,activation="softmax")
#         self.trans = P.Transpose()
#         self.perm = (1, 0, 2)
#
#         self.assessment_method = assessment_method
#
#     def construct(self, input_ids, input_mask, token_type_id):
#         sequence_output, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
#         embeddings = self.trans(sequence_output, self.perm)
#         output, _ = self.encoder(embeddings)
#         encoding = self.concat((self.squeeze(output[0:1:1]), self.squeeze(output[-2:-1:1])))
#         logits = self.decoder(encoding)
#         if self.assessment_method != "spearman_correlation":
#             logits = self.log_softmax(logits)
#         return logits
#
# class BertLSTMModelEval(nn.Cell):
#     """
#     This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
#     LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
#     logits as the results of log_softmax is proportional to that of softmax.
#     """
#
#     def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
#                  assessment_method=""):
#         super(BertLSTMModelEval, self).__init__()
#         if not is_training:
#             config.hidden_dropout_prob = 0.0
#             config.hidden_probs_dropout_prob = 0.0
#         self.bert = BertModel(config, is_training, use_one_hot_embeddings)
#         self.cast = P.Cast()
#         self.weight_init = TruncatedNormal(config.initializer_range)
#         self.log_softmax = P.LogSoftmax(axis=-1)
#         self.dtype = config.dtype
#         self.num_labels = num_labels
#         self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
#                                 has_bias=True).to_float(config.compute_type)
#         self.dropout = nn.Dropout(1 - dropout_prob)
#
#         self.encoder = nn.LSTM(input_size=config.hidden_size,
#                                hidden_size=128,
#                                num_layers=2,
#                                has_bias=True,
#                                bidirectional=True,
#                                dropout=0.0)
#         self.concat = P.Concat(1)
#         self.squeeze = P.Squeeze(axis=0)
#         self.decoder = nn.Dense(128 *4, 2,activation="softmax")
#         self.trans = P.Transpose()
#         self.perm = (1, 0, 2)
#
#         self.assessment_method = assessment_method
#
#     def construct(self, input_ids, input_mask, token_type_id):
#         sequence_output, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
#         embeddings = self.trans(sequence_output, self.perm)
#         output, _ = self.encoder(embeddings)
#         encoding = self.concat((self.squeeze(output[0:1:1]), self.squeeze(output[-2:-1:1])))
#         logits = self.decoder(encoding)
#         if self.assessment_method != "spearman_correlation":
#             logits = self.log_softmax(logits)
#         return logits,sequence_output, pooled_output