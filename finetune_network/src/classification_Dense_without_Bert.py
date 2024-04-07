import os
import argparse

import mindspore.dataset as ds
from mindspore import  Model
from mindspore.nn import  Recall, F1
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Accuracy
from tqdm import tqdm
from mindspore.train.callback import LossMonitor
import numpy as np
from mindspore import  context
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore as ms
from mindspore.dataset import transforms
import os
import math
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from sklearn import metrics

def generate_dataset_train():
    onehot_op = C.OneHot(num_classes=2)
    type_cast_op = C.TypeCast(mstype.float32)

    train_set = ds.MindDataset(os.path.join(args.input_file,"train.mindrecord"),
                              columns_list=["input_ids","label_ids"])
    train_set_1=train_set.batch(32, False)
    train_set_2=train_set_1.map(operations=[onehot_op,type_cast_op], input_columns=["label_ids"])
    # train_set_3=train_set_2.map(operations=type_cast_op, input_columns=["input_ids"])

    return train_set_2

def generate_dataset_test():
    onehot_op = C.OneHot(num_classes=2)
    type_cast_op = C.TypeCast(mstype.float32)

    test_set = ds.MindDataset(os.path.join(args.input_file,"predict.mindrecord"),
                              columns_list=["input_ids","label_ids"])
    test_set_1=test_set.batch(1, False)
    test_set_2=test_set_1.map(operations=[onehot_op,type_cast_op], input_columns=["label_ids"])
    # test_set_3=test_set_2.map(operations=type_cast_op, input_columns=["input_ids"])
    return test_set_2

class DenseNet(nn.Cell):
    """Sentiment network structure."""

    def __init__(self,):
        super(DenseNet, self).__init__()
        # Mapp words to vectors
        self.decoder = nn.Dense(1024, 2)
        self.softmax=nn.Softmax()

    def construct(self, x):

        x = self.decoder(x)
        return self.softmax(x)

class SentimentNet_without_bert(nn.Cell):
    """Sentiment network structure."""

    def __init__(self,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 bidirectional):
        super(SentimentNet_without_bert, self).__init__()
        # Mapp words to vectors
        self.embedding=nn.OneHot(depth=embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               has_bias=True,
                               bidirectional=bidirectional,
                               dropout=0.0)
        #self.dropout = nn.Dropout(0.5)
        self.concat = P.Concat(1)
        self.squeeze = P.Squeeze(axis=0)
        self.decoder = nn.Dense(128 * 4, 2)
        self.trans = P.Transpose()
        self.perm = (1, 0, 2)


    def construct(self, inputs):
        # input：(64,500,300)
        embeddings = self.embedding(inputs)
        # #embeddings = self.dropout(self.embedding(inputs))
        embeddings = self.trans(embeddings, self.perm)
        output, _ = self.encoder(embeddings)
        # states[i] size(64,200)  -> encoding.size(64,400)
        encoding = self.concat((self.squeeze(output[0:1:1]), self.squeeze(output[999:1000:1])))
        decoding = self.decoder(encoding)
        return decoding



def run_train():
    data_train=generate_dataset_train()
    data_test=generate_dataset_test()
    # network = DenseNet()

    network=SentimentNet_without_bert(
                            embed_size=25,
                            num_hiddens=128,
                            num_layers=2,
                            bidirectional=True,)

    loss = nn.BCELoss(reduction='mean')
    opt = nn.optim.Adam(network.trainable_params())
    # opt = nn.Momentum(network.trainable_params(), 0.0001, 0.9)
    loss_cb = LossMonitor()
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        enable_graph_kernel=False,
        device_target="CPU")
    model = Model(network, loss, opt, metrics={'acc': Accuracy()})
    config_ck = CheckpointConfig(save_checkpoint_steps=7800,
                                 keep_checkpoint_max=1)
    # ckpoint_cb = ModelCheckpoint(prefix="Dense", directory=args.output_file, config=config_ck)
    time_cb = TimeMonitor(data_size=data_train.get_dataset_size())
    # model.train(20, data_train, callbacks=[time_cb, loss_cb])

    true_labels=[]
    pred_labels=[]
    for data in tqdm(data_test.create_dict_iterator(num_epochs=1), total=data_test.get_dataset_size()):
        input_data = []
        columns_list = ["input_ids", "label_ids"]
        for i in columns_list:
            input_data.append(data[i])
        # if len(true_labels)==10:
        #     break
        input_ids, label_ids = input_data
        logits= model.predict(input_ids)
        true_labels.append(np.argmax(label_ids.asnumpy()))
        pred_labels.append(np.argmax(logits.asnumpy()[0]))
    print("==============================================================")
    ACC=metrics.accuracy_score(np.array(true_labels),np.array(pred_labels))
    AUC=metrics.roc_auc_score(np.array(true_labels), np.array(pred_labels)[:,1])
    CM=metrics.confusion_matrix(np.array(true_labels),np.argmax(np.array(pred_labels),axis=1))
    F1=metrics.f1_score(np.array(true_labels),np.argmax(np.array(pred_labels),axis=1))
    Recall=metrics.recall_score(np.array(true_labels),np.argmax(np.array(pred_labels),axis=1))

    print("accuracy {:.4f}".format(ACC))
    print("AUC {:.4f}".format(AUC))
    print("F1 {:.4f}".format(F1))
    print("Recall {:.4f}".format(Recall))
    print("Confusion Matrix\n"
          " \t0\t1\n"
          "0\t{}\t{}\n"
          "1\t{}\t{}".format(CM[0][0],CM[0,1],CM[1,0],CM[1][1]))

    print("==============================================================")



def parse_args():
    parser = argparse.ArgumentParser(description="Generate MindRecord for bert")
    parser.add_argument("--input_file", type=str, default="E:/bert/data_for_classification/SignalP/",help="Input raw text file (or comma-separated list of files).")
    # parser.add_argument("--input_file", type=str, default="/data1/bert/Mindspore_bert/datas/bert_CN_data/wiki_processed/AS/wiki_31",help="Input raw text file (or comma-separated list of files).")

    # parser.add_argument("--output_file", type=str, default="E:/bert/data_for_classification/SignalP/",
    #                     help="Output MindRecord file (or comma-separated list of files).")
    args_opt = parser.parse_args()
    return args_opt


if __name__ == '__main__':
    args=parse_args()

    print("START")

    run_train()



