import math
import random
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.python.saved_model import nested_structure_coder
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


class HistoryExt(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(HistoryExt, self).__init__(**kwargs)
        self.records = dict()

    def on_train_begin(self, logs={}):
        self.records = {"batch": dict(), "epoch": dict()}

    def on_batch_end(self, batch, logs={}):
        for n, v in logs.items():
            if n not in self.records["batch"]:
                self.records["batch"][n] = list()
            self.records["batch"][n].append(v)

    def on_epoch_end(self, epoch, logs={}):
        for n, v in logs.items():
            if n not in self.records["epoch"]:
                self.records["epoch"][n] = list()
            self.records["epoch"][n].append(v)

    def get(self, value, mode="batch", fuzzy=True):
        """
        value: str type as model compiler metrics, like `loss`, `auc`, `accuracy` ...
        mode: `batch` or `epoch`, which show the processing results of each batch or epoch in model training
        """
        if fuzzy:
            whs = [i for i in self.records.get(mode, dict()).keys() if i.startswith(value)]
            if len(whs) < 1:
                print(f"value `{value}` not in records, records contains: {self.records.get(mode, dict()).keys()}")
                return list()
            return self.records.get(mode, dict()).get(whs[0], list())
        return self.records.get(mode, dict()).get(value, list())


class Plot:
    def __init__(self, fig_size=(5, 4), xlabel=None, ylabel=None, title=None, fixed="auto", fixed_num=None):
        self.xlabel = xlabel or "x"
        self.ylabel = ylabel or "y"
        self.title = title or "title"
        self.fixed = fixed
        self.fixed_num = fixed_num or 1
        self.label2dpairs = list()
        self.sub_xlabels = list()
        self.sub_ylabels = list()
        self.sub_titles = list()
        self.fig_size = fig_size

    def add_plot(self, label2dpair, sub_xlabel=None, sub_ylabel=None, sub_title=None,
                 hide_sub_title=False, hide_sub_xlabel=False, hide_sub_ylabel=False):
        sub_xlabel = "" if hide_sub_xlabel else sub_xlabel or self.xlabel
        sub_ylabel = "" if hide_sub_ylabel else sub_ylabel or self.ylabel
        sub_title = "" if hide_sub_title else sub_title or self.title
        self.label2dpairs.append(label2dpair)
        self.sub_xlabels.append(sub_xlabel)
        self.sub_ylabels.append(sub_ylabel)
        self.sub_titles.append(sub_title)

    def plot(self, save_path=None):
        if self.fixed == "auto":
            row, column = min(4, len(self.label2dpairs)), math.ceil(len(self.label2dpairs) / 4)
        elif self.fixed == "row":
            row, column = self.fixed_num, math.ceil(len(self.label2dpairs) / self.fixed_num)
        elif self.fixed == "column":
            row, column == math.ceil(len(self.label2dpairs) / self.fixed_num), self.fixed_num
        else:
            raise ValueError(f"`fixed` should be `auto`, `row` or `column`, but given `{self.fixed}`")

        fig, axs = plt.subplots(column, row, figsize=(self.fig_size[0]*row,self.fig_size[1]*column))
        for i, label2dpair in enumerate(self.label2dpairs):
            ax = axs[i]
            for c_i, (l, v) in enumerate(label2dpair.items()):
                if len(v) == 2:
                    x, y = v
                else:
                    x, y = range(len(v)), v
                ax.plot(x, y, 'b', label=l, color=f"C{c_i}")
            ax.set_title(self.sub_titles[i])
            ax.set_xlabel(self.sub_xlabels[i])
            ax.set_ylabel(self.sub_ylabels[i])
            ax.legend()
        fig.tight_layout()
        fig.suptitle(self.title)
        if save_path:
            plt.savefig(save_path)
        plt.show()


def save_model_as_savedmodel(model, path):
    if int(tf.version.VERSION.split(".")[0]) > 1:
        tf.saved_model.save(model, path)
        loaded = tf.saved_model.load(path)
        loaded = loaded.signatures["serving_default"]
        coder = nested_structure_coder.StructureCoder()
        encoder = coder.encode_structure(loaded.structured_input_signature)
        decoder = coder.decode_proto(encoder)
        print("Input:\n", decoder)
        print("Output:\n", loaded.structured_outputs)
    else:
        inputs = {i.name.split(":")[0]: i for i in model.inputs}
        outputs = {'outputs': model.outputs[0]}
        with tf.keras.backend.get_session() as sess:
            tf.saved_model.simple_save(
                sess, path,
                inputs=inputs,
                outputs=outputs
            )
        print("Input:\n", inputs)
        print("Output:\n", outputs)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if int(tf.version.VERSION.split(".")[0]) > 1:
        tf.random.set_seed(seed)
    else:
        tf.compat.v1.set_random_seed(seed)


def train_test_split(df, sort_col=None, group_col=None, ascending=True, test_size=0.2, shuffle=True, random_state=2020):
    if sort_col is not None:
        df = df.sort_values(sort_col, ascending=ascending)
    if group_col is not None:
        sort_col = sort_col or list()
        df = df.sort_values(sort_col + [g for g in group_col if g not in sort_col])
        sample_col = None
        for c in df.columns:
            if c not in sort_col and c not in group_col:
                sample_col = c
                break
        g = df.groupby(group_col, as_index=False, sort=False).count()[sample_col].values
        train_size = int(len(g) * (1 - test_size))
        train_size = sum(g[:train_size])
    else:
        train_size = int(df.shape[0] * (1 - test_size))
    train = df.iloc[:train_size,]
    test = df.iloc[train_size:,]

    print("train", train.shape, "test", test.shape)

    if shuffle:
        train = sklearn.utils.shuffle(train, random_state=random_state)
        test = sklearn.utils.shuffle(test, random_state=random_state)
    return train, test


if __name__ == "__main__":
    pass



