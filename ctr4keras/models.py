from typing import Union, List
from collections import OrderedDict
import math

import keras.models
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, models, layers, initializers, regularizers

from ctr4keras import FeatureType, FeatureAttributes
from ctr4keras.layers import get_inputs, LinearEmbedding, DenseEmbedding, SequenceEmbedding,  DNN, FMCross, \
    AFMCross, DCNCross, GlobalSumPooling, KMaxPooling, BiInteractionPooling, MultiHeadAttention,\
    MultiHeadInsterestAttention, Transformer
from ctr4keras.metrics import idcg, rank_evaluate


def LR(features, regularizer=1e-3):
    inputs_dict = get_inputs(features=features)
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]

    outputs = LinearEmbedding(
        features=features,
        activation="sigmoid",
        name="outputs",
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    return models.Model(list(inputs_dict.values()), outputs)


def Deep(features, dense_emb_dim=4, regularizer=1e-3):
    inputs_dict = get_inputs(features=features)
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]

    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    flatten = layers.Flatten(name="dense_flatten")(dense_embedding)

    outputs = DNN(output_dim=1, hidden_dims=[], name="outputs", regularizer=regularizer)(flatten)

    return models.Model(list(inputs_dict.values()), dense_embedding)


def DWL(features, dense_emb_dim=8, dense_hidden_dims=[128, 64, 32], regularizer=1e-3):
    inputs_dict = get_inputs(features=features)
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]

    lr_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    flatten = layers.Flatten(name="dense_flatten")(dense_embedding)
    dnn_output = DNN(output_dim=1, hidden_dims=dense_hidden_dims, regularizer=regularizer)(flatten)

    outputs = layers.Concatenate(axis=1, name="last_concat")([lr_output, dnn_output])
    outputs = layers.Dense(units=1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)


def FM(features, dense_emb_dim=8, regularizer=1e-3):
    inputs_dict = get_inputs(features=features)
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]

    # fm 1st order output
    linear_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=use_type
    )(inputs=inputs_dict)

    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=use_type
    )(inputs=inputs_dict)

    # fm 2nd order output
    cross_output = FMCross(name="cross")(dense_embedding)

    outputs = layers.Concatenate(axis=1, name="last_concat")([linear_output, cross_output])
    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)


def DeepFM(features, dense_emb_dim=8, dense_hidden_dims=[128, 64, 32], regularizer=1e-3):
    inputs_dict = get_inputs(features=features)
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]

    # fm 1st order output
    linear_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=use_type
    )(inputs=inputs_dict)

    # fm 2nd order output
    cross_output = FMCross(name="cross")(dense_embedding)

    # dnn output
    flatten = layers.Flatten(name="dense_flatten")(dense_embedding)
    dnn_output = DNN(output_dim=1, hidden_dims=dense_hidden_dims, regularizer=regularizer)(flatten)

    outputs = layers.Concatenate(axis=1, name="last_concat")([linear_output, cross_output, dnn_output])
    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)


def _DCN(features, structure="parallel", cross_mode="vector", cross_layer_num=3,
         dense_emb_dim=8, dense_hidden_dims=[128, 64, 32], regularizer=1e-3):
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]
    # TODO: MoE + uv decomposition

    inputs_dict = get_inputs(features=features)

    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    flatten = layers.Flatten(name="dense_flatten")(dense_embedding)

    cross = DCNCross(layer_num=cross_layer_num, mode=cross_mode, regularizer=regularizer)(flatten)
    if structure == "parallel":
        dnn = DNN(output_dim=dense_hidden_dims[-1], hidden_dims=dense_hidden_dims[:-1],
                  regularizer=regularizer)(flatten)
        stack = layers.Concatenate(axis=1, name="last_concat")([cross, dnn])
    elif structure == "stacked":
        stack = DNN(output_dim=dense_hidden_dims[-1], hidden_dims=dense_hidden_dims[:-1],
                    regularizer=regularizer)(cross)
    else:
        raise ValueError(f"`structure` should be `parallel` or `stacked`, but curr value is {structure}.")

    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(stack)

    return models.Model(list(inputs_dict.values()), outputs)


def DCN(features, cross_layer_num=3, dense_emb_dim=8, dense_hidden_dims=[128, 64, 32], regularizer=1e-3):
    return _DCN(features=features, structure="parallel", cross_mode="vector", cross_layer_num=cross_layer_num,
                dense_emb_dim=dense_emb_dim, dense_hidden_dims=dense_hidden_dims, regularizer=regularizer)


def DCNv2(features, structure="parallel", cross_mode="vector", cross_layer_num=3,
          dense_emb_dim=8, dense_hidden_dims=[128, 64, 32]):
    raise NotImplementedError("To be continue...")


def FNN(features, dense_emb_dim=8, dense_hidden_dims=[128, 64, 32], regularizer=1e-3):
    """
        FIXME: pseudo fnn, calc embedding replacing the loaded embedding
    """
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]
    inputs_dict = get_inputs(features=features)
    linear_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=use_type
    )(inputs=inputs_dict)

    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=use_type
    )(inputs=inputs_dict)

    flatten = layers.Flatten(name="dense_flatten")(dense_embedding)
    dnn_output = DNN(output_dim=1, hidden_dims=dense_hidden_dims, regularizer=regularizer)(flatten)

    outputs = layers.Concatenate(axis=1, name="last_concat")([linear_output, dnn_output])
    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)


def CCPM(features, dense_emb_dim=8, conv_filters=[10, 10], conv_kernel_width=5, dense_hidden_dims=[128, 128],
         regularizer=1e-3):
    if conv_kernel_width > len(features.keys()):
        raise ValueError("number of features must bigger than convolution kernel width.")
    inputs_dict = get_inputs(features=features)
    use_type = [
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]

    linear_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    # ccpm convolutional layer just use categoirical features
    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        use_type=use_type,
        concat=True,
        regularizer=regularizer
    )(inputs=inputs_dict)  # batch, ftr_num, ftr_emb

    conv_result = tf.expand_dims(dense_embedding, axis=3)  # batch, ftr_num, ftr_emb, 1  -> for conv input

    n_ftr = int(dense_embedding.shape[1])
    l = len(conv_filters)
    for i, filter in enumerate(conv_filters):
        # flexible p-max pooling
        k = max(1, int((1 - pow((i + 1) / l, l - i - 1)) * n_ftr)) if i < l - 1 else 3

        conv_result = layers.Conv2D(filters=filter, kernel_size=(conv_kernel_width, 1), strides=(1, 1), padding='same',
                                    activation='tanh', use_bias=True)(conv_result)  # batch, ftr_num, ftr_emb, filter
        conv_result = KMaxPooling(k=min(k, int(conv_result.shape[1])), axis=1)(conv_result)  # batch, k, ftr_emb, filter

    flatten = layers.Flatten(name="dense_flatten")(conv_result)
    dnn_output = DNN(output_dim=1, hidden_dims=dense_hidden_dims, regularizer=regularizer)(flatten)

    outputs = layers.Concatenate(axis=1, name="last_concat")([linear_output, dnn_output])
    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)


def NFM(features, dense_emb_dim=8, dense_hidden_dims=[128, 128], regularizer=1e-3):
    inputs_dict = get_inputs(features=features)

    linear_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=[
            FeatureType.Continuous.name,
            FeatureType.Ordered.name,
            FeatureType.Categorical.name,
            FeatureType.MultiCategorical.name
        ]
    )(inputs=inputs_dict)

    cont_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        use_type=[
            FeatureType.Continuous.name,
            FeatureType.Ordered.name
        ],
        regularizer=regularizer
    )(inputs=inputs_dict)  # batch, ftr_num, ftr_emb

    cate_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        use_type=[
            FeatureType.Categorical.name,
            FeatureType.MultiCategorical.name,
        ],
        regularizer=regularizer
    )(inputs=inputs_dict)  # batch, ftr_num, ftr_emb

    bi_pooling = BiInteractionPooling(name="bi_interaction_pooling")(cate_embedding)  # batch, 1, ftr_emb
    dnn_input = layers.Concatenate(axis=1, name="bi_cont_concat")(
        [bi_pooling, cont_embedding])  # batch, cont_ftr_num+1, ftr_emb
    dnn_input = layers.Flatten(name="dense_flatten")(dnn_input)  # batch, (cont_ftr_num+1)*ftr_emb

    dnn_output = DNN(output_dim=1, hidden_dims=dense_hidden_dims, regularizer=regularizer)(dnn_input)

    outputs = layers.Concatenate(axis=1, name="last_concat")([linear_output, dnn_output])
    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)


def AFM(features, dense_emb_dim=4, atten_dim=5, regularizer=1e-3):
    inputs_dict = get_inputs(features=features)
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]

    linear_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    atten_output = AFMCross(atten_dim=atten_dim, name="attention", regularizer=regularizer)(dense_embedding)

    outputs = layers.Concatenate(axis=1, name="last_concat")([linear_output, atten_output])
    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)


def DeepAFM(features, dense_emb_dim=4, dense_hidden_dims=[128, 128], atten_dim=5, regularizer=1e-3):
    inputs_dict = get_inputs(features=features)
    use_type = [
        FeatureType.Continuous.name,
        FeatureType.Ordered.name,
        FeatureType.Categorical.name,
        FeatureType.MultiCategorical.name
    ]

    linear_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    dense_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=use_type,
    )(inputs=inputs_dict)

    atten_output = AFMCross(atten_dim=atten_dim, name="attention", regularizer=regularizer)(dense_embedding)

    flatten = layers.Flatten(name="dense_flatten")(dense_embedding)
    dnn_output = DNN(output_dim=1, hidden_dims=dense_hidden_dims, regularizer=regularizer)(flatten)

    outputs = layers.Concatenate(axis=1, name="last_concat")([linear_output, dnn_output, atten_output])
    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)


def AutoInt(features, dense_emb_dim=4, dense_hidden_dims=[128, 128, 32],
            atten_head_num=4, atten_head_dim=8, atten_layer_num=2, regularizer=1e-3):
    inputs_dict = get_inputs(features=features)

    linear_output = LinearEmbedding(
        features=features,
        activation="sigmoid",
        regularizer=regularizer,
        use_type=[
            FeatureType.Continuous.name,
            FeatureType.Ordered.name,
            FeatureType.Categorical.name,
            FeatureType.MultiCategorical.name
        ],
    )(inputs=inputs_dict)

    cont_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=[
            FeatureType.Continuous.name,
            FeatureType.Ordered.name
        ],
    )(inputs=inputs_dict)

    cate_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=[
            FeatureType.Categorical.name,
            FeatureType.MultiCategorical.name
        ],
    )(inputs=inputs_dict)

    # cont output
    cont_flatten = layers.Flatten(name="cont_flatten")(cont_embedding)
    cont_output = DNN(output_dim=1, hidden_dims=dense_hidden_dims, regularizer=regularizer)(cont_flatten)

    # multi-layer multi-head residual attention
    residual_output = cate_embedding
    for _ in range(atten_layer_num):
        residual_inputs = [residual_output] * 3
        atten_output = MultiHeadAttention(
            head_num=atten_head_num, head_dim=atten_head_dim)(residual_inputs)  # batch, seq, heads*head_size

        residual_output = layers.Concatenate(axis=-1)([atten_output, cate_embedding])

    multi_atten_flatten = layers.Flatten(name="atten_flatten")(residual_output)
    multi_atten_output = DNN(output_dim=1, hidden_dims=dense_hidden_dims, regularizer=regularizer)(multi_atten_flatten)

    outputs = layers.Concatenate(axis=1, name="last_concat")([linear_output, cont_output, multi_atten_output])
    outputs = layers.Dense(1, activation='sigmoid', name="outputs")(outputs)

    return models.Model(list(inputs_dict.values()), outputs)

def DIN(features, dense_emb_dim=4, dense_hidden_dims=[128, 64, 32], seq_emb_dim=8,
        atten_head_num=2, atten_head_dim=8, regularizer=1e-5):
    inputs_dict = get_inputs(features=features)

    cate_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=[
            FeatureType.Categorical.name,
            FeatureType.MultiCategorical.name
        ],
    )(inputs=inputs_dict)

    cate_flatten = layers.Flatten(name="cate_flatten")(cate_embedding)

    cont_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=[
            FeatureType.Continuous.name,
            FeatureType.Ordered.name
        ],
    )(inputs=inputs_dict)

    cont_flatten = layers.Flatten(name="cont_flatten")(cont_embedding)

    # interest interactive
    attentions = list()
    for f in features.values():
        if f.target_sequence is not None:
            seq_input = inputs_dict[f.target_sequence]
            curr_input = inputs_dict[f.name]
            seq_len = int(seq_input.shape[1])

            curr_emb = DenseEmbedding(
                features={f.name: f},
                units=seq_emb_dim,
                regularizer=regularizer,
            )({f.name: curr_input}) # batch, 1, emb
            curr_emb = tf.tile(curr_emb, [1, seq_len, 1])  # batch, seq_len, emb

            seq_emb = SequenceEmbedding(
                feature=features[f.target_sequence],
                units=seq_emb_dim,
                emb_type=None
            )(seq_input)

            attention = MultiHeadInsterestAttention(
                head_num=atten_head_num,
                head_dim=atten_head_dim,
            )(inputs=[curr_emb, seq_emb, seq_emb])

            attentions.append(attention)

    attent_poolings = [GlobalSumPooling(axis=1)(a) for a in attentions]

    last_flatten = layers.Concatenate(axis=-1)(attent_poolings + [cate_flatten, cont_flatten])

    output = DNN(output_dim=1, hidden_dims=dense_hidden_dims,
                 activation="sigmoid", hidden_activation=layers.PReLU(),
                 regularizer=regularizer, name="outputs")(last_flatten)

    return models.Model(list(inputs_dict.values()), output)


def BST(features, dense_emb_dim=4, dense_hidden_dims=[128, 64, 32], seq_emb_dim=8,
        atten_head_num=2, atten_head_dim=8, atten_layer_num=2, regularizer=1e-5):
    inputs_dict = get_inputs(features=features)

    cate_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=[
            FeatureType.Categorical.name,
            FeatureType.MultiCategorical.name
        ],
    )(inputs=inputs_dict)

    cate_flatten = layers.Flatten(name="cate_flatten")(cate_embedding)

    cont_embedding = DenseEmbedding(
        features=features,
        units=dense_emb_dim,
        regularizer=regularizer,
        use_type=[
            FeatureType.Continuous.name,
            FeatureType.Ordered.name
        ],
    )(inputs=inputs_dict)

    cont_flatten = layers.Flatten(name="cont_flatten")(cont_embedding)

    # interest interactive
    transformers = list()
    for f in features.values():
        if f.target_sequence is not None:
            seq_input = inputs_dict[f.target_sequence]
            curr_input = inputs_dict[f.name]
            seq_len = int(seq_input.shape[1])

            curr_emb = DenseEmbedding(
                features={f.name: f},
                units=seq_emb_dim,
                regularizer=regularizer,
            )({f.name: curr_input})  # batch, 1, emb

            seq_emb = SequenceEmbedding(
                feature=features[f.target_sequence],
                units=seq_emb_dim,
                emb_type=None
            )(seq_input)  # batch, seq_len, emb

            x = tf.concat([seq_emb, curr_emb], axis=1)  # batch, seq_len+1, emb

            transformer = Transformer(
                layer_num=atten_layer_num,
                head_num=atten_head_num,
                head_dim=atten_head_dim,
            )(inputs=x)

            transformers.append(transformer)

    trans_poolings = [GlobalSumPooling(axis=1)(a) for a in transformers]

    last_flatten = layers.Concatenate(axis=-1)(trans_poolings + [cate_flatten, cont_flatten])

    output = DNN(output_dim=1, hidden_dims=dense_hidden_dims,
                 activation="sigmoid", hidden_activation=layers.LeakyReLU(),
                 regularizer=regularizer, name="outputs")(last_flatten)

    return models.Model(list(inputs_dict.values()), output)


class Ranker:
    def __init__(self, module=None, features=None, mode='rank', **kwargs):
        self.module = module
        self.features = features or dict()
        self.inputs_dict1 = get_inputs(features=features)
        self.inputs_dict2 = get_inputs(features=features, subffix="a")
        self.mode = mode
        if module is not None:
            self.model = self._build_model(**kwargs)

    def _build_model(self, **kwargs):

        self.module_ins = self.module(self.features, **kwargs)
        s1 = self.module_ins(self.inputs_dict1)
        s2 = self.module_ins(self.inputs_dict2)

        subtracted = layers.Subtract(name='Subtract_layer')([s1, s2])
        out = layers.Activation('sigmoid', name='Activation_layer')(subtracted)

        model = models.Model(
            inputs=list(self.inputs_dict1.values()) + list(self.inputs_dict2.values()),
            outputs=out
        )
        self.pred_model = models.Model(
            inputs=list(self.inputs_dict1.values()),
            outputs=s1
        )
        return model

    def calc_gradient(self, si, sj, i, j, idcg):
        origin = ((2 ** si) - 1) / math.log2(i + 2) + ((2 ** sj) - 1) / math.log2(j + 2)
        curr = ((2 ** si) - 1) / math.log2(j + 2) + ((2 ** sj) - 1) / math.log2(i + 2)
        return abs(curr - origin) / (idcg + 1e-8)

    def calc_steps(self, y, group, batch_size):
        acc_group = [0]
        for g in group:
            acc_group.append(acc_group[-1] + g)
        count = 0
        for start, end in zip(acc_group[: -1], acc_group[1:]):
            y_slice = y[start: end].tolist()
            for i in range(end - start):
                for j in range(end - start):
                    if y_slice[i] > y_slice[j]:
                        count += 1
        return count // batch_size

    def data_generator(self, x, y, group, eval_at=None, batch_size=256):
        x1, x2, label, weight = list(), list(), list(), list()

        acc_group = [0]
        for g in group:
            acc_group.append(acc_group[-1] + g)

        while True:
            for start, end in zip(acc_group[: -1], acc_group[1:]):
                x_slice = {k: v[start: end] for k, v in x.items()}
                y_slice = y[start: end].tolist()
                for i in range(end - start):
                    for j in range(end - start):
                        if y_slice[i] <= y_slice[j]:
                            continue

                        if self.mode == 'rank':
                            w = 1
                        elif self.mode == 'lambda':
                            w = self.calc_gradient(y_slice[i], y_slice[j], i, j, idcg(y_slice, eval_at))
                        else:
                            w = 1

                        if (i + j) % 2 == 0:
                            x1.append({k: v[i: i + 1] for k, v in x_slice.items()})
                            x2.append({k: v[j: j + 1] for k, v in x_slice.items()})
                            label.append(1)
                            weight.append(w)
                        else:
                            x1.append({k: v[j: j + 1] for k, v in x_slice.items()})
                            x2.append({k: v[i: i + 1] for k, v in x_slice.items()})
                            label.append(0)
                            weight.append(w)

                        if len(x1) == batch_size:
                            x1 = {k: np.concatenate([d[k] for d in x1]) for k in x1[0].keys()}
                            x2 = {f'{k}a': np.concatenate([d[k] for d in x2]) for k in x2[0].keys()}
                            ret = {**x1, **x2}
                            label = np.array(label)
                            weight = np.array(weight)
                            yield ret, label, weight
                            x1, x2, label, weight = list(), list(), list(), list()

    def fit(self, x, y, group, batch_size=None, epochs=1, verbose=1,
            eval_x=None, eval_y=None, eval_group=None, eval_at=None, callbacks=None):
        generator = self.data_generator(x, y, group, eval_at, batch_size=batch_size)

        steps_per_epoch = self.calc_steps(y, group, batch_size)
        if eval_x is not None and eval_y is not None and eval_group is not None:
            valid_generator = self.data_generator(eval_x, eval_y, eval_group, eval_at, batch_size)
            validation_steps = self.calc_steps(eval_y, eval_group, batch_size)
        else:
            valid_generator = None
            validation_steps = None
        self.model.fit_generator(
            generator,
            epochs=epochs,
            verbose=verbose,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        if eval_x is not None and eval_y is not None and eval_group is not None:
            self.evaluate(eval_x, eval_y, eval_group, eval_at)

    def predict(self, x, batch_size=256):
        ranker_output = self.pred_model
        return ranker_output.predict(x, batch_size=batch_size)

    def evaluate(self, x, y, group, eval_at=None):
        eval_at = eval_at or [1, 3, 5, 10]
        y_pred = self.predict(x)
        rank_evaluate(y_pred, y, group, eval_at)

    def compile(self, loss='binary_crossentropy', optimizer='adam', metrics=None):
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )
        return self

    def summary(self):
        self.model.summary()

    def save(self, path):
        self.pred_model.save(path)
        return self

    @staticmethod
    def load(path):
        self = Ranker()
        self.pred_model = keras.models.load_model(path)
        return self


class LambdaRanker(Ranker):
    def __init__(self, module, features, **kwargs):
        super(LambdaRanker, self).__init__(module, features, mode='lambda', **kwargs)




