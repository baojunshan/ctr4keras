from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

from . import FeatureType, FeatureAttributes


def get_inputs(features, subffix=""):
    """
    features:
        OrderedDict[str, FeatureAttributes]
    -----------------------------------------------
    return:
        OrderedDict[str, layers.Input]
    """
    inputs_dict = OrderedDict()
    for feat in features.values():
        dtype = tf.int32 if feat.type != FeatureType.Continuous.name else tf.float32
        inputs_dict[feat.name] = layers.Input(shape=(feat.input_size,), name=f"{feat.name}{subffix}", dtype=dtype)
    return inputs_dict


class LinearEmbedding(layers.Layer):
    """
        features:
            OrderedDict[str, FeatureAttributes]
        use_type:
            ["Continuous", "Categorical", "MultiCategorical", "Ordered", "Sequence"]
        activation:
            last activation of the 1 dim output
        prefix:
    """

    def __init__(self,
                 features,
                 use_type=None,
                 activation=None,
                 regularizer=1e-3,
                 prefix="linear_emb",
                 **kwargs):
        super(LinearEmbedding, self).__init__(**kwargs)
        self.features = features
        self.use_type = use_type or ["Continuous", "Categorical", "MultiCategorical", "Ordered", "Sequence"]
        self.activation = activation
        self.regularizer = regularizer
        self.prefix = prefix

    def build(self, input_shape):
        super(LinearEmbedding, self).build(input_shape)
        self.cont_concat = layers.Concatenate(name=f"{self.prefix}_cont_concat")
        self.cont_dense = layers.Dense(1, kernel_regularizer=l2(self.regularizer), name=f"{self.prefix}_cont_dense")
        self.order_concat = layers.Concatenate(name=f"{self.prefix}_order_concat")
        self.order_dense = layers.Dense(1, kernel_regularizer=l2(self.regularizer), name=f"{self.prefix}_order_dense")

        self.emb_dict = dict()
        for f in self.features.values():
            if f.type not in self.use_type:
                continue
            if f.type == FeatureType.Categorical.name:
                self.emb_dict[f.name] = \
                    models.Sequential([
                        layers.Embedding(f.dim, 1, embeddings_regularizer=l2(self.regularizer),
                                         name=f"{self.prefix}_cate_emb_{f.name}"),
                        layers.Reshape((1,), name=f"{self.prefix}_cate_emb_reshape_{f.name}")
                    ])
            elif f.type in (FeatureType.MultiCategorical.name, FeatureType.Sequential.name):
                self.emb_dict[f.name] = \
                    models.Sequential([
                        layers.Embedding(f.dim, 1, embeddings_regularizer=l2(self.regularizer),
                                         name=f"{self.prefix}_multicate_emb_{f.name}"),
                        GlobalAveragePooling(axis=1, name=f"{self.prefix}_multicate_emb_mean_{f.name}"),
                        layers.Reshape((1,), name=f"{self.prefix}_multicate_emb_reshape_{f.name}")
                    ])
            else:
                continue
        self.add = layers.Add(name=f"{self.prefix}_last_add")

        act = self.activation if self.activation is not None else "sigmoid"
        self.activate = layers.Activation(act)

    def call(self, inputs, **kwargs):
        """
        inputs_dict or inputs:
            OrderedDict[str, layers.Input]
        """
        inputs_dict = inputs
        linears = list()
        if FeatureType.Continuous.name in self.use_type:
            cont_input_list = [inputs_dict[f.name] for f in self.features.values() if
                               f.type == FeatureType.Continuous.name]
            if len(cont_input_list) > 0:
                cont_output = self.cont_dense(self.cont_concat(cont_input_list))
                linears.append(cont_output)
        if FeatureType.Ordered.name in self.use_type:
            order_input_list = [inputs_dict[f.name] for f in self.features.values() if
                                f.type == FeatureType.Ordered.name]
            if len(order_input_list) > 0:
                order_output = self.order_dense(self.order_concat(order_input_list))
                linears.append(order_output)

        for f in self.features.values():
            if f.type in self.use_type and f.name in self.emb_dict:
                emb = self.emb_dict[f.name](inputs_dict[f.name])
                linears.append(emb)

        output = self.add(linears)
        if self.activation is not None:
            output = self.activate(output)
        return output

    def compute_output_shape(self, input_shape):
        return (1,)


class DenseEmbedding(layers.Layer):
    def __init__(self,
                 features,
                 units,
                 use_type=None,
                 regularizer=1e-3,
                 dropout=0,
                 prefix="dense_emb",
                 **kwargs):
        super(DenseEmbedding, self).__init__(**kwargs)
        self.features = features
        self.units = units
        self.use_type = use_type or [FeatureType.Continuous.name, FeatureType.Ordered.name,
                                     FeatureType.Categorical.name, FeatureType.MultiCategorical.name,
                                     FeatureType.Sequential.name]
        self.regularizer = regularizer
        self.dropout_rate = dropout
        self.prefix = prefix

    def build(self, input_shape):
        super(DenseEmbedding, self).build(input_shape)
        self.emb_dict = dict()
        self.emb_name = list()
        for f in self.features.values():
            if f.type not in self.use_type:
                continue
            if f.type == FeatureType.Continuous.name:
                self.emb_dict[f.name] = \
                    models.Sequential([
                        layers.Dense(
                            self.units,
                            kernel_regularizer=l2(self.regularizer),
                            name=f"{self.prefix}_cont_emb_{f.name}"
                        ),
                        layers.Reshape(
                            (1, self.units),
                            name=f"{self.prefix}_cont_emb_reshape_{f.name}"
                        )
                    ])
            elif f.type == FeatureType.Ordered.name:
                self.emb_dict[f.name] = \
                    models.Sequential([
                        layers.Dense(
                            self.units,
                            kernel_regularizer=l2(self.regularizer),
                            name=f"{self.prefix}_order_emb_{f.name}"
                        ),
                        layers.Reshape(
                            (1, self.units),
                            name=f"{self.prefix}_order_emb_reshape_{f.name}"
                        )
                    ])
            elif f.type == FeatureType.Categorical.name:
                self.emb_dict[f.name] = \
                    layers.Embedding(
                        f.dim, self.units,
                        embeddings_regularizer=l2(self.regularizer),
                        name=f"{self.prefix}_cate_emb_{f.name}"
                    )
            elif f.type in (FeatureType.MultiCategorical.name, FeatureType.Sequential.name):
                self.emb_dict[f.name] = \
                    models.Sequential([
                        layers.Embedding(
                            f.dim, self.units,
                            embeddings_regularizer=l2(self.regularizer),
                            mask_zero=True,
                            name=f"{self.prefix}_multicate_emb_{f.name}"
                        ),
                        GlobalAveragePooling(
                            axis=1,
                            name=f"{self.prefix}_multicate_emb_mean_{f.name}"
                        ),
                        layers.Reshape(
                            (1, self.units),
                            name=f"{self.prefix}_multicate_emb_reshape_{f.name}"
                        )
                    ])
            self.emb_name.append(f.name)
        self.dropout = layers.Dropout(rate=self.dropout_rate)
        self.concat_layer = layers.Concatenate(axis=1, name=f"{self.prefix}_last_concat")

    @property
    def _embed_names(self):
        return self.emb_name

    def call(self, inputs, **kwargs):
        inputs_dict = inputs

        denses = list()
        for f in self.features.values():
            if f.type in self.use_type and f.name in self.emb_dict:
                emb = self.emb_dict[f.name](inputs_dict[f.name])
                emb = self.dropout(emb)
                denses.append(emb)
        if len(denses) > 1:
            output = self.concat_layer(denses)
        else:
            output = denses[0]
        return output  # batch, ftr, emb

    def compute_output_shape(self, input_shape):
        return len(self.emb_dict.keys()), self.units


class SequenceEmbedding(layers.Layer):
    def __init__(self, feature, units, emb_type=None, regularizer=1e-3, **kwargs):
        super(SequenceEmbedding, self).__init__(**kwargs)
        self.feature = feature
        self.units = units
        self.emb_type = emb_type
        self.regularizer = regularizer

    def build(self, input_shape):
        super(SequenceEmbedding, self).build(input_shape)
        self.embedding = layers.Embedding(
            self.feature.dim, self.units,
            embeddings_regularizer=l2(self.regularizer),
            mask_zero=True,
            name=f"seq_emb_{self.feature.name}"
        )


    def call(self, inputs,  **kwargs):
        emb = self.embedding(inputs)
        return emb

    def compute_output_shape(self, input_shape):
        if self.emb_type is None:
            return input_shape[0], self.feature.input_size, self.units
        else: # lstm, gru, cnn, avg, sum
            return input_shape[0], 1, self.units



class DNN(layers.Layer):
    def __init__(self, output_dim, hidden_dims, hidden_activation="relu", activation="relu",
                 dropout=0.0, batch_norm=True, regularizer=1e-3, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims]
        self.hidden_activation = hidden_activation
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.regularizer = regularizer

    def build(self, input_shape):
        super(DNN, self).build(input_shape)
        self.denses = models.Sequential()
        for dim in self.hidden_dims:
            self.denses.add(layers.Dense(dim, kernel_regularizer=l2(self.regularizer), ))
            if self.batch_norm:
                self.denses.add(layers.BatchNormalization())
            if isinstance(self.hidden_activation, str):
                self.denses.add(layers.Activation(self.hidden_activation))
            else:
                self.denses.add(self.hidden_activation)
            self.denses.add(layers.Dropout(rate=self.dropout))
        self.last_dense = layers.Dense(self.output_dim, activation=self.activation,
                                       kernel_regularizer=l2(self.regularizer))

    def call(self, inputs, **kwargs):
        outputs = self.denses(inputs)
        outputs = self.last_dense(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (self.output_dim,)


class FMCross(layers.Layer):
    def __init__(self, **kwargs):
        super(FMCross, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FMCross, self).build(input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"Unexpected inputs dimensions {len(input_shape)}, expect to be 3 dimensions")

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(f"Unexpected inputs dimensions {K.ndim(inputs)}, expect to be 3 dimensions")

        x_sum = tf.reduce_sum(inputs, axis=1)
        x_square_sum = tf.reduce_sum(tf.pow(inputs, 2), axis=1)
        outputs = 0.5 * tf.reduce_sum(
            tf.subtract(tf.pow(x_sum, 2), x_square_sum),
            axis=1, keepdims=True)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1


class AFMCross(layers.Layer):
    def __init__(self, atten_dim, regularizer=1e-3, **kwargs):
        super(AFMCross, self).__init__(**kwargs)
        self.atten_dim = atten_dim
        self.regularizer = regularizer

    def build(self, input_shape):
        super(AFMCross, self).build(input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"Unexpected inputs dimensions {len(input_shape)}, expect to be 3 dimensions")
        ftr_dim = int(input_shape[2])
        self.attention = Attention(dim=self.atten_dim)
        self.proj_p = self.add_weight(
            name='proj_p',
            shape=(ftr_dim, 1),
            initializer='glorot_uniform',
            regularizer=l2(self.regularizer),
            trainable=True
        )

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(f"Unexpected inputs dimensions {K.ndim(inputs)}, expect to be 3 dimensions")

        embeddings = inputs  # batch, ftr_n, ftr_emb
        n_ftr = int(inputs.shape[1])
        element_wise_product_list = list()
        for i in range(0, n_ftr):
            for j in range(i + 1, n_ftr):
                element_wise_product_list.append(
                    tf.multiply(embeddings[:, i, :], embeddings[:, j, :]))  # batch, ftr_emb

        element_wise_product = tf.stack(element_wise_product_list)  # ftr_n*(ftr_n-1), batch, ftr_emb
        element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2])  # batch, ftr_n*(ftr_n-1), ftr_emb
        atten_score = self.attention(element_wise_product)  # batch, ftr_n*(ftr_n-1), ftr_emb
        atten_emb = tf.reduce_sum(atten_score, axis=1)  # batch, ftr_emb
        output = tf.matmul(atten_emb, self.proj_p)  # batch, 1
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class DCNCross(layers.Layer):
    def __init__(self, layer_num=3, regularizer=1e-3, mode="vector", **kwargs):
        super(DCNCross, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.mode = mode
        self.regularizer = regularizer

    def build(self, input_shape):
        super(DCNCross, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(f"Unexpected inputs dimensions {len(input_shape)}, expect to be 2 dimensions")

        dim = int(input_shape[-1])
        if self.mode == 'vector':
            self.kernels = [self.add_weight(
                name='crossnet_kernel' + str(i),
                shape=(dim, 1),
                initializer='glorot_uniform',
                regularizer=l2(self.regularizer),
                trainable=True
            ) for i in range(self.layer_num)]
        elif self.mode == 'matrix':
            self.kernels = [self.add_weight(
                name='crossnet_kernel' + str(i),
                shape=(dim, dim),
                initializer='glorot_uniform',
                regularizer=self.regularizer,
                trainable=True) for i in range(self.layer_num)]
        else:  # error
            raise ValueError("`mode` should be 'vector' or 'matrix'")
        self.bias = [self.add_weight(name='crossnet_bias' + str(i),
                                     shape=(dim, 1),
                                     initializer='zeros',
                                     regularizer=l2(self.regularizer),
                                     trainable=True) for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(f"Unexpected inputs dimensions {K.ndim(inputs)}, expect to be 2 dimensions")

        x_0 = tf.expand_dims(inputs, axis=2)
        x_i = x_0
        for i in range(self.layer_num):
            if self.mode == 'vector':
                x_i_w = tf.tensordot(x_i, self.kernels[i], axes=(1, 0))
                x_i = tf.matmul(x_0, x_i_w) + x_i + self.bias[i]  # x_0 * w_i * x_i + x_i + b_i
            elif self.mode == 'matrix':
                x_i_w = tf.einsum('bjk,ij->bik', x_i, self.kernels[i])
                x_i = x_0 * (x_i_w + self.bias[i]) + x_i  # x_0 * (w_i * x_i + b_i) + x_i
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        output = tf.squeeze(x_i, axis=2)  ## deserilize expand_dims
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class Attention(layers.Layer):
    def __init__(self, dim, regularizer=1e-3, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dim = dim
        self.regularizer = regularizer

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"Unexpected inputs dimensions {len(input_shape)}, expect to be 3 dimensions")

        ftr_dim = int(input_shape[-1])
        self.weight = self.add_weight(name='atten_weight',
                                      shape=(ftr_dim, self.dim),
                                      initializer='glorot_uniform',
                                      regularizer=l2(self.regularizer),
                                      trainable=True)
        self.bias = self.add_weight(name='atten_bias',
                                    shape=(1, self.dim),
                                    initializer='glorot_uniform',
                                    regularizer=l2(self.regularizer),
                                    trainable=True)
        self.proj = self.add_weight(name='atten_proj',
                                    shape=(self.dim, 1),
                                    initializer='glorot_uniform',
                                    regularizer=l2(self.regularizer),
                                    trainable=True)
        self.activate = layers.ReLU()
        self.softmax = layers.Softmax(axis=1)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(f"Unexpected inputs dimensions {K.ndim(inputs)}, expect to be 3 dimensions")
        # batch, ftr_n, ftr_emb
        attention_mul = tf.einsum("bnf,fd->bnd", inputs, self.weight)  # batch, ftr_n, atten_dim
        attention = attention_mul + self.bias  # batch, ftr_n, atten_dim
        attention = self.activate(attention)
        attention_h = tf.tensordot(attention, self.proj, axes=(-1, 0))  # batch, ftr_n, atten_dim
        attention_score = tf.reduce_sum(attention_h, axis=2, keepdims=True)  # batch, ftr_n, 1
        attention_score = self.softmax(attention_score)  # batch, ftr_n, 1
        atten_emb = tf.multiply(inputs, attention_score)  # batch, ftr_n*(ftr_n-1), ftr_emb
        return atten_emb

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 1


class GlobalAveragePooling(layers.Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(GlobalAveragePooling, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x) != K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
        else:
            return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        return output_shape


class GlobalMaxPooling(layers.Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(GlobalMaxPooling, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x) != K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.max(x, axis=self.axis)
        else:
            return K.max(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        return output_shape


class GlobalSumPooling(layers.Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(GlobalSumPooling, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x) != K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis)
        else:
            return K.sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        return output_shape


class KMaxPooling(layers.Layer):
    def __init__(self, k=1, axis=-1, **kwargs):

        self.k = k
        self.axis = axis
        self.dims = -1
        super(KMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.axis < 1 or self.axis > len(input_shape):
            raise ValueError(f"axis must be 1~{len(input_shape)}, now is {self.axis}")

        if self.k < 1 or self.k > input_shape[self.axis]:
            raise ValueError("k must be in 1 ~ {input_shape[self.axis]},now k is {self.k}")
        self.dims = len(input_shape)
        super(KMaxPooling, self).build(input_shape)

    def call(self, inputs):
        # swap the last and the axis dimensions since top_k will be applied along the last dimension
        perm = list(range(self.dims))
        perm[-1], perm[self.axis] = perm[self.axis], perm[-1]
        shifted_input = tf.transpose(inputs, perm)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        output = tf.transpose(top_k, perm)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.k
        return tuple(output_shape)


class BiInteractionPooling(layers.Layer):
    def __init__(self, **kwargs):
        super(BiInteractionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BiInteractionPooling, self).build(input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"Unexpected inputs dimensions {len(input_shape)}, expect to be 3 dimensions")

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(f"Unexpected inputs dimensions {K.ndim(inputs)}, expect to be 3 dimensions")

        concated_embeds_value = inputs
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1, input_shape[-1]


class SinusoidalPositionEmbedding(layers.Layer):
    """定义Sin-Cos位置Embedding
    pe(2i) = sin(pos * pow(10000, -2i/d))
    pe(2i+1) = cos(pos * pow(10000, -2i/d))
    """

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False, **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = K.shape(inputs)[1]
            inputs, position_ids = inputs
            if 'float' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, K.floatx())
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]

        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        indices = K.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = tf.einsum('bn,d->bnd', position_ids, indices)  # 1, seq, dim/2
        embeddings = K.stack([K.sin(embeddings), K.cos(embeddings)], axis=-1)  # concat + reshape 达到 sin-cos间隔
        embeddings = K.reshape(embeddings, (-1, seq_len, self.output_dim))  # 1, seq, dim

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Transformer(layers.Layer):
    def __init__(
            self,
            layer_num,
            head_num,
            head_dim,
            dropout_rate=0.3,
            intermediate_dim=None,
            use_position=True,
            **kwargs
    ):
        super(Transformer, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.intermediate_dim = intermediate_dim
        self.use_position = use_position

    def build(self, input_shape):
        super(Transformer, self).build(input_shape)

        self.intermediate_dim = self.intermediate_dim or int(input_shape[2])
        self.multi_head_attentions = list()
        self.attent_dropouts = list()
        self.attent_adds = list()
        self.attent_layer_normalizations = list()
        self.feed_forwards = list()
        self.feed_dropouts = list()
        self.feed_adds = list()
        self.feed_layer_normalizations = list()

        if self.use_position:
            self.position_embedding = SinusoidalPositionEmbedding(output_dim=input_shape[2], name="position-embedding")

        for i in range(self.layer_num):
            self.multi_head_attentions.append(MultiHeadAttention(head_num=self.head_num, head_dim=self.head_dim,
                                                                 out_dim=input_shape[2],
                                                                 name=f"multi-head-attention-layer-{i}"))
            self.attent_dropouts.append(layers.Dropout(self.dropout_rate, name=f"attent-dropout-layer-{i}"))
            self.attent_adds.append(layers.Add(name=f"attent-add-layer-{i}"))
            self.attent_layer_normalizations.append(LayerNormalization(name=f"attent-layer-normalization-layer-{i}"))
            self.feed_forwards.append(FeedForward(units=self.intermediate_dim, name=f"feed-forward-layer-{i}"))
            self.feed_dropouts.append(layers.Dropout(self.dropout_rate, name=f"feed-forward-dropout-layer-{i}"))
            self.feed_adds.append(layers.Add(name=f"feed-forward-add-layer-{i}"))
            self.feed_layer_normalizations.append(LayerNormalization(name=f"feed-forward-layer-normalization-layer-{i}"))

    def call(self, inputs, mask=None, **kwargs):
        if self.use_position:
            inputs = inputs + self.position_embedding(inputs)

        x = inputs
        for i in range(self.layer_num):
            x, x_ori = [x, x, x], x
            x = self.multi_head_attentions[i](x, mask=mask)
            x = self.attent_dropouts[i](x)
            x = self.attent_adds[i]([x, x_ori])
            x = self.attent_layer_normalizations[i](x)
            x_ori = x
            x = self.feed_forwards[i](x)
            x = self.feed_dropouts[i](x)
            x = self.feed_adds[i]([x, x_ori])
            x = self.feed_layer_normalizations[i](x)

        return x


class FeedForward(layers.Layer):
    """FeedForward层
    如果activation不是一个list，那么它就是两个Dense层的叠加；如果activation是
    一个list，那么第一个Dense层将会被替换成门控线性单元（Gated Linear Unit）。
    参考论文: https://arxiv.org/abs/2002.05202
    """
    def __init__(
        self,
        units,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        if not isinstance(activation, list):
            activation = [activation]
        self.activation = [tf.keras.activations.get(act) for act in activation]
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]

        for i, activation in enumerate(self.activation):
            i_dense = layers.Dense(
                units=self.units,
                activation=activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, 'i%s_dense' % i, i_dense)

        self.o_dense = layers.Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs):
        x = self.i0_dense(inputs)
        for i in range(1, len(self.activation)):
            x = x * getattr(self, 'i%s_dense' % i)(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [
                activations.serialize(act) for act in self.activation
            ],
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiHeadAttention(layers.Layer):
    """多头注意力机制
    """

    def __init__(
            self,
            head_num,
            head_dim,
            out_dim=None,
            key_size=None,
            use_bias=True,
            attention_scale=True,
            return_attention_scores=False,
            kernel_initializer='glorot_uniform',
            **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.head_num = head_num
        self.head_dim = head_dim
        self.out_dim = out_dim or head_num * head_dim
        self.key_size = key_size or head_dim
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = layers.Dense(
            units=self.key_size * self.head_num,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = layers.Dense(
            units=self.key_size * self.head_num,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = layers.Dense(
            units=self.head_dim * self.head_num,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = layers.Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        这里不需要v_mask
        """
        q, k, v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            q_mask, v_mask = mask[0], mask[2]
        # 线性变换
        qw = self.q_dense(q)  # batch, seq_len, key_size*heads
        kw = self.k_dense(k)  # batch, seq_len, key_size*heads
        vw = self.v_dense(v)  # batch, seq_len, head_size*heads
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.head_num, self.key_size))  # batch, seq_len, heads, key_size
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.head_num, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.head_num, self.head_dim))
        # Attention
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_masks = [q_mask, v_mask]
        o, a = self.pay_attention_to(qkv_inputs, qv_masks, **kwargs)
        # 完成输出
        o = K.reshape(o, (-1, K.shape(o)[1], self.head_dim * self.head_num))
        o = self.o_dense(o)
        # 返回结果
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    # def pay_interest_in(self, inputs, mask=None, **kwargs):

    def pay_attention_to(self, inputs, mask=None, **kwargs):
        """实现标准的乘性多头注意力
        a_bias: 对attention矩阵的bias。
                不同的attention bias对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。
        说明: 这里单独分离出pay_attention_to函数，是为了方便
              继承此类来定义不同形式的atttention；此处要求
              返回o.shape=(batch_size, seq_len, heads, head_size)。
        """
        (qw, kw, vw), n = inputs[:3], 3
        q_mask, v_mask = mask
        # a_bias, p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        # if a_bias:
        #     a_bias = inputs[n]
        #     n += 1
        # if p_bias == 'rotary':
        #     cos_pos = K.repeat_elements(inputs[n][..., None, 1::2], 2, -1)
        #     sin_pos = K.repeat_elements(inputs[n][..., None, ::2], 2, -1)
        #     qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
        #     qw2 = K.reshape(qw2, K.shape(qw))
        #     qw = qw * cos_pos + qw2 * sin_pos
        #     kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
        #     kw2 = K.reshape(kw2, K.shape(kw))
        #     kw = kw * cos_pos + kw2 * sin_pos
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)  # batch, head, [matmul(jd, hd)] -> seq_len, seq_len
        # 处理位置编码
        # if p_bias == 'typical_relative':
        #     position_bias = inputs[n]
        #     a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        # elif p_bias == 't5_relative':
        #     position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
        #     a = a + K.expand_dims(position_bias, 0)
        # Attention（续）
        if self.attention_scale:
            a = a / self.key_size ** 0.5
        # if a_bias is not None:
        #     a = a + a_bias
        a = sequence_masking(a, v_mask, '-inf', -1)
        A = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', A, vw)
        # if p_bias == 'typical_relative':
        #     o = o + tf.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.head_num, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.head_num,
            'head_size': self.head_dim,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiHeadInsterestAttention(MultiHeadAttention):
    def __init__(
            self,
            head_num,
            head_dim,
            out_dim=None,
            key_size=None,
            use_bias=True,
            attention_scale=True,
            return_attention_scores=False,
            kernel_initializer='glorot_uniform',
            **kwargs
    ):
        super(MultiHeadInsterestAttention, self).__init__(
            head_num=head_num,
            head_dim=head_dim,
            out_dim=out_dim,
            key_size=key_size,
            use_bias=use_bias,
            attention_scale=attention_scale,
            return_attention_scores=return_attention_scores,
            kernel_initializer=kernel_initializer,
            **kwargs
        )
    def build(self, input_shape):
        super(MultiHeadInsterestAttention, self).build(input_shape)
        self.linear = layers.Dense(1)
        self.activate = layers.PReLU(shared_axes=[1])

    def pay_attention_to(self, inputs, mask=None, **kwargs):
        q, k, v = inputs  # b, s, h, d
        att_input = tf.concat([q, k, q-k, q*k], axis=-1) # b, s, h, d*4
        att_input = self.activate(att_input)
        weight = self.linear(att_input)
        output = k * weight
        return output, None


class LayerNormalization(layers.Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """

    def __init__(
            self,
            center=True,
            scale=True,
            epsilon=None,
            conditional=False,
            hidden_units=None,
            hidden_activation='linear',
            hidden_initializer='glorot_uniform',
            **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = tf.keras.activations.get(hidden_activation)
        self.hidden_initializer = tf.keras.initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = layers.Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = layers.Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sequence_masking(x, mask, value=0.0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        return x * mask + value * (1 - mask)
