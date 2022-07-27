from typing import Union
from collections import namedtuple, OrderedDict
from abc import abstractmethod

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

from ctr4keras import FeatureType, FeatureAttributes


def to_list(x):
    if isinstance(x, (int, float, str)):
        return x
    if isinstance(x, pd.Series):
        x_l = x.values.tolist()
    elif isinstance(x, np.ndarray):
        x_l = x.tolist()
    elif isinstance(x, (list, set)):
        x_l = list(x)
    else:
        raise ValueError("input should be pd.Series, np.ndarray, set or list")
    x_l = [to_list(i) for i in x_l]
    return x_l


def fillna(x, features, cont_na=0.0, cate_na="NULL"):
    """
    x: pd.DataFrame or dict[name, value_lists]
    features: OrderedDict[name, FeatureType]
    """
    df = x
    if isinstance(x, dict):
        df = pd.DataFrame(x)
    df = df.copy()
    columns = set(df.columns.tolist())
    for n, v in features.items():
        if n not in columns:
            continue
        if v.name in (FeatureType.Continuous.name, FeatureType.Ordered.name):
            df[n] = df[n].fillna(cont_na)
        elif v.name == FeatureType.Categorical.name:
            df[n] = df[n].fillna(cate_na)
        else:
            df[n] = df[n].apply(lambda x: x if not isinstance(x, float) else list())
            df[n] = df[n].apply(lambda x: x if len(x)>0 else [cate_na])
    return df


class LabelEncoder:
    def __init__(self, unk="UNK", non="NULL"):
        self.unk = unk
        self.non = non
        self.index2label = {1:self.non, 2:self.unk}
        self.label2index = {self.non:1, self.unk:2}

    @property
    def size(self):
        return len(self.label2index)

    def _flatten(self, x):
        x = to_list(x)
        flatten_list = lambda x: [y for l in x for y in flatten_list(l)] if isinstance(x, (list, set)) else [x]
        return flatten_list(x)

    def fit(self, x):
        x1 = self._flatten(x)
        for i in x1:
            if i not in self.label2index:
                curr_index = self.size + 1
                self.label2index[i] = curr_index
                self.index2label[curr_index] = i
        return self

    def transform(self, y):
        classes = set(self.label2index.keys())
        y1 = self._flatten(y)
        y1 = [i if i in classes else self.unk for i in y1]
        res = [self.label2index[i] for i in y1]

        if isinstance(to_list(y)[0], list):
            idx = [0]
            for i in [len(i) for i in y]:
                idx.append(idx[-1] + i)
            res = [res[i:j] for i, j in zip(idx[:-1], idx[1:])]

        return res

    def inverse_transform(self, y, filter_value=0):
        y1 = to_list(y)

        if isinstance(y1[0], list):
            y1 = [[j for j in i if j != filter_value] for i in y1]
        else:
            y1 = [i for i in y1 if i != filter_value]

        res = [self.index2label[i] for i in self._flatten(y1)]

        if isinstance(to_list(y1)[0], list):
            idx = [0]
            for i in [len(i) for i in y1]:
                idx.append(idx[-1] + i)
            res = [res[i:j] for i, j in zip(idx[:-1], idx[1:])]

        return res

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class BasePreprocessor:
    def __init__(self,
                 feat_type_dict: Union[dict],
                 multi_cate_maxlen_dict: Union[dict]=None,
                 feat_pretrain_embedding_dict: Union[dict]=None,
                 feat_vocab_dict: Union[dict]=None,
                 feat_target_seq_dict: Union[dict]=None
                 ):
        self.features = OrderedDict()
        self.processors = OrderedDict()

        multi_cate_maxlen_dict = multi_cate_maxlen_dict or dict()
        feat_pretrain_embedding_dict = feat_pretrain_embedding_dict or dict()
        feat_vocab_dict = feat_vocab_dict or dict()
        feat_target_seq_dict = feat_target_seq_dict or dict()

        for n, t in feat_type_dict.items():
            if isinstance(t, FeatureType):
                t = t.name
            self.features[n] = FeatureAttributes(
                name=n,
                type=t,
                input_size=multi_cate_maxlen_dict.get(n, 1),
                embedding=feat_pretrain_embedding_dict.get(n, None),
                vocab=feat_vocab_dict.get(n, None),
                target_sequence=feat_target_seq_dict.get(n, None),
            )
        self.multi_cate_maxlen_dict = multi_cate_maxlen_dict or dict()

    @abstractmethod
    def fit(self, data: Union[dict, pd.DataFrame]):
        raise NotImplementedError

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def generator(self, data, mode="train"):
        if mode not in ("train", "service"):
            raise ValueError("mode must be `train` or `service`.")

        res = dict()
        df = self.trans_input_to_pd(data)
        for f in self.features.values():
            if f.type not in (FeatureType.MultiCategorical.name, FeatureType.Sequential.name):
                res[f.name] = df[f.name].values
            else:
                res[f.name] = np.array(df[f.name].tolist())

        if mode == "service":
            for k, v in res.items():
                res[k] = [i if isinstance(i, list) else [i] for i in to_list(v)]

        return res

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def trans_input_to_pd(data: Union[dict, pd.DataFrame]):
        df = data
        if isinstance(data, list):
            df = pd.DataFrame(data)
        return df


class SimplePreprocessor(BasePreprocessor):
    def __init__(self,
                 feat_type_dict: Union[dict],
                 multi_cate_maxlen_dict: Union[dict]=None,
                 feat_pretrain_embedding_dict: Union[dict]=None,
                 feat_vocab_dict: Union[dict]=None,
                 feat_target_seq_dict: Union[dict]=None):
        super(SimplePreprocessor, self).__init__(feat_type_dict=feat_type_dict,
                                                 multi_cate_maxlen_dict=multi_cate_maxlen_dict,
                                                 feat_pretrain_embedding_dict=feat_pretrain_embedding_dict,
                                                 feat_vocab_dict=feat_vocab_dict,
                                                 feat_target_seq_dict=feat_target_seq_dict)

    def _padding(self, x, maxlen, value=0):
        if not isinstance(x[0], list):
            return x[:maxlen] + [value] * max(0, maxlen - len(x))
        return [self._padding(i, maxlen, value) for i in x]

    def fit(self, data: Union[dict, pd.DataFrame]):
        df = self.trans_input_to_pd(data)

        for feat in self.features.values():
            if feat.type in (FeatureType.Continuous.name, FeatureType.Ordered.name):
                p = MinMaxScaler(feature_range=(0, 1))
                p.fit(df[[feat.name]])
                self.features[feat.name].dim = 1
            elif feat.type == FeatureType.Categorical.name:
                p = LabelEncoder(non="NULL")
                if feat.vocab is not None:
                    p.fit(feat.vocab)
                else:
                    p.fit(df[feat.name])
                self.features[feat.name].dim = p.size + 1
            elif feat.type in (FeatureType.MultiCategorical.name, FeatureType.Sequential.name):
                p = LabelEncoder(non="NULL")
                if feat.vocab is not None:
                    p.fit(feat.vocab)
                else:
                    x = [j for i in df[feat.name].tolist() for j in i]
                    p.fit(x)
                self.features[feat.name].dim = p.size + 1

                maxlen = max([len(i) for i in df[feat.name].tolist()])
                maxlen = max(self.features[feat.name].input_size, maxlen)
                self.features[feat.name].input_size = maxlen
                self.multi_cate_maxlen_dict[feat.name] = maxlen
            else:
                p = None

            if p is not None:
                self.processors[feat.name] = p
        return self

    def transform(self, data):
        df = self.trans_input_to_pd(data)
        columns = df.columns.tolist()
        for feat in self.features.values():
            if feat.name not in columns:
                continue
            p = self.processors[feat.name]
            if feat.type in (FeatureType.Continuous.name, FeatureType.Ordered.name):
                df[feat.name] = p.transform(df[[feat.name]])
            elif feat.type == FeatureType.Categorical.name:
                df[feat.name] = p.transform(df[feat.name])
            elif feat.type in (FeatureType.MultiCategorical.name, FeatureType.Sequential.name):
                x = p.transform(df[feat.name])
                x = self._padding(x, maxlen=self.multi_cate_maxlen_dict[feat.name], value=0)
                df[feat.name] = x
        return df

    def inverse_transform(self, data):
        df = self.trans_input_to_pd(data)
        columns = df.columns.tolist()
        for feat in self.features.values():
            if feat.name not in columns:
                continue
            p = self.processors[feat.name]
            if feat.type == FeatureType.Continuous.name or feat.type == FeatureType.Ordered.name:
                df[feat.name] = p.inverse_transform(df[[feat.name]])
            elif feat.type == FeatureType.Categorical.name:
                # print(df[feat.name])
                # print(p.inverse_transform(df[feat.name]))
                # print("------------------")
                df[feat.name] = p.inverse_transform(df[feat.name])
            elif feat.type == FeatureType.MultiCategorical.name:
                df[feat.name] = p.inverse_transform(df[feat.name])
        return df

if __name__ == "__main__":
    test_cases = [
        [1,2,3,4,5],
        ["a", "b", "c", "a", "e"],
        [["d", "f"], ["e"], ["e","d", "f"]]
    ]
    for t in test_cases:
        l = LabelEncoder()
        res = l.fit_transform(t)
        print(l.inverse_transform(res))
        # print(l.size, l.label2index, res)


