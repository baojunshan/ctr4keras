from enum import Enum, unique


@unique
class FeatureType(Enum):
    Continuous = 0
    Categorical = 1
    MultiCategorical = 2
    Ordered = 3
    Sequential = 4


class FeatureAttributes:
    """
    name:
        str : feature name
    type:
        str:  feature type
    dim:
        int:  feature embedding size, = feature_type_num + 1
    input_size:
        int:  feature input size
    embedding:
        list[list] or np.ndarray: pre-trained embedding of curr feature
    vocab:
        list: the pre-trained embedding vocabulary for labelencoding
    target_sequence:
        list: the sequence feature which will atentioned by it for inteset computation
    """

    def __init__(self, name: str, type: str, dim: int = -1, input_size: int = 1, embedding=None, vocab: list = None,
                 target_sequence: list = None):
        self.name = name
        self.type = type
        self.dim = dim
        self.input_size = input_size
        self.embedding = embedding
        self.vocab = vocab
        self.target_sequence = target_sequence
        if embedding is not None:
            if vocab is None:
                raise ValueError("`vocab` should not be none if `embedding` has value.")
