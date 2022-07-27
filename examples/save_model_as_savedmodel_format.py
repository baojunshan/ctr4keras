import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.models import DCN
from ctr4keras.snippets import save_model_as_savedmodel


def main():
    # 1. feature preprocess
    data = pd.read_parquet("../data/train_data_210413.parquet")

    feat_type_dict = dict()
    for i in data.columns:
        if i.startswith("C"):
            feat_type_dict[i] = FeatureType.Categorical
        elif i.startswith("I"):
            feat_type_dict[i] = FeatureType.Continuous
        elif i.startswith("M"):
            feat_type_dict[i] = FeatureType.MultiCategorical
        elif i.startswith("O"):
            feat_type_dict[i] = FeatureType.Ordered
        elif i.startswith("S"):
            feat_type_dict[i] = FeatureType.Sequential
    preprocessor = SimplePreprocessor(feat_type_dict=feat_type_dict)

    df = preprocessor.fit_transform(data)

    print(df.head())


    # 3. build model
    model = DCN(features=preprocessor.features, cross_layer_num=5, dense_emb_dim=4, dense_hidden_dims=[512, 128, 32])
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )

    # 4. train model
    train, test = train_test_split(df, test_size=0.2, random_state=2020)
    train_input = preprocessor.generator(train)
    test_input = preprocessor.generator(test)

    model.fit(train_input, train["label"].values,
              batch_size=256, epochs=10, verbose=2, validation_split=0.2)

    # 5. save model as saved-model-format for service
    save_model_as_savedmodel(model, "../service/some_model/1/")