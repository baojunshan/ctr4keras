import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.models import FM
from ctr4keras.metrics import evaluate_report
from ctr4keras.snippets import set_seed


set_seed(2021)

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. feature preprocess
    data = pd.read_parquet("../data/train_data_210419.parquet")

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

    # 2. build model
    model = FM(features=preprocessor.features, dense_emb_dim=4)
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )

    # 3. train model
    train, test = train_test_split(df, test_size=0.2, random_state=2020)
    train_input = preprocessor.generator(train)
    test_input = preprocessor.generator(test)

    model.fit(
        train_input, train["label"].values,
        batch_size=256, epochs=10, verbose=2, validation_split=0.2,
    )

    # 5. evaluate or predict
    pred_res = model.predict(test_input, batch_size=256)
    evaluate_report(y_true=test["label"].values, y_score=pred_res)


if __name__ == "__main__":
    main()
