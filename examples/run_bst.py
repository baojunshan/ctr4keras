import pandas as pd
import tensorflow as tf

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.models import BST
from ctr4keras.snippets import set_seed, train_test_split
from ctr4keras.metrics import evaluate_report

set_seed(2021)

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. feature preprocess
    data = pd.read_pickle("../data/train_data_210423_seq.pkl", compression="zip")
    # C20 : S1  -> image_id : history image ids

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
    preprocessor = SimplePreprocessor(feat_type_dict=feat_type_dict, feat_target_seq_dict={"C20": "S1"})

    df = preprocessor.fit_transform(data)
    print(df.head())

    # 2. build model
    model = BST(
        features=preprocessor.features,
        dense_emb_dim=4, seq_emb_dim=8, atten_head_num=2, atten_head_dim=8
    )
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        # optimizer="adam",
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )

    train, test = train_test_split(df, test_size=0.1, random_state=2020, sort_col="stime", ascending=True)
    train_model_input = preprocessor.generator(train)
    test_model_input = preprocessor.generator(test)

    model.fit(train_model_input, train["label"].values,
              batch_size=256, epochs=20, verbose=2, validation_split=0.2,)

    # 4. evaluate or predict
    pred_res = model.predict(test_model_input, batch_size=256)
    evaluate_report(y_true=test["label"].values, y_score=pred_res)


if __name__ == "__main__":
    main()
