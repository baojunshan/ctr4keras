import pandas as pd
import tensorflow as tf

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.models import AutoInt
from ctr4keras.snippets import HistoryExt, Plot, set_seed, train_test_split
from ctr4keras.metrics import evaluate_report, gauc

set_seed(2021)

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. feature preprocess
    data = pd.read_pickle("../data/train_data_210423_seq.pkl", compression="zip")

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
    model = AutoInt(
        features=preprocessor.features,
        dense_emb_dim=4,
        dense_hidden_dims=[128, 128, 32],
        atten_head_num=4,
        atten_head_dim=16,
        atten_layer_num=2,
        regularizer=1e-3
    )
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )

    # 3. train model
    train, test = train_test_split(df, test_size=0.1, random_state=2020, sort_col="stime", ascending=True)
    train_model_input = preprocessor.generator(train)
    test_model_input = preprocessor.generator(test)

    model.fit(train_model_input, train["label"].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2,)

    # 4. evaluate or predict
    pred_res = model.predict(test_model_input, batch_size=256)
    # print("GAUC:", gauc(y_true=test["label"].values, y_score=pred_res, indicators=test["usr_id"]))
    evaluate_report(y_true=test["label"].values, y_score=pred_res)


if __name__ == "__main__":
    main()
