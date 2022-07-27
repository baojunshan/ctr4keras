import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.models import DeepAFM
from ctr4keras.snippets import HistoryExt, Plot, set_seed
from ctr4keras.metrics import evaluate_report, gauc


set_seed(2021)

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. feature preprocess
    data = pd.read_parquet("../data/train_data_210421.parquet")

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
    model = DeepAFM(features=preprocessor.features, dense_emb_dim=4, atten_dim=5, regularizer=1e-5)
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

    history_callback = HistoryExt()
    model.fit(
        train_input, train["label"].values,
        batch_size=256, epochs=10, verbose=1, validation_split=0.2,
        callbacks=[history_callback]
    )

    # 5. evaluate or predict
    pred_res = model.predict(test_input, batch_size=256)
    print("GAUC:", gauc(y_true=test["label"].values, y_score=pred_res, indicators=test["usr_id"]))
    evaluate_report(y_true=test["label"].values, y_score=pred_res)

    plot = Plot(title="DeepAFM eval metrics")
    for t in ["loss", "auc"]:
        plot.add_plot(
            label2dpair={
                "train_loss": history_callback.get(value=t, mode="epoch"),
                "valid_loss": history_callback.get(value=f"val_{t}", mode="epoch")
            }, sub_xlabel="Epoch", sub_ylabel=t.capitalize(), sub_title=f"LR {t}"
        )
    plot.plot(save_path=None)


if __name__ == "__main__":
    main()
