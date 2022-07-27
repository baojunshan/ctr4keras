import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.models import LR
from ctr4keras.snippets import HistoryExt, Plot, set_seed
from ctr4keras.metrics import evaluate_report
from ctr4keras.losses import binary_focal_loss


set_seed(2021)

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. feature preprocess
    data = pd.read_parquet("../data/train_data_210421.parquet").head(100)

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
    model = LR(features=preprocessor.features)
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        # loss=binary_focal_loss(alpha=0.25, gamma=2.0),
        optimizer='adam',
        metrics=[tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.AUC()]
    )

    # 4. train model
    train, test = train_test_split(df, test_size=0.2, random_state=2020)
    train_model_input = preprocessor.generator(train)
    test_model_input = preprocessor.generator(test)

    history_callback = HistoryExt()
    model.fit(train_model_input, train["label"].values,
              batch_size=256, epochs=10, verbose=2, validation_split=0.2,
              callbacks=[history_callback])

    # 5. evaluate or predict
    pred_res = model.predict(test_model_input, batch_size=256)
    auc, log_loss_, accuracy, precision, recall, report = evaluate_report(y_true=test["label"].values, y_score=pred_res)

    print(report.keys())

    plot = Plot(title="LR eval metrics")
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
