import pandas as pd
import tensorflow as tf

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.models import DCN
from ctr4keras.snippets import HistoryExt, Plot, save_model_as_savedmodel, set_seed, train_test_split
from ctr4keras.metrics import evaluate_report, gauc

set_seed(2021)

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. feature preprocess
    data = pd.read_pickle("../data/train_data_210423_seq.pkl", compression="zip")
    print(data.head())

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
    model = DCN(
        features=preprocessor.features,
        cross_layer_num=5,
        dense_emb_dim=4,
        dense_hidden_dims=[128, 128, 32],
        regularizer=1e-5
    )
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )

    # 4. train model
    train, test = train_test_split(df, test_size=0.1, random_state=2020, sort_col="stime", ascending=True)
    train_input = preprocessor.generator(train)
    test_input = preprocessor.generator(test)

    history_callback = HistoryExt()
    model.fit(train_input, train["label"].values,
              batch_size=256, epochs=2, verbose=2, validation_split=0.1,
              callbacks=[history_callback])

    # 5. save model as saved-model-format for service
    # save_model_as_savedmodel(model, "../service/serving_model/1/")

    # 5. evaluate or predict
    pred_res = model.predict(test_input, batch_size=256)
    evaluate_report(y_true=test["label"].values, y_score=pred_res)

    plot = Plot(title="DCN eval metrics")
    for t in ["loss", "auc"]:
        plot.add_plot(
            label2dpair={
                f"train_{t}": history_callback.get(value=t, mode="epoch"),
                f"valid_{t}": history_callback.get(value=f"val_{t}", mode="epoch")
            }, sub_xlabel="Epoch", sub_ylabel=t.capitalize(), sub_title=f"{t}"
        )
    plot.plot(save_path=None)


if __name__ == "__main__":
    main()
