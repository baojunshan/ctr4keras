import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.models import DeepFM, FM, LR, DWL, DCN, FNN, NFM, CCPM, AFM, DeepAFM
from ctr4keras.snippets import HistoryExt, Plot, set_seed
from ctr4keras.metrics import evaluate_report, gauc

set_seed(2021)

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    models = {
        "LR": LR(
            features=preprocessor.features
        ),
        "FM": FM(
            features=preprocessor.features,
            dense_emb_dim=4
        ),
        "DWL": DWL(
            features=preprocessor.features,
            dense_emb_dim=4,
            dense_hidden_dims=[128, 128, 32]
        ),
        "FNN": FNN(
            features=preprocessor.features,
            dense_emb_dim=4,
            dense_hidden_dims=[128, 128, 32]
        ),
        "DeepFM": DeepFM(
            features=preprocessor.features,
            dense_emb_dim=4,
            dense_hidden_dims=[128, 128, 32],
            regularizer=5e-4
        ),
        "DCN": DCN(
            features=preprocessor.features,
            cross_layer_num=5,
            dense_emb_dim=4,
            dense_hidden_dims=[128, 128, 32]
        ),
        "NFM": NFM(
            features=preprocessor.features,
            dense_emb_dim=4,
            dense_hidden_dims=[128, 128, 8]
        ),
        "CCPM": CCPM(
            features=preprocessor.features,
            dense_emb_dim=4,
            conv_filters=[5, 5],
            conv_kernel_width=5,
            dense_hidden_dims=[32, 32, 8],
            regularizer=1e-5
        ),
        "AFM": AFM(
            features=preprocessor.features,
            dense_emb_dim=4,
            atten_dim=5,
            regularizer=1e-5
        ),
        "DeepAFM": DeepAFM(
            features=preprocessor.features,
            dense_emb_dim=4,
            atten_dim=5,
            regularizer=1e-5
        ),
    }

    for model in models.values():
        model.compile(
            loss='binary_crossentropy',
            optimizer="adam",
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
        )

    # 4. train model
    train, test = train_test_split(df, test_size=0.2, random_state=2020)
    train_model_input = preprocessor.generator(train)
    test_model_input = preprocessor.generator(test)

    history_callbacks = dict()
    for name, model in models.items():
        print(name)
        history_callbacks[name] = HistoryExt()
        model.fit(train_model_input, train["label"].values,
                  batch_size=256, epochs=10, verbose=2, validation_split=0.2,
                  callbacks=[history_callbacks[name]])
        print("\n")

    # 5. evaluate or predict

    eval_results, names = list(), list()
    for name, model in models.items():
        names.append(name)
        pred_res = model.predict(test_model_input, batch_size=256)
        gauc_score = gauc(y_true=test["label"].values, y_score=pred_res, indicators=test["usr_id"])
        auc, log_loss, accuracy, precision, recall, report = \
            evaluate_report(y_true=test["label"].values, y_score=pred_res, report_output_dict=True)
        eval_results.append({
            "GAUC": gauc_score,
            "AUC": auc,
            "Log Loss": log_loss,
            "ACC": accuracy,
            "Precision-1": report["1"]["precision"],
            "Recall-1": report["1"]["recall"],
            "Precision-0": report["0"]["precision"],
            "Recall-0": report["0"]["recall"]
        })
    eval_results_df = pd.DataFrame(eval_results, index=names)
    print(eval_results_df)

    plot_train = Plot(title="Train Metrics of Models")
    plot_train.add_plot(
        label2dpair={
            n: history_callbacks[n].get("auc", "epoch")
            for n, h in history_callbacks.items()
        }, sub_xlabel="Epoch", sub_ylabel="AUC", sub_title=f"AUC Curve of Models"
    )
    plot_train.add_plot(
        label2dpair={
            n: history_callbacks[n].get("loss", "epoch")
            for n, h in history_callbacks.items()
        }, sub_xlabel="Epoch", sub_ylabel="Loss", sub_title=f"Loss Curve of Models"
    )
    plot_train.plot("./models_train.png")

    plot_val = Plot(title="Validate Metrics of Models")
    plot_val.add_plot(
        label2dpair={
            n: history_callbacks[n].get("val_auc", "epoch")
            for n, h in history_callbacks.items()
        }, sub_xlabel="Epoch", sub_ylabel="AUC", sub_title=f"AUC Curve of Models"
    )
    plot_val.add_plot(
        label2dpair={
            n: history_callbacks[n].get("val_loss", "epoch")
            for n, h in history_callbacks.items()
        }, sub_xlabel="Epoch", sub_ylabel="Loss", sub_title=f"Loss Curve of Models"
    )
    plot_val.plot("./models_val.png")


if __name__ == "__main__":
    main()
