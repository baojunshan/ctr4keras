import json
import pandas as pd
# from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor
from ctr4keras.metrics import evaluate_report
from ctr4keras.snippets import train_test_split

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. feature preprocess   lightGBM 不能接受多值离散型和时间序列型特征
    data = pd.read_pickle("../data/train_data_210423_seq.pkl", compression="zip")
    cols = [c for c in data.columns if not c.startswith("M") and not c.startswith("S")]
    data = data[cols]

    feat_type_dict = dict()
    for i in data.columns:
        if i.startswith("C"):
            feat_type_dict[i] = FeatureType.Categorical
        elif i.startswith("I"):
            feat_type_dict[i] = FeatureType.Continuous
        elif i.startswith("M"):
            feat_type_dict[i] = FeatureType.MultiCategorical
    preprocessor = SimplePreprocessor(feat_type_dict=feat_type_dict)

    df = preprocessor.fit_transform(data)

    print(df.head())

    # 3.train model
    feature_columns = list(preprocessor.features.keys())

    train, test = train_test_split(df, test_size=0.1, random_state=2020, sort_col="stime", ascending=True)
    train, eval = train_test_split(train, test_size=0.1, random_state=2020, sort_col="stime", ascending=True)

    x_train, y_train = train[[c for c in train.columns if c in feature_columns]], train["label"]
    x_eval, y_eval = eval[[c for c in eval.columns if c in feature_columns]], eval["label"]
    x_test, y_test = test[[c for c in test.columns if c in feature_columns]], test["label"]

    params = {
        "importance_type": "gain",
        "is_unbalance": True,
        "n_estimators": 1000,
    }

    model = LGBMRegressor(**params)
    model.fit(X=x_train, y=y_train, eval_set=[(x_eval, y_eval)], eval_metric=["binaray_logloss", "auc"],
              early_stopping_rounds=30)

    y_pred = model.predict(x_test, num_iteration=model.best_iteration_)

    evaluate_report(y_true=y_test, y_score=y_pred)

    ftr_imp_df = pd.DataFrame({"ftr": x_train.columns.tolist(), "importance": model.feature_importances_.tolist()}) \
        .sort_values("importance", ascending=False)
    ftr_imp_df["importance"] = ftr_imp_df["importance"].apply(lambda x: x / sum(model.feature_importances_))

    with open("../data/ftr_name_210423_seq.json", "r") as f:
        name2ftr = json.load(f)
        ftr2name = {v: k for k, v in name2ftr.items()}
    ftr_imp_df["ftr_name"] = ftr_imp_df["ftr"].apply(lambda x: ftr2name.get(x, ""))
    ftr_imp_df =ftr_imp_df.reset_index(drop=True)

    print(ftr_imp_df.head(200))

    print(ftr_imp_df["ftr"].tolist())


if __name__ == "__main__":
    main()
