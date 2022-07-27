import pandas as pd
import tensorflow as tf
import json

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor, fillna
from ctr4keras.models import DCN, LambdaRanker, DIN, DeepFM
from ctr4keras.snippets import HistoryExt, Plot, save_model_as_savedmodel, set_seed, train_test_split
from ctr4keras.metrics import evaluate_report, rank_evaluate

set_seed(2021)

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. feature preprocess
    data = pd.read_pickle("/home/baojunshan/douya-ctr/ltr/train_data_220607_lgb.pkl")
    ftr_names = json.load(open('/home/baojunshan/douya-ctr/ltr/train_data_ftr_name_220607_lgb.json'))
    data = data.rename(columns=ftr_names)
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

    data = fillna(data, feat_type_dict, cont_na=0.0, cate_na="NULL")
    preprocessor = SimplePreprocessor(feat_type_dict=feat_type_dict)
    df = preprocessor.fit_transform(data)

    print(df.head())
    # df.to_pickle('/home/baojunshan/douya-ctr/ltr/train_data_220607_lgb_processed.pkl')
    # preprocessor.save('/home/baojunshan/douya-ctr/ltr/train_data_220607_lgb_processor.pkl')
    # df = pd.read_pickle('/home/baojunshan/douya-ctr/ltr/train_data_220607_lgb_processed.pkl')
    # preprocessor = SimplePreprocessor.load('/home/baojunshan/douya-ctr/ltr/train_data_220607_lgb_processor.pkl')

    # 3. build model
    # model = LambdaRanker(
    #     module=DCN,
    #     features=preprocessor.features,
    #     cross_layer_num=5,
    #     dense_emb_dim=4,
    #     dense_hidden_dims=[128, 128, 32],
    #     regularizer=1e-5
    # )
    model = LambdaRanker(
        module=DeepFM,
        features=preprocessor.features,
        dense_emb_dim=8,
        dense_hidden_dims=[128, 64, 32],
        regularizer=1e-3
    )

    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )

    # 4. train model
    feature_columns = list(preprocessor.features.keys())

    train, test = train_test_split(df, test_size=0.1, shuffle=False, sort_col=["ts"], group_col=['uid', 'ts'],
                                   ascending=True)
    train, valid = train_test_split(train, test_size=0.1, shuffle=False, sort_col=["ts"], group_col=['uid', 'ts'],
                                    ascending=True)

    x_train, y_train = train[[c for c in train.columns if c in feature_columns]], train["label"]
    g_train = train.groupby(['uid', 'ts'], as_index=False, sort=False).count()['label'].values

    x_valid, y_valid = valid[[c for c in valid.columns if c in feature_columns]], valid["label"]
    g_valid = valid.groupby(['uid', 'ts'], as_index=False, sort=False).count()['label'].values

    x_test, y_test = test[[c for c in test.columns if c in feature_columns]], test["label"]
    g_test = test.groupby(['uid', 'ts'], as_index=False, sort=False).count()['label'].values

    train_input = preprocessor.generator(x_train)
    valid_input = preprocessor.generator(x_valid)
    test_input = preprocessor.generator(x_test)

    history_callback = HistoryExt()

    model.fit(
        train_input,
        y_train,
        g_train,
        batch_size=4096,
        epochs=10,
        verbose=1,
        eval_x=valid_input,
        eval_y=y_valid,
        eval_group=g_valid,
        callbacks=[history_callback]
    )

    # 5. save model as saved-model-format for service
    # save_model_as_savedmodel(model, "../service/serving_model/1/")

    # 5. evaluate or predict
    y_pred = model.predict(test_input)
    evaluate_report(y_true=y_test, y_score=y_pred)
    rank_evaluate(y_pred=y_pred, y_true=y_test, group=g_test, eval_at=[1, 3, 5, 10])

    plot = Plot(title="LambdaDCN eval metrics")
    for t in ["loss", "auc"]:
        plot.add_plot(
            label2dpair={
                f"train_{t}": history_callback.get(value=t, mode="epoch"),
                f"valid_{t}": history_callback.get(value=f"val_{t}", mode="epoch")
            }, sub_xlabel="Batch", sub_ylabel=t.capitalize(), sub_title=f"{t}"
        )
    plot.plot(save_path="./lambda_dcn_220609.jpg")


if __name__ == "__main__":
    main()
