import time
import pandas as pd

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor


pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)

data = pd.read_parquet("../data/train_data_210421.parquet")

print(data.loc[368:369,])

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
# preprocessor = SimplePreprocessor(feat_type_dict=feat_type_dict)
#
# print(data.head())
#
# df = preprocessor.fit_transform(data)
# print(df.head())
#
# preprocessor.save("./dictionary.pkl")
#
# exit()

preprocessor2 = SimplePreprocessor.load("./dictionary.pkl")
# data2 = preprocessor2.inverse_transform(df)

start_time = time.time()

res = preprocessor2.transform(data)
print("time", time.time() - start_time)

print(res.loc[368:369,])

# print(data2.head())