import json
import requests
import time
import pandas as pd

from ctr4keras import FeatureType
from ctr4keras.preprocessing import SimplePreprocessor

pd.set_option("display.max_row", 200)
pd.set_option("display.max_column", 200)
pd.set_option('display.width', 100)
pd.set_option('max_colwidth', 100)


def main():
    # 1. 特征工程造的特征
    data = pd.read_parquet("../data/train_data_210421.parquet").head(200)

    processor = SimplePreprocessor.load("./dictionary.pkl")
    df = processor.transform(data)

    # 3. 把encode过的特征转换成tf serving需要的格式
    test_input = processor.generator(df.head(2), mode="service")
    # print(test_input)

    print(test_input)
    print(len(test_input.keys()))
    # 4. 调用tf serving
    start_time = time.time()
    url = f"http://localhost:8501/v1/models/serving_model:predict"
    headers = {"content-type": "application/json"}
    data = json.dumps({
        "signature_name": "serving_default",
        "inputs": test_input
    })
    res = requests.post(url, headers=headers, data=data).json()
    scores = res["outputs"]
    print(time.time() - start_time)
    print(scores)


if __name__ == "__main__":
    main()
