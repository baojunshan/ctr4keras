# ctr4keras

- 更清晰、更轻量的keras版本的ctr模型库
- 使用方法详见`ctr4keras/examples`

# 说明
感谢浅梦大佬的[DeepCTR](https://github.com/shenweichen/DeepCTR) 和苏神的[bert4keras](https://github.com/bojone/bert4keras) ,本实现有不少地方借鉴了DeepCTR和bert4keras的代码。

# 功能
- 支持多种特征结构的输入，包含`连续型（dense）`、`离散型（sparse）`、`有序型`、`多值离散型` 和 `时间序列型` 特征
- 支持各种CTR模型的训练 
- 支持常见ctr损失函数，如`focal loss`
- 支持各种组件（layer）的拼接，得到新模型（model）
- 支持结果可视化和tf serving的部署
- 支持各种深度学习模型与`lambda`结合的排序模型

# 主要框架
![Frame](./docs/ctr4keras_frame.png)

# 模型

|Finished|  Model | Paper |
|:---| :--- | :---|
| &#9745;| LR | [1986] Logistic Regression |
| &#9745;| FM | [IEEE 2010][Factorization Machines](https://ieeexplore.ieee.org/document/5694074)|
| &#9745;| CCPM | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf) |
| &#9745;| FNN | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf) |
| &#9745;| DWL | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)  |
| &#9745;| DeepFM | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf) |
| &#9745;| DCN | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)  |
| &#9745;| NFM | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)|
| &#9745;| AFM | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
| &#9745;| DIN | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf) |
| &#9745;| AutoInt | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) |
| &#9745;| BST | [DLP-KDD 2019][Behavior sequence transformer for e-commerce recommendation in Alibaba](https://arxiv.org/pdf/1905.06874.pdf) |
| &#9744;| DIEN| [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)  |
| &#9744;| DSIN | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482) |
| &#9745;| DeepAFM |  |


# 更新

- **2022.06.09**: 新增rank和lambda两种pairwise的训练方法，适用于所有深度学习模型
- **2021.04.25**: 加入`DIN`，`BST`
- **2021.04.22**: 加入`AutoInt`，特征可以定制vocab
- **2021.04.20**: 加入`gauc`评估指标，加入全局种子便于结果复现
- **2021.04.16**: 支持`NFM`，`CCPM`，`AFM`，`DeepAFM`
- **2021.04.14**: 支持`LR`，`FM`，`DWL`，`FNN`，`DeepFM`，`DCN`，加入`focal loss`

# 使用
Lambda使用：模型生成时插入深度学习模型即可，详情见`examples/run_lambda_dcn.py`
```python
model = LambdaRanker(
    module=DCN,  # 可以是其他任意模型
    features=preprocessor.features,
    cross_layer_num=5,
    dense_emb_dim=4,
    dense_hidden_dims=[128, 128, 32],
    regularizer=1e-5
)
```


