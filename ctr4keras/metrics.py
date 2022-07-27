import math
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss



def gauc(y_true, y_score, indicators=None):
    y_true = np.array(y_true).reshape(-1)
    y_score = np.array(y_score).reshape(-1)
    if indicators is None:
        indicators = np.array([1] * y_true.shape[-1])
    else:
        indicators = np.array(indicators).reshape(-1)

    uniq_indicators = list(set(indicators))

    auc_dict = dict()
    total_cnt = 0
    for ind in uniq_indicators:
        indices = indicators == ind
        curr_y_true = y_true[indices]
        curr_y_score = y_score[indices]

        y_true_sum, cnt = curr_y_true.sum(), sum(indices)
        if y_true_sum == cnt or y_true_sum == 0:
            # if y all are 1 or 0, do not calc auc
            continue

        total_cnt += cnt
        auc_dict[ind] = cnt * roc_auc_score(y_true=curr_y_true, y_score=curr_y_score)  # auc * cnt

    gauc = sum(auc_dict.values()) / total_cnt + 1e-8
    return gauc


def evaluate_report(y_true, y_score, threshold=0.5, verbose=1, report_output_dict=False):
    y_pred = np.where(y_score > threshold, 1, 0)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    log_loss_ = log_loss(y_true=y_true, y_pred=y_score)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=report_output_dict)
    if verbose > 0:
        print("AUC:", auc)
        print("Log loss:", log_loss_)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print(report)
    return auc, log_loss_, accuracy, precision, recall, report


def dcg(scores, eval_at=None):
    return sum([((2 ** s) - 1) / math.log2(i + 2) for i, s in enumerate(scores[:eval_at or len(scores)])])


def idcg(scores, eval_at=None):
    return dcg(sorted(scores, reverse=True), eval_at)


def ndcg(scores, eval_at=None):
    return dcg(scores, eval_at) / (idcg(scores, eval_at) + 1e-8)


def rank_evaluate(y_pred, y_true, group, eval_at=[1, 3, 5, 10]):
    g = [0]
    for i in group:
        g.append(g[-1] + i)

    indice = list()
    length = list()
    ndcg_scores = list()
    for s, e in zip(g[:-1], g[1:]):
        scores = y_pred[s: e].tolist()
        labels = y_true[s: e].tolist()
        if sum(labels) > 0:
            score = scores[labels.index(1)]
            scores = sorted(scores, reverse=True)
            idx = scores.index(score)
            indice.append(idx)
            length.append(len(labels))

            sidx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            pred_label = [labels[i] for i in sidx]
            ndcg_score = [ndcg(pred_label, ea) for ea in eval_at]
            ndcg_scores.append(ndcg_score)

    avg_rank = sum(indice) / len(indice)
    len_rank = sum(length) / len(length)
    top_k_num = [len([i for i in indice if i < ea]) for ea in eval_at]
    top_k_rate = [n / len(indice) for n in top_k_num]
    avg_ndcgs = [sum([n[i] for n in ndcg_scores]) / len(ndcg_scores) for i in range(len(ndcg_scores[0]))]

    topk_str = ', '.join([f'top@{j} rate: {n:.4f}' for n, j in zip(top_k_rate, eval_at)])
    ndcg_str = ', '.join([f'ndcg@{j} rate: {n:.4f}' for n, j in zip(avg_ndcgs, eval_at)])
    print(f'Average rank: {avg_rank:.2f}({len_rank:.2f}), {topk_str}, {ndcg_str}')