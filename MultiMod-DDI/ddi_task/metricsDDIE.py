import csv
import sys
import logging
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support, \
    classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import os
# 它使用scikit-learn库来计算机器学习模型的性能指标，并提供了一些用于数据预处理和可视化的函数
logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support, classification_report
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    save_count = 0
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix


    def ddie_compute_metrics(preds, labels, output_dir, plot_curves=True):
        label_list = ('other', 'mechanism', 'effect', 'advise', 'int')
        global save_count
        # 获取实际存在的类别（过滤掉没有样本的类别）
        unique_labels = np.unique(labels)
        valid_labels = sorted(unique_labels)  # 实际出现的类别索引

        # 计算基础指标（只考虑实际存在的类别）
        p, r, f, _ = precision_recall_fscore_support(
            y_pred=preds,
            y_true=labels,
            labels=valid_labels,  # 只计算有样本的类别
            average='macro'
        )
        result = {"Precision": p, "Recall": r, "macroF": f}

        # 计算ROC曲线和AUC（只考虑实际存在的类别）
        num_classes = len(label_list)
        one_hot = np.eye(num_classes)
        preds_onehot = one_hot[preds]
        labels_onehot = one_hot[labels]

        fpr, tpr, roc_auc = dict(), dict(), dict()
        precision, recall, average_precision = dict(), dict(), dict()

        for i in valid_labels:  # 只处理有样本的类别
            # ROC曲线
            fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], preds_onehot[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            # PR曲线
            precision[i], recall[i], _ = precision_recall_curve(labels_onehot[:, i], preds_onehot[:, i])
            average_precision[i] = average_precision_score(labels_onehot[:, i], preds_onehot[:, i])

        # 计算micro平均ROC和PR曲线
        fpr["micro"], tpr["micro"], _ = roc_curve(labels_onehot.ravel(), preds_onehot.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        precision["micro"], recall["micro"], _ = precision_recall_curve(labels_onehot.ravel(), preds_onehot.ravel())
        average_precision["micro"] = average_precision_score(labels_onehot, preds_onehot, average="micro")

        # 计算macro平均ROC曲线
        all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_labels]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in valid_labels:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(valid_labels)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # 计算macro平均PR曲线（修正版）
        # 首先找到所有类别的统一召回率点
        all_recall = np.unique(np.concatenate([recall[i] for i in valid_labels]))
        mean_precision = np.zeros_like(all_recall)

        # 对每个类别进行插值
        for i in valid_labels:
            mean_precision += np.interp(all_recall, np.flip(recall[i]), np.flip(precision[i]))

        # 计算平均精度
        mean_precision /= len(valid_labels)
        recall["macro"] = all_recall
        precision["macro"] = mean_precision

        # 计算macro平均AP
        average_precision["macro"] = np.mean([average_precision[i] for i in valid_labels])

        result["micro_auc"] = roc_auc["micro"]
        result["macro_auc"] = roc_auc["macro"]
        result["micro_avg_precision"] = average_precision["micro"]
        result["macro_avg_precision"] = average_precision["macro"]

        # 绘制ROC曲线和PR曲线
        if plot_curves:
            # 创建两个图像
            plt.figure(figsize=(12, 5))

            # 设置全局背景色
            plt.rcParams['axes.facecolor'] = '#e6e6fa'
            # 设置边框颜色为白色
            plt.rcParams['axes.edgecolor'] = 'white'
            # 开启网格线
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.color'] = 'white'

            # 绘制ROC曲线
            plt.subplot(1, 2, 1)
            # plt.plot(fpr["micro"], tpr["micro"],
            #          label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
            #          color='deeppink', linestyle=':', linewidth=4)
            #
            # plt.plot(fpr["macro"], tpr["macro"],
            #          label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
            #          color='navy', linestyle=':', linewidth=4)

            colors = ['#3467a9', '#e59233', '#37a967', '#d93025', '#7c3fa9']
            for i, color in zip(valid_labels, colors[:len(valid_labels)]):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of class {label_list[i]} (area = {roc_auc[i]:0.2f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")

            # 绘制PR曲线（修正版）
            plt.subplot(1, 2, 2)
            # plt.plot(recall["micro"], precision["micro"],
            #          label=f'micro-average PR curve (AP = {average_precision["micro"]:0.2f})',
            #          color='deeppink', linestyle=':', linewidth=4)
            #
            # plt.plot(recall["macro"], precision["macro"],
            #          label=f'macro-average PR curve (AP = {average_precision["macro"]:0.2f})',
            #          color='navy', linestyle=':', linewidth=4)

            for i, color in zip(valid_labels, colors[:len(valid_labels)]):
                plt.plot(recall[i], precision[i], color=color, lw=2,
                         label=f'PR curve of class {label_list[i]} (AP = {average_precision[i]:0.2f})')

            # 绘制随机分类器的基线
            chance = np.sum(labels_onehot) / labels_onehot.size
            # plt.plot([0, 1], [chance, chance], 'k--', lw=2, label=f'Random (AP = {chance:0.2f})')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall (PR) Curve')
            plt.legend(loc="lower left")

            plt.tight_layout()

            if output_dir:
                ROC_folder_path = os.path.join(output_dir, 'ROC')
                # 判断文件夹是否存在，不存在则创建
                if not os.path.exists(ROC_folder_path):
                    os.makedirs(ROC_folder_path)
                save_count += 1
                file_name = f"{ROC_folder_path}/roc_pr_curves_{save_count}.png"
                plt.savefig(file_name)
            plt.show()

        cm = confusion_matrix(labels, preds, labels=valid_labels)
        plt.figure(figsize=(8, 6))
        plt.rcParams['axes.grid'] = False
        # 使用与归一化混淆矩阵相同的紫色系颜色映射
        cmap = plt.get_cmap('Purples')
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=[label_list[i] for i in valid_labels],
                    yticklabels=[label_list[i] for i in valid_labels],
                    cbar_kws={'shrink': 0.5},
                    linewidths=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        if output_dir:
            confusion_folder_path = os.path.join(output_dir, 'confusion')
            if not os.path.exists(confusion_folder_path):
                os.makedirs(confusion_folder_path)
            save_count += 1
            cm_file_name = f"{confusion_folder_path}/confusion_matrix_{save_count}.png"
            plt.savefig(cm_file_name)
        plt.show()

        # 计算归一化混淆矩阵
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        norm_cm = cm / row_sums
        norm_cm[np.isnan(norm_cm)] = 0  # 处理可能出现的NaN值（如果某行总和为0）

        # 绘制归一化混淆矩阵
        plt.figure(figsize=(8, 6))
        plt.rcParams['axes.grid'] = False
        # 使用Purples颜色映射
        cmap = plt.get_cmap('Purples')
        sns.heatmap(norm_cm, annot=True, fmt='.3f', cmap=cmap,
                    xticklabels=[label_list[i] for i in valid_labels],
                    yticklabels=[label_list[i] for i in valid_labels],
                    cbar_kws={'shrink': 0.5},
                    linewidths=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Normalized Confusion Matrix')
        if output_dir:
            norm_confusion_folder_path = os.path.join(output_dir, 'normalized_confusion')
            if not os.path.exists(norm_confusion_folder_path):
                os.makedirs(norm_confusion_folder_path)
            save_count += 1
            norm_cm_file_name = f"{norm_confusion_folder_path}/normalized_confusion_matrix_{save_count}.png"
            plt.savefig(norm_cm_file_name)
        plt.show()

        # 生成每个类别的详细指标（只处理实际存在的类别）
        # （这部分代码保持不变，被注释掉了）

        return result


    def pretraining_compute_metrics(task_name, preds, labels, every_type=False):
        acc = accuracy_score(y_pred=preds, y_true=labels)
        result = {
            "Accuracy": acc,
        }
        return result
