# Start with some documentation and a bunch of imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

def eval_AUC(y_true, y_preds):
    """
    return overall AUC, float
    """
    return roc_auc_score(y_true, y_preds)

def eval_AP(y_true, y_preds):
    """
    return average precision, float
    """
    return average_precision_score(y_true, y_preds)

def eval_func_AUC(n_classes, class_names, y_true, y_preds):
    """
    plot the AUC-ROC curve
    n_classes: number of classes, integer
    class_names: classes names, list
    y_true: groud-truth targets, array
    y_preds: predicted targets, array
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(class_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for multiclasses data')
    plt.legend(loc='lower right')
    plt.show()

def eval_func_PR(n_classes, class_names, y_true, y_preds):
    """
    plot the PR-AUC curve
    """
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_preds[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_preds[:, i])
    for i in range(n_classes):
        plt.plot(precision[i], recall[i],
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(class_names[i], average_precision[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR-AUC for multiclasses data')
    plt.legend(loc='lower right')
    plt.show()

def eval_func_classification_report(y_true, y_preds):
    print(metrics.classification_report(y_true, y_preds))
    return

def eval_func_confusion_matrix(y_true, y_preds, class_names):
    """
    plot confusion matrix as heatmap
    """
    cm = confusion_matrix(y_true.argmax(axis=1), y_preds.argmax(axis=1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, columns=class_names, index=class_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (8,5))
    sns.set(font_scale=1.5)
    sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 16})
    plt.show()