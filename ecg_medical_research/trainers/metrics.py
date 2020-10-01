"""Evaluation metrics for ECG classification models."""
import logging
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from sklearn.metrics import precision_recall_curve
import os
import numpy as np
from ecg_medical_research.evaluation import evaluate_doctor_answers


def plt_roc_curve(y_true, y_pred, classes, writer, total_iters, split, model_dir):
    """Calculate ROC curve from predictions and ground truths.

    writes the roc curve into tensorboard writer object.

    :param y_true:[[1,0,0,0,0], [0,1,0,0], [1,0,0,0,0],...]
    :param y_pred: [0.34,0.2,0.1] , 0.2,...]
    :param classes:5
    :param writer: tensorboard summary writer.
    :param total_iters: total number of training iterations when the predictions where generated.
    :return: List of area-under-curve (AUC) of the ROC curve.
    """
    logging.info("%s: Number of Ground Truths: %d. Number of Predictions: %d", split, len(y_true), len(y_pred))
    fpr = {}
    tpr = {}
    roc_auc = {}
    roc_auc_res = []
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], probs = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])
        roc_auc_res.append(roc_auc[classes[i]])
        fig = plt.figure()
        lw = 2
        bokeh_plot_roc(fpr[classes[i]], tpr[classes[i]], probs, i, total_iters, split, model_dir)
        plt.plot(fpr[classes[i]], tpr[classes[i]], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[classes[i]])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic beat {}'.format(classes[i]))
        plt.legend(loc="lower right")
        writer.add_figure('{}/roc_curve_beat_{}'.format(split, classes[i]), fig, total_iters)
        plt.close()
        fig.clf()
        fig.clear()
    return roc_auc_res


def bokeh_plot_roc(fpr, tpr, probabilites, class_num, total_iters, split, model_dir):

    source = ColumnDataSource(data=
                              dict(FPR=fpr,
                                   TPR=tpr,
                                   probs=probabilites))

    TOOLTIPS = [
        ("FPR, TPR", "$x, $y"),
        ("threshold", "@probs"),
    ]

    p = figure(plot_width=600, plot_height=600,
               x_axis_label='FPR', y_axis_label='TPR', tooltips=TOOLTIPS,
               x_range =(0.0, 1.05),
               y_range=(0.0, 1.05))

    p.line('FPR', 'TPR', line_color="black", line_width=2, source=source)
    output_file(os.path.join(model_dir, "outputs", f"ROC_CLASS_{class_num}_{total_iters}_{split}.html"))
    show(p)


def plt_precision_recall_curve(y_true, y_pred, classes, writer, total_iters, split, model_dir):
    # For each class
    precision = dict()
    recall = dict()
    n_classes = len(classes)
    # average_precision = dict()
    i = 1
    # for i in range(n_classes):
        # logging.info(classes[i])
    precision[classes[i]], recall[classes[i]], probabilities = precision_recall_curve(y_true[:, i],
                                                        y_pred[:, i])

    max_f1 = bokeh_plot_precision_recall(precision[classes[i]], recall[classes[i]], probabilities, i, total_iters, split,
                                model_dir, y_true[:, 1], y_pred[:, 1])
    fig = plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall graph')
    plt.plot(recall[classes[i]], precision[classes[i]])
    writer.add_figure('{}/precision_recall_{}'.format(split, classes[i]), fig, total_iters)
    plt.close()
    fig.clf()
    fig.clear()
    return max_f1


def bokeh_plot_precision_recall(precision, recall, probabilities, class_num, total_iters, split,
                                    model_dir, y_true, y_pred):
    source = ColumnDataSource(data=
                              dict(recall=recall,
                                   precision=precision,
                                   probs=probabilities,
                                   f1=2 * (precision * recall) / (precision + recall)))

    TOOLTIPS = [
        ("Recall, Precision", "$x, $y"),
        ("threshold", "@probs"),
        ("F1", '@f1'),
    ]

    p = figure(plot_width=600, plot_height=600,
               x_axis_label='Recall', y_axis_label='Precision', tooltips=TOOLTIPS,
               x_range=(0.0, 1.05),
               y_range=(0.0, 1.05))

    p.line('recall', 'precision', line_color="black", line_width=2, source=source)
    output_file(os.path.join(model_dir, "outputs",  f"PRECISION_RECALL_CLASS_{class_num}_{total_iters}_{split}.html"))
    show(p)
    f1 = 2 * (precision * recall) / (precision + recall)
    max_f1 = max(f1)
    logging.info("Max F1: %s", max_f1)
    max_f1_ind = np.argmax(f1)
    max_f1_threshold = probabilities[max_f1_ind]
    predictions_at_thr = [1 if pred >= max_f1_threshold else 0 for pred in y_pred]
    tp, fp, tn, fn = evaluate_doctor_answers.perf_measure(y_true, predictions_at_thr)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1_score = tp / (tp + 0.5 * (fp + fn))
    logging.info("F1 Score calculated at threshold %f: %f", max_f1_threshold, f1_score)
    logging.info("TP: %s, FP: %s, TN: %s, FN: %s. TPR: %s, FPR: %s", tp, fp, tn, fn, tpr, fpr)
    return max_f1
