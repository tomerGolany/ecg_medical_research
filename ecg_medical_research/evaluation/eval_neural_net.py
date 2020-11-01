"""Evaluate Neural Networks on ECG datasets."""
import numpy as np
import torch
import logging
from ecg_medical_research.evaluation import metrics
from ecg_medical_research.inference import inference_main
import logging
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import Label
import os
from ecg_medical_research.evaluation import evaluate_doctor_answers, quality
from matplotlib import pyplot as plt


def bokeh_plot_roc(fpr, tpr, roc_auc, probabilites, html_output_dir, step:str = None):

    source = ColumnDataSource(data=
                              dict(FPR=fpr,
                                   TPR=tpr,
                                   probs=probabilites))
    TOOLTIPS = [
        ("FPR, TPR", "$x, $y"),
        ("threshold", "@probs"),
    ]

    p = figure(title="ROC Curve", plot_width=600, plot_height=600,
               x_axis_label='FPR', y_axis_label='TPR', tooltips=TOOLTIPS,
               x_range =(0.0, 1.05),
               y_range=(0.0, 1.05))

    p.line('FPR', 'TPR', line_color="black", line_width=2, source=source)

    auc_label = Label(
                     text=f'AUC = {roc_auc}', render_mode='css',
                     border_line_color='black', border_line_alpha=1.0,
                     background_fill_color='white', background_fill_alpha=1.0)
    p.add_layout(auc_label)
    if step is not None:
        name = f"ROC_CURVE_{step}.html"
    else:
        name = f"ROC_CURVE.html"
    output_file(os.path.join(html_output_dir, name))
    show(p)


def bokeh_plot_precision_recall(precision, recall, probabilities, html_output_dir, step: str=None):
    source = ColumnDataSource(data=
                              dict(recall=recall,
                                   precision=precision,
                                   probs=probabilities,
                                   f1=2 * (precision * recall) / (precision + recall)))
    tooltips = [
        ("Recall, Precision", "$x, $y"),
        ("threshold", "@probs"),
        ("F1", '@f1'),
    ]
    p = figure(title="Precision Recall", plot_width=600, plot_height=600,
               x_axis_label='Recall', y_axis_label='Precision', tooltips=tooltips,
               x_range=(0.0, 1.05),
               y_range=(0.0, 1.05))

    p.line('recall', 'precision', line_color="black", line_width=2, source=source)
    if step is not None:
        name = f"PRECISION_RECALL_{step}.html"
    else:
        name = f"PRECISION_RECALL.html"
    output_file(os.path.join(html_output_dir, name))
    show(p)


def tensorboard_plot_precision_recall(precision, recall, writer, split_name, writer_step):
    fig = plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision Recall graph')
    plt.plot(recall, precision)
    writer.add_figure('{}/precision_recall'.format(split_name), fig, writer_step)
    plt.close()
    fig.clf()
    fig.clear()


def tensorboard_plot_roc(fpr, tpr, roc_auc, writer, split_name, writer_step):
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    writer.add_figure('{}/roc_curve_beat'.format(split_name), fig, writer_step)
    plt.close()
    fig.clf()
    fig.clear()


def evaluation_metrics(predictions_df, html_output_dir, split_name: str, writer=None, writer_step=None):
    #
    # Accuracy:
    #
    total_number_of_ecgs = len(predictions_df)
    binary_ground_truths = predictions_df['ground_truth']
    number_of_correct_predictions = len(predictions_df[predictions_df['binary_predictions'] == binary_ground_truths])
    accuracy = number_of_correct_predictions / total_number_of_ecgs
    logging.info("Accuracy: %.2f", accuracy)
    if writer is not None:
        writer.add_scalars('accuracy', {split_name: accuracy}, global_step=writer_step)
    # writer.add_scalars('cross_entropy_loss', {f'validation_{split}': val_loss}, global_step=global_step)

    #
    # Precision Recall
    #
    predictions = list(predictions_df['prediction'])
    precision, recall, probabilities_pr = precision_recall_curve(list(binary_ground_truths), predictions)
    bokeh_plot_precision_recall(precision, recall, probabilities_pr, html_output_dir, writer_step)
    if writer is not None:
        tensorboard_plot_precision_recall(precision, recall, writer, split_name, writer_step)

    #
    # F1
    #
    f1 = 2 * (precision * recall) / (precision + recall)
    max_f1 = max(f1)
    logging.info("Max F1 score: %s", max_f1)

    max_f1_ind = np.argmax(f1)
    max_f1_threshold = probabilities_pr[max_f1_ind]
    predictions_at_thr = predictions_df['prediction'].apply(
        lambda pred: 1 if pred >= max_f1_threshold else 0)
    tp, fp, tn, fn = evaluate_doctor_answers.perf_measure(list(binary_ground_truths), list(predictions_at_thr))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1_score = tp / (tp + 0.5 * (fp + fn))
    total = tp + fp + tn + fn
    logging.info("F1 Score calculated at threshold %s: %s", max_f1_threshold, f1_score)
    logging.info("TP: %s, FP: %s, TN: %s, FN: %s. TPR: %.4f, FPR: %.4f, total: %d", tp, fp, tn, fn, tpr, fpr, total)

    #
    # ROC
    #
    fpr, tpr, probabilities_roc = roc_curve(list(binary_ground_truths), list(predictions))
    roc_auc = auc(fpr, tpr)
    bokeh_plot_roc(fpr, tpr, roc_auc, probabilities_roc, html_output_dir, writer_step)
    if writer is not None:
        tensorboard_plot_roc(fpr, tpr, roc_auc, writer, split_name, writer_step)

    return max_f1


def evaluate_test_set(checkpoint_path, html_output_dir: str, prediction_threshold: float = 0.5, network=None,
                      device='cpu', writer=None, writer_step=None):
    """Load checkpoint and evaluate it on different metrics on the test set."""
    predictions_df = inference_main.run_inference(checkpoint_path=checkpoint_path, network=network, device=device)

    logging.info("\n\nEvaluating all Test-Set...")
    predictions_df['binary_predictions'] = predictions_df['prediction'].apply(
        lambda pred: 1 if pred >= prediction_threshold else 0)
    evaluation_metrics(predictions_df, html_output_dir, split_name='all_test_set', writer=writer,
                       writer_step=writer_step)

    logging.info("\n\nEvaluating Perfect ECGs...")
    only_perfect_ecgs_numbers = quality.test_set_quality_keep(quality.EcgQuality.PERFECT)
    only_perfect_ecgs_df = predictions_df[predictions_df['ecg_number'].isin(only_perfect_ecgs_numbers)]
    logging.info("Number of Perfect ECGs to evaluate: %d", len(only_perfect_ecgs_df))
    if not os.path.isdir(os.path.join(html_output_dir, 'perfect')):
        os.makedirs(os.path.join(html_output_dir, 'perfect'))
    evaluation_metrics(only_perfect_ecgs_df, os.path.join(html_output_dir, 'perfect'), split_name='perfect_ecgs_only',
                       writer=writer,
                       writer_step=writer_step)

    logging.info("\n\nEvaluating ECGs without artifacts...")
    all_ecgs_except_artifacts_numbers = quality.test_set_quality_filter(quality.EcgQuality.SEVERE_ARTIFACTS)
    all_ecgs_except_artifacts_df = predictions_df[predictions_df['ecg_number'].isin(all_ecgs_except_artifacts_numbers)]
    logging.info("Number of ECGs without artifacts to evaluate: %d", len(all_ecgs_except_artifacts_df))
    if not os.path.isdir(os.path.join(html_output_dir, 'without_artifacts')):
        os.makedirs(os.path.join(html_output_dir, 'without_artifacts'))
    max_f1 = evaluation_metrics(all_ecgs_except_artifacts_df, os.path.join(html_output_dir, 'without_artifacts'),
                                split_name='no_artifacts',
                                writer=writer,
                                writer_step=writer_step)
    return max_f1


def eval_network(dataset_loader, device, net, global_step, writer, loss_fn, split, model_dir):
    total = 0
    correct = 0
    total_loss = 0
    y_preds = np.array([]).reshape((0, 2))
    y_true = np.array([]).reshape((0, 2))
    for i_val, data_val in enumerate(dataset_loader):
        val_inputs, val_labels = (
            data_val['ecg_signal_filterd'].to(device), data_val['echo'].to(device))
        outputs = net(val_inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += val_labels.size(0)
        correct += (predicted == val_labels).sum().item()
        total_loss += loss_fn(outputs, val_labels).item()
        if (i_val + 1) % 5 == 0:
            logging.info("Evaluated on %d/%d...", total, len(dataset_loader.dataset))

        labels_one_hot = torch.nn.functional.one_hot(val_labels, 2)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        y_true = np.concatenate((y_true, labels_one_hot.cpu().detach().numpy()))
        y_preds = np.concatenate((y_preds, probs.cpu().detach().numpy()))
    val_accuracy = 100 * correct / total
    val_loss = total_loss / total
    logging.info("Validation Accuracy: %.2f.\t Loss: %.2f", val_accuracy, val_loss)
    writer.add_scalars('accuracy', {f'validation_{split}': val_accuracy}, global_step=global_step)
    writer.add_scalars('cross_entropy_loss', {f'validation_{split}': val_loss}, global_step=global_step)
    max_f1 = metrics.plt_precision_recall_curve(y_true, y_preds, ['sick', 'health'], writer, global_step, split, model_dir)
    roc_results = metrics.plt_roc_curve(y_true, y_preds, ['sick', 'health'], writer, global_step, split, model_dir)
    return max_f1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_test_set(
      "/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/trainers/mayo_1/checkpoints/checkpoint_epoch_18_iters_16646",
    html_output_dir="/Users/tomer.golany/PycharmProjects/ecg_medical_research/ecg_medical_research/trainers/mayo_4/")