"""Train module."""
import os
from typing import Optional
from ecg_medical_research.data_reader import pytorch_dataset
from ecg_medical_research.evaluation import metrics
from ecg_medical_research.architectures import tcn, lenet, resnet, mayo
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
from ecg_medical_research.data_reader.dataset import ecg_to_echo_dataset
from ecg_medical_research.evaluation import eval_neural_net
import numpy as np
import pandas as pd
from ecg_medical_research.inference import inference_main


def eval_network(dataset_loader, device, net, global_step, writer, loss_fn, split, model_dir):
    total = 0
    correct = 0
    total_loss = 0
    y_preds = np.array([]).reshape((0, 2))
    y_true = np.array([]).reshape((0, 2))
    dicom_files = []
    for i_val, data_val in enumerate(dataset_loader):
        val_inputs, val_labels = (
            data_val['ecg_signal_filtered'].to(device), data_val['echo'].to(device))
        dicom_files += data_val['dicom_file']
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
    val_accuracy =  correct / total
    val_loss = total_loss / total
    logging.info("Validation Accuracy: %.2f.\t Loss: %.2f", val_accuracy, val_loss)
    writer.add_scalars('accuracy', {f'validation_{split}': val_accuracy}, global_step=global_step)
    writer.add_scalars('cross_entropy_loss', {f'validation_{split}': val_loss}, global_step=global_step)
    max_f1 = metrics.plt_precision_recall_curve(y_true, y_preds, ['sick', 'health'], writer, global_step, split, model_dir)
    roc_results = metrics.plt_roc_curve(y_true, y_preds, ['sick', 'health'], writer, global_step, split, model_dir)
    df = pd.DataFrame(list(zip(y_true[:, 1], y_preds[:, 1], dicom_files)),
                      columns=['ground_truth', 'prediction', 'dicom file'])
    df['label'] = df['ground_truth'].apply(lambda gt: 'normal' if gt == 1 else 'sick')
    df.to_csv("inference_v2.csv", index=False)
    return max_f1


def accuracy_metric(network_obj: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> float:
    """Calculate Accuracy.

    :param network_obj: Pytorch module network which accepts inputs and masks.
    :param inputs: Input of shape [batch_size, 12, sequence_length]
    :param labels: Ground truth labels of shape [batch_size, 2]
    :param mask: Optional mask for the inputs.
    :return:
        Accuracy.
    """
    logits = network_obj(inputs)
    _, max_indices = torch.max(logits, 1)
    accuracy = (max_indices == labels).sum().item()/max_indices.size()[0]
    return accuracy


def train(excel_file: str, dicom_dir: str, model_dir: str, batch_size: int, num_iterations: int,
          device) -> None:

    writer = SummaryWriter(log_dir=os.path.join(model_dir, 'tensorboard'))
    os.mkdir(os.path.join(model_dir, 'checkpoints'))
    os.mkdir(os.path.join(model_dir, 'outputs'))
    # net = tcn.SimpleTCN(num_f_maps=64, num_classes=2).to(device)
    # net = lenet.Lenet(input_features_size=10000)
    # net = resnet.resnet50().to(device)
    net = mayo.MayoNet().to(device)

    train_ecg_dataset = ecg_to_echo_dataset.ECGToEchoDataset(excel_path=excel_file, dicom_dir=dicom_dir,
                                                             split_name=None,
                                                             transform=ecg_to_echo_dataset.ToTensor(), threshold_35=False)

    train_loader = DataLoader(train_ecg_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                              # collate_fn=ecg_to_echo_dataset.collate_fn_simetric_padding)

    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.RMSprop(net.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    running_loss = 0
    epoch = 1
    global_step = 0
    best_roc = -1
    while global_step <= num_iterations:
        num_correct = 0
        num_examples = 0
        for i, data in enumerate(train_loader):
            num_examples += data['echo'].size()[0]
            # net.train()
            global_step += 1
            # inputs, labels, mask = data['ecg_signal'].to(device), data['echo'].to(device), data['mask'].to(device)
            inputs, labels = data['ecg_signal_filtered'].to(device), data['echo'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs = net(inputs, mask)
            # print(inputs.shape)
            outputs = net(inputs)
            _, max_indices = torch.max(outputs, 1)
            num_correct += (max_indices == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalars('cross_entropy_loss', {'train_batch': loss}, global_step=global_step)
            train_batch_accuracy = accuracy_metric(net, inputs, labels)

            if global_step % 25 == 0:
                writer.add_scalars('accuracy', {'train_batch': train_batch_accuracy}, global_step=global_step)
            # print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                logging.info('Epoch [%d]: Global iteration: [%d] loss: %.5f. Accuracy: %.5f.', epoch, global_step,
                             running_loss / (global_step * inputs.size()[0]), train_batch_accuracy)
                running_loss = 0.0
                # Calculate Metrics on validation:
                logging.info("Start evaluation on validation set...")
                with torch.no_grad():
                    # net.eval()
                    # max_f1 = eval_network(validation_loader, device, net, global_step, writer, criterion,
                    #                            split='test_all', model_dir=model_dir)

                    # max_f1 = eval_network(perfect_loader, device, net, global_step, writer, criterion,
                    #                            split='test_on_perfect', model_dir=model_dir)
                    #
                    # max_f1 = eval_network(artifacts_loader, device, net, global_step, writer, criterion,
                    #                            split='test_without_artifacts', model_dir=model_dir)
                    max_f1 = eval_neural_net.evaluate_test_set(checkpoint_path=None, html_output_dir=model_dir,
                                                               network=net, device=device, writer=writer,
                                                               writer_step=global_step)

                    if max_f1 > best_roc:
                        best_step = 'checkpoints/checkpoint_epoch_{}_iters_{}'.format(epoch, global_step)
                        best_roc = max_f1
                        logging.info("New best F1 on Test set: %f", best_roc)
                        logging.info("Saving model...")
                        torch.save({
                            'global_step': global_step,
                            'epoch': epoch,
                            'resnet': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': criterion,
                        }, os.path.join(model_dir, 'checkpoints/checkpoint_epoch_{}_iters_{}'.format(epoch,
                                                                                                     global_step)))
                        chk_path = os.path.join(model_dir, 'checkpoints/checkpoint_epoch_{}_iters_{}'.format(epoch,
                                                                                                     global_step))
                        # Verify that we get the same F1 value after exporting the checkpoint:
                        # logging.info("**Verification.**")
                        # eval_neural_net.evaluate_test_set(chk_path, model_dir)

        epoch += 1
        accuracy_epoc = num_correct / num_examples
        writer.add_scalars('accuracy', {'train_epoch': accuracy_epoc}, global_step=global_step)
    # Calculate ROC on train:
    logging.info("Runing evaluation on train...")
    with torch.no_grad():
        roc_results = eval_network(train_loader, device, net, global_step, writer, criterion, split='train', model_dir=model_dir)
    logging.info("Training done...Completed %d epochs. Best F1 = %f best ste: %s", epoch, best_roc, best_step)



