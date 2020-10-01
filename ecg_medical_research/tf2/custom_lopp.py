import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
from ecg_medical_research.tf2.datasets import dataset
from ecg_medical_research.tf2.architectures import resnet50v2
import glob
import os
import numpy as np
from ecg_medical_research.trainers import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import io


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def plt_roc_curve(y_true, y_pred, classes, writer, total_iters, split):
    """Calculate ROC curve from predictions and ground truths.

    writes the roc curve into tensorboard writer object.

    :param y_true:[[1,0,0,0,0], [0,1,0,0], [1,0,0,0,0],...]
    :param y_pred: [0.34,0.2,0.1] , 0.2,...]
    :param classes:5
    :param writer: tensorboard summary writer.
    :param total_iters: total number of training iterations when the predictions where generated.
    :return: List of area-under-curve (AUC) of the ROC curve.
    """
    fpr = {}
    tpr = {}
    roc_auc = {}
    roc_auc_res = []
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[classes[i]], tpr[classes[i]], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])
        roc_auc_res.append(roc_auc[classes[i]])
        fig = plt.figure()
        lw = 2
        plt.plot(fpr[classes[i]], tpr[classes[i]], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[classes[i]])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic beat {}'.format(classes[i]))
        plt.legend(loc="lower right")
        with writer.as_default():
            tf.summary.image('{}/roc_curve_beat_{}'.format(split, classes[i]), plot_to_image(fig), total_iters)
        plt.close()
        fig.clf()
        fig.clear()
    return roc_auc_res



def eval_network(dataset_loader, net, global_step, writer, loss_fn, split):
    total = 0
    correct = 0
    total_loss = 0
    y_preds = np.array([]).reshape((0, 2))
    y_true = np.array([]).reshape((0, 2))
    for x, y in dataset_loader:
        val_inputs, val_labels = x, y
        # print(val_inputs)
        # print("v")
        # print(val_labels)
        labels_one_hot = val_labels
        y_true = np.concatenate((y_true, labels_one_hot.numpy()))
        outputs = net(val_inputs, training=True)
        predicted = tf.argmax(outputs, axis=1, output_type=tf.int32)
        val_labels = tf.argmax(val_labels, axis=1, output_type=tf.int32)
        total += val_labels.numpy().shape[0]
        correct += sum(predicted.numpy() == val_labels.numpy())
        # print(predicted.numpy() == val_labels.numpy())
        # print(correct)
        total_loss += loss_fn(labels_one_hot.numpy(), outputs).numpy()
        probs = tf.keras.activations.softmax(outputs, axis=1)
        y_preds = np.concatenate((y_preds, probs.numpy()))
    val_accuracy = 100 * correct / total
    val_loss = total_loss / total
    print("Test Accuracy: %.2f.\t Loss: %.2f" % (val_accuracy, val_loss))
    # writer.add_scalars('accuracy', {'validation': val_accuracy}, global_step=global_step)
    # writer.add_scalars('cross_entropy_loss', {'validation': val_loss}, global_step=global_step)
    # plt_precision_recall_curve(y_true, y_preds, ['health', 'sick'], writer, global_step)
    roc_results = plt_roc_curve(y_true, y_preds, ['health', 'sick'], writer, global_step, split)
    return roc_results


def grad(model, inputs, targets, loss_object):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True, loss_object=loss_object)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def loss(model, x, y, training, loss_object):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


def train(batch_size, num_epochs, model_dir):
    # Create Tensorboard Writer:
    train_summary_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'tensorboard', 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(model_dir, 'tensorboard', 'test'))

    # Create datasets:
    train_files = glob.glob("datasets/data/train*.record")
    print(train_files)
    train_ds = dataset.create_tf_dataset(filenames=train_files)
    train_ds = train_ds.shuffle(buffer_size=100)
    train_ds = train_ds.batch(batch_size)
    # train_ds = train_ds

    test_files = glob.glob("datasets/data/test*.record")
    print(test_files)
    test_ds = dataset.create_tf_dataset(filenames=test_files)
    test_ds = test_ds.batch(batch_size)

    # Create loss and optimizers:
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Create Model:
    model = resnet50v2.ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(5499, 12),
        pooling=None,
        classes=2,
        classifier_activation=None)

    global_step = 0
    # batch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    eval_network(test_ds, model, global_step, test_summary_writer, loss_object, split='test')
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Training loop
        for x, y in train_ds:
            global_step += 1
            # Optimize the model
            loss_value, grads = grad(model, x, y, loss_object)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            with train_summary_writer.as_default():
                tf.summary.scalar("batch_loss", loss_value, step=global_step)
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True))
            if global_step % 25 == 0:
                print(f"Epoch: [{epoch}]. Iteration [{global_step}]: Train loss: {loss_value}")

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', epoch_accuracy.result(), step=epoch)

        # End epoch
        # train_loss_results.append(epoch_loss_avg.result())
        # train_accuracy_results.append(epoch_accuracy.result())
        test_accuracy = tf.keras.metrics.Accuracy()
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

        for (x, y) in test_ds:
            # training=False is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            logits = model(x, training=False)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            test_accuracy(prediction, tf.argmax(y, axis=1, output_type=tf.int32))
            loss_batch = loss_object(y, logits)
            test_loss.update_state(loss_batch)
        eval_network(test_ds, model, global_step, test_summary_writer, loss_object, split='test')
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
        print("Epoch {:03d}: Loss: {:.3f}, Train Average Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(),
                                                                                  epoch_accuracy.result()))
        print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


if __name__ == "__main__":

    batch_size = 32
    num_epochs = 50
    model_dir = 'model_outputs/custom_loop_11'
    train(batch_size, num_epochs, model_dir)