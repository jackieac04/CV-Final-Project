"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import io
import os
import re
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import hyperparameters as hp


def plot_to_image(figure):
    """ Converts a pyplot figure to an image tensor. """

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


class ImageLabelingLogger(tf.keras.callbacks.Callback):
    """ Keras callback for logging a plot of test images and their
    predicted labels for viewing in Tensorboard. """

    def __init__(self, logs_path, datasets):
        super(ImageLabelingLogger, self).__init__()

        self.datasets = datasets
        self.task = datasets.task
        self.logs_path = logs_path

        print("Done setting up image labeling logger.")

    def on_epoch_end(self, epoch, logs=None):
        self.log_image_labels(epoch, logs)

    def log_image_labels(self, epoch_num, logs):
        """ Writes a plot of test images and their predicted labels
        to disk. """

        fig = plt.figure(figsize=(9, 9))
        count_all = 0
        count_misclassified = 0
        
        for batch in self.datasets.test_data:
            misclassified = []
            correct_labels = []
            wrong_labels = []

            for i, image in enumerate(batch[0]):
                plt.subplot(5, 5, min(count_all+1, 25))

                correct_class_idx = batch[1][i]
                probabilities = self.model(np.array([image])).numpy()[0]
                predict_class_idx = np.argmax(probabilities)

                if self.task == '1':
                    image = np.clip(image, 0., 1.)
                    plt.imshow(image, cmap='gray')
                else:
                    # Undo VGG preprocessing
                    mean = [103.939, 116.779, 123.68]
                    image[..., 0] += mean[0]
                    image[..., 1] += mean[1]
                    image[..., 2] += mean[2]
                    image = image[:, :, ::-1]
                    image = image / 255.
                    image = np.clip(image, 0., 1.)

                    plt.imshow(image)

                is_correct = correct_class_idx == predict_class_idx

                title_color = 'b' if is_correct else 'r'

                plt.title(
                    self.datasets.idx_to_class[predict_class_idx],
                    color=title_color)
                plt.axis('off')
                
                # output individual images with wrong labels
                if not is_correct:
                    count_misclassified += 1
                    misclassified.append(image)
                    correct_labels.append(correct_class_idx)
                    wrong_labels.append(predict_class_idx)

                count_all += 1
                
                # ensure there are >= 2 misclassified images
                if count_all >= 25 and count_misclassified >= 2:
                    break

            if count_all >= 25 and count_misclassified >= 2:
                break

        figure_img = plot_to_image(fig)

        file_writer_il = tf.summary.create_file_writer(
            self.logs_path + os.sep + "image_labels")

        misclassified_path = "misclassified" + self.logs_path[self.logs_path.index(os.sep):]
        if not os.path.exists(misclassified_path):
            os.makedirs(misclassified_path)
        for correct, wrong, img in zip(correct_labels, wrong_labels, misclassified):
            wrong = self.datasets.idx_to_class[wrong]
            correct= self.datasets.idx_to_class[correct]
            image_name = wrong + "_predicted" + ".png"
            if not os.path.exists(misclassified_path + os.sep + correct):
                os.makedirs(misclassified_path + os.sep + correct)
            plt.imsave(misclassified_path + os.sep + correct + os.sep + image_name, img)

        with file_writer_il.as_default():
            tf.summary.image("0 Example Set of Image Label Predictions (blue is correct; red is incorrect)",
                             figure_img, step=epoch_num)
            for label, wrong, img in zip(correct_labels, wrong_labels, misclassified):
                img = tf.expand_dims(img, axis=0)
                tf.summary.image("1 Example @ epoch " + str(epoch_num) + ": " + self.datasets.idx_to_class[label] + " misclassified as " + self.datasets.idx_to_class[wrong], 
                                 img, step=epoch_num)

class ConfusionMatrixLogger(tf.keras.callbacks.Callback):
    """ Keras callback for logging a confusion matrix for viewing
    in Tensorboard. """

    def __init__(self, logs_path, datasets):
        super(ConfusionMatrixLogger, self).__init__()

        self.datasets = datasets
        self.logs_path = logs_path

    def on_epoch_end(self, epoch, logs=None):
        self.log_confusion_matrix(epoch, logs)

    def log_confusion_matrix(self, epoch, logs):
        """ Writes a confusion matrix plot to disk. """

        test_pred = []
        test_true = []
        count = 0
        for i in self.datasets.test_data:
            test_pred.append(self.model.predict(i[0]))
            test_true.append(i[1])
            count += 1
            if count >= 1500 / hp.batch_size:
                break

        test_pred = np.array(test_pred)
        test_pred = np.argmax(test_pred, axis=-1).flatten()
        test_true = np.array(test_true).flatten()

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        cm = sklearn.metrics.confusion_matrix(test_true, test_pred)
        figure = self.plot_confusion_matrix(
            cm, class_names=self.datasets.classes)
        cm_image = plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer(
            self.logs_path + os.sep + "confusion_matrix")

        with file_writer_cm.as_default():
            tf.summary.image(
                "Confusion Matrix (on validation set)", cm_image, step=epoch)

    def plot_confusion_matrix(self, cm, class_names):
        """ Plots a confusion matrix returned by
        sklearn.metrics.confusion_matrix(). """

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        cm = np.around(cm.astype('float') / cm.sum(axis=1)
                       [:, np.newaxis], decimals=2)

        threshold = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure


class CustomModelSaver(tf.keras.callbacks.Callback):
    """ Custom Keras callback for saving weights of networks. """

    def __init__(self, checkpoint_dir, task, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.task = task
        self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
        """ At epoch end, weights are saved to checkpoint directory. """

        min_acc_file, max_acc_file, max_acc, num_weights = \
            self.scan_weight_files()

        cur_acc = logs["val_sparse_categorical_accuracy"]

        # Only save weights if test accuracy exceeds the previous best
        # weight file
        if cur_acc > max_acc:
            save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(
                epoch, cur_acc)

            if self.task == '1':
                save_location = self.checkpoint_dir + os.sep + "your." + save_name
                print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                       "maximum TEST accuracy.\nSaving checkpoint at {location}")
                       .format(epoch + 1, cur_acc, location = save_location))
                self.model.save_weights(save_location)
            else:
                save_location = self.checkpoint_dir + os.sep + "vgg." + save_name
                print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                       "maximum TEST accuracy.\nSaving checkpoint at {location}")
                       .format(epoch + 1, cur_acc, location = save_location))
                # Only save weights of classification head of VGGModel
                self.model.head.save_weights(save_location)

            # Ensure max_num_weights is not exceeded by removing
            # minimum weight
            if self.max_num_weights > 0 and \
                    num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir + os.sep + min_acc_file)
        else:
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) DID NOT EXCEED "
                   "previous maximum TEST accuracy.\nNo checkpoint was "
                   "saved").format(epoch + 1, cur_acc))


    def scan_weight_files(self):
        """ Scans checkpoint directory to find current minimum and maximum
        accuracy weights files as well as the number of weights. """

        min_acc = float('inf')
        max_acc = 0
        min_acc_file = ""
        max_acc_file = ""
        num_weights = 0

        files = os.listdir(self.checkpoint_dir)

        for weight_file in files:
            if weight_file.endswith(".h5"):
                num_weights += 1
                file_acc = float(re.findall(
                    r"[+-]?\d+\.\d+", weight_file.split("acc")[-1])[0])
                if file_acc > max_acc:
                    max_acc = file_acc
                    max_acc_file = weight_file
                if file_acc < min_acc:
                    min_acc = file_acc
                    min_acc_file = weight_file

        return min_acc_file, max_acc_file, max_acc, num_weights
