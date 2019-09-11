import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
import shutil

from data_input import DataInput
from CNN_models.simple_model_1 import simple_model_1
from CNN_models.simple_model_2 import simple_model_2
from CNN_models.inception import inception
from CNN_models.resnet import resnet
from CNN_models.inception_resnet import inception_resnet


def main():

    config = json.load(open("config.json", "r"))

    DATA_PATH = config["DATA_PATH"]
    INITIAL_LR = float(config["INITIAL_LR"])
    DECAY_STEPS_LR = int(config["DECAY_STEPS_LR"])
    DECAY_FACTOR_LR = float(config["DECAY_FACTOR_LR"])
    BATCH_SIZE = int(config["BATCH_SIZE"])
    NUM_STEPS = int(config["NUM_STEPS"])
    OUTPUT_DIR = config["OUTPUT_DIR"]
    VAL_SET_SIZE = int(config["VAL_SET_SIZE"])
    KEEP_DROPOUT_PROB = float(config["KEEP_DROPOUT_PROB"])
    WEIGHT_DECAY = float(config["WEIGHT_DECAY"])
    MODEL = config["MODEL"]
    AUGMENT_PROB = float(config["AUGMENT_PROB"])
    LOSS = config["LOSS"]

    # create model output folder and copy corresponding config file to it
    now = datetime.now()
    current_time = now.strftime("%D_%H%M%S")
    current_time = current_time.replace("/", "")
    model_path = os.path.join(OUTPUT_DIR, "_".join([MODEL, current_time]))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    shutil.copy("config.json", os.path.join(model_path, "config.json"))
    shutil.copy("CNN_models/%s.py" % MODEL, os.path.join(model_path, "model.py"))

    # placeholders
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="learning_rate_ph")
    images_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="input_images_ph")
    labels_ph = tf.placeholder(tf.int32, shape=[None], name="labels_ph")
    accuracy_ph = tf.placeholder(tf.float32, shape=[], name="accuracy_ph")
    accuracy_per_class_phs = [tf.placeholder(tf.float32, shape=[], name="accuracy_per_classs/class_%s_ph" % str(i)) for i in range(10)]
    training_ph = tf.placeholder(tf.bool, shape=[], name="training_ph")

    # choose model
    if MODEL == "simple_model_1":
        logits = simple_model_1(images_ph, dropout_prob=KEEP_DROPOUT_PROB, weight_decay=WEIGHT_DECAY, is_training=training_ph)
    elif MODEL == "simple_model_2":
        logits = simple_model_2(images_ph, dropout_prob=KEEP_DROPOUT_PROB, weight_decay=WEIGHT_DECAY, is_training=training_ph)
    elif MODEL == "inception":
        logits = inception(images_ph, dropout_prob=KEEP_DROPOUT_PROB, weight_decay=WEIGHT_DECAY, is_training=training_ph)
    elif MODEL == "resnet":
        logits = resnet(images_ph, dropout_prob=KEEP_DROPOUT_PROB, weight_decay=WEIGHT_DECAY, is_training=training_ph)
    elif MODEL == "inception_resnet":
        logits = inception_resnet(images_ph, dropout_prob=KEEP_DROPOUT_PROB, weight_decay=WEIGHT_DECAY, is_training=training_ph)

    # create loss
    if LOSS == "CROSS_ENTROPY":
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph, name="c_entropy"))
    elif LOSS == "CROSS_ENTROPY_WEIGHTED":
        class_weights = tf.constant([2, 1, 2, 1, 2, 1, 2, 1, 1, 1])
        weights = tf.gather(class_weights, labels_ph)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_ph, logits=logits, weights=weights)
    elif LOSS == "FOCAL":
        gamma = 2
        preds = tf.nn.softmax(logits, dim=-1)
        labels_one_hot = tf.one_hot(labels_ph, depth=preds.shape[1])
        loss = -labels_one_hot * ((1 - preds) ** gamma) * tf.log(preds)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    optimizer = tf.train.AdamOptimizer(learning_rate_ph)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    # create input reader object
    data_input = DataInput(DATA_PATH, BATCH_SIZE, VAL_SET_SIZE, AUGMENT_PROB)
    val_images, val_labels = data_input.get_val_set()

    # saver
    saver = tf.train.Saver()

    # summaries
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("learning_rate", learning_rate_ph)
    tf.summary.scalar("accuracy", accuracy_ph)
    for i in range(10):
        tf.summary.scalar("accuracy_per_class/class_%s" % str(i), accuracy_per_class_phs[i])

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(os.path.join(model_path, "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(model_path, "validation"), sess.graph)

        # number of parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("total_params: " + str(total_parameters))

        train_summary_pred = []
        train_summary_labels = []

        for i in range(1, NUM_STEPS+1):

            print("Step: " + str(i))

            train_images, train_labels = data_input.get_batch()

            power = i // DECAY_STEPS_LR
            learning_rate = INITIAL_LR * DECAY_FACTOR_LR ** power

            train_feed_dict = {images_ph: train_images, labels_ph: train_labels, learning_rate_ph: learning_rate, training_ph: True}
            sess.run(train_op, feed_dict=train_feed_dict)

            if i % 5000 == 0:
                saver.save(sess, os.path.join(model_path, "model.ckpt"), i)

            if i % 4 == 0:
                train_logits = sess.run(logits, feed_dict={images_ph: train_images, training_ph: False})
                pred = np.argmax(train_logits, axis=1)
                train_summary_pred += [p for p in pred]
                train_summary_labels += [l for l in train_labels]

            if i % 500 == 0:
                train_summary_pred = np.array(train_summary_pred)
                train_summary_labels = np.array(train_summary_labels)
                hits = train_summary_pred == train_summary_labels
                accuracy = np.round(np.sum(hits) / len(hits) * 100, decimals=2)
                accuracy_per_class = []
                for cl in range(10):
                    accuracy_cl = np.round(np.sum(hits[train_summary_labels == cl]) / np.sum(train_summary_labels == cl) * 100, decimals=2)
                    accuracy_per_class.append(accuracy_cl)

                train_feed_dict[accuracy_ph] = accuracy
                for cl in range(10):
                    train_feed_dict[accuracy_per_class_phs[cl]] = accuracy_per_class[cl]
                summary_train = sess.run(summary_op, feed_dict=train_feed_dict)
                train_writer.add_summary(summary_train, i)
                train_summary_labels = []
                train_summary_pred = []

                # validation summary
                val_logits = sess.run(logits, feed_dict={images_ph: val_images, training_ph: False})
                pred = np.argmax(val_logits, axis=1)
                hits = pred == val_labels
                accuracy = np.round(np.sum(hits) / len(pred) * 100, decimals=2)
                accuracy_per_class = []
                for cl in range(10):
                    accuracy_cl = np.round(np.sum(hits[val_labels == cl]) / np.sum(val_labels == cl) * 100, decimals = 2)
                    accuracy_per_class.append(accuracy_cl)

                val_feed_dict = {images_ph: val_images, labels_ph: val_labels, accuracy_ph: accuracy, learning_rate_ph: learning_rate, training_ph: True}
                for cl in range(10):
                    val_feed_dict[accuracy_per_class_phs[cl]] = accuracy_per_class[cl]
                summary_val = sess.run(summary_op, feed_dict=val_feed_dict)
                val_writer.add_summary(summary_val, i)


if __name__ == "__main__":
    main()