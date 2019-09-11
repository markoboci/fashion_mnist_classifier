import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
import shutil

from data_input import load_mnist

model_dir = 'final_model'

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == "__main__":

    images, labels = load_mnist('data/fashion', 't10k')
    print("Number of test examples: " + str(len(labels)))
    images = np.array([np.expand_dims(img.reshape(28, 28) / 255, axis=2) for img in images])

    graph = load_graph(os.path.join('output_dir', model_dir, 'frozen_inference_graph.pb'))

    ops = graph.get_operations()
    all_tensor_names = [output.name for op in ops for output in op.outputs]

    is_training_tensor = graph.get_tensor_by_name('prefix/training_ph:0')
    input_tensor = graph.get_tensor_by_name('prefix/input_images_ph:0')
    output_tensor = graph.get_tensor_by_name('prefix/prediction/BatchNorm/Reshape_1:0')

    with tf.Session(graph=graph) as sess:

        preds = sess.run([output_tensor], feed_dict={input_tensor: images, is_training_tensor: False})
        hits = np.argmax(preds[0], axis=1) == labels

        accuracy = np.round(np.sum(hits) / len(hits) * 100, decimals=2)
        print("Overall accuracy: " + str(accuracy))

        print("\n Accuracy per class: ")
        for cl in range(10):
            accuracy_cl = np.round(np.sum(hits[labels == cl]) / np.sum(labels == cl) * 100, decimals=2)
            print("Class %s: %s" % (str(cl), str(accuracy_cl)))






