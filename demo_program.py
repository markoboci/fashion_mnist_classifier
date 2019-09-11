import cv2
import tensorflow as tf
import numpy as np

# relative box height
relative_height = 0.6
frozen_graph_filename = 'output_dir/resnet_090919_154503/frozen_inference_graph.pb'
class_dict = {0: 'T-shity',
              1: 'Trouser',
              2: 'Pullover',
              3: 'Dress',
              4: 'Coat',
              5: 'Sandal',
              6: 'Shirt',
              7: 'Sheaker',
              8: 'Bag',
              9: 'Ankle boot'}

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def main():

    stream = cv2.VideoCapture(0)

    state, frame = stream.read()
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', frame_w, frame_h)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out_video = cv2.VideoWriter('demo_video.avi', fourcc, 25, (frame_w, frame_h))

    graph = load_graph(frozen_graph_filename)

    is_training_tensor = graph.get_tensor_by_name('prefix/training_ph:0')
    input_tensor = graph.get_tensor_by_name('prefix/input_images_ph:0')
    output_tensor = graph.get_tensor_by_name('prefix/prediction/BatchNorm/Reshape_1:0')

    with tf.Session(graph=graph) as sess:

        while state:

            state, frame = stream.read()

            box_size = int(frame_h * relative_height)
            x = int(frame_w / 2 - box_size / 2)
            y = int(frame_h / 2 - box_size / 2)

            model_input = frame[y:(y+box_size), x:(x+box_size), :]
            model_input = cv2.resize(model_input, (28, 28), interpolation=cv2.INTER_CUBIC)
            model_input_grey = cv2.cvtColor(model_input, cv2.COLOR_BGR2GRAY)

            model_input_grey = np.expand_dims(model_input_grey, 0)
            model_input_grey = np.expand_dims(model_input_grey, 3) / 255

            pred = sess.run([output_tensor], feed_dict={input_tensor: model_input_grey, is_training_tensor: False})
            pred = np.argmax(pred[0], axis=1)

            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (255, 0, 0), 2)
            cv2.putText(frame, 'Predicted class: ' + class_dict[pred[0]], (int(frame_h * 0.1), int(frame_h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('frame', frame)
            out_video.write(frame)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()