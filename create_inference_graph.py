import tensorflow as tf
import os

model_dir = 'final_model'

def main(argv=None):

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(os.path.join('output_dir', model_dir))
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored (%s)...' % ckpt.model_checkpoint_path)
        else:
           raise Exception('Model at (%s) could not be restored...' % model_dir)

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                                input_graph_def=tf.get_default_graph().as_graph_def(),
                                                                output_node_names=['prediction/BatchNorm/Reshape_1'])

        with tf.gfile.GFile(name=os.path.join('output_dir', model_dir, 'frozen_inference_graph.pb'),mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == "__main__":
    tf.app.run()