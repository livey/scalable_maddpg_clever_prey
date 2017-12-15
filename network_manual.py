import tensorflow as tf
import numpy as np

class NetworkManual:
    def __init__(self,sess):
        self.sess = sess
        self.saver = tf.train.Saver()

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state("saved_network")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print('Could not find old network weights')

    def save_network(self,step):
        # do not processing under Dropbox
        #  exit drop box then run
        print('save network...',step)
        self.saver.save(self.sess, 'saved_network/' + 'network', global_step=step)