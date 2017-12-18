import tensorflow as tf
import numpy as np
import math

# target updating rate
TAU = .001
L2 = .00001
LEARNING_RATE = 1e-3

Layer1_size = 10
Layer2_size = 10
Layer3_size = 10
Layer4_size = 10
Layer5_size = 10
Layer6_size = 5

SUMMARY_DIR ='summaries/'

class CriticNetwork:

    ''''for critic network,
    the input is the (states,actions) for every agents,
    output is the Q(s,a) value for each agents'''
    def __init__(self,sess,stateDimension,actionDimension):
        self.time_step = 0
        self.sess = sess
        self.actionDimension = actionDimension
        self.stateDimension  = stateDimension

        # create critic network
        self.stateInputs,\
        self.actionInputs,\
        self.q_value_outputs,\
        self.nets = self.createQNetwork(stateDimension,actionDimension)

        # construct target q network
        self.target_q_value_outputs, \
        self.target_update = self.create_target_network(self.q_value_outputs, self.nets)

        # create training methods
        self.create_training_method()

        # merge all the summaries

        self.summaries_writer,\
            self.merge_summaries = self.collect_summaries()

        self.init_new_variables()

        self.update_target()

    def createQNetwork(self,stateDimension,actionDimension):

        with tf.variable_scope('prey_criticNetwork') as scope:
            # the input state training data  is batchSize*numOfAgents*stateDimension
            stateInputs = tf.placeholder('float',[None, stateDimension])
            # the input action training data is batchSize*numOfAgents*stateDimension
            actionInputs = tf.placeholder('float',[None, actionDimension])

            W1 = tf.get_variable('W1', [self.stateDimension, Layer1_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('b1', [Layer1_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            W2 = tf.get_variable('W2', [Layer1_size, Layer2_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable('b2', [Layer2_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            W3 = tf.get_variable('W3', [Layer2_size,Layer3_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable('b3', [Layer3_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            action_W3 = tf.get_variable('action_W3', [actionDimension,Layer3_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            W4 = tf.get_variable('W4', [Layer3_size, Layer4_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            b4 = tf.get_variable('b4', [Layer4_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            W5 = tf.get_variable('W5', [Layer4_size, Layer5_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.get_variable('b5', [Layer5_size],
                                  initializer=tf.contrib.layers.xavier_initializer())

            W6 = tf.get_variable('W6', [Layer5_size, Layer6_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b6 = tf.get_variable('b6', [Layer6_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            W7 = tf.get_variable('W7', [Layer6_size, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b7 = tf.get_variable('b7', [1],
                                 initializer=tf.contrib.layers.xavier_initializer())

            layer1 = tf.nn.selu(tf.matmul(stateInputs,W1)+b1)
            layer2 = tf.nn.selu(tf.matmul(layer1,W2)+b2)
            layer3 = tf.nn.selu(tf.matmul(layer2,W3)+tf.matmul(actionInputs,action_W3)+b3)
            layer4 = tf.nn.selu(tf.matmul(layer3,W4)+b4)
            layer5 = tf.nn.selu(tf.matmul(layer4, W5)+b5)
            layer6 = tf.nn.selu(tf.matmul(layer5, W6)+b6)
            q_value = tf.identity(tf.matmul(layer6,W7)+b7)


        nets = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prey_criticNetwork')
        return stateInputs, actionInputs, q_value, nets


    def create_target_network(self,q_output, nets):
        #state_input = tf.placeholder('float', [None,None,stateDimension])
        #action_input = tf.placeholder('float', [None,None,actionDimension])
        ##https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        ## how to use https://stackoverflow.com/questions/45206910/tensorflow-exponential-moving-average
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU,zero_debias=True)
        target_update = ema.apply(nets)
        # reference using
        #http://web.stanford.edu/class/cs20si/lectures/slides_14.pdf
        # on page 22
        # get the after averaged weights
        # copy this the Q network, but with the target network weights

        # the difference between operation and the result of that operation
        # Variable has the function value()
        replace_ts = {}
        for tt in nets:
            temp_ts = ema.average(tt)
            replace_ts.update({tt.value(): temp_ts.value()}) # Tensor to Tensor
        # graph_replace
        # https://www.tensorflow.org/api_docs/python/tf/contrib/graph_editor/graph_replace
        target_q_value = tf.contrib.graph_editor.graph_replace(q_output, replace_ts)

        return target_q_value, target_update

    def create_training_method(self):
        # the expected size of Rt is batch_size* agents
        self.Rt = tf.placeholder('float', [None])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.nets])
        self.cost = tf.reduce_mean(tf.square(self.Rt - self.q_value_outputs)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        mean_rewards = tf.reduce_mean(self.q_value_outputs)
        #mean_rewards = self.q_value_outputs
        tf.summary.scalar('mean_Q_value', mean_rewards)
        self.action_gradients = tf.gradients(mean_rewards, self.actionInputs)

    def train(self,Rt,state_batch,action_batch):
        self.time_step += 1
        self.sess.run(
            self.optimizer,feed_dict={
                self.Rt : Rt,
                self.stateInputs: state_batch,
                self.actionInputs: action_batch
            }
        )

    def target_q(self,state_batch,action_batch):
        return self.sess.run(
            self.target_q_value_outputs, feed_dict={
            self.stateInputs:  state_batch,
            self.actionInputs: action_batch})

    def printnets(self):
        for nn in self.nets:
            print(nn)

    def q_value(self, stateInputs, actionInputs):
        return self.sess.run(self.q_value_outputs,feed_dict={
            self.stateInputs: stateInputs, self.actionInputs: actionInputs})

    def update_target(self):
        self.sess.run(self.target_update)

    def gradients(self,state_batch,action_batch):
        return self.sess.run(
            self.action_gradients,feed_dict={
                self.stateInputs: state_batch,
                self.actionInputs: action_batch
            }
        )[0]

    def q_value(self,state_batch,action_batch):
        return self.sess.run(self.q_value_outputs,feed_dict={
            self.stateInputs: state_batch,
            self.actionInputs: action_batch
        })

    def collect_summaries(self):
        summaries = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)
        return summary_writer, summaries

    def write_summaries(self,state_batch, action_batch, record_num):
        summ = self.sess.run(self.merge_summaries, feed_dict={self.stateInputs: state_batch,
                                                              self.actionInputs: action_batch})
        self.summaries_writer.add_summary(summ, record_num)

    def init_new_variables(self):
        '''init the new add variables, instead of all the variables
           it is convenient to add new agents
           https://asyoulook.com/computers%20&%20internet/tensorflow-how-to-get-the-list-of-uninitialized-variables-from-tf-report-uninitialized-variables/1730337
           https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
        '''
        list_of_variables = tf.global_variables()
        # this method returns b'strings' , so decode to string for comparison
        uninit_names = set(self.sess.run(tf.report_uninitialized_variables()))
        # https://stackoverflow.com/questions/606191/convert-bytes-to-a-string
        uninit_names = [v.decode('utf-8') for v in uninit_names]
        uninit_variables =  [v for v in list_of_variables if
             v.name.split(':')[0] in uninit_names]
        ss = tf.variables_initializer(uninit_variables)
        self.sess.run(ss)

    # def load_network(self):
    #     checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
    #     if checkpoint and checkpoint.model_checkpoint_path:
    #         self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
    #         print("Successfully loaded:", checkpoint.model_checkpoint_path)
    #     else:
    #         print('Could not find old network weights')
    #
    # def save_network(self,time_step):
    #     print('save critic-network...',time_step)
    #     self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step=time_step)
