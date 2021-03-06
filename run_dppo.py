from running_mean_std import RunningMeanStd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from environment import centauro_env
import scipy.signal
import math
import time
import matplotlib.pyplot as plt

IT_MAX = 100000
EP_LEN = 100
N_WORKER = 1               # parallel workers
GAMMA = 0.99                 # reward discount factor
LAM = 0.9
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
LR = 0.001

BATCH_SIZE = 64
MIN_BATCH_SIZE = 63       # minimum batch size for updating PPO

UPDATE_STEP = 5            # loop update operation n-steps
EPSILON = 0.2              # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = centauro_env.observation_space, centauro_env.action_space 

G_ITERATION = 0

Goal_states = []
Goal_return = []
Goal_step = 60
Goal_count = 0
Goal_buffer_full = False 

Crash_states = []
Crash_return = []
Crash_step = 20
Crash_count = 0
Crash_buffer_full = False 

RP_buffer_size = MIN_BATCH_SIZE * 10

History_states = []
History_adv = []
History_return = []
History_count = 0
History_buffer_full = False
History_buffer_size = MIN_BATCH_SIZE * 10

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(self.sess, shape=S_DIM)

        # critic
        # l1 = self.feature #tf.layers.dense(self.feature, 100, tf.nn.relu)
        self.feature = self._build_feature_net('feature', self.tfs, reuse = False)
        self.v = self._build_cnet('value', self.feature, reuse = False)

        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.diff_r_v = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.diff_r_v))
        # self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # # for continue action
        self.tfa = tf.placeholder(tf.float32, [None, 1], 'action')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        self.ratio = self.pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        self.entropy = self.pi.entropy()
        self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)  # operation of choosing action
        self.sample_op_stochastic = self.pi.loc
        self.std = self.pi.scale

        # # descrete action
        # self.tfa = tf.placeholder(tf.int32, [None], 'action')
        # self.pi_prob = tf.reduce_sum((self.pi) * tf.one_hot(self.tfa, A_DIM, dtype=tf.float32), axis=1, keep_dims=True)
        # oldpi_prob = tf.reduce_sum((oldpi) * tf.one_hot(self.tfa, A_DIM, dtype=tf.float32), axis=1, keep_dims=True)
        # self.ratio = self.pi_prob / (oldpi_prob + 1e-5) #tf.exp(self.log_pi - log_oldpi)
        # self.entropy = -tf.reduce_sum(self.pi * tf.log(self.pi + 1e-5), axis=1, keep_dims=True) 

        self.surr1 = self.ratio * self.tfadv
        self.surr2 = tf.clip_by_value(self.ratio, 1. - EPSILON, 1. + EPSILON)
        self.surr = tf.minimum(self.surr1, self.surr2) + 0.0*self.entropy
        self.aloss = -tf.reduce_mean(self.surr)

        # value replay
        self.tfs_history = tf.placeholder(tf.float32, [None, S_DIM], 'state_history') # for value replay
        self.return_history = tf.placeholder(tf.float32, [None, 1], 'history_return') # for value replay

        self.feature_history = self._build_feature_net('feature', self.tfs_history, reuse = True) # for value replay
        self.v_history = self._build_cnet('value', self.feature_history, reuse = True)
        self.diff_history = self.return_history - self.v_history
        self.loss_history = tf.reduce_mean(tf.square(self.diff_history))

        # reward predict
        self.tfs_label = tf.placeholder(tf.float32, [None, S_DIM], 'state_label') # for reward prediction
        self.label = tf.placeholder(tf.int32, [None], 'true_label')

        self.feature_label = self._build_feature_net('feature', self.tfs_label, reuse = True)  # for reward prediction
        self.pred_label = tf.layers.dense(self.feature_label, 2)
        self.loss_pred = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_label, labels=self.label))

        ###########################################################################################
        self.total_loss = self.aloss + (self.closs*1 + self.loss_pred*0 + self.loss_history*0)
        self.base_loss = self.aloss + self.closs*1 + self.loss_history*0

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = LR
        end_learning_rate = LR/10
        decay_steps = 10
        learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                                decay_steps, end_learning_rate,
                                                power=0.5)

        # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.train_base_op = optimizer.minimize(self.base_loss, global_step=global_step)

        # self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('./log', self.sess.graph)
        # self.load_model()

    def get_entropy(self):
        a0 = self.pi - self.max(self.pi, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = self.sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        entropy = self.sum(p0 * (tf.log(z0) - a0), axis=-1)   
        return entropy     


    def neglogp(self, pi, a):
        one_hot_actions = tf.one_hot(a, pi.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(
            logits=pi,
            labels=one_hot_actions)

    def sum(self, x, axis=None, keepdims=False):
        axis = None if axis is None else [axis]
        return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        axis = None if axis is None else [axis]
        return tf.reduce_max(x, axis=axis, keep_dims=keepdims)

    def load_model(self):
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state('./model/rl/')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print ('loaded')
        else:
            print ('no model file')  

    def write_summary(self, summary_name, value):
        summary = tf.Summary()
        summary.value.add(tag=summary_name, simple_value=float(value))
        self.summary_writer.add_summary(summary, GLOBAL_EP)
        self.summary_writer.flush()  

    def get_rp_buffer(self, sample_goal_num, sample_crash_num):
        rp_states = []
        rp_label = []
        rp_return = []

        sample_goal_num = int(sample_goal_num)
        sample_crash_num = int(sample_crash_num)

        size = RP_buffer_size
        replace = False
        if Goal_buffer_full == False:
            size = Goal_count
            replace = True
        if size > 0 and sample_goal_num > 0:
            if sample_goal_num > size*2:
                sample_goal_num = size*2
            goal_selected = np.random.choice(size, sample_goal_num, replace=replace)
            for index in goal_selected:
                rp_states.append(Goal_states[index])
                rp_label.append(0)
                rp_return.append(Goal_return[index])

        size = RP_buffer_size
        replace = False
        if Crash_buffer_full == False:
            size = Crash_count
            replace = True
        if size > 0 and sample_crash_num > 0:
            if sample_crash_num > size*2:
                sample_crash_num = size*2            
            crash_selected = np.random.choice(size, sample_crash_num, replace=replace)
            for index in crash_selected:
                rp_states.append(Crash_states[index])
                rp_label.append(1)
                rp_return.append(Crash_return[index])

        return np.array(rp_states), np.array(rp_label), np.array(rp_return)[:, np.newaxis]

    def get_vr_buffer(self, sample_num):
        vr_states = []
        vr_returns = []

        sample_num = int(sample_num)
        size = History_buffer_size
        replace = False
        if History_buffer_full == False:
            size = History_count
            replace = True
        if size > 0:
            if sample_num > size*2:
                sample_num = size*2

            index_selected = np.random.choice(size, sample_num, replace=replace)
            for index in index_selected:
                vr_states.append(History_states[index])
                vr_returns.append(History_return[index])        

        return np.array(vr_states), np.array(vr_returns)[:, np.newaxis]

    def update_base_task(self, s, a, r, adv, vr_states, vr_returns):
        feed_dict = {
            self.tfs: s, 
            self.tfa: a, 
            self.tfdc_r: r, 
            self.tfadv: adv, 

            self.tfs_history: vr_states,
            self.return_history: vr_returns            
        }
        # st = self.sess.run(self.aloss, feed_dict = feed_dict)
        # ratio = self.sess.run(self.ratio, feed_dict = feed_dict)
        # # st2 = self.sess.run(self.surr, feed_dict = feed_dict)
        # print('aloss', st.flatten())
        # print('ratio',ratio.flatten())
        # # print(st2)
        # # print(np.mean(st2))

        vr_loss = 0
        # tloss, aloss, vloss, entropy, _ = self.sess.run([self.base_loss, self.aloss, self.closs, self.entropy, self.train_base_op]        
        tloss, aloss, vloss, vr_loss, entropy, _ = self.sess.run([self.base_loss, self.aloss, self.closs, self.loss_history, self.entropy, self.train_base_op]
        , feed_dict = feed_dict)

        return tloss, aloss, vloss, 0, vr_loss, np.mean(entropy)

    def update_all_task(self, s, a, r, adv, rp_states, rp_labels, vr_states, vr_returns):
        feed_dict = {
            self.tfs: s, 
            self.tfa: a, 
            self.tfdc_r: r, 
            self.tfadv: adv, 
            
            self.tfs_label: rp_states, 
            self.label: rp_labels, 
            
            self.tfs_history: vr_states,
            self.return_history: vr_returns
        }
        # st = self.sess.run(self.aloss, feed_dict = feed_dict)
        # print(st)
        tloss, aloss, vloss, rp_loss, vr_loss, entropy, _ = self.sess.run([self.total_loss, self.aloss, self.closs, self.loss_pred, self.loss_history, self.entropy, self.train_op]
        , feed_dict = feed_dict)

        return tloss, aloss, vloss, rp_loss, vr_loss, np.mean(entropy)

    def shuffel_data(self, s, a, r, adv):
        index_shuffeled = np.random.choice(len(r), len(r), replace=False)
        s_shuf, a_shuf, r_shuf, adv_shuf = [], [], [], []

        for i in index_shuffeled:
            s_shuf.append(s[i])
            a_shuf.append(a[i])
            r_shuf.append(r[i])
            adv_shuf.append(adv[i])

        return s_shuf, a_shuf, r_shuf, adv_shuf

    def shuffel_history(self, history_states, history_returns):
        index_shuffeled = np.random.choice(len(history_returns), len(history_returns), replace=False)
        s_shuf, r_shuf = [], []

        for i in index_shuffeled:
            s_shuf.append(history_states[i])
            r_shuf.append(history_returns[i])

        return s_shuf, r_shuf #, np.array(r_shuf)[:, np.newaxis]
 

    def get_vr_batch(self, s, r):
        # combined_states = s 
        # combined_returns = r
        # history buffer
        if History_buffer_full or History_count > 0:
            if History_buffer_full:
                his_size = History_buffer_size
            else:
                his_size = History_count

            combined_states = History_states[:his_size]
            combined_returns = np.array(History_return[:his_size])[:, np.newaxis]

        # goal buffer
        if Goal_buffer_full or Goal_count > 0:
            if Goal_buffer_full:
                his_size = RP_buffer_size
            else:
                his_size = Goal_count

            combined_states = np.concatenate((combined_states, Goal_states[:his_size]), axis=0)
            combined_returns = np.concatenate((combined_returns, np.array(Goal_return[:his_size])[:, np.newaxis]), axis=0)  

        #crash buffer
        if Crash_buffer_full or Crash_count > 0:
            if Crash_buffer_full:
                his_size = RP_buffer_size
            else:
                his_size = Crash_count

            combined_states = np.concatenate((combined_states, Crash_states[:his_size]), axis=0)
            combined_returns = np.concatenate((combined_returns, np.array(Crash_return[:his_size])[:, np.newaxis]), axis=0)  

        return combined_states, combined_returns

    def update(self):
        global GLOBAL_UPDATE_COUNTER, G_ITERATION
        while not COORD.should_stop():
            UPDATE_EVENT.wait()                     # wait until get batch of data
            self.sess.run(self.update_oldpi_op)     # copy pi to old pi
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
            data = np.vstack(data)
            # s, a, r, adv = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, S_DIM + A_DIM: S_DIM + A_DIM + 1], data[:, -1:]
            s, a, r, reward, adv = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1], data[:, S_DIM + 1: S_DIM + 2], data[:, S_DIM + 2: S_DIM + 3], data[:, -1:]
            self.ob_rms.update(s)
            if adv.std() != 0:                
                adv = (adv - adv.mean())/adv.std()
                print('adv min max', adv.min(), adv.max())

            # print('adv', adv)
            # adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
            # update actor and critic in a update loop

            mean_return = np.mean(r)
            print(G_ITERATION, '  --------------- update! batch size:', len(a), '-----------------')
            print('--------------------------------------------------------------------------------------')

            # combined_states, combined_returns = self.get_vr_batch(s, r)
            combined_states, combined_returns = s, r

            print('a batch', len(r), 'v batch', len(combined_returns))

            for iteration in range(UPDATE_STEP):
                # construct reward predict data
                tloss, aloss, vloss, rp_loss, vr_loss = [], [], [], [], []
                tloss_sum, aloss_sum, vloss_sum, rp_loss_sum, vr_loss_sum, entropy_sum = 0, 0, 0, 0, 0, 0  

                # s, a, r, adv = self.shuffel_data(s, a, r, adv)      

                combined_states, combined_returns = self.shuffel_history(combined_states, combined_returns)

                count = 0
                for start in range(0, len(combined_returns), MIN_BATCH_SIZE):
                    # print('update',iteration, count)
                    end = start + MIN_BATCH_SIZE
                    if end > len(combined_returns) - 1:
                        break
                    count += 1
                    sub_s = combined_states[start:end]
                    # sub_a = a[start:end]
                    sub_r = combined_returns[start:end]
                    # sub_adv = adv[start:end]

                    rp_states, rp_labels, rp_returns = self.get_rp_buffer(MIN_BATCH_SIZE*1, MIN_BATCH_SIZE*1)
                    # vr_states, vr_returns = self.get_vr_buffer(MIN_BATCH_SIZE*1)

                    # vr_states = np.concatenate((vr_states, s), axis=0)
                    # vr_returns = np.concatenate((vr_returns, r), axis=0) 
                    # if len(rp_states) != 0:
                    #     vr_states = np.concatenate((vr_states, rp_states), axis=0)
                    #     vr_returns = np.concatenate((vr_returns, rp_returns), axis=0)    


                    # if len(rp_states) != 0:     
                    #     tloss, aloss, vloss, rp_loss, vr_loss, entropy = self.update_all_task(sub_s, sub_a, sub_r, sub_adv, rp_states, rp_labels, vr_states, vr_returns)
                    # else:
                    # tloss, aloss, vloss, rp_loss, vr_loss, entropy = self.update_base_task(sub_s, sub_a, sub_r, sub_adv, vr_states, vr_returns)
                    tloss, aloss, vloss, rp_loss, vr_loss, entropy = self.update_base_task(s, a, r, adv, sub_s, sub_r)
                    
                    tloss_sum += tloss
                    aloss_sum += aloss 
                    vloss_sum += vloss 
                    rp_loss_sum += rp_loss
                    vr_loss_sum += vr_loss
                    entropy_sum += entropy

                if count == 0:
                    count = 1
                    print('---------------  need more sample  --------------- ')
                    break 

                print("aloss: %7.4f|, vloss: %7.4f|, rp_loss: %7.4f|, vr_loss: %7.4f|, entropy: %7.4f" % \
                                    (aloss_sum/count, vloss_sum/count, rp_loss_sum/count, vr_loss_sum/count, entropy_sum/count))

            # [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
            # [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]

            print(Goal_count, Crash_count, History_count)
            print(Goal_buffer_full, Crash_buffer_full, History_buffer_full)
            entropy = self.sess.run(self.entropy, {self.tfs: s})              
            self.write_summary('Loss/entropy', np.mean(entropy))  
            self.write_summary('Loss/a loss', aloss_sum/count) 
            self.write_summary('Loss/v loss', vloss_sum/count) 
            self.write_summary('Loss/rp loss', rp_loss_sum/count) 
            self.write_summary('Loss/vr loss', vr_loss_sum/count) 
            self.write_summary('Loss/t loss', tloss_sum/count) 
            self.write_summary('Perf/mean_reward', np.mean(reward))  

            self.saver.save(self.sess, './model/rl/model.cptk') 

            UPDATE_EVENT.clear()        # updating finished
            GLOBAL_UPDATE_COUNTER = 0   # reset counter
            G_ITERATION += 1
            ROLLING_EVENT.set()         # set roll-out available
                

    def _build_feature_net(self, name, input_state, reuse = False):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        with tf.variable_scope(name, reuse=reuse):
            state_size = 5
            num_img = S_DIM - state_size - 1# 
            img_size = int(math.sqrt(num_img))
            print(num_img, img_size)

            input_state = (input_state - self.ob_rms.mean)/self.ob_rms.std
            ob_grid = tf.slice(input_state, [0, 0], [-1, num_img])
            # tp_state = tf.slice(self.tfs, [0, num_img], [-1, 2])
            # rp_state = tf.slice(self.tfs, [0, num_img+2], [-1, 3])
            # action_taken = tf.slice(self.tfs, [0, num_img+4], [-1, 1])
            # index_in_ep = tf.slice(self.tfs, [0, num_img+5], [-1, 1])

            ob_state = tf.slice(input_state, [0, num_img], [-1, state_size])
            # ob_state = tf.concat([ob_state , index_in_ep], 1, name = 'concat_ob')
            # reshaped_grid = tf.reshape(ob_grid,shape=[-1, img_size, img_size, 1]) 
            ob_state = tf.reshape(ob_state,shape=[-1, state_size])  

            x = (ob_grid - 0.5)*2
            x = tf.layers.dense(x, 100, tf.nn.tanh, kernel_initializer=w_init, name='x_fc1' )
            x = tf.layers.dense(x, 50, tf.nn.tanh, kernel_initializer=w_init, name='x_fc2' )

            # process state
            state_rt = tf.layers.dense(ob_state, state_size*10, tf.nn.tanh, kernel_initializer=w_init, name='rt_fc1' )
            # state_rt = tf.layers.dense(state_rt, state_size*10, tf.nn.tanh, name='rt_fc2' )
            
            feature = tf.concat([x , state_rt], 1, name = 'concat')
            # feature = state_rt
            # feature = tf.layers.dense(state_concat, 100, tf.nn.tanh, name='feature_fc' )
        return feature

    def _build_anet(self, name, trainable):
        # w_init = tf.random_normal_initializer(0., .1)
        # w_init = tf.zeros_initializer()
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.feature, 100, tf.nn.tanh, trainable=trainable)
            # l1 = self.feature

            mu = tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            # logstd = tf.get_variable(name="logstd", shape=[1, A_DIM], initializer=tf.zeros_initializer(), trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)  #   tf.exp(logstd))

            # norm_dist = tf.layers.dense(l1, A_DIM, tf.nn.softmax, kernel_initializer=w_init, trainable=trainable)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_cnet(self, name, input_state, reuse = False):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(input_state, 100, tf.nn.tanh, kernel_initializer=w_init)
            # l1 = input_state
            v = tf.layers.dense(l1, 1)
        return v

    def choose_action(self, s, stochastic = True, show_plot = False): 
        s = s[np.newaxis, :]
        if stochastic:
            a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        else:
            a = self.sess.run(self.sample_op_stochastic, {self.tfs: s})[0]

        mean, scale = self.sess.run([self.sample_op_stochastic, self.std], {self.tfs: s})

        mean = mean[0]
        scale = scale[0]
        np.append(scale, 0)

        scale = np.pi * (20 * scale)**2
        a = np.clip(a, -1, 1)
        if show_plot:
            plt.clf()
            plt.scatter(range(A_DIM+1), np.append(a, 1.0).flatten(), s=scale, c=[10, 10, 10, 10] )
            plt.pause(0.01)
            # print(prob)

        return a, 0

    # def choose_action(self, s, stochastic = True, show_plot = False):  # run by a local
    #     prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[np.newaxis, :]})

    #     if stochastic:
    #         action = np.random.choice(range(prob_weights.shape[1]),
    #                               p=prob_weights.ravel())  # select action w.r.t the actions prob
    #     else:
    #         action = np.argmax(prob_weights.ravel())

    #     if show_plot:
    #         prob = prob_weights.ravel()
    #         plt.clf()
    #         plt.scatter(range(A_DIM+1), np.append(prob, 0.5).flatten() )
    #         plt.pause(0.01)
    #         # print(s[-6:])
    #         # print(prob)
    #     return action, prob_weights.ravel()

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = centauro_env.Simu_env(20000 + wid)
        self.ppo = GLOBAL_PPO

    def discount(self, x, gamma):
        """ Calculate discounted forward sum of a sequence at each point """
        x = np.asarray(x)
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    def compute_gae(self, rewards, vpreds):
        rewards = np.asarray(rewards)
        vpreds = np.asarray(vpreds)
        tds = rewards - vpreds + np.append(vpreds[1:] * GAMMA, 0)
        advantages = self.discount(tds, GAMMA * LAM)
        return advantages

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER \
            , Goal_states, Goal_count, Goal_return, Goal_buffer_full \
            , Crash_states, Crash_count, Crash_return, Crash_buffer_full \
            , History_states, History_count, History_adv, History_return, History_buffer_full

        self.env.save_ep()
        s = self.env.reset(EP_LEN, 0, 1, 0)

        Goal_states = np.array([s for _ in range(RP_buffer_size)])
        Goal_return = np.zeros(RP_buffer_size, 'float32')

        Crash_states = np.array([s for _ in range(RP_buffer_size)])
        Crash_return = np.zeros(RP_buffer_size, 'float32')

        History_states = np.array([s for _ in range(History_buffer_size)])
        History_return = np.zeros(History_buffer_size, 'float32')
        History_adv = np.zeros(History_buffer_size, 'float32')

        while not COORD.should_stop():
            buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_return, buffer_adv, buffer_aprob = [], [], [], [], [], [], []
            ep_crash = 0
            ep_step = 0

            # if Goal_buffer_full == False and Goal_count < 200:
            #     s = self.env.reset(EP_LEN, 0, 4, 0)
            # else:
            s = self.env.reset(EP_LEN, 0, 1, 0)

            # s[-1] = (ep_step/EP_LEN-0.5)*2

            if self.wid == 0 and N_WORKER==100:
                for t in range(EP_LEN):
                    a, p_prob = self.ppo.choose_action(s, stochastic = False, show_plot = True)
                    s_, r, done, info = self.env.step(a)
                    vpred = self.ppo.get_v(s)
                    vpred_ = self.ppo.get_v(s_)
                    td = r + GAMMA*vpred_ - vpred
                    print("action: %5i| r: %7.4f| vpred: %7.4f| v_next: %7.4f| TD: %7.4f| " %(a, r, vpred, vpred_, td))
                    s = s_
                    if info == 'goal' or t == EP_LEN-1 or info == 'out' or info == 'crash':
                        break

            else:
                # for t in range(EP_LEN):
                t = 0
                while(1):
                    if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                        ROLLING_EVENT.wait()                        # wait until PPO is updated
                        GLOBAL_UPDATE_COUNTER -= len(buffer_r)
                        buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_return, buffer_adv = [], [], [], [], [], []

                    if self.wid == 0:
                        a, a_prob = self.ppo.choose_action(s, stochastic = True, show_plot = True)
                    else:
                        a, a_prob = self.ppo.choose_action(s, stochastic = True, show_plot = False)

                    s_, r, done, info = self.env.step(a)

                    vpred = self.ppo.get_v(s)
                    # if self.wid == 0:
                    #     vpred_ = self.ppo.get_v(s_)
                    #     td = r + GAMMA*vpred_ - vpred
                    #     print("action: %5i| r: %7.4f| vpred: %7.4f| v_next: %7.4f| TD: %7.4f| " %(a, r, vpred, vpred_, td), self.env.action_list[a])

                    buffer_s.append(s*1)
                    buffer_a.append(a)
                    buffer_r.append(r)                    # normalize reward, find to be useful
                    # buffer_aprob.append(a_prob[a])
                    buffer_vpred.append(vpred)

                    s = s_
                    ep_step += 1
                    t += 1
                    
                    GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                    if done or GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                        # if info == 'unfinish':
                        buffer_r[-1] += GAMMA * self.ppo.get_v(s_)
                        buffer_return = self.discount(buffer_r, GAMMA)
                        buffer_adv = self.compute_gae(buffer_r, buffer_vpred)

                        # self.write_summary('Perf/ep_length', len(a))  
                        if self.wid == 0:
                        #     print('---------------------------------------')
                        #     print(buffer_r)
                        #     print(buffer_return)
                        #     print(buffer_vpred)
                        #     print(buffer_adv)
                            print(info)
            

                        # update terminal buffer
                        if info == 'goal':
                            for i in range(len(buffer_r)-1, len(buffer_r)-Goal_step-1, -1):
                                if i < 0:
                                    break 
                                else:
                                    Goal_states[Goal_count] = buffer_s[i]
                                    Goal_return[Goal_count] = buffer_return[i]
                                    Goal_count = (Goal_count+1)%RP_buffer_size
                                    if Goal_count == RP_buffer_size - 1:
                                        Goal_buffer_full = True
                        elif info == 'crash':
                            for i in range(len(buffer_r)-1, len(buffer_r)-Crash_step-1, -1):
                                if i < 0:
                                    break 
                                else:
                                    Crash_states[Crash_count] = buffer_s[i]
                                    Crash_return[Crash_count] = buffer_return[i]
                                    Crash_count = (Crash_count+1)%RP_buffer_size
                                    if Crash_count == RP_buffer_size - 1:
                                        Crash_buffer_full = True
                        # add to history buffer
                        # elif info == 'unfinish':
                        else:
                            for i in range(0, len(buffer_r)):
                                History_states[History_count] = buffer_s[i]
                                History_return[History_count] = buffer_return[i]
                                History_adv[History_count] = buffer_adv[i]
                                History_count = (History_count+1)%History_buffer_size
                                if History_count == History_buffer_size - 1:
                                    History_buffer_full = True            

                        # print(Goal_count, Crash_count)

                        # print(buffer_return)
                        # print(buffer_vpred)
                        # print(buffer_adv)
                        # print(self.ppo.get_v(s_))

                        bs, ba, bret, brew, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_return)[:, np.newaxis], np.array(buffer_r)[:, np.newaxis], np.array(buffer_adv)[:, np.newaxis]

                        buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_return, buffer_adv = [], [], [], [], [], []
                    
                        QUEUE.put(np.hstack((bs, ba, bret, brew, badv)))          # put data in the queue
                        if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                            ROLLING_EVENT.clear()       # stop collecting data
                            UPDATE_EVENT.set()          # globalPPO update

                        if G_ITERATION >= IT_MAX:         # stop training
                            COORD.request_stop()
                            break

                        if done: #info == 'goal' or info == 'out' or info == 'nostep':
                            break 

            GLOBAL_EP += 1


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.daemon = True
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    t = threading.Thread(target=GLOBAL_PPO.update,)
    t.daemon = True
    threads.append(t)
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
    env = gym.make('Pendulum-v0')
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]