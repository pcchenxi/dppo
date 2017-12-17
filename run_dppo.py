from running_mean_std import RunningMeanStd
from distributions import make_pdtype
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from environment import centauro_env
import scipy.signal
import math
import time
import matplotlib.pyplot as plt

EP_MAX = 500000
EP_LEN = 500
N_WORKER = 1               # parallel workers
GAMMA = 0.98                # reward discount factor
LAM = 0.95
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0005               # learning rate for critic
LR = 0.0001

BATCH_SIZE = 512
MIN_BATCH_SIZE = 64       # minimum batch size for updating PPO

UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2              # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = centauro_env.observation_space, centauro_env.action_space 
Action_Space = centauro_env.action_type

G_ITERATION = 0

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.tf_is_train = tf.placeholder(tf.bool, None)

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(self.sess, shape=S_DIM)

        # actor
        pi, self.v, net_params = self._build_anet('net', self.tfs, trainable=True)

        oldpi, oldv, oldnet_params = self._build_anet('oldnet', self.tfs, trainable=False)
        self.sample_op = pi.sample() #tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(net_params, oldnet_params)]
        # self.restorepi_op = [p.assign(oldp) for p, oldp in zip(net_params, oldnet_params)]

        if isinstance(Action_Space, gym.spaces.Box):
            self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        else:
            self.tfa = tf.placeholder(tf.int32, [None], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None], 'advantage')

        a_prob = tf.reduce_sum(tf.nn.softmax(pi.logits) * tf.one_hot(self.tfa, A_DIM, dtype=tf.float32), axis=1, keep_dims=True)
        olda_prob = tf.reduce_sum(tf.nn.softmax(oldpi.logits) * tf.one_hot(self.tfa, A_DIM, dtype=tf.float32), axis=1, keep_dims=True)
        # self.ratio = a_prob/(olda_prob + 1e-8)
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        # ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        self.ratio = tf.exp(oldpi.neglogp(self.tfa) - pi.neglogp(self.tfa))

        self.surr = self.ratio * self.tfadv                       # surrogate loss
        self.surr2 = tf.clip_by_value(self.ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv
        l2 = tf.reduce_mean(tf.square(self.ratio - 1)) * 10
        self.aloss = -tf.reduce_mean(tf.minimum(self.surr, self.surr2))

        self.entropy = tf.reduce_mean(pi.entropy())

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net/actor')
        a_grads = tf.gradients(self.aloss, a_params)
        a_grads, a_grad_norm = tf.clip_by_global_norm(a_grads, 5)

        a_grads = list(zip(a_grads, a_params))
        a_optimizer = tf.train.AdamOptimizer(learning_rate=A_LR, epsilon=1e-5)
        # self.atrain_op = a_optimizer.apply_gradients(a_grads)
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss, var_list = a_params)
        
        # critic
        # self.feature = self._build_feature_net('feature', self.tfs)
        # l1 = tf.layers.dense(self.feature, 100, tf.nn.relu)
        # self.v = tf.layers.dense(self.feature, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None], 'discounted_r')
        vpredclipped = oldv + tf.clip_by_value(self.v - oldv, - EPSILON, EPSILON)

        self.closs1 = tf.square(self.tfdc_r - self.v)
        self.closs2 = tf.square(self.tfdc_r - vpredclipped)

        self.closs = tf.reduce_mean(self.closs1) #tf.reduce_mean(tf.maximum(self.closs1, self.closs2))
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net/value')
        c_grads = tf.gradients(self.closs, c_params)
        c_grads, c_grad_norm = tf.clip_by_global_norm(c_grads, 0.5)

        c_grads = list(zip(c_grads, c_params))
        c_optimizer = tf.train.AdamOptimizer(learning_rate=C_LR, epsilon=1e-5)
        # self.ctrain_op = c_optimizer.apply_gradients(c_grads)
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs, var_list = c_params)

        self.total_loss = self.aloss + self.closs*0.5 - 0.0*self.entropy

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op = tf.train.AdamOptimizer(LR).minimize(self.total_loss)

        total_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.total_loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('./log', self.sess.graph)    
        self.pi_prob = tf.nn.softmax(pi.logits)
        self.oldpi_prob = tf.nn.softmax(oldpi.logits)

        self.a_prob = a_prob
        self.olda_prob = olda_prob
        self.load_model()   

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

    def shuffel_data(self, s, a, r, adv):
        index_shuffeled = np.random.choice(len(r), len(r), replace=False)
        s_shuf, a_shuf, r_shuf, adv_shuf = [], [], [], []

        for i in index_shuffeled:
            s_shuf.append(s[i])
            a_shuf.append(a[i])
            r_shuf.append(r[i])
            adv_shuf.append(adv[i])

        return s_shuf, a_shuf, r_shuf, adv_shuf
                
    def add_layer(self, x, out_size, name, ac=None):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        x = tf.layers.dense(x, out_size, kernel_initializer=w_init, name = name)
        # the momentum plays important rule. the default 0.99 is too high in this case!
        x = tf.layers.batch_normalization(x, momentum=0.99, training=self.tf_is_train)    # when have BN
        out = x if ac is None else ac(x)
        return out

    def _build_feature_net(self, name, input_state, trainable):
        # w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(name):
            state_size = 4
            num_img = S_DIM - state_size 
            img_size = int(math.sqrt(num_img))
            print(num_img, img_size)

            input_state = (input_state - self.ob_rms.mean)/self.ob_rms.std
            # input_state = tf.layers.batch_normalization(input_state, training=self.tf_is_train)
            print(input_state.shape)
            ob_grid = tf.slice(input_state, [0, 0], [-1, num_img], name = 'slice_grid')
            print('ob_grib', ob_grid.shape)

            ob_state = tf.slice(input_state, [0, num_img], [-1, state_size], name = 'slice_ob')
            print('ob_state', ob_state.shape)
            # ob_state = tf.concat([ob_state , index_in_ep], 1, name = 'concat_ob')
            reshaped_grid = tf.reshape(ob_grid,shape=[-1, img_size, img_size, 1]) 

            x = tf.layers.conv2d(inputs=reshaped_grid, filters=16, kernel_size=[3, 3], padding="valid", activation=tf.nn.tanh, trainable=trainable )
            x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.tanh, trainable=trainable )
            x = tf.contrib.layers.flatten(x)
            # x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.tanh)

            # x = (ob_grid - 0.5)*2
            # x = tf.layers.dense(ob_grid, 50, tf.nn.tanh, kernel_initializer=w_init, name='x_fc1', trainable=trainable )
            # x = tf.layers.dense(x, 25, tf.nn.tanh, kernel_initializer=w_init, name='x_fc2', trainable=trainable )
            # x = self.add_layer(ob_grid, 50, 'x_fc1', ac = tf.nn.tanh)
            # x = self.add_layer(x, 25, 'x_fc2', ac = tf.nn.tanh)

            # # process state
            # state_rt = tf.layers.dense(ob_state, state_size*5, tf.nn.tanh, kernel_initializer=w_init, name='rt_fc1', trainable=trainable )
            # state_rt = tf.layers.dense(state_rt, state_size*3, tf.nn.tanh, name='rt_fc2', trainable=trainable )
            
            feature = tf.concat([x , ob_state], 1, name = 'concat')
            # feature = state_rt
            # feature = tf.layers.dense(feature, 256, tf.nn.tanh, name='feature_fc', trainable=trainable )
            # feature = tf.layers.dense(feature, 256, tf.nn.tanh, name='feature_fc2', trainable=trainable )
            # feature = self.add_layer(feature, 50, 'feature_fc', tf.nn.tanh)
            # feature = self.add_layer(feature, 25, 'feature_fc2', tf.nn.tanh)
        return feature

    def _build_anet(self, name, input_state, trainable):
        # w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(name):
            self.feature = self._build_feature_net('feature', input_state, trainable=trainable)
            with tf.variable_scope('actor'):
                l1 = tf.layers.dense(self.feature, 256, tf.nn.tanh, trainable=trainable)
                l1 = tf.layers.dense(l1, 256, tf.nn.tanh, trainable=trainable)
                # mu = 1 * tf.layers.dense(self.feature, A_DIM, tf.nn.tanh, kernel_initializer=w_init, trainable=trainable)
                # sigma = tf.layers.dense(self.feature, A_DIM, tf.nn.softplus, kernel_initializer=w_init, trainable=trainable)
                # norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
                # self.feature_a = self._build_feature_net('feature_a', input_state, trainable=trainable)
                if isinstance(Action_Space, gym.spaces.Box):
                    print('continue action')
                    pi = tf.layers.dense(l1, A_DIM, kernel_initializer=w_init, trainable=trainable)
                    logstd = tf.get_variable(name="logstd", shape=[1, A_DIM], initializer=tf.zeros_initializer(), trainable=trainable)
                    pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
                else:
                    print('discrate action')
                    # pdparam = tf.layers.dense(self.feature_a, A_DIM, kernel_initializer=w_init, trainable=trainable)  
                    pdparam = tf.layers.dense(l1, A_DIM, kernel_initializer=w_init, trainable=trainable)      

            with tf.variable_scope('value'):
                # self.feature_v = self._build_feature_net('feature_v', input_state, trainable=trainable)
                l2 = tf.layers.dense(self.feature, 256, tf.nn.tanh, trainable=trainable)
                l2 = tf.layers.dense(l2, 256, tf.nn.tanh, trainable=trainable)
                v = tf.layers.dense(l2, 1, trainable=trainable)
                # v = tf.clip_by_value(v, 0, centauro_env.REWARD_GOAL)

        pdtype = make_pdtype(Action_Space)
        pd = pdtype.pdfromflat(pdparam)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pd, v, params


    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s, self.tf_is_train: False})[0]
        # prob_weights = self.sess.run(self.pi_prob, feed_dict={self.tfs: s})
        # a = np.random.choice(range(prob_weights.shape[1]),
        #                           p=prob_weights.ravel())

        if isinstance(Action_Space, gym.spaces.Box):
            a = np.clip(a, -1, 1)
        return a

    def get_action_prob(self, s):
        s = s[np.newaxis, :]
        prob = self.sess.run(self.pi_prob, {self.tfs: s})[0]

        plt.clf()
        plt.scatter(range(A_DIM+1), np.append(prob, 1.0).flatten())
        plt.pause(0.01)

        return prob

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s, self.tf_is_train: False})[0, 0]

    def check_overall_loss(self, s, a, r, adv):
        feed_dict = {
            self.tfs: s, 
            self.tfa: a, 
            self.tfdc_r: r, 
            self.tfadv: adv, 
            self.tf_is_train: False
        }

        tloss, aloss, vloss, entropy = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy], feed_dict = feed_dict)
        return tloss, aloss, vloss, entropy 

    def seperate_adv(self, buffer_adv):
        possitive_index = []
        negative_index = []
        for index in range(len(buffer_adv)):
            if buffer_adv[index] == 0:
                continue
            if buffer_adv[index] > 0:
                possitive_index.append(index)
            else:
                negative_index.append(index)
        return possitive_index, negative_index

    def get_minibatch(self, s, a, r, adv, possitive_index, negative_index):
        sub_s, sub_a, sub_r, sub_adv = [], [], [], []
        if len(possitive_index) != 0 and len(negative_index) != 0:
            size = int(MIN_BATCH_SIZE/2)
            selected_index = np.random.choice(len(possitive_index), size)
            for i in selected_index:
                index = possitive_index[i]
                sub_s.append(s[index])
                sub_a.append(a[index])
                sub_r.append(r[index])
                sub_adv.append(adv[index])

            selected_index = np.random.choice(len(negative_index), size)
            for i in selected_index:
                index = negative_index[i]
                sub_s.append(s[index])
                sub_a.append(a[index])
                sub_r.append(r[index])
                sub_adv.append(adv[index])
        else:
            selected_index = np.random.choice(len(r), MIN_BATCH_SIZE)
            for i in selected_index:
                index = i
                sub_s.append(s[index])
                sub_a.append(a[index])
                sub_r.append(r[index])
                sub_adv.append(adv[index])
        return sub_s, sub_a, sub_r, sub_adv

    def balance_minibatch(self, s, a, r, adv, sub_s, sub_a, sub_r, sub_adv, possitive_index, negative_index):
        sum_adv_init = sub_adv.sum()
        count = 0
        while sum_adv_init * sub_adv.sum() >= 0:
            if sum_adv_init < 0:
                if len(possitive_index) == 0:
                    return sub_s, sub_a, sub_r, sub_adv
                randint = np.random.randint(len(possitive_index))
                index = possitive_index[randint]
            else:
                if len(negative_index) == 0:
                    return sub_s, sub_a, sub_r, sub_adv
                randint = np.random.randint(len(negative_index))
                index = negative_index[randint]

            sub_s.append(s[index])
            sub_a = np.append(sub_a, a[index])
            sub_r = np.append(sub_r, r[index])
            sub_adv = np.append(sub_adv, adv[index])

        # print(sum_adv_init, sub_adv.sum())
        return sub_s, sub_a, sub_r, sub_adv


    def update(self):
        global GLOBAL_UPDATE_COUNTER, G_ITERATION
        while not COORD.should_stop():
            UPDATE_EVENT.wait()                     # wait until get batch of data

            self.sess.run(self.update_oldpi_op)     # copy pi to old pi
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
            data = np.vstack(data)
            # s, a, r, adv = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, S_DIM + A_DIM: S_DIM + A_DIM + 1], data[:, -1:]
            if isinstance(Action_Space, gym.spaces.Box):
                s, a, r, reward, adv = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, S_DIM + A_DIM: S_DIM + A_DIM+1], data[:, S_DIM + A_DIM+1: S_DIM + A_DIM+2], data[:, -1:]
            else:
                s, a, r, reward, adv = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1], data[:, S_DIM + 1: S_DIM + 1+1], data[:, S_DIM + 1+1: S_DIM + 1+2], data[:, -1:]
                a = a.flatten()
                r = r.flatten()
                adv = adv.flatten()

            # if adv.std() != 0:         
            adv_ori = adv*1       
            # adv = (adv - adv.mean())/adv.std()
            # adv = np.clip(adv, -5, 5)
            #     # adv = ((adv - adv.min())/(adv.max() - adv.min()) - 0.5)*2
            #     print('adv min max', adv.mean(), adv.min(), adv.max())

            # for index in range(len(adv)):
            #     if adv[index] > -0.5 and adv[index] < 0.5:
            #         adv[index] = 0

            possitive_index, negative_index = self.seperate_adv(adv)

            mean_return = np.mean(r)
            print(G_ITERATION, '  --------------- update! batch size:', GLOBAL_EP, '-----------------', len(r))
            vpred, olda_prob = self.sess.run([self.v, self.oldpi_prob], feed_dict = {self.tfs:s, self.tfa:a})
            vpred = vpred.flatten()

            # adv_filtered = adv = (adv - adv.mean())/adv.std()

            for iteration in range(UPDATE_STEP):
                # construct reward predict data                
                s_, a_, r_, adv_ = self.shuffel_data(s, a, r, adv)   
                count = 0
                for start in range(0, len(r), MIN_BATCH_SIZE):
                    end = start + MIN_BATCH_SIZE
                    if end > len(r) - 1 and count != 0:
                        break
                    if  end > len(r) - 1 and count == 0:
                        end = len(r)-1
                    count += 1

                    sub_s = s_[start:end]
                    sub_a = a_[start:end]
                    sub_r = r_[start:end]
                    sub_adv = np.asarray(adv_[start:end])

                    # sub_s, sub_a, sub_r, sub_adv = self.balance_minibatch(s, a, r, adv, sub_s, sub_a, sub_r, sub_adv, possitive_index, negative_index)
                    # sub_s, sub_a, sub_r, sub_adv = self.get_minibatch(s, a, r, adv, possitive_index, negative_index)
                    # print(sub_adv)
                    feed_dict = {
                        self.tfs: sub_s, 
                        self.tfa: sub_a, 
                        self.tfdc_r: sub_r, 
                        self.tfadv: sub_adv, 
                        self.tf_is_train: True
                    }
                    self.sess.run(self.train_op, feed_dict = feed_dict)
                    # self.sess.run([self.atrain_op, self.ctrain_op], feed_dict = feed_dict)

                tloss, aloss, vloss, entropy = self.check_overall_loss(s, a, r, adv)
                print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, entropy))

            feed_dict = {
                self.tfs: s, 
                self.tfa: a, 
                self.tfdc_r: r, 
                self.tfadv: adv, 
                self.tf_is_train: False
            }
            # onehota_prob, oldhota_prob, ratio = self.sess.run([self.a_prob, self.olda_prob, self.ratio], feed_dict = feed_dict)
            # for i in range(len(r)):
            #     print(onehota_prob[i], oldhota_prob[i], ratio[i])

            ratio, surr, surr2, a_prob, vpred_new = self.sess.run([self.ratio, self.surr, self.surr2, self.pi_prob, self.v], feed_dict = feed_dict)
            ratio = ratio.flatten()
            for i in range(len(r)): #range(25):
                act = int(a[i])
                output = []
                output.append(reward[i][0])
                output.append(r[i])
                output.append(vpred[i])
                output.append(vpred_new[i][0])
                output.append(adv[i])
                output.append(ratio[i])
                output.append(a[i])
                output.append(0)
                for prob in olda_prob[i]:
                    output.append(prob)
                output.append(0)
                for prob in a_prob[i]:
                    output.append(prob)
                # print(output)

                ratio_prob = a_prob[i][act] / olda_prob[i][act]
                # print("%8.4f, %8.4f, %8.4f, %8.4f, %8.4f, %8.4f, %6.0i, %8.4f, %8.4f"%(reward[i], r[i], vpred[i], vpred_new[i], adv[i], ratio[i], a[i], olda_prob[i][act],a_prob[i][act]))
                print("%8.4f, %8.4f, %8.4f, %8.4f, %8.4f, %8.4f, %6.0i, %8.4f"%(reward[i], r[i], vpred[i], vpred_new[i], adv[i], ratio[i], a[i], a_prob[i][act]), a_prob[i])

            # ratio_clip = np.clip(ratio, 1-EPSILON, 1+EPSILON)
            # surr_ = ratio*adv.flatten()
            # surr2_ = ratio_clip*adv.flatten()
            # # -np.minimum(ratio*adv.flatten(), ratio_clip*adv.flatten())

            # print((surr))
            # print(surr_)
            tloss, aloss, vloss, entropy = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy], feed_dict = feed_dict)
            print('-------------------------------------------------------------------------------')
            print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, np.mean(entropy)))

            # self.write_summary('Loss/entropy', np.mean(entropy))  
            # self.write_summary('Loss/a loss', aloss) 
            # self.write_summary('Loss/v loss', vloss) 
            # self.write_summary('Loss/t loss', tloss) 
            # self.write_summary('Perf/mean_reward', np.mean(reward))  

            self.saver.save(self.sess, './model/rl/model.cptk') 

            UPDATE_EVENT.clear()        # updating finished
            GLOBAL_UPDATE_COUNTER = 0   # reset counter
            G_ITERATION += 1
            ROLLING_EVENT.set()         # set roll-out available
            self.ob_rms.update(s)

class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = centauro_env.Simu_env(20000 + wid)
        self.ppo = GLOBAL_PPO

    def process_demo_path(self, demo_a, start_ob):
        buffer_s, buffer_a, buffer_r, buffer_er, buffer_vpred = [], [], [], [], []
        s = start_ob
        for i in range(len(demo_a)-1, -1, -1):
            a = demo_a[i]
            full_a = self.env.action_list[a]
            full_a = -np.array(full_a)
            for index in range(len(self.env.action_list)):
                env_action = np.array(self.env.action_list[index])
                if np.array_equal(env_action, full_a):
                    a = index 
                    break
            # print('reversed action', full_a, a)
            # for continues action
            # a = -a

            s_, r, event_r, done, info = self.env.step(a)
            vpred = self.ppo.get_v(s)

            buffer_s.append(s*1)
            buffer_a.append(a)
            buffer_r.append(r)                    # normalize reward, find to be useful
            buffer_er.append(event_r)
            buffer_vpred.append(vpred)

            s = s_

            if info != 'goal':
                # self.env.save_ep()
                dd = 1
            else:
                break 

        return buffer_s, buffer_a, buffer_r, buffer_er, buffer_vpred, s_

    def filter_crash(self, buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_info):
        s, a, r, vpred, info = [], [], [], [], []
        for i in range(len(buffer_r)):
            if buffer_info[i] != 'crash':
                s.append(buffer_s[i])
                a.append(buffer_a[i])
                r.append(buffer_r[i])
                vpred.append(buffer_vpred[i])
                info.append(buffer_info[i])

        vpred.append(buffer_vpred[-1])
        return s, a, r, vpred, info

    def compute_adv_return(self, buffer_reward, buffer_vpred, buffer_info):
        lastgaelam = 0.0
        buffer_return = np.zeros(len(buffer_reward))
        buffer_adv = np.zeros(len(buffer_reward))

        # print(len(buffer_reward))

        for index in reversed(range(len(buffer_reward))):
            if buffer_info[index] == 'crash' or buffer_info[index] == 'goal':
            # if buffer_info[index] == 'goal':
                nonterminal = 0
            else:
                nonterminal = 1
            delta = buffer_reward[index] + GAMMA * buffer_vpred[index+1] * nonterminal - buffer_vpred[index]

            # nonterminal = 1
            # delta = buffer_reward[index] + GAMMA * buffer_vpred[index+1] - buffer_vpred[index]

            lastgaelam = delta + GAMMA * LAM * nonterminal * lastgaelam
            buffer_adv[index] = lastgaelam
            # print (index, len(buffer_r), nonterminal, delta, lastgaelam, buffer_adv[index])
        buffer_return = buffer_adv + buffer_vpred[:-1]
        # print(buffer_reward)
        # print(buffer_return)

        return buffer_adv, buffer_return

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER \
            , Goal_states, Goal_count, Goal_return, Goal_buffer_full \
            , Crash_states, Crash_count, Crash_return, Crash_buffer_full \
            , History_states, History_count, History_adv, History_return, History_buffer_full

        self.env.save_ep()
        # for _ in range(5):
        #     s = self.env.reset( 0, 0, 0)
        #     self.env.save_ep()

        while not COORD.should_stop():
            buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_info = [], [], [], [], []
            info = 'unfinish'

            s = self.env.reset(0, 1, 0)
            t = 0

            has_crash = False

            test_actions = [3, 3, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 4, 4]
            # test_actions = [3, 3, 5, 5, 5, 5, 5, 4, 4, 3, 3, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 4, 4]
            ep_length = len(test_actions) #EP_LEN #len(test_actions)
            # for a in test_actions:
            while(1):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_info = [], [], [], [], []
                    # break
                
                # if self.wid == 0:
                #     self.ppo.get_action_prob(s)
                a = self.ppo.choose_action(s)
                s_, r, event_r, done, info = self.env.step(a)
                vpred = self.ppo.get_v(s)

                if self.wid == 0:
                    vpred_ = self.ppo.get_v(s_)
                    td = event_r + GAMMA*vpred_ - vpred
                    print("a: %i | event_r: %7.4f| vpred: %7.4f| TD: %7.4f| " %(a, r, vpred, td))

                buffer_s.append(s*1)
                buffer_a.append(a)
                buffer_r.append(r)                    # normalize reward, find to be useful
                buffer_vpred.append(vpred)
                buffer_info.append(info)
                s_demo = s_*1
                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers

                s = s_
                t += 1

                if info == 'crash':
                    has_crash = True

                if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE or t == ep_length-1 or info == 'goal' or info == 'out':

                    vpred_ = self.ppo.get_v(s_)
                    buffer_vpred.append(vpred_)

                    buffer_r_backup = buffer_r*1
                    for index in range(len(buffer_r)):
                        if buffer_info[index] != 'goal':
                            buffer_r[index] = 0

                    buffer_adv, buffer_return = self.compute_adv_return(buffer_r, buffer_vpred, buffer_info)

                    for index in range(len(buffer_r_backup)):
                        if buffer_info[index] != 'goal':
                            buffer_adv[index] += buffer_r_backup[index]
                            buffer_return[index] += buffer_r_backup[index]

                    # print(buffer_r_backup)
                    # print(buffer_return)

                    bs, ba, bret, brew, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_return)[:, np.newaxis], np.array(buffer_r_backup)[:, np.newaxis], np.array(buffer_adv)[:, np.newaxis]                
                    QUEUE.put(np.hstack((bs, ba, bret, brew, badv)))          # put data in the queue

                    # if has_crash:
                    #     buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_info = self.filter_crash(buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_info)
                    #     if len(buffer_r) > 1:
                    #         buffer_adv, buffer_return = self.compute_adv_return(buffer_r, buffer_vpred, buffer_info)
                    #         bs, ba, bret, brew, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_return)[:, np.newaxis], np.array(buffer_r)[:, np.newaxis], np.array(buffer_adv)[:, np.newaxis]                
                    #         QUEUE.put(np.hstack((bs, ba, bret, brew, badv)))          # put data in the queue
                    #         GLOBAL_UPDATE_COUNTER += len(buffer_r)

                    buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_info = [], [], [], [], []
                    if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if G_ITERATION >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    if done or t == ep_length-1:
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