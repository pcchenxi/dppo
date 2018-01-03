from vec_normalize import VecNormalize
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
EP_LEN = 200
N_WORKER = 4               # parallel workers
GAMMA = 0.99                # reward discount factor
LAM = 0.99
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0005               # learning rate for critic
LR = 0.00005

EP_BATCH_SIZE = 5
UPDATE_L_STEP = 30
BATCH_SIZE = 5120
MIN_BATCH_SIZE = 64       # minimum batch size for updating PPO

UPDATE_STEP = 1            # loop update operation n-steps
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

        # actor
        pi, self.vs, self.vl, net_params = self._build_anet('net', self.tfs, trainable=True)

        oldpi, oldvs, oldvl, oldnet_params = self._build_anet('oldnet', self.tfs, trainable=False)
        self.sample_op = pi.sample() #tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.sample_op_det = pi.mode()

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(net_params, oldnet_params)]

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
        
        # critic
        self.tfdc_rs = tf.placeholder(tf.float32, [None], 'discounted_rs')
        self.tfdc_rl = tf.placeholder(tf.float32, [None], 'discounted_rl')
        # vpredclipped = oldv + tf.clip_by_value(self.v - oldv, - EPSILON, EPSILON)

        self.closs_s = tf.square(self.tfdc_rs - self.vs)
        self.closs_l = tf.square(self.tfdc_rl - self.vl)
        # self.closs2 = tf.square(self.tfdc_r - vpredclipped)

        self.closs = tf.reduce_mean(self.closs_s)*0.5 + tf.reduce_mean(self.closs_l)*0.5 #tf.reduce_mean(tf.maximum(self.closs1, self.closs2)) #

        self.total_loss = self.aloss*1 + self.closs*0.25 - 0.0*self.entropy

        # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # grads = tf.gradients(self.total_loss, params)
        # grads, self.grad_norm = tf.clip_by_global_norm(grads, 1)            
        # grads = list(zip(grads, params))

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     optimizer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5) 
        #     # self.train_op = optimizer.apply_gradients(grads)
        #     self.train_op = optimizer.minimize(self.total_loss)
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.total_loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('./log', self.sess.graph)    
        self.pi_prob = tf.nn.softmax(pi.logits)
        self.oldpi_prob = tf.nn.softmax(oldpi.logits)

        # self.load_model()   

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

    def shuffel_data(self, s, a, rs, rl, adv):
        index_shuffeled = np.random.choice(len(rs), len(rs), replace=False)
        s_shuf, a_shuf, rs_shuf, rl_shuf, adv_shuf = [], [], [], [], []

        for i in index_shuffeled:
            s_shuf.append(s[i])
            a_shuf.append(a[i])
            rs_shuf.append(rs[i])
            rl_shuf.append(rl[i])
            adv_shuf.append(adv[i])

        return s_shuf, a_shuf, rs_shuf, rl_shuf, adv_shuf
                
    def add_layer(self, x, out_size, name, ac=None, trainable=True):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        x = tf.layers.dense(x, out_size, kernel_initializer=w_init, name = name, trainable = trainable)
        # the momentum plays important rule. the default 0.99 is too high in this case!
        x = tf.layers.batch_normalization(x, momentum=0.7, training=self.tf_is_train)    # when have BN
        out = x if ac is None else ac(x)
        return out

    def _build_feature_net(self, name, input_state, trainable):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        # w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(name):
            state_size = 4
            num_img = S_DIM - state_size 
            img_size = int(math.sqrt(num_img))

            ob_grid = tf.slice(input_state, [0, 0], [-1, num_img], name = 'slice_grid')
            ob_state = tf.slice(input_state, [0, num_img], [-1, state_size], name = 'slice_ob') 

            reshaped_grid = tf.reshape(ob_grid,shape=[-1, img_size, img_size, 1]) 
            conv1 = tf.layers.conv2d(inputs=reshaped_grid, filters=32, kernel_size=[8, 8], strides = 4, name='conv1', kernel_initializer=w_init, padding="valid", activation=tf.nn.relu, trainable=trainable )
            # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            # x = tf.layers.batch_normalization(x, momentum=0.7, training=self.tf_is_train)    # when have BN
            # x = tf.nn.tanh(x)

            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4, 4], strides = 2, name = 'conv2', padding="valid", kernel_initializer=w_init, activation=tf.nn.relu, trainable=trainable )
            # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            # x = tf.layers.batch_normalization(x, momentum=0.7, training=self.tf_is_train)    # when have BN
            # x = tf.nn.tanh(x)

            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3, 3], strides = 1, name = 'conv3', padding="valid", kernel_initializer=w_init, activation=tf.nn.relu, trainable=trainable )
            # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
            # x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, trainable=trainable )

            flat = tf.contrib.layers.flatten(conv1)

            ob_fc = tf.layers.dense(ob_state, 32, tf.nn.relu, kernel_initializer=w_init, name = 'fc_state', trainable=trainable)
            feature = tf.concat([flat , ob_fc], 1, name = 'concat')
            # x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)

        return feature

    def _build_anet(self, name, input_state, trainable):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        # w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(name):
            self.feature = self._build_feature_net('feature', input_state, trainable=trainable)
            with tf.variable_scope('actor'):
                l1 = tf.layers.dense(self.feature, 1024, tf.nn.relu, kernel_initializer=w_init, name = 'fc_1', trainable=trainable)
                # l2 = tf.layers.dense(l1, 1024, tf.nn.relu, kernel_initializer=w_init, trainable=trainable)
                # l1 = self.add_layer(self.feature, 256, 'l1_fc', tf.nn.tanh, trainable=trainable)
                # l1 = self.add_layer(l1, 128, 'l1_fc2', tf.nn.tanh, trainable=trainable)

                if isinstance(Action_Space, gym.spaces.Box):
                    print('continue action')
                    pi = tf.layers.dense(l1, A_DIM, kernel_initializer=w_init, trainable=trainable)
                    logstd = tf.get_variable(name="logstd", shape=[1, A_DIM+2], initializer=tf.zeros_initializer(), trainable=trainable)
                    out = tf.concat([pi, pi * 0.0 + logstd], axis=1)
                else:
                    print('discrate action')
                    # out = tf.layers.dense(self.feature_a, A_DIM, kernel_initializer=w_init, trainable=trainable)  
                    out = tf.layers.dense(l1, A_DIM+2, name='fc_2', kernel_initializer=w_init, trainable=trainable)      

                vs = tf.slice(out, [0, A_DIM], [-1, 1], name = 'slice_vs')
                vl = tf.slice(out, [0, A_DIM+1], [-1, 1], name = 'slice_vl')
                pdparam = tf.slice(out, [0, 0], [-1, A_DIM], name = 'slice_a')

        pdtype = make_pdtype(Action_Space)
        pd = pdtype.pdfromflat(pdparam)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pd, vs, vl, params


    def choose_action(self, s, det=False):
        s = s[np.newaxis, :]

        if det:
            a = self.sess.run(self.sample_op_det, {self.tfs: s, self.tf_is_train: False})[0]
        else:
            a = self.sess.run(self.sample_op, {self.tfs: s, self.tf_is_train: False})[0]
        # prob_weights = self.sess.run(self.pi_prob, feed_dict={self.tfs: s})
        # a = np.random.choice(range(prob_weights.shape[1]),
        #                           p=prob_weights.ravel())

        if isinstance(Action_Space, gym.spaces.Box):
            a = np.clip(a, -1, 1)
        return a

    def get_action_prob(self, s):
        s = s[np.newaxis, :]
        prob = self.sess.run(self.pi_prob, {self.tfs: s, self.tf_is_train: False})[0]

        # plt.clf()
        # plt.scatter(range(A_DIM+1), np.append(prob, 1.0).flatten())
        # plt.pause(0.01)

        return prob

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        vs, vl = self.sess.run([self.vs, self.vl], {self.tfs: s, self.tf_is_train: False})
        return vs[0,0], vl[0,0]

    def check_overall_loss(self, s, a, rs, rl, adv):
        feed_dict = {
            self.tfs: s, 
            self.tfa: a, 
            self.tfdc_rs: rs, 
            self.tfdc_rl: rl, 
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

    def get_minibatch(self, s, a, rs, rl, adv, possitive_index, negative_index):
        sub_s, sub_a, sub_rs, sub_rl, sub_adv = [], [], [], [], []
        if len(possitive_index) != 0 and len(negative_index) != 0:
            size = int(MIN_BATCH_SIZE/2)
            selected_index = np.random.choice(len(possitive_index), size)
            for i in selected_index:
                index = possitive_index[i]
                sub_s.append(s[index])
                sub_a.append(a[index])
                sub_rs.append(rs[index])
                sub_rl.append(rl[index])
                sub_adv.append(adv[index])

            selected_index = np.random.choice(len(negative_index), size)
            for i in selected_index:
                index = negative_index[i]
                sub_s.append(s[index])
                sub_a.append(a[index])
                sub_rs.append(rs[index])
                sub_rl.append(rl[index])
                sub_adv.append(adv[index])
        else:
            selected_index = np.random.choice(len(rs), MIN_BATCH_SIZE)
            for i in selected_index:
                index = i
                sub_s.append(s[index])
                sub_a.append(a[index])
                sub_rs.append(rs[index])
                sub_rl.append(rl[index])
                sub_adv.append(adv[index])
        return sub_s, sub_a, sub_rs, sub_rl, sub_adv

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
        global GLOBAL_UPDATE_COUNTER, G_ITERATION, GLOBAL_EP
        update_count = 1
        s_all, a_all, rs_all, rl_all, adv_s_all, adv_l_all = [], [], [], [], [], []

        while not COORD.should_stop():
            UPDATE_EVENT.wait()                     # wait until get batch of data

            self.sess.run(self.update_oldpi_op)     # copy pi to old pi
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
            data = np.vstack(data)
            # s, a, r, adv = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, S_DIM + A_DIM: S_DIM + A_DIM + 1], data[:, -1:]
            if isinstance(Action_Space, gym.spaces.Box):
                s, a, rs, rl, adv_s, adv_l = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, S_DIM + A_DIM: S_DIM + A_DIM+1], data[:, S_DIM + A_DIM+1: S_DIM + A_DIM+2], data[:, S_DIM + A_DIM+2: S_DIM + A_DIM+3], data[:, -1:]
            else:
                s, a, rs, rl, adv_s, adv_l = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1], data[:, S_DIM + 1: S_DIM + 2], data[:, S_DIM + 2: S_DIM + 3], data[:, S_DIM + 3: S_DIM + 4], data[:, -1:]
                a = a.flatten()
                rs = rs.flatten()
                rl = rl.flatten()
                adv_s = adv_s.flatten()
                adv_l = adv_l.flatten()

            # if a_all == []:
            #     s_all, a_all, rs_all, rl_all, adv_s_all, adv_l_all = s, a, rs, rl, adv_s, adv_l
            # else:
            #     s_all = np.concatenate((s_all, s), axis = 0)
            #     a_all = np.concatenate((a_all, a), axis = 0)
            #     rs_all = np.concatenate((rs_all, rs), axis = 0)
            #     rl_all = np.concatenate((rl_all, rl), axis = 0)
            #     adv_s_all = np.concatenate((adv_s_all, adv_s), axis = 0)
            #     adv_l_all = np.concatenate((adv_l_all, adv_l), axis = 0)
            # print('all size', len(a_all))

            # if update_count % UPDATE_L_STEP == 0:
            #     print('update long adv', update_count, UPDATE_L_STEP)
            #     s, a, rs, rl, adv_s, adv_l = s_all, a_all, rs_all, rl_all, adv_s_all, adv_l_all 
            #     adv = adv_l_all 
            #     update_iter = UPDATE_STEP*8
            #     print(len(a), len(a))
            # else:
            #     print('update short adv', update_count, UPDATE_L_STEP)
            #     adv = adv_s
            #     update_iter = UPDATE_STEP

            # adv_s = (adv_s - adv_s.mean())/adv_s.std()
            # adv_l = (adv_l - adv_l.mean())/adv_l.std()

            adv = adv_s + adv_l
            if adv.std() != 0:
                adv = (adv - adv.mean())/adv.std()
            # possitive_index, negative_index = self.seperate_adv(adv)

            print(G_ITERATION, '  --------------- update! batch size:', GLOBAL_EP, '-----------------', len(rs))
            # vpred, olda_prob = self.sess.run([self.v, self.oldpi_prob], feed_dict = {self.tfs:s, self.tfa:a})
            # vpred = vpred.flatten()

            # adv_filtered = adv = (adv - adv.mean())/adv.std()

            for iteration in range(UPDATE_STEP):
                # construct reward predict data                
                s_, a_, rs_, rl_, adv_ = self.shuffel_data(s, a, rs, rl, adv)   
                count = 0
                for start in range(0, len(rs), MIN_BATCH_SIZE):
                    end = start + MIN_BATCH_SIZE
                    if end > len(rs) - 1 and count != 0:
                        break
                    if  end > len(rs) - 1 and count == 0:
                        end = len(rs)-1
                    count += 1

                    sub_s = s_[start:end]
                    sub_a = a_[start:end]
                    sub_rs = rs_[start:end]
                    sub_rl = rl_[start:end]
                    sub_adv = np.asarray(adv_[start:end])

                    # sub_s, sub_a, sub_r, sub_adv = self.balance_minibatch(s, a, r, adv, sub_s, sub_a, sub_r, sub_adv, possitive_index, negative_index)
                    # sub_s, sub_a, sub_rs, sub_rl, sub_adv = self.get_minibatch(s, a, rs, rl, adv, possitive_index, negative_index)
                    # print(sub_adv)
                    feed_dict = {
                        self.tfs: sub_s, 
                        self.tfa: sub_a, 
                        self.tfdc_rs: sub_rs,
                        self.tfdc_rl: sub_rl, 
                        self.tfadv: sub_adv, 
                        self.tf_is_train: True
                    }
                    self.sess.run(self.train_op, feed_dict = feed_dict)
                    # self.sess.run([self.atrain_op, self.ctrain_op], feed_dict = feed_dict)

                tloss, aloss, vloss, entropy = self.check_overall_loss(s, a, rs, rl, adv)
                print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, entropy))

            feed_dict = {
                self.tfs: s, 
                self.tfa: a, 
                self.tfdc_rs: rs, 
                self.tfdc_rl: rl, 
                self.tfadv: adv, 
                self.tf_is_train: True
            }
            # onehota_prob, oldhota_prob, ratio = self.sess.run([self.a_prob, self.olda_prob, self.ratio], feed_dict = feed_dict)
            # for i in range(len(r)):
            #     print(onehota_prob[i], oldhota_prob[i], ratio[i])

            ratio, surr, surr2, a_prob, olda_prob = self.sess.run([self.ratio, self.surr, self.surr2, self.pi_prob, self.oldpi_prob], feed_dict = feed_dict)
            vs, vl = self.sess.run([self.vs, self.vl], feed_dict = feed_dict)
            ratio = ratio.flatten()
            for i in range(len(rs)): #range(25):
                act = int(a[i])
                print("%8.4f, %8.4f|, %8.4f, %8.4f|, %8.4f, %8.4f, %8.4f|, %8.4f|, %6.0i|, %8.4f, %8.4f"%(rs[i], rl[i], vs[i], vl[i], adv_s[i], adv_l[i], adv[i], ratio[i], a[i], olda_prob[i][act],a_prob[i][act]))
                
                # print("%8.4f, %8.4f, %8.4f, %8.4f, %8.4f, %8.4f, %6.0i, %8.4f"%(reward[i], r[i], vpred[i], vpred_new[i], adv[i], ratio[i], a[i], a_prob[i][act]), a_prob[i])

            print(rs.mean(), rl.mean())
            # ratio_clip = np.clip(ratio, 1-EPSILON, 1+EPSILON)
            # surr_ = ratio*adv.flatten()
            # surr2_ = ratio_clip*adv.flatten()
            # # -np.minimum(ratio*adv.flatten(), ratio_clip*adv.flatten())

            # print((surr))
            # print(surr_)
            # tloss, aloss, vloss, entropy = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy], feed_dict = feed_dict)
            print('-------------------------------------------------------------------------------')
            print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, entropy))

            # self.write_summary('Loss/entropy', np.mean(entropy))  
            # self.write_summary('Loss/a loss', aloss) 
            # self.write_summary('Loss/v loss', vloss) 
            # self.write_summary('Loss/t loss', tloss) 
            # self.write_summary('Perf/mean_reward', np.mean(reward))  

            self.saver.save(self.sess, './model/rl/model.cptk') 

            UPDATE_EVENT.clear()        # updating finished
            GLOBAL_UPDATE_COUNTER = 0   # reset counter
            G_ITERATION += 1
            GLOBAL_EP = 0
            ROLLING_EVENT.set()         # set roll-out available
            
            update_count += 1
            if update_count % UPDATE_L_STEP == 1:
                update_count = 1
                s_all, a_all, rs_all, rl_all, adv_s_all, adv_l_all = [], [], [], [], [], []
                print('reset')


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = centauro_env.Simu_env(20000 + wid)
        self.env = VecNormalize(self.env)
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
            vpred_s, vpred_l = self.ppo.get_v(s)

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

    def compute_adv_return(self, buffer_reward, buffer_vpred, buffer_info, mode='long'):
        lastgaelam = 0.0
        buffer_return = np.zeros(len(buffer_reward))
        buffer_adv = np.zeros(len(buffer_reward))

        # print(len(buffer_reward))
        if mode == 'long':
            gamma = GAMMA
        else:
            gamma = GAMMA*0.7
            
        for index in reversed(range(len(buffer_reward))):
            # if buffer_info[index] == 'crash' or buffer_info[index] == 'goal':
            if mode == 'long':
                if buffer_info[index] == 'goal':
                    nonterminal = 0
                else:
                    nonterminal = 1
            else:
            #     if buffer_info[index] == 'crash':
            #         nonterminal = 0
            #     else:
                nonterminal = 1                
            delta = buffer_reward[index] + gamma * buffer_vpred[index+1] * nonterminal - buffer_vpred[index]

            # nonterminal = 1
            # delta = buffer_reward[index] + gamma * buffer_vpred[index+1] - buffer_vpred[index]

            lastgaelam = delta + gamma * LAM * nonterminal * lastgaelam
            buffer_adv[index] = lastgaelam
            # print (index, len(buffer_r), nonterminal, delta, lastgaelam, buffer_adv[index])

        buffer_return = buffer_adv + buffer_vpred[:-1]
        # for index in range(len(buffer_reward)):
        #     print(buffer_reward[index], buffer_return[index], buffer_vpred[index], buffer_adv[index])

        # print('-----------------')
        return buffer_adv, buffer_return

    def process_and_send(self, buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_info, s_, end):
        buffer_vpred_s, buffer_vpred_l = self.ppo.sess.run([self.ppo.vs, self.ppo.vl], feed_dict = {self.ppo.tfs: buffer_s, self.ppo.tf_is_train: False})
        buffer_vpred_s = buffer_vpred_s.flatten()
        buffer_vpred_l = buffer_vpred_l.flatten()
        
        vpred_s_, vpred_l_ = self.ppo.get_v(s_)
        if end != -1 and end > vpred_l_:
            vpred_l_ = end
        # else:
        #     vpred_l_ = 0
        buffer_vpred_s = np.append(buffer_vpred_s, vpred_s_)
        buffer_vpred_l = np.append(buffer_vpred_l, vpred_l_)
        buffer_adv_s, buffer_return_s = self.compute_adv_return(buffer_rs, buffer_vpred_s, buffer_info, 'short')
        buffer_adv_l, buffer_return_l = self.compute_adv_return(buffer_rl, buffer_vpred_l, buffer_info, 'long')

        # for i in range(len(buffer_adv_s)):
        #     if buffer_adv_s[i] > -0.5 and buffer_adv_s[i] < 0.5:
        #         buffer_adv_s[i] = 0

        buffer_adv = buffer_adv_s + buffer_adv_l

        # for i in range(len(buffer_adv)):
            # print("rs: %7.4f| rl: %7.4f| vs: %7.4f| vl: %7.4f| adv_s: %7.4f| adv_l: %7.4f| r_s: %7.4f| r_l: %7.4f" %(buffer_rs[i], buffer_rl[i], buffer_vpred_s[i], buffer_vpred_l[i], buffer_adv_s[i], buffer_adv_l[i], buffer_return_s[i], buffer_return_l[i]), buffer_info[i])
        bs, ba, bret_s, bret_l, badv_s, badv_l = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_return_s)[:, np.newaxis], np.array(buffer_return_l)[:, np.newaxis], np.array(buffer_adv_s)[:, np.newaxis], np.array(buffer_adv_l)[:, np.newaxis]                 
        QUEUE.put(np.hstack((bs, ba, bret_s, bret_l, badv_s, badv_l)))          # put data in the queue

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER \
            , Goal_states, Goal_count, Goal_return, Goal_buffer_full \
            , Crash_states, Crash_count, Crash_return, Crash_buffer_full \
            , History_states, History_count, History_adv, History_return, History_buffer_full

        # reply buffer:
        g_s, g_a, g_rs, g_rl, g_info, g_s_, g_end = [], [], [], [], [], [], []
        g_index = 0
        g_max = 20

        # self.env.save_ep()
        # for _ in range(3):
        #     s = self.env.reset( 0, 0, 1)
        #     self.env.save_ep()

        update_counter = 0
        ep_count = 0

        while not COORD.should_stop():
            buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_vpred_s, buffer_vpred_l, buffer_info = [], [], [], [], [], [], []
            info = 'unfinish'

            s = self.env.reset(0, 1, 1)
            t = 0

            has_crash = False

            test_actions = [3, 3, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 4, 4]
            # test_actions = [3, 3, 5, 5, 5, 5, 5, 4, 4, 3, 3, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 4, 4]
            ep_length = EP_LEN #len(test_actions)
            # for a in test_actions:
            GLOBAL_EP += 1
            ep_count += 1
            saved_ep = False
            # if ep_count == 5:
            #     ep_count = 0
            #     self.env.clear_history()
            #     s = self.env.reset( 0, 0, 0)
            #     self.env.save_ep()

            while(1):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_vpred_s, buffer_vpred_l, buffer_info = [], [], [], [], [], [], []
                    # if update_counter%5 == 0:
                    #     self.env.clear_history()
                    #     s = self.env.reset( 0, 0, 0)
                    #     self.env.save_ep()
                    # self.env.clear_history_leave_one()
                    print(len(g_a))
                    update_counter += 1

                    # if len(g_a) > 0:
                    #     num = int(len(g_a)/2 + 1)
                    #     for _ in range(num):
                    #         index = np.random.randint(len(g_a))
                    #         self.process_and_send(g_s[index], g_a[index], g_rs[index], g_rl[index], g_info[index], g_s_[index])
                    # break

                # if self.wid == 0:
                #     self.ppo.get_action_prob(s)
                # if self.wid == 0 or self.wid == 1:
                #     a = self.ppo.choose_action(s, True)
                # else:
                a = self.ppo.choose_action(s, False)
                s_, r_short, r_long, done, info = self.env.step(a)
                vpred_s, vpred_l = self.ppo.get_v(s)

                # img = self.ppo.sess.run(self.ppo.img, feed_dict = {self.ppo.tfs:[s]})[0]
                # plt.clf()
                # plt.imshow(img)
                # plt.pause(0.01)

                # if self.wid == 0:
                #     prob = self.ppo.get_action_prob(s)
                #     print("a: %6i | rs: %7.4f| rl: %7.4f| rs: %7.4f| rl: %7.4f| prob: %7.4f" %(a, r_short, r_long, vpred_s, vpred_l, prob[a]), info)

                buffer_s.append(s*1)
                buffer_a.append(a)
                buffer_rs.append(r_short)                    # normalize reward, find to be useful
                buffer_rl.append(r_long)
                buffer_vpred_s.append(vpred_s)
                buffer_vpred_l.append(vpred_l)
                buffer_info.append(info)
                s_demo = s_*1
                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers

                s = s_
                t += 1

                if info == 'crash' and saved_ep == False:
                    has_crash = True
                    self.env.save_ep()
                    saved_ep = True

                # if self.wid == 0:
                #     if t == ep_length-1 or done:
                #         buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_vpred_s, buffer_vpred_l, buffer_info = [], [], [], [], [], [], []
                #         break
                #     else:
                #         continue

                if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE or t == ep_length-1 or done:
                # if (GLOBAL_EP != 0 and GLOBAL_EP%EP_BATCH_SIZE == 0) or t == ep_length-1 or done:
                    
                    if info == 'goal' and len(buffer_a) > 5:
                        if len(g_a) < g_max:
                            g_s.append(buffer_s)
                            g_a.append(buffer_a)
                            g_rs.append(buffer_rs)
                            g_rl.append(buffer_rl)
                            g_info.append(buffer_info)
                            g_s_.append(s_)
                            g_end.append(self.env.return_end)
                        else:
                            index = g_index % len(g_a)
                            g_s[index] = buffer_s
                            g_a[index] = buffer_a
                            g_rs[index] = buffer_rs
                            g_rl[index] = buffer_rl
                            g_info[index] = buffer_info
                            g_s_[index] = s_
                            g_end[index] = self.env.return_end
                            g_index += 1 

                    if len(g_a) > 0 and np.random.rand() > 0.7:
                        index = np.random.randint(len(g_a))
                        self.process_and_send(g_s[index], g_a[index], g_rs[index], g_rl[index], g_info[index], g_s_[index], g_end[index])

                    self.process_and_send(buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_info, s_, self.env.return_end)

                    buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_vpred_s, buffer_vpred_l, buffer_info = [], [], [], [], [], [], []
                    if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                    # if (GLOBAL_EP != 0 and GLOBAL_EP%EP_BATCH_SIZE == 0):
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if G_ITERATION >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    if done or t == ep_length-1:
                        break 


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 1

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