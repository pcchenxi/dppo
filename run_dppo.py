from vec_normalize import VecNormalize
from distributions import make_pdtype
from utils import ortho_init

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from environment import centauro_env
import scipy.signal
import cv2, math, time
import matplotlib.pyplot as plt

import joblib, time
import cv2

EP_MAX = 500000
EP_LEN = 50
N_WORKER = 8               # parallel workers
GAMMA = 0.98                # reward discount factor
LAM = 0.95
LR = 0.0001

BATCH_SIZE = 5000
MIN_BATCH_SIZE = 1000 #int(BATCH_SIZE/5)-1       # minimum batch size for updating PPO

UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2              # for clipping surrogate objective
S_DIM, A_DIM = centauro_env.observation_space, centauro_env.action_space 
Action_Space = centauro_env.action_type

G_ITERATION = 0

t_s = time.time()
G_lift_s, G_straight_s, G_avoid_s, G_open_s = [], [], [], []
# G_lift_s = joblib.load('./guided_tra/lift_s_test.pkl')
# G_lift_a = joblib.load('./guided_tra/lift_a_test.pkl')
# G_lift_ret = joblib.load('./guided_tra/lift_ret_test.pkl')

# G_straight_s = joblib.load('./guided_tra/straight_s.pkl')[:100000]
# G_straight_a = joblib.load('./guided_tra/straight_a.pkl')[:100000]
# G_straight_ret = joblib.load('./guided_tra/straight_ret.pkl')[:100000]

# G_avoid_s = joblib.load('./guided_tra/avoid_s.pkl')
# G_avoid_a = joblib.load('./guided_tra/avoid_a.pkl')
# G_avoid_ret = joblib.load('./guided_tra/avoid_ret.pkl')

# G_open_s = joblib.load('./guided_tra/open_s.pkl')
# G_open_a = joblib.load('./guided_tra/open_a.pkl')
# G_open_ret = joblib.load('./guided_tra/open_ret.pkl')

print('loaded', len(G_lift_s), len(G_straight_s), len(G_avoid_s), len(G_open_s))
print(time.time() - t_s)

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        self.tfdc_r1 = tf.placeholder(tf.float32, [None], 'discounted_r1')
        self.tfdc_r2 = tf.placeholder(tf.float32, [None], 'discounted_r2')

        if isinstance(Action_Space, gym.spaces.Box):
            self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        else:
            self.tfa = tf.placeholder(tf.int32, [None], 'action')

        self.tfadv = tf.placeholder(tf.float32, [None], 'advantage')
        self.tf_is_train = tf.placeholder(tf.bool, None)

        print('init place holder')
        # actor
        pd, self.v1, self.v2, net_params = self._build_anet('net', self.tfs, trainable=True)
        oldpd, oldv1, oldv2, oldnet_params = self._build_anet('oldnet', self.tfs, trainable=False)
        print('inti net')

        self.sample_op = pd.sample() #tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.sample_op_det = pd.mode()

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(net_params, oldnet_params)]

        neglogpac = pd.neglogp(self.tfa)
        OLDNEGLOGPAC = oldpd.neglogp(self.tfa)
        self.entropy = tf.reduce_mean(pd.entropy())

        vf_loss1 = tf.square(self.v1 - self.tfdc_r1)

        vpredclipped2 = oldv2 + tf.clip_by_value(self.v2 - oldv2, - EPSILON, EPSILON)
        vf_losses12 = tf.square(self.v2 - self.tfdc_r2)
        vf_losses22 = tf.square(vpredclipped2 - self.tfdc_r2)
        vf_loss2 = 0.5 * tf.reduce_mean(tf.maximum(vf_losses12, vf_losses22))

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -self.tfadv * ratio
        pg_losses2 = -self.tfadv * tf.clip_by_value(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), EPSILON)))

        loss = pg_loss - self.entropy * 0.001 + vf_loss2

        params = tf.trainable_variables(scope='net')
        # print(params)

        grads = tf.gradients(loss, params)
        max_grad_norm = 1
        if max_grad_norm is not None:
            grads_clipped, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_params = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        self.train_op = trainer.apply_gradients(grads_and_params)

        self.train_op1 = tf.train.AdamOptimizer(0.001).minimize(vf_loss1)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('./log', self.sess.graph)   

        # self.pi_prob = tf.nn.softmax(pd.logits)
        self.aloss = pg_loss 
        self.closs = tf.reduce_mean(vf_loss2)
        self.closs1 = vf_loss1
        self.total_loss = loss
        self.ratio = ratio
        self.grad_norm = _grad_norm

        # self.load_model()   

    def load_model(self):
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state('./model/rl/')
        # ckpt = tf.train.get_checkpoint_state('./all_model/log_gppo/')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print ('loaded')
        else:
            print ('no model file')  

    def write_summary(self, summary_name, value):
        summary = tf.Summary()
        summary.value.add(tag=summary_name, simple_value=float(value))
        self.summary_writer.add_summary(summary, GLOBAL_STEP)
        self.summary_writer.flush()  
        # print(value, GLOBAL_EP, 'summary')

    def shuffel_data(self, s, a, r1, r2, adv):
        index_shuffeled = np.random.choice(len(r1), len(r1), replace=False)
        s_shuf, a_shuf, r1_shuf, r2_shuf, adv_shuf = [], [], [], [], []

        for i in index_shuffeled:
            s_shuf.append(s[i])
            a_shuf.append(a[i])
            r1_shuf.append(r1[i])
            r2_shuf.append(r2[i])
            adv_shuf.append(adv[i])

        return s_shuf, a_shuf, r1_shuf, r2_shuf, adv_shuf

    def _build_feature_net(self, name, input_state, trainable):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        # w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope(name):
            state_size = 4
            num_img = S_DIM - state_size 
            img_size = int(math.sqrt(num_img))
            print('image size', img_size)

            ob_grid = tf.slice(input_state, [0, 0], [-1, num_img], name = 'slice_grid')
            ob_state = tf.slice(input_state, [0, num_img], [-1, state_size], name = 'slice_ob') 

            reshaped_grid = tf.reshape(ob_grid,shape=[-1, img_size, img_size, 1]) 
            h = tf.layers.conv2d(inputs=reshaped_grid, filters=16, kernel_size=[3, 3], strides = 2, name = 'conv1', padding="valid", kernel_initializer=tf.orthogonal_initializer(2), activation=tf.nn.relu, trainable=trainable )
            h1 = tf.layers.conv2d(inputs=h, filters=32, kernel_size=[3, 3], strides = 1, name = 'conv2', padding="valid", kernel_initializer=tf.orthogonal_initializer(2), activation=tf.nn.relu, trainable=trainable )
            h2 = tf.layers.conv2d(inputs=h1, filters=32, kernel_size=[3, 3], strides = 1, name = 'conv3', padding="valid", kernel_initializer=tf.orthogonal_initializer(2), activation=tf.nn.relu, trainable=trainable )
            h3 = tf.contrib.layers.flatten(h2)

            h_concat = tf.concat([h3, ob_state], axis=1)

            # ob_fc = tf.layers.dense(ob_state, 32, tf.nn.relu, kernel_initializer=w_init, name = 'fc_state', trainable=trainable)
            # feature = tf.concat([flat , ob_state], 1, name = 'concat')
            # x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)

        return h_concat

    def _build_anet(self, name, input_state, trainable):
        w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        # w_init = tf.random_normal_initializer(0., 0.01)
        # w_init_big = tf.random_normal_initializer(0., 0.01)
        # w_init = tf.orthogonal_initializer(2)
        with tf.variable_scope(name):
            # feature = self._build_feature_net('feature', input_state, trainable=trainable)
            h1 = tf.layers.dense(input_state, 256, tf.nn.relu, kernel_initializer=w_init, name = 'fc1', trainable=trainable)
            h2 = tf.layers.dense(h1, 256, tf.nn.relu, kernel_initializer=w_init, name = 'fc2', trainable=trainable)
            h3 = tf.layers.dense(h2, 256, tf.nn.relu, kernel_initializer=w_init, name = 'fc3', trainable=trainable)

            state_t = tf.slice(input_state, [0, S_DIM-3], [-1, 3], name = 'slice_ob') 

            with tf.variable_scope('actor_critic'):
                if isinstance(Action_Space, gym.spaces.Box):
                    print('continue action',A_DIM, S_DIM)
                    pi = tf.layers.dense(h3, A_DIM, kernel_initializer=w_init, name = 'fc_pi', trainable=trainable)
                    logstd = tf.get_variable(name="logstd", shape=[1, A_DIM], initializer=tf.zeros_initializer())
                    pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
                    pdtype = make_pdtype(Action_Space)
                    pd = pdtype.pdfromflat(pdparam)
                else:
                    print('discrate action',A_DIM, S_DIM)
                    pi = tf.layers.dense(h4, A_DIM, kernel_initializer=w_init, name = 'fc_pi', trainable=trainable)
                    pdtype = make_pdtype(Action_Space)
                    pd = pdtype.pdfromflat(pi)   

                v1_h1 = tf.layers.dense(state_t, 64, tf.nn.relu, kernel_initializer=w_init, name = 'v1_h1', trainable=trainable)
                v1_h2 = tf.layers.dense(v1_h1, 64, tf.nn.relu, kernel_initializer=w_init, name = 'v1_h2', trainable=trainable)
                v1 = tf.layers.dense(v1_h2, 1, kernel_initializer=w_init, name = 'v1', trainable=trainable)

                v2 = tf.layers.dense(h3, 1, kernel_initializer=w_init, name = 'fc_v', trainable=trainable)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pd, v1, v2, params


    def choose_action(self, s, det=False):
        s = s[np.newaxis, :]

        if det:
            a = self.sess.run(self.sample_op_det, {self.tfs: s, self.tf_is_train: False})[0]
        else:
            a = self.sess.run(self.sample_op, {self.tfs: s, self.tf_is_train: False})[0]

        if isinstance(Action_Space, gym.spaces.Box):
            a = np.clip(a, -1, 1)
        return a

    def get_action_prob(self, s):
        s = s[np.newaxis, :]
        prob = 0
        prob = self.sess.run(self.pi_prob, {self.tfs: s, self.tf_is_train: False})[0]

        plt.clf()
        plt.scatter(range(A_DIM+1), np.append(prob, 1.0).flatten())
        plt.pause(0.01)

        return prob

    def get_v(self, s):
        s_ndim = s.ndim
        if s.ndim < 2: s = s[np.newaxis, :]
        v1, v2 = self.sess.run([self.v1, self.v2], {self.tfs: s, self.tf_is_train: False})

        if s_ndim < 2:
            return v1[0][0], v2[0][0]
        else:
            return v1, v2

    def check_overall_loss(self, s, a, r1, r2, adv1, adv2):
        feed_dict = {
            self.tfs: s, 
            self.tfa: a, 
            self.tfdc_r1: r1, 
            self.tfdc_r2: r2, 
            self.tfadv_1: adv1, 
            self.tfadv_2: adv2, 
            self.tf_is_train: False
        }

        tloss, aloss, vloss, entropy, grad_norm = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy, self.grad_norm], feed_dict = feed_dict)
        return tloss, aloss, vloss, entropy, grad_norm

    def load_guid_tra(self, s, a, rl, adv_l, info):
        g_s, g_a, g_rl, g_info = [], [], [], []

        size = 2560 #int(len(a)/4) #int(BATCH_SIZE*0.3)
        # selected_index = np.random.choice(len(G_lift_a), size, replace=False)
        # selected_index = np.random.randint(len(G_lift_a) - size - 1)
        simi_threshold = 2
        print(len(a))
        for index_s in range(0, len(s), 3):
            sub_s = s[index_s]
            found_num = 0
            for _ in range(800):
                index_lift = np.random.randint(len(G_lift_s))
                index_straight = np.random.randint(len(G_straight_s))
                index_open = np.random.randint(len(G_open_s))
                index_avoid = np.random.randint(len(G_avoid_s))

                diff = np.sum(abs(G_lift_s[index_lift] - sub_s))
                if diff < simi_threshold:
                    g_s.append(G_lift_s[index_lift])
                    g_a.append(G_lift_a[index_lift])
                    g_rl.append(G_lift_ret[index_lift])
                    # g_rl.append(0)
                    g_info.append(11)
                    found_num += 1       

                diff = np.sum(abs(G_straight_s[index_straight] - sub_s))
                if diff < simi_threshold:
                    g_s.append(G_straight_s[index_straight])
                    g_a.append(G_straight_a[index_straight])
                    g_rl.append(G_straight_ret[index_straight])
                    # g_rl.append(0)
                    g_info.append(12)
                    found_num += 1   

                diff = np.sum(abs(G_open_s[index_open] - sub_s))
                if diff < simi_threshold:
                    g_s.append(G_open_s[index_open])
                    g_a.append(G_open_a[index_open])
                    g_rl.append(G_open_ret[index_open])
                    # g_rl.append(0)
                    g_info.append(13)
                    found_num += 1  

                diff = np.sum(abs(G_avoid_s[index_avoid] - sub_s))
                if diff < simi_threshold:
                    g_s.append(G_avoid_s[index_avoid])
                    g_a.append(G_avoid_a[index_avoid])
                    g_rl.append(G_avoid_ret[index_avoid])
                    # g_rl.append(0)
                    g_info.append(14)
                    found_num += 1  

                if found_num > 1:
                    break  

        # for index in range(selected_index, selected_index+size):    
        #     diff = np.sum(abs(G_lift_s[index] - s[0]))
        #     # if diff < 10:
        #     #     img = s[0][:-4].reshape((30,30))
        #     #     img2 = G_lift_s[index][:-4].reshape((30,30))
        #     #     plt.clf()
        #     #     plt.imshow(img)
        #     #     plt.pause(0.5)
        #     #     plt.imshow(img2)
        #     #     plt.pause(0.5)   
        #     #     print(G_lift_a[index], diff)

        # # for index in selected_index:
        #     g_s.append(G_lift_s[index])
        #     g_a.append(G_lift_a[index])
        #     g_rl.append(G_lift_ret[index])
        #     # g_rl.append(0)
        #     g_info.append(11)

        # selected_index = np.random.choice(len(G_avoid_a), size, replace=False)
        # for index in selected_index:
        #     g_s.append(G_avoid_s[index])
        #     g_a.append(G_avoid_a[index])
        #     g_rl.append(G_avoid_ret[index])
        #     # g_rl.append(0)
        #     g_info.append(12)

        # selected_index = np.random.choice(len(G_open_a), size, replace=False)
        # for index in selected_index:
        #     g_s.append(G_open_s[index])
        #     g_a.append(G_open_a[index])
        #     g_rl.append(G_open_ret[index])
        #     # g_rl.append(0)
        #     g_info.append(13)
                    
        # selected_index = np.random.choice(len(G_straight_a), size, replace=False)
        # selected_index = np.random.randint(len(G_straight_a) - size - 1)
        # for index in range(selected_index, selected_index+size):
        #     g_s.append(G_straight_s[index])
        #     g_a.append(G_straight_a[index])
        #     g_rl.append(G_straight_ret[index])
        #     # g_rl.append(0)
        #     g_info.append(14)

        print('g_tra num', len(g_s))
        g_vpred_l = self.sess.run(self.vl, feed_dict = {self.tfs: g_s})
        g_adv = g_rl-g_vpred_l

        s = np.concatenate((s, g_s), axis = 0)
        a = np.concatenate((a, g_a), axis = 0)
        rl = np.concatenate((rl, g_rl), axis = 0)
        adv_l = np.concatenate((adv_l, g_adv), axis = 0)
        info = np.concatenate((info, g_info), axis = 0)

        return s, a, rl, adv_l, info

    def update(self):
        global GLOBAL_UPDATE_COUNTER, G_ITERATION, GLOBAL_EP
        update_count = 1

        while not COORD.should_stop():
            UPDATE_EVENT.wait()                     # wait until get batch of data

            self.sess.run(self.update_oldpi_op)     # copy pi to old pi
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
            data = np.vstack(data)
            # s, a, r, adv = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, S_DIM + A_DIM: S_DIM + A_DIM + 1], data[:, -1:]
            shift = S_DIM + A_DIM
            if isinstance(Action_Space, gym.spaces.Box):
                s, a, r1, r2, adv_l, rew, info = data[:, :S_DIM], data[:, S_DIM: shift], data[:, shift: shift+1], data[:, shift+1: shift+2], data[:, shift+2: shift+3], data[:, shift+3: shift+4], data[:, shift+4: shift+5]
            else:
                s, a, r1, r2, adv_l, rew, info = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1], data[:, S_DIM + 1: S_DIM + 2], data[:, S_DIM + 2: S_DIM + 3], data[:, S_DIM + 3: S_DIM + 4], data[:, S_DIM + 4: S_DIM + 5], data[:, S_DIM + 5: S_DIM + 6]
                a = (a.flatten())
            
            r1 = (r1.flatten())
            r2 = (r2.flatten())
            adv_l = (adv_l.flatten())
            info = info.flatten()
            rew = rew.flatten()
            ret_mean = r1.mean()
            rew_mean = rew.mean()
            # s, a, rl, adv_l, info = self.load_guid_tra(s, a, rl, adv_l, info)

            adv = adv_l

            if adv.std() != 0:
                adv = (adv - adv.mean())/adv.std()

            print(G_ITERATION, '  --------------- update! batch size:', GLOBAL_EP, '-----------------', len(r1))

            for iteration in range(UPDATE_STEP):
                # construct reward predict data                
                # s_, a_, r1_, r2_, adv_ = self.shuffel_data(s, a, r1, r2, adv)   
                s_, a_, r1_, r2_, adv_ = s, a, r1, r2, adv
                count = 0
                for start in range(0, len(r1), MIN_BATCH_SIZE):
                    end = start + MIN_BATCH_SIZE -1
                    if end >= len(r1) and count != 0:
                        break
                    if  end >= len(r1) and count == 0:
                        end = len(r1)-1
                    count += 1

                    sub_s = s_[start:end]
                    sub_a = a_[start:end]
                    sub_r1 = r1_[start:end]
                    sub_r2 = r2_[start:end]
                    sub_adv = np.asarray(adv_[start:end])
                    sub_info = info[start:end]
                    # sub_s, sub_a, sub_rs, sub_rl, sub_adv, sub_adv, sub_info = self.load_guid_tra(sub_s, sub_a, sub_rs, sub_rl, sub_adv, sub_adv, sub_info)
                    # sub_s, sub_a, sub_r, sub_adv = self.balance_minibatch(s, a, r, adv, sub_s, sub_a, sub_r, sub_adv, possitive_index, negative_index)
                    # sub_s, sub_a, sub_rs, sub_rl, sub_adv = self.get_minibatch(s, a, rs, rl, adv, possitive_index, negative_index)
                    # print(sub_adv)
                    feed_dict = {
                        self.tfs: s, 
                        self.tfa: a, 
                        self.tfdc_r1: r1, 
                        self.tfdc_r2: r2, 
                        self.tfadv: adv, 
                        self.tf_is_train: False
                    }
                    self.sess.run([self.train_op, self.train_op1], feed_dict = feed_dict)
                    # self.sess.run([self.atrain_op, self.ctrain_op], feed_dict = feed_dict)

                # tloss, aloss, vloss, entropy, grad_norm = self.check_overall_loss(s, a, rs, rl, adv)
                # print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, entropy), grad_norm)
                print(iteration)

            feed_dict = {
                self.tfs: s, 
                self.tfa: a, 
                self.tfdc_r1: r1, 
                self.tfdc_r2: r2, 
                self.tfadv: adv, 
                self.tf_is_train: False
            }
            # onehota_prob, oldhota_prob, ratio = self.sess.run([self.a_prob, self.olda_prob, self.ratio], feed_dict = feed_dict)
            # for i in range(len(r)):
            #     print(onehota_prob[i], oldhota_prob[i], ratio[i])

            ratio, v1, v2 = self.sess.run([self.ratio, self.v1, self.v2], feed_dict = feed_dict)
            # ratio = ratio.flatten()
            for i in range(len(r1)): #range(25):
                print("rew %8.4f, r1 %8.4f, r2 %8.4f|, v1 %8.4f  v2 %8.4f|, adv %8.4f|, ratio %8.4f|"%(rew[i], r1[i], r2[i], v1[i], v2[i], adv[i], ratio[i]), info[i])
                # print(a[i])

            tloss, aloss, vloss, entropy, grad_norm = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy, self.grad_norm], feed_dict = feed_dict)
            print('-------------------------------------------------------------------------------')
            print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, entropy), grad_norm)

            self.write_summary('Loss/entropy', entropy)  
            self.write_summary('Loss/a loss', aloss) 
            self.write_summary('Loss/v loss', vloss) 
            self.write_summary('Loss/grad norm', grad_norm) 
            self.write_summary('Perf/mean_ret', ret_mean)  
            self.write_summary('Perf/mean_rew', rew_mean)  

            model_index = int(G_ITERATION / 10)
            self.saver.save(self.sess, './model/rl/model_' + str(model_index) + '.cptk') 

            UPDATE_EVENT.clear()        # updating finished
            GLOBAL_UPDATE_COUNTER = 0   # reset counter
            G_ITERATION += 1
            # GLOBAL_EP = 0
            ROLLING_EVENT.set()         # set roll-out available
            
            update_count += 1

            # # if entropy < 1:         # stop training
            # if GLOBAL_STEP > 620000:
            #     COORD.request_stop()


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = centauro_env.Simu_env(20000 + wid)
        self.env = VecNormalize(self.env)
        self.ppo = GLOBAL_PPO

    def get_transfered_pose(self, current_pose, next_pose):
        x_diff = next_pose[0] - current_pose[0]
        y_diff = next_pose[1] - current_pose[1]
        theta = -current_pose[2]

        x = math.cos(theta)*x_diff - math.sin(theta)*y_diff + centauro_env.map_shift
        y = math.sin(theta)*x_diff + math.cos(theta)*y_diff + centauro_env.map_shift

        col = centauro_env.map_pixel - y/centauro_env.grid_size
        row = centauro_env.map_pixel - x/centauro_env.grid_size

        return int(row), int(col)

    def get_pose_mask(self, pose_list):
        pose_mask = np.zeros((centauro_env.map_pixel, centauro_env.map_pixel), np.float32)

        radius = int(0.4/centauro_env.grid_size)
        for i in range(len(pose_list)):
            row, col = self.get_transfered_pose(pose_list[0], pose_list[i])
            if col > 0 and col < centauro_env.map_pixel and row > 0 and row < centauro_env.map_pixel:
                cv2.circle(pose_mask, (col,row), radius, 1, -1)
        return pose_mask

    def modifly_state(self, pose_mask, pose_list, s):
        success = False
        img = s[:-4].reshape(centauro_env.map_pixel, centauro_env.map_pixel)
        obs_num = np.random.randint(4)
        obs_index = np.random.choice(3, obs_num)
        obs_x = np.random.choice(8, obs_num, replace=False) * 0.25 - 1
        obs_y = np.random.choice(8, obs_num, replace=False) * 0.25 - 1

        heights_list = [0.2, 0.5, 1]
        radius_list = [0.27, 0.1, 0.18]

        for i in range(len(obs_index)):
            index = obs_index[i]
            ox = obs_x[i]
            oy = obs_y[i]
            h = heights_list[index]
            r = int(radius_list[index]/centauro_env.grid_size )
            row, col = self.get_transfered_pose(pose_list[0], [ox, oy])
            # check pose mask
            if col > 0 and col < centauro_env.map_pixel and row > 0 and row < centauro_env.map_pixel:
                mask_v = pose_mask[row, col]
                if mask_v == 0:
                    cv2.circle(img, (col, row), r, h, -1)
                    print(img[row, col])
                    success = True

        new_img = img.flatten()
        s_new = s
        s_new[:-4] = new_img

        if success:
            plt.clf()
            plt.imshow(img)
            plt.pause(0.5)
            plt.imshow(pose_mask)
            plt.pause(0.5)            

        return s_new, success

    def get_aurgm_batch(self, iteration, s, a, ret_l, info):
        aurg_s, aurg_a, aurg_ret_l, aurg_adv_l, aurg_info = [], [], [], [], []

        for i in range(len(s)-2):
            pose_mask = self.get_pose_mask(self.env.robot_pose_list[i:])
            for _ in range(iteration):
                s_modf, success = self.modifly_state(pose_mask, self.env.robot_pose_list[i:], s[i]*1)
                if success:
                    aurg_s.append(s_modf)
                    aurg_a.append(a[i])
                    aurg_ret_l.append(ret_l[i])
                    aurg_info.append(info[i])

        if aurg_s != []:
            aurg_vpred_s, aurg_vpred_l = self.ppo.get_v(np.asarray(aurg_s))
            aurg_adv_l = aurg_ret_l - aurg_vpred_l

        # print(len(aurg_s[0]), len(aurg_a), len(aurg_ret_l), len(aurg_adv_l), len(aurg_adv_s), len(aurg_info))
        return aurg_s, aurg_a, aurg_ret_l, aurg_adv_l, aurg_info

    def compute_adv_return(self, buffer_reward, buffer_vpred, buffer_info):
        if buffer_info[-1] == 'goal':
            nonterminal = 0
        else:
            nonterminal = 1              

        v_s_ = buffer_vpred[-1]*nonterminal
        buffer_return = []                           # compute discounted reward
        for r in buffer_reward[::-1]:
            v_s_ = r + GAMMA * v_s_
            buffer_return.append(v_s_)
        buffer_return.reverse()

        buffer_adv = buffer_return - buffer_vpred[:-1]

        # for index in range(len(buffer_reward)):
        #     print(buffer_reward[index], buffer_return[index], buffer_vpred[index], buffer_adv[index])

        # print('-----------------')
        return buffer_adv, buffer_return

    def process_and_send(self, buffer_s, buffer_a, buffer_r, buffer_info, s_):
        buffer_vpred_1, buffer_vpred_2 = self.ppo.sess.run([self.ppo.v1, self.ppo.v2], feed_dict = {self.ppo.tfs: buffer_s, self.ppo.tf_is_train: False})
        buffer_vpred_1, buffer_vpred_2 = buffer_vpred_1.flatten(), buffer_vpred_2.flatten()
        
        vpred_1_, vpred_2_ = self.ppo.get_v(s_)
        buffer_vpred_1, buffer_vpred_2 = np.append(buffer_vpred_1, vpred_1_), np.append(buffer_vpred_2, vpred_2_)
        buffer_adv, buffer_return = self.compute_adv_return(buffer_r, buffer_vpred_2, buffer_info)
        buffer_adv_1, buffer_return_1 = self.compute_adv_return(buffer_r, buffer_vpred_1, buffer_info)

        if self.wid == 0:
            print('----')
            for i in range(len(buffer_a)):
                print("rl: %7.4f| v1: %7.4f| v2: %7.4f| adv_1: %7.4f| adv_2: %7.4f| ret_1: %7.4f, ret_2: %7.4f" %(buffer_r[i], buffer_vpred_1[i], buffer_vpred_2[i], buffer_adv_1[i], buffer_adv[i], buffer_return_1[i], buffer_return[i]), buffer_info[i])
            
        info_num = []
        for info in buffer_info:
            if info == 'goal':
                info_num.append(1)
            elif info == 'crash':
                info_num.append(-2)
            elif info == 'crash_a':
                info_num.append(-1)
            else:
                info_num.append(0)
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return buffer_return_1, buffer_return, buffer_adv_1, buffer_adv, info_num

    def evaluate_model(self, ep_num):
        goal_num = 0
        goal_num_nocrash = 0
        failed_num = 0

        sum_reward = 0
        sum_len = 0
        step_count = 0

        max_step = int(EP_LEN)
        print('evaluating model', ep_num) 
        for test_num in range(ep_num):
            s = self.env.reset(0, 0, 1)
            vpred_s, vpred_l = self.ppo.get_v(s)
            print(vpred_l)

            crashed = False
            tra_len = 0
            # if test_num > 0 and test_num % 100 == 0:
            #     print(test_num)
            #     print ('success rate', goal_num/ep_num, 'failed rate', failed_num/ep_num, 'average reward', sum_reward/step_count)
            for step in range(max_step):
                a = self.ppo.choose_action(s, False)
                # print('action generated:', a)
                s_, r_short, r_long, done, info = self.env.step(a)

                tra_len += 1

                step_count += 1
                sum_reward += r_long
                s = s_
                if info == 'crash':
                    crashed = True
                #     self.env.save_start_end_ep()
                #     saved_ep += 1
                #     break                    
                if done or step == max_step-1:
                    # print(test_num, info)
                    sum_len += tra_len
                    if info == 'goal':
                        goal_num += 1   
                        if crashed == False:
                            goal_num_nocrash += 1
                    else:
                        failed_num += 1  
                    break

        self.ppo.write_summary('Perf/success_rate', goal_num/ep_num)  
        self.ppo.write_summary('Perf/success_rate_nocrash', goal_num_nocrash/ep_num)  
        self.ppo.write_summary('Perf/avg_reward', sum_reward/step_count)  
        self.ppo.write_summary('Perf/avg_len', sum_len/ep_num)  
        print ('success rate', goal_num/ep_num, 'failed rate', failed_num/ep_num, 'average reward', sum_reward/step_count)

    def test_model(self, ep_num):
        goal_num = 0
        saved_ep = 0
        count = 0
        max_step = int(EP_LEN/3 * 2)
        print('searching for failed ep', ep_num) 
        for test_num in range(100):
            count += 1
            s = self.env.reset(0, 0, 1)

            for step in range(max_step):
                a = self.ppo.choose_action(s, False)
                s_, r_short, r_long, done, info = self.env.step(a)

                s = s_
                # if info == 'crash':
                #     self.env.save_start_end_ep()
                #     saved_ep += 1
                #     break                    
                if done or step == max_step-1:
                    # print(test_num, info)
                    if info == 'goal':
                        goal_num += 1     
                    else:
                        self.env.save_start_end_ep()
                        saved_ep += 1
                    break
            if saved_ep > ep_num:
                break

        if saved_ep < ep_num:
            for test_num in range(100):
                count += 1
                s = self.env.reset(0, 0, 1)
                for step in range(max_step):
                    a = self.ppo.choose_action(s, False)
                    s_, r_short, r_long, done, info = self.env.step(a)

                    s = s_
                    if info == 'crash':
                        self.env.save_start_end_ep()
                        saved_ep += 1
                        break                    
                    if done or step == max_step-1:
                        # print(test_num, info)
                        if info != 'goal':
                            self.env.save_start_end_ep()
                            saved_ep += 1
                        break
                if saved_ep > ep_num:
                    break

        if saved_ep < ep_num:
            for test_num in range(100):
                count += 1
                s = self.env.reset(0, 0, 1)
                for step in range(max_step):
                    a = self.ppo.choose_action(s, True)
                    s_, r_short, r_long, done, info = self.env.step(a)

                    s = s_
                    if info == 'crash':
                        self.env.save_start_end_ep()
                        saved_ep += 1
                        break                    
                    if done or step == max_step-1:
                        # print(test_num, info)
                        if info != 'goal':
                            self.env.save_start_end_ep()
                            saved_ep += 1
                        break
                if saved_ep > ep_num:
                    break
        print (goal_num/count)


    def work(self):
        global GLOBAL_EP, GLOBAL_STEP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER \
            , Goal_states, Goal_count, Goal_return, Goal_buffer_full \
            , Crash_states, Crash_count, Crash_return, Crash_buffer_full \
            , History_states, History_count, History_adv, History_return, History_buffer_full

        # self.env.save_ep()
        # if self.wid != 0:
        #     self.test_model(5)
        # if self.wid == 0:
        #     while not COORD.should_stop():
        #         self.evaluate_model(50)
        # for _ in range(2):
        #     s = self.env.reset( 0, 0, 1)
        #     self.env.save_ep()

        update_counter = 0
        ep_count = 0
        if self.wid == N_WORKER - 1:
            time.sleep(1)

        while not COORD.should_stop():
            buffer_s, buffer_a, buffer_rl, buffer_info = [], [], [], []
            info = 'unfinish'

            t = 0
            ep_length = EP_LEN #len(test_actions)
            # for a in test_actions:
            GLOBAL_EP += 1
            ep_count += 1
            saved_ep = False
            
            s = self.env.reset( 0, 1, 1)
            a = self.ppo.choose_action(s, False)
            while(1):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_rl, buffer_info = [], [], [], []
                    # if update_counter%1 == 0:
                        # self.env.clear_history()
                    # self.test_model(5)
                    update_counter += 1
                    break

                # if t%3 == 0:
                a = self.ppo.choose_action(s, False)
                # for i in range(len(a)-4):
                #     a[i] = 0
                
                # state_v = self.ppo.sess.run(self.ppo.state_v, {self.ppo.tfs: s[np.newaxis, :]})

                s_, r_long, done, info = self.env.step(a)
                # if self.wid == 0:
                #     print(a[-2:])
                # print('action generated:', a)
                # self.ppo.get_action_prob(s)
                vpred_1, vpred_2 = self.ppo.get_v(s)

                GLOBAL_STEP += 1

                # img = self.ppo.sess.run(self.ppo.img, feed_dict = {self.ppo.tfs:[s]})[0]
                # plt.clf()
                # plt.imshow(img)
                # plt.pause(0.01)

                buffer_s.append(s*1)
                buffer_a.append(a)
                buffer_rl.append(r_long)
                buffer_info.append(info)
                s_demo = s_*1
                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers

                s = s_
                t += 1       
                            

                if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE or t >= ep_length-1 or done:
                    if done and info == 'goal':
                        buffer_a[-1] = np.zeros(len(a))
                    
                    for i in range(len(buffer_rl)):
                        buffer_rl[i] = buffer_rl[i] * 0.5

                    buffer_return_1, buffer_return_2, buffer_adv_1, buffer_adv_2, info_num = self.process_and_send(buffer_s, buffer_a, buffer_rl, buffer_info, s_)
                    adv = buffer_adv_1 #(buffer_adv_1 + buffer_adv_2)/2
                    bs, ba, bret_1, bret_2, badv, br, binfo = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_return_1)[:, np.newaxis], np.array(buffer_return_2)[:, np.newaxis], np.array(adv)[:, np.newaxis], np.vstack(buffer_rl), np.vstack(info_num)     
                    QUEUE.put(np.hstack((bs, ba, bret_1, bret_2, badv, br, binfo)))          # put data in the queue

                    # if info == 'goal':
                    #     aug_s, aug_a, aug_return_s, aug_return_l, aug_adv_l, info_num = self.get_aurgm_batch(5, buffer_s, buffer_a, buffer_return_l, info_num)
                    #     if (aug_s != []):
                    #         bs, ba, bret_l, badv_l, binfo = np.vstack(aug_s), np.vstack(aug_a), np.array(aug_return_l)[:, np.newaxis], np.array(aug_adv_l)[:, np.newaxis], np.vstack(info_num)     
                    #         QUEUE.put(np.hstack((bs, ba, bret_l, badv_l, binfo)))          # put data in the queue
                    #         GLOBAL_UPDATE_COUNTER += len(aug_a)
                    #         print('generage new batch:', len(aug_a))

                    buffer_s, buffer_a, buffer_rl, buffer_vpred_l, buffer_info = [], [], [], [], []
                    if self.wid == 0:
                        print("step collected:: %7.4f| %7.4f| %7.4f" %(GLOBAL_UPDATE_COUNTER/BATCH_SIZE, GLOBAL_UPDATE_COUNTER, BATCH_SIZE))
                    if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if G_ITERATION >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    if done or t == ep_length-1:
                        # if done != 'goal':
                        #     self.env.save_start_end_ep()                            
                        break 


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP, GLOBAL_STEP = 0, 1, 540700

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