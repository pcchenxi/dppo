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

EP_MAX = 500000
EP_LEN = 50
N_WORKER = 4               # parallel workers
GAMMA = 0.95                # reward discount factor
LAM = 1
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0001               # learning rate for critic
LR = 0.0005

EP_BATCH_SIZE = 5
UPDATE_L_STEP = 30
BATCH_SIZE = 5120
MIN_BATCH_SIZE = 128       # minimum batch size for updating PPO

UPDATE_STEP = 5            # loop update operation n-steps
EPSILON = 0.2              # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = centauro_env.observation_space, centauro_env.action_space 
Action_Space = centauro_env.action_type

G_ITERATION = 0

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        self.tfdc_rs = tf.placeholder(tf.float32, [None], 'discounted_rs')
        self.tfdc_rl = tf.placeholder(tf.float32, [None], 'discounted_rl')

        if isinstance(Action_Space, gym.spaces.Box):
            self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        else:
            self.tfa = tf.placeholder(tf.int32, [None], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None], 'advantage')

        self.tf_is_train = tf.placeholder(tf.bool, None)

        print('init place holder')
        # actor
        pd, self.vs, self.vl, net_params = self._build_anet('net', self.tfs, trainable=True)
        oldpd, oldvs, oldvl, oldnet_params = self._build_anet('oldnet', self.tfs, trainable=False)
        print('inti net')

        self.sample_op = pd.sample() #tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.sample_op_det = pd.mode()

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(net_params, oldnet_params)]

        neglogpac = pd.neglogp(self.tfa)
        OLDNEGLOGPAC = oldpd.neglogp(self.tfa)
        self.entropy = tf.reduce_mean(pd.entropy())

        vpredclipped = oldvl + tf.clip_by_value(self.vl - oldvl, - EPSILON, EPSILON)
        vf_losses1 = tf.square(self.vl - self.tfdc_rl)
        vf_losses2 = tf.square(vpredclipped - self.tfdc_rl)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -self.tfadv * ratio
        pg_losses2 = -self.tfadv * tf.clip_by_value(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), EPSILON)))
        loss = pg_loss - self.entropy * 0.001 + vf_loss

        params = tf.trainable_variables(scope='net')
        print(params)

        grads = tf.gradients(loss, params)
        max_grad_norm = 0.5
        if max_grad_norm is not None:
            grads_clipped, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_params = list(zip(grads_clipped, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        self.train_op = trainer.apply_gradients(grads_and_params)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('./log', self.sess.graph)   

        self.pi_prob = tf.nn.softmax(pd.logits)
        self.aloss = pg_loss 
        self.closs = tf.reduce_mean(vf_losses1) 
        self.total_loss = loss
        self.ratio = ratio
        self.grad_norm = _grad_norm

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
        # summary.value.add(tag=summary_name, simple_value=float(value))
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
        # w_init = tf.contrib.layers.xavier_initializer()
        # w_init = tf.zeros_initializer()
        # w_init = tf.random_normal_initializer(0., 0.01)
        # w_init_big = tf.random_normal_initializer(0., 0.01)
        with tf.variable_scope(name):
            feature = self._build_feature_net('feature', input_state, trainable=trainable)
            with tf.variable_scope('actor_critic'):
                h4 = tf.layers.dense(feature, 512, tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(2), name = 'fc1', trainable=trainable)

                if isinstance(Action_Space, gym.spaces.Box):
                    print('continue action',A_DIM, S_DIM)
                    pi = tf.layers.dense(h4, A_DIM, kernel_initializer=tf.orthogonal_initializer(0.01), name = 'fc_pi', trainable=trainable)
                    logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())
                    pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
                    pdtype = make_pdtype(Action_Space)
                    pd = pdtype.pdfromflat(pdparam)
                else:
                    print('discrate action',A_DIM, S_DIM)
                    pi = tf.layers.dense(h4, A_DIM, kernel_initializer=tf.orthogonal_initializer(0.01), name = 'fc_pi', trainable=trainable)
                    pdtype = make_pdtype(Action_Space)
                    pd = pdtype.pdfromflat(pi)   

                vl = tf.layers.dense(h4, A_DIM, name = 'fc_v', trainable=trainable)[:,0]
                vs = vl #fc(h4, 'v', 1, act=lambda x:x)[:,0]

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
        prob = 0
        prob = self.sess.run(self.pi_prob, {self.tfs: s, self.tf_is_train: False})[0]

        plt.clf()
        plt.scatter(range(A_DIM+1), np.append(prob, 1.0).flatten())
        plt.pause(0.01)

        return prob

    def get_v(self, s):
        s_ndim = s.ndim
        if s.ndim < 2: s = s[np.newaxis, :]
        vs, vl = self.sess.run([self.vs, self.vl], {self.tfs: s, self.tf_is_train: False})

        if s_ndim < 2:
            return vs[0], vl[0]
        else:
            return vs, vl

    def check_overall_loss(self, s, a, rs, rl, adv):
        feed_dict = {
            self.tfs: s, 
            self.tfa: a, 
            self.tfdc_rs: rs, 
            self.tfdc_rl: rl, 
            self.tfadv: adv, 
            self.tf_is_train: False
        }

        tloss, aloss, vloss, entropy, grad_norm = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy, self.grad_norm], feed_dict = feed_dict)
        return tloss, aloss, vloss, entropy, grad_norm

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
                s, a, rs, rl, adv_s, adv_l, info = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, S_DIM + A_DIM: S_DIM + A_DIM+1], data[:, S_DIM + A_DIM+1: S_DIM + A_DIM+2], data[:, S_DIM + A_DIM+2: S_DIM + A_DIM+3], data[:, S_DIM + A_DIM+3: S_DIM + A_DIM+4], data[:, -1:]
            else:
                s, a, rs, rl, adv_s, adv_l, info = data[:, :S_DIM], data[:, S_DIM: S_DIM + 1], data[:, S_DIM + 1: S_DIM + 2], data[:, S_DIM + 2: S_DIM + 3], data[:, S_DIM + 3: S_DIM + 4], data[:, S_DIM + 4: S_DIM + 5], data[:, -1:]
                a = (a.flatten())
            
            rs = (rs.flatten())
            rl = (rl.flatten())
            adv_s = (adv_s.flatten())
            adv_l = (adv_l.flatten())

            adv = adv_l*1

            adv_s_scale = adv_s * abs(adv.min())

            for i in range(len(adv)):
                if adv_s[i] < -0.5:
                    adv[i] = (adv_s_scale[i] + adv[i])/2

            if adv.std() != 0:
                adv = (adv - adv.mean())/adv.std()

            print(G_ITERATION, '  --------------- update! batch size:', GLOBAL_EP, '-----------------', len(rs))

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

                tloss, aloss, vloss, entropy, grad_norm = self.check_overall_loss(s, a, rs, rl, adv)
                print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, entropy), grad_norm)
                print(iteration)

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

            ratio, vs, vl = self.sess.run([self.ratio, self.vs, self.vl], feed_dict = feed_dict)
            ratio = ratio.flatten()
            for i in range(len(rs)): #range(25):
                print("%8.4f, %8.4f|, %8.4f, %8.4f|, %8.4f, %8.4f, %8.4f|, %8.4f|"%(rs[i], rl[i], vs[i], vl[i], adv_s[i], adv_l[i], adv[i], ratio[i]), a[i], info[i])
                # print("%8.4f, %8.4f, %8.4f, %8.4f, %8.4f, %8.4f, %6.0i, %8.4f"%(reward[i], r[i], vpred[i], vpred_new[i], adv[i], ratio[i], a[i], a_prob[i][act]), a_prob[i])

            print(rs.mean(), rl.mean())

            tloss, aloss, vloss, entropy, grad_norm = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy, self.grad_norm], feed_dict = feed_dict)
            print('-------------------------------------------------------------------------------')
            print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, entropy), grad_norm)

            self.write_summary('Loss/entropy', entropy)  
            self.write_summary('Loss/a loss', aloss) 
            self.write_summary('Loss/v loss', vloss) 
            self.write_summary('Loss/grad norm', grad_norm) 
            self.write_summary('Loss/avg_rew', rl.mean()) 
            # self.write_summary('Perf/mean_reward', np.mean(reward))  

            self.saver.save(self.sess, './model/rl/model.cptk') 

            UPDATE_EVENT.clear()        # updating finished
            GLOBAL_UPDATE_COUNTER = 0   # reset counter
            G_ITERATION += 1
            # GLOBAL_EP = 0
            ROLLING_EVENT.set()         # set roll-out available
            
            update_count += 1
            if update_count % UPDATE_L_STEP == 1:
                update_count = 1
                s_all, a_all, rs_all, rl_all, adv_s_all, adv_l_all = [], [], [], [], [], []
                print('reset')

            if entropy < 1:         # stop training
                COORD.request_stop()


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
                mask_v = pose_mask[col, row]
                if mask_v == 0:
                    cv2.circle(img, (col, row), r, h, -1)
                    success = True

        new_img = img.flatten()
        s_new = s
        s_new[:-4] = new_img

        # if success:
        #     plt.clf()
        #     plt.imshow(img)
        #     plt.pause(0.0)

        return s_new, success

    def get_aurgm_batch(self, iteration, s, a, ret_s, ret_l, info):
        aurg_s, aurg_a, aurg_ret_s, aurg_ret_l, aurg_adv_s, aurg_adv_l, aurg_info = [], [], [], [], [], [], []

        for i in range(len(s)-2):
            pose_mask = self.get_pose_mask(self.env.robot_pose_list[i:])
            for _ in range(iteration):
                s_modf, success = self.modifly_state(pose_mask, self.env.robot_pose_list[i:], s[i]*1)
                if success:
                    aurg_s.append(s_modf)
                    aurg_a.append(a[i])
                    aurg_ret_s.append(ret_s[i])
                    aurg_ret_l.append(ret_l[i])
                    aurg_info.append(info[i])

        if aurg_s != []:
            aurg_vpred_s, aurg_vpred_l = self.ppo.get_v(np.asarray(aurg_s))
            aurg_adv_s = aurg_ret_s - aurg_vpred_s
            aurg_adv_l = aurg_ret_l - aurg_vpred_l

        # print(len(aurg_s[0]), len(aurg_a), len(aurg_ret_l), len(aurg_ret_s), len(aurg_adv_l), len(aurg_adv_s), len(aurg_info))
        return aurg_s, aurg_a, aurg_ret_s, aurg_ret_l, aurg_adv_s, aurg_adv_l, aurg_info

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
            if mode != 'long':
                if buffer_info[index] == 'crash' or buffer_info[index] == 'crash_a' or buffer_info[index] == 'out':
                    nonterminal = 0
                else:
                    nonterminal = 1                
            delta = buffer_reward[index] + gamma * buffer_vpred[index+1] * nonterminal - buffer_vpred[index]
            lastgaelam = delta + gamma * LAM * nonterminal * lastgaelam
            buffer_adv[index] = lastgaelam

        #     delta = buffer_reward[index] + gamma * buffer_vpred[index+1] * nonterminal
        #     lastgaelam = delta + gamma * nonterminal * lastgaelam
        #     buffer_return[index] = lastgaelam
        # buffer_adv = buffer_return - buffer_vpred[:-1]

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
        vpred_s_ = 0
        # vpred_l_ = 0
        buffer_vpred_s = np.append(buffer_vpred_s, vpred_s_)
        buffer_vpred_l = np.append(buffer_vpred_l, vpred_l_)
        buffer_adv_s, buffer_return_s = self.compute_adv_return(buffer_rs, buffer_vpred_s, buffer_info, 'short')
        buffer_adv_l, buffer_return_l = self.compute_adv_return(buffer_rl, buffer_vpred_l, buffer_info, 'long')

        # for i in range(len(buffer_return_s)):
        #     buffer_return_l[i] = buffer_return_l[i] + (buffer_return_s[i])
            # print(before, buffer_rl[i], buffer_return_s[i])

        buffer_adv_s = buffer_return_s*1

        # if buffer_adv_l.max() > 0:
        #     buffer_adv_l = buffer_adv_s * buffer_adv_l.max() + buffer_adv_l
        # else:
        #     buffer_adv_l = buffer_adv_s * abs(buffer_adv_l.min()) + buffer_adv_l
        
        # for i in range(len(buffer_adv_s)):
        #     if buffer_adv_s[i] > -0: # and buffer_adv_s[i] < 0.5:
        #         buffer_adv_s[i] = 0

        # buffer_adv_l = buffer_return_l - buffer_vpred_l[:-1]
        buffer_adv = buffer_adv_s + buffer_adv_l

        if self.wid == 0:
            print('----')
            for i in range(len(buffer_adv)):
                print("rs: %7.4f| rl: %7.4f| vs: %7.4f| vl: %7.4f| adv_s: %7.4f| adv_l: %7.4f| r_s: %7.4f| r_l: %7.4f" %(buffer_rs[i], buffer_rl[i], buffer_vpred_s[i], buffer_vpred_l[i], buffer_adv_s[i], buffer_adv_l[i], buffer_return_s[i], buffer_return_l[i]), buffer_info[i])
            
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
        
        return buffer_return_s, buffer_return_l, buffer_adv_s, buffer_adv_l, info_num

    def test_model(self, ep_num):
        goal_num = 0
        saved_ep = 0
        max_step = EP_LEN
        print('searching for failed ep', ep_num) 
        for test_num in range(100):
            s = self.env.reset(0, 0, 1)
            for step in range(max_step):
                a = self.ppo.choose_action(s, False)
                s_, r_short, r_long, done, info = self.env.step(a)

                s = s_
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
        print (goal_num/10)

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER \
            , Goal_states, Goal_count, Goal_return, Goal_buffer_full \
            , Crash_states, Crash_count, Crash_return, Crash_buffer_full \
            , History_states, History_count, History_adv, History_return, History_buffer_full

        # self.env.save_ep()
        # self.test_model(5)
        # for _ in range(2):
        #     s = self.env.reset( 0, 0, 1)
        #     self.env.save_ep()

        update_counter = 0
        ep_count = 0

        while not COORD.should_stop():
            buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_vpred_s, buffer_vpred_l, buffer_info = [], [], [], [], [], [], []
            info = 'unfinish'

            t = 0
            ep_length = EP_LEN #len(test_actions)
            # for a in test_actions:
            GLOBAL_EP += 1
            ep_count += 1
            saved_ep = False
            
            s = self.env.reset( 0, 1, 1)

            while(1):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_vpred_s, buffer_vpred_l, buffer_info = [], [], [], [], [], [], []
                    # if update_counter%1 == 0:
                        # self.env.clear_history()
                    # self.test_model(5)
                    update_counter += 1

                a = self.ppo.choose_action(s, False)
                # self.ppo.get_action_prob(s)
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

                # if info == 'crash' and saved_ep == False:
                #     # self.env.save_ep()
                #     # self.env.save_start_end_ep()
                #     saved_ep = True

                # if self.wid == 0:
                #     if t == ep_length-1 or done:
                #         buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_vpred_s, buffer_vpred_l, buffer_info = [], [], [], [], [], [], []
                #         break
                #     else:
                #         continue

                if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE or t == ep_length-1 or done:
                # if (GLOBAL_EP != 0 and GLOBAL_EP%EP_BATCH_SIZE == 0) or t == ep_length-1 or done:

                    buffer_return_s, buffer_return_l, buffer_adv_s, buffer_adv_l, info_num = self.process_and_send(buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_info, s_, self.env.return_end)
                    bs, ba, bret_s, bret_l, badv_s, badv_l, binfo = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_return_s)[:, np.newaxis], np.array(buffer_return_l)[:, np.newaxis], np.array(buffer_adv_s)[:, np.newaxis], np.array(buffer_adv_l)[:, np.newaxis], np.vstack(info_num)     
                    QUEUE.put(np.hstack((bs, ba, bret_s, bret_l, badv_s, badv_l, binfo)))          # put data in the queue

                    # if info == 'goal':
                    #     aug_s, aug_a, aug_return_s, aug_return_l, aug_adv_s, aug_adv_l, info_num = self.get_aurgm_batch(5, buffer_s, buffer_a, buffer_return_s, buffer_return_l, info_num)
                    #     if (aug_s != []):
                    #         bs, ba, bret_s, bret_l, badv_s, badv_l, binfo = np.vstack(aug_s), np.vstack(aug_a), np.array(aug_return_s)[:, np.newaxis], np.array(aug_return_l)[:, np.newaxis], np.array(aug_adv_s)[:, np.newaxis], np.array(aug_adv_l)[:, np.newaxis], np.vstack(info_num)     
                    #         QUEUE.put(np.hstack((bs, ba, bret_s, bret_l, badv_s, badv_l, binfo)))          # put data in the queue
                    #         GLOBAL_UPDATE_COUNTER += len(aug_a)
                    #         print('generage new batch:', len(aug_a))

                    buffer_s, buffer_a, buffer_rs, buffer_rl, buffer_vpred_s, buffer_vpred_l, buffer_info = [], [], [], [], [], [], []
                    if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
                    # if (GLOBAL_EP != 0 and GLOBAL_EP%EP_BATCH_SIZE == 0):
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if G_ITERATION >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    if info == 'goal' or info == 'out' or t == ep_length-1:
                        # if done != 'goal':
                        #     self.env.save_start_end_ep()
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