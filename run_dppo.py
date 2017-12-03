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
EP_LEN = 50
N_WORKER = 7               # parallel workers
GAMMA = 0.97                 # reward discount factor
LAM = 0.7
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
LR = 0.0001

BATCH_SIZE = 10240
MIN_BATCH_SIZE = 64       # minimum batch size for updating PPO

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
        self.tf_is_train = tf.placeholder(tf.bool, None)

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(self.sess, shape=S_DIM)

        # actor
        pi, self.v, net_params = self._build_anet('net', self.tfs, trainable=True)
        training_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')

        oldpi, oldv, oldnet_params = self._build_anet('oldnet', self.tfs, trainable=False)
        self.sample_op = pi.sample() #tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(net_params, oldnet_params)]
        # self.restorepi_op = [p.assign(oldp) for p, oldp in zip(net_params, oldnet_params)]

        if isinstance(Action_Space, gym.spaces.Box):
            self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        else:
            self.tfa = tf.placeholder(tf.int32, [None], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        # ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        self.ratio = tf.exp(oldpi.neglogp(self.tfa) - pi.neglogp(self.tfa))
        self.surr = self.ratio * self.tfadv                       # surrogate loss
        self.surr2 = tf.clip_by_value(self.ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv
        self.aloss = -tf.reduce_mean(tf.minimum(self.surr, self.surr2))

        self.entropy = tf.reduce_mean(pi.entropy())
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # critic
        # self.feature = self._build_feature_net('feature', self.tfs)
        # l1 = tf.layers.dense(self.feature, 100, tf.nn.relu)
        # self.v = tf.layers.dense(self.feature, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None], 'discounted_r')
        vpredclipped = oldv + tf.clip_by_value(self.v - oldv, - EPSILON, EPSILON)

        self.closs1 = tf.square(self.tfdc_r - self.v)
        self.closs2 = tf.square(self.tfdc_r - vpredclipped)

        self.closs = 0.5 * tf.reduce_mean(tf.maximum(self.closs1, self.closs2))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        self.total_loss = self.aloss + self.closs*0.3

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op = tf.train.AdamOptimizer(LR).minimize(self.total_loss)

        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.total_loss, var_list = training_params)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('./log', self.sess.graph)     
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
            state_size = 5
            num_img = S_DIM - state_size - 1# 
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

            x = tf.layers.conv2d(inputs=reshaped_grid, filters=32, kernel_size=[3, 3], padding="valid", activation=tf.nn.tanh)
            x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="valid", activation=tf.nn.tanh)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.tanh)

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
            feature = tf.layers.dense(feature, 50, tf.nn.tanh, name='feature_fc', trainable=trainable )
            feature = tf.layers.dense(feature, 50, tf.nn.tanh, name='feature_fc2', trainable=trainable )
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
                # l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
                # mu = 1 * tf.layers.dense(self.feature, A_DIM, tf.nn.tanh, kernel_initializer=w_init, trainable=trainable)
                # sigma = tf.layers.dense(self.feature, A_DIM, tf.nn.softplus, kernel_initializer=w_init, trainable=trainable)
                # norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
                if isinstance(Action_Space, gym.spaces.Box):
                    print('continue action')
                    pi = tf.layers.dense(self.feature, A_DIM, kernel_initializer=w_init, trainable=trainable)
                    logstd = tf.get_variable(name="logstd", shape=[1, A_DIM], initializer=tf.zeros_initializer(), trainable=trainable)
                    pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
                else:
                    print('discrate action')
                    pdparam = tf.layers.dense(self.feature, A_DIM, kernel_initializer=w_init, trainable=trainable)        

            with tf.variable_scope('value'):
                v = tf.layers.dense(self.feature, 1, trainable=trainable)
                # v = tf.clip_by_value(v, 0, centauro_env.REWARD_GOAL)

        pdtype = make_pdtype(Action_Space)
        pd = pdtype.pdfromflat(pdparam)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pd, v, params


    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s, self.tf_is_train: False})[0]
        if isinstance(Action_Space, gym.spaces.Box):
            a = np.clip(a, -1, 1)
        return a

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s, self.tf_is_train: False})[0, 0]


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
            self.ob_rms.update(s)
            
            # adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

            if adv.std() != 0:                
                adv = (adv - adv.mean())/adv.std()
                # adv = ((adv - adv.min())/(adv.max() - adv.min()) - 0.5)*2
                print('adv min max', adv.mean(), adv.min(), adv.max())

            # adv = adv * 1 + reward * 0
            # for i in range(len(r)):
            #     print(a[i], 'ret', r[i], 'adv', adv[i], reward[i])

            mean_return = np.mean(r)
            print(G_ITERATION, '  --------------- update! batch size:', GLOBAL_EP, '-----------------', len(r))

            for iteration in range(UPDATE_STEP):
                # construct reward predict data
                tloss, aloss, vloss, rp_loss, vr_loss = [], [], [], [], []
                tloss_sum, aloss_sum, vloss_sum, rp_loss_sum, vr_loss_sum, entropy_sum = 0, 0, 0, 0, 0, 0  
                
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
                    sub_adv = adv_[start:end]
                    feed_dict = {
                        self.tfs: sub_s, 
                        self.tfa: sub_a, 
                        self.tfdc_r: sub_r, 
                        self.tfadv: sub_adv, 
                        self.tf_is_train: True
                    }

                    tloss, aloss, vloss, entropy, _ = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy, self.train_op], feed_dict = feed_dict)
                    
                    tloss_sum += tloss
                    aloss_sum += aloss 
                    vloss_sum += vloss 
                    entropy_sum += np.mean(entropy)

                print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss_sum/count, vloss_sum/count, entropy_sum/count))

            feed_dict = {
                self.tfs: s, 
                self.tfa: a, 
                self.tfdc_r: r, 
                self.tfadv: adv, 
                self.tf_is_train: False
            }
            # ratio, surr, surr2 = self.sess.run([self.ratio, self.surr, self.surr2], feed_dict = feed_dict)
            # ratio_clip = np.clip(ratio, 1-EPSILON, 1+EPSILON)
            # surr_ = ratio*adv.flatten()
            # surr2_ = ratio_clip*adv.flatten()
            # # -np.minimum(ratio*adv.flatten(), ratio_clip*adv.flatten())

            # print((surr))
            # print(surr_)
            tloss, aloss, vloss, entropy = self.sess.run([self.total_loss, self.aloss, self.closs, self.entropy], feed_dict = feed_dict)
            print('-------------------------------------------------------------------------------')
            print("aloss: %7.4f|, vloss: %7.4f| entropy: %7.4f" % (aloss, vloss, np.mean(entropy)))

            self.write_summary('Loss/entropy', np.mean(entropy))  
            self.write_summary('Loss/a loss', aloss) 
            self.write_summary('Loss/v loss', vloss) 
            self.write_summary('Loss/t loss', tloss) 
            self.write_summary('Perf/mean_reward', np.mean(reward))  

            self.saver.save(self.sess, './model/rl/model.cptk') 

            UPDATE_EVENT.clear()        # updating finished
            GLOBAL_UPDATE_COUNTER = 0   # reset counter
            G_ITERATION += 1
            ROLLING_EVENT.set()         # set roll-out available

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
            # full_a = self.env.action_list[a]
            # full_a = -np.array(full_a)
            # for index in range(len(self.env.action_list)):
            #     env_action = np.array(self.env.action_list[index])
            #     if np.array_equal(env_action, full_a):
            #         a = index 
            #         break
            # print('reversed action', full_a, a)
            # a = full_a
            a = -a
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


    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER \
            , Goal_states, Goal_count, Goal_return, Goal_buffer_full \
            , Crash_states, Crash_count, Crash_return, Crash_buffer_full \
            , History_states, History_count, History_adv, History_return, History_buffer_full

        self.env.save_ep()
        s = self.env.reset( 0, 1, 0)

        ep_count = 0
        while not COORD.should_stop():
            buffer_s, buffer_a, buffer_r, buffer_er, buffer_vpred, buffer_return, buffer_adv, buffer_done = [], [], [], [], [], [], [], []
            ep_count += 1
            ep_step = 0
            a_demo = np.random.rand(A_DIM)# np.array([-1, 0, 0, 0, 0])
            # a_demo[3] = 0
            # a_demo[4] = 0
            info = 'unfinish'
            if ep_count%10 == 1 or ep_count == 1:
                self.env.clear_history()
                # self.env.clear_history_leave_one()
                # self.env.reset(0, 0, 0)
                # self.env.save_ep()
                num_saved = 0

            if ep_count%5 == 0 and False:
                s = self.env.reset(0, 4, 0)
                ep_length = int(EP_LEN)
                DEMO_MODE = True
                # self.env.clear_history_leave_one()
                ep_batch_size = BATCH_SIZE
            else:
                s = self.env.reset(0, 1, 0)
                ep_length = EP_LEN
                ep_batch_size = BATCH_SIZE
                DEMO_MODE = False
            t = 0
            mb_t = 0
            retry = False
            while(1):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r, buffer_er, buffer_vpred, buffer_return, buffer_adv = [], [], [], [], [], [], []
                    # break

                if DEMO_MODE:
                    # a = np.random.randint(A_DIM, size=1)
                    # a = a[0]
                    if info == 'crash' or np.random.rand() > 0.5:
                        a = np.random.rand(A_DIM)
                        a = (a-0.5)*2
                    # elif info == 'crash':
                    #     s_demo = s*1
                    #     s_demo[-5] = -s_demo[-5]
                    #     s_demo[-6] = -s_demo[-6]
                    #     a = self.ppo.choose_action(s)
                    else:
                        a = a_demo
                    # print('demo action', a)
                else:
                    a = self.ppo.choose_action(s)
                
                s_, r, event_r, done, info = self.env.step(a)
                vpred = self.ppo.get_v(s)

                if DEMO_MODE and (info == 'goal' or info == 'crash'):
                    done = False

                if self.wid == 0:
                    vpred_ = self.ppo.get_v(s_)
                    td = event_r + GAMMA*vpred_ - vpred
                    print("vpred_: %7.4f| event_r: %7.4f| vpred: %7.4f| TD: %7.4f| " %(vpred_, event_r, vpred, td))

                if DEMO_MODE == False or (DEMO_MODE and info != 'crash'):
                    buffer_s.append(s*1)
                    buffer_a.append(a)
                    buffer_r.append(r)                    # normalize reward, find to be useful
                    buffer_er.append(event_r)
                    buffer_vpred.append(vpred)
                    buffer_done.append(done)
                    s_demo = s_*1
                    GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers

                s = s_
                ep_step += 1
                t += 1
                # mb_t += 1
                
                # if info == 'crash':
                #     retry = True
                #     mb_t = 0

                if done or retry or GLOBAL_UPDATE_COUNTER >= ep_batch_size or t == ep_length-1:
                    # if num_saved < 5 and done == False:
                    #     num_saved += 1
                    #     self.env.save_ep()

                    if DEMO_MODE:
                        if len(buffer_a) == 0:
                            buffer_s, buffer_a, buffer_r, buffer_er, buffer_vpred, buffer_return, buffer_adv, buffer_done = [], [], [], [], [], [], [], []
                            break
                        else:
                            buffer_s, buffer_a, buffer_r, buffer_er, buffer_vpred, s_ = self.process_demo_path(buffer_a, s_demo)

                    # if done:
                    #     vpred_ = 0
                    # else:
                    vpred_ = self.ppo.get_v(s_)
                    buffer_done.append(0)
                    buffer_vpred.append(vpred_)
                    buffer_return = np.zeros_like(buffer_r)
                    buffer_adv = np.zeros_like(buffer_r)
                    lastgaelam = 0

                    for index in reversed(range(len(buffer_r))):
                        nonterminal = 1-buffer_done[index+1]
                        delta = buffer_er[index] + GAMMA * buffer_vpred[index+1] * nonterminal - buffer_vpred[index]
                        buffer_adv[index] = lastgaelam = delta + GAMMA * LAM * nonterminal * lastgaelam
                    buffer_return = buffer_adv + buffer_vpred[:-1]

                    # buffer_return = self.discount(buffer_er, GAMMA)
                    # buffer_adv = self.compute_gae(buffer_er, buffer_vpred)
        
                    # for i in range(len(buffer_r)):
                    #     print(buffer_a[i], 'ret', buffer_return[i], 'pred', buffer_vpred[i], 'adv', buffer_adv[i], buffer_r[i])
                    # print(DEMO_MODE, info)

                    # if self.wid == 0:
                    #     print(buffer_er)
                    #     print (buffer_return)
                    #     # print(buffer_vpred)
                    #     print (buffer_adv)
                    bs, ba, bret, brew, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_return)[:, np.newaxis], np.array(buffer_r)[:, np.newaxis], np.array(buffer_adv)[:, np.newaxis]

                    buffer_s, buffer_a, buffer_r, buffer_er, buffer_vpred, buffer_return, buffer_adv, buffer_done = [], [], [], [], [], [], [], []
                
                    QUEUE.put(np.hstack((bs, ba, bret, brew, badv)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= ep_batch_size:
                        # if DEMO_MODE == False:
                        #     self.env.clear_history_leave_one()

                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if G_ITERATION >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    # if DEMO_MODE == False and (info == 'goal' or info == 'out' or t == ep_length-1):
                    #     break 

                    if DEMO_MODE == False and (t == ep_length-1 or done):
                        break 

                    # if DEMO_MODE == False and retry and t < ep_length-1:
                    #     s = self.env.reset(0, 5, 0) 
                    #     retry = False


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