
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from environment import centauro_env
import scipy.signal
import math

IT_MAX = 100000
EP_LEN = 150
N_WORKER = 7                # parallel workers
GAMMA = 0.9                 # reward discount factor
LAM = 0.99
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
LR = 0.0001
MIN_BATCH_SIZE = 128         # minimum batch size for updating PPO
UPDATE_STEP = 5            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
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
Crash_step = 10
Crash_count = 0
Crash_buffer_full = False 

RP_buffer_size = 2000

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        self.tfs_label = tf.placeholder(tf.float32, [None, S_DIM], 'state_label')
        self.label = tf.placeholder(tf.int32, [None], 'true_label')
        self.label_return = tf.placeholder(tf.float32, [None, 1], 'label_return')

        # feature
        self.feature = self._build_feature_net('feature', self.tfs, reuse = False)
        self.feature_label = self._build_feature_net('feature', self.tfs_label, reuse = True)

        # label predict
        self.label_pred = tf.layers.dense(self.feature_label, 2)
        self.loss_pred = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.label_pred, labels=self.label))

        # critic
        # l1 = self.feature #tf.layers.dense(self.feature, 100, tf.nn.relu)
        self.v = self._build_cnet('value', self.feature, reuse = False)

        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.diff_r_v = self.tfdc_r - self.v
        self.closs = tf.nn.l2_loss(self.diff_r_v)
        # self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.entropy = -tf.reduce_sum(pi.entropy())

        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.sample_op_stochastic = pi.loc

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)

        surr1 = ratio * self.tfadv                       # surrogate loss
        surr2 = tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv

        self.aloss = -tf.reduce_mean(tf.minimum(surr1, surr2)) + self.entropy*0.001

        # value replay
        self.v_label = self._build_cnet('value', self.feature_label, reuse = True)
        self.diff_r_vr = self.label_return - self.v_label
        self.loss_vr = tf.nn.l2_loss(self.diff_r_vr)

        #############################################################
        self.total_loss = self.aloss + self.closs #+ self.loss_pred*0 + self.loss_vr*0
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.total_loss)

        # self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
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

    def get_rp_buffer(self, sample_num):
        rp_states = []
        rp_label = []
        rp_return = []

        size = RP_buffer_size
        replace = False
        if Goal_buffer_full == False:
            size = Goal_count
            replace = True
        if size > 0:
            goal_selected = np.random.choice(size, sample_num, replace=replace)
            for index in goal_selected:
                rp_states.append(Goal_states[index])
                rp_label.append(0)
                rp_return.append(Goal_return[index])

        size = RP_buffer_size
        replace = False
        if Crash_buffer_full == False:
            size = Crash_count
            replace = True
        if size > 0:
            crash_selected = np.random.choice(size, sample_num, replace=replace)
            for index in crash_selected:
                rp_states.append(Crash_states[index])
                rp_label.append(1)
                rp_return.append(Crash_return[index])

        return rp_states, rp_label, np.array(rp_return)[:, np.newaxis]

    def update(self):
        global GLOBAL_UPDATE_COUNTER, G_ITERATION
        while not COORD.should_stop():
            UPDATE_EVENT.wait()                     # wait until get batch of data
            self.sess.run(self.update_oldpi_op)     # copy pi to old pi
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
            data = np.vstack(data)
            s, a, r, adv = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, S_DIM + A_DIM: S_DIM + A_DIM + 1], data[:, -1:]
            adv = (adv - adv.mean())/adv.std()
            # adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
            # update actor and critic in a update loop

            print(G_ITERATION, '  --------------- update! batch size:', len(a), '-----------------')
            tloss, aloss, vloss, rp_loss, vr_loss = [], [], [], [], []
            for _ in range(UPDATE_STEP):
                # construct reward predict data
                rp_states, rp_label, rp_return = self.get_rp_buffer(int(len(r)))
                if len(rp_return) == 0:
                    break
                tloss, aloss, vloss, rp_loss, vr_loss, entropy, _ = self.sess.run([self.total_loss, self.aloss, self.closs, self.loss_pred, self.loss_vr, self.entropy, self.train_op]
                , {self.tfs: s, self.tfa: a, self.tfdc_r: r, self.tfadv: adv, self.tfs_label: rp_states, self.label: rp_label, self.label_return: rp_return})
                
                print("aloss: %7.4f|, vloss: %7.4f|, rp_loss: %7.4f|, vr_loss: %7.4f|, entropy: %7.4f" % (np.mean(aloss), np.mean(vloss), np.mean(rp_loss), np.mean(rp_return), np.mean(entropy)))

            # [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
            # [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]

            entropy = self.sess.run(self.entropy, {self.tfs: s})                
            self.write_summary('Loss/entropy', np.mean(entropy))  
            self.write_summary('Loss/a loss', np.mean(aloss)) 
            self.write_summary('Loss/v loss', np.mean(vloss)) 
            self.write_summary('Loss/rp loss', np.mean(rp_loss)) 
            self.write_summary('Loss/vr loss', np.mean(vr_loss)) 
            self.write_summary('Loss/t loss', np.mean(tloss)) 
            self.write_summary('Perf/return', np.mean(r))  

            self.saver.save(self.sess, './model/rl/model.cptk') 

            UPDATE_EVENT.clear()        # updating finished
            GLOBAL_UPDATE_COUNTER = 0   # reset counter
            G_ITERATION += 1
            ROLLING_EVENT.set()         # set roll-out available
                

    def _build_feature_net(self, name, input_state, reuse = False):
        with tf.variable_scope(name, reuse=reuse):
            state_size = 5
            num_img = S_DIM - state_size - 1# 
            img_size = int(math.sqrt(num_img))
            print(num_img, img_size)

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
            x = tf.layers.dense(x, 100, tf.nn.relu, name='x_fc1' )
            x = tf.layers.dense(x, 50, tf.nn.relu, name='x_fc2' )

            # process state
            state_rt = tf.layers.dense(ob_state, state_size*5, tf.nn.relu, name='rt_fc1' )
            state_rt = tf.layers.dense(state_rt, state_size*3, tf.nn.relu, name='rt_fc2' )
            
            feature = tf.concat([x , state_rt], 1, name = 'concat')
            # feature = tf.layers.dense(state_concat, 100, tf.nn.relu, name='feature_fc' )
        return feature

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.feature, 100, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l1, A_DIM, tf.nn.tanh, kernel_initializer=tf.zeros_initializer(), trainable=trainable)
            # logstd = tf.get_variable(name="logstd", shape=[1, A_DIM], initializer=tf.zeros_initializer(), trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)  #   tf.exp(logstd))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_cnet(self, name, input_state, reuse = False):
        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(input_state, 100, tf.nn.relu)
            v = tf.layers.dense(l1, 1)
        return v

    def choose_action(self, s, stochastic = True):
        s = s[np.newaxis, :]
        if stochastic:
            a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        else:
            a = self.sess.run(self.sample_op_stochastic, {self.tfs: s})[0]

        return np.clip(a, -1, 1)

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
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, Goal_states, Crash_states, Goal_return, Crash_return, Goal_count, Crash_count
        self.env.save_ep()
        s = self.env.reset(0, 1, 0)
        Goal_states = np.array([s for _ in range(RP_buffer_size)])
        Goal_return = np.zeros(RP_buffer_size, 'float32')

        Crash_states = np.array([s for _ in range(RP_buffer_size)])
        Crash_return = np.zeros(RP_buffer_size, 'float32')

        while not COORD.should_stop():
            buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_return, buffer_adv = [], [], [], [], [], []
            ep_crash = 0
            ep_step = 0
            s = self.env.reset(0, 4, 0)
            s[-1] = (ep_step/EP_LEN-0.5)*2

            if self.wid == 0:
                for t in range(EP_LEN):
                    a = self.ppo.choose_action(s, stochastic = True)
                    s_, r, done, info = self.env.step(a)
                    vpred = self.ppo.get_v(s)
                    print(vpred, a, r, info)
                    s = s_

                    if info == 'goal' or info == 'out' or t == EP_LEN - 1:
                        self.env.reset(0,1,0)
            else:
                for t in range(EP_LEN):
                    if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                        ROLLING_EVENT.wait()                        # wait until PPO is updated
                        GLOBAL_UPDATE_COUNTER -= len(buffer_r)
                        buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_return, buffer_adv = [], [], [], [], [], []
                    a = self.ppo.choose_action(s)
                    vpred = self.ppo.get_v(s)
                    # print(vpred)
                    s_, r, done, info = self.env.step(a)
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)                    # normalize reward, find to be useful
                    buffer_vpred.append(vpred)

                    s = s_
                    ep_step += 1
                    # if info == 'crash':
                    #     ep_crash += 1
                    # else:
                    #     ep_crash = 0

                    # if ep_crash > 10:
                    #     done = True

                    GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                    if t == EP_LEN - 1 or done or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        # if info == 'unfinish' and t != EP_LEN - 1:
                        buffer_r[-1] += GAMMA * self.ppo.get_v(s_)
                        buffer_return = self.discount(buffer_r, GAMMA)
                        buffer_adv = self.compute_gae(buffer_r, buffer_vpred)

                        # self.write_summary('Perf/ep_length', len(a))  
                        # print('---------------------------------------')
                        # print(buffer_r)
                        # print(buffer_return)
                        # print(buffer_vpred)
                        # print(buffer_adv)

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

                        if info == 'crash':
                            for i in range(len(buffer_r)-1, len(buffer_r)-Crash_step-1, -1):
                                if i < 0:
                                    break 
                                else:
                                    Crash_states[Crash_count] = buffer_s[i]
                                    Crash_return[Crash_count] = buffer_return[i]
                                    Crash_count = (Crash_count+1)%RP_buffer_size
                                    if Crash_count == RP_buffer_size - 1:
                                        Crash_buffer_full = True

                        # print(Goal_count, Crash_count)

                        bs, ba, bret, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.array(buffer_return)[:, np.newaxis], np.array(buffer_adv)[:, np.newaxis]

                        buffer_s, buffer_a, buffer_r, buffer_vpred, buffer_return, buffer_adv = [], [], [], [], [], []
                    
                        QUEUE.put(np.hstack((bs, ba, bret, badv)))          # put data in the queue
                        if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                            ROLLING_EVENT.clear()       # stop collecting data
                            UPDATE_EVENT.set()          # globalPPO update

                        if G_ITERATION >= IT_MAX:         # stop training
                            COORD.request_stop()
                            break

                        if info == 'goal' or info == 'out' or t == EP_LEN - 1:
                            if Goal_count > 100:
                                self.env.reset(0,1,0)
                            else:
                                self.env.reset(0,4,0)

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
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
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