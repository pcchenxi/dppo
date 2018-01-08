# from baselines.common.vec_env import VecEnv
from running_mean_std import RunningMeanStd
import numpy as np

class VecNormalize(object):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.venv = venv
        self._observation_space = self.venv.observation_space
        self._action_space = venv.action_space
        self.ob_rms = RunningMeanStd(shape=self._observation_space.shape) if ob else None
        self.ret_s_rms = RunningMeanStd(shape=()) if ret else None
        self.ret_l_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret_s = np.zeros(self.num_envs)
        self.ret_l = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews_s, rews_l, news, infos = self.venv.step(vac)
        self.ret_s = self.ret_s * self.gamma*0.7 + rews_s
        self.ret_l = self.ret_l * self.gamma + rews_l
        obs = self._obfilt(obs)
        # if self.ret_s_rms: 
        #     self.ret_s_rms.update((self.ret_s + self.ret_l)/2)
        #     rews_s = np.clip(rews_s / np.sqrt(self.ret_s_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        #     rews_l = np.clip(rews_l / np.sqrt(self.ret_s_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        # if self.ret_l_rms: 
        #     self.ret_l_rms.update(self.ret_l)
        #     rews_l = np.clip(rews_l / np.sqrt(self.ret_l_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        return obs, rews_s, rews_l, news, infos
    def _obfilt(self, obs):
        if self.ob_rms: 
            self.ob_rms.update(obs)
            img_ori = obs[:-4]
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            obs[:-4] = img_ori
            return obs
        else:
            return obs
    def reset(self, env_mode, reset_mode, save_ep):
        """
        Reset all environments
        """
        obs = self.venv.reset(env_mode, reset_mode, save_ep)
        return self._obfilt(obs)

    def save_ep(self):
        return self.venv.save_ep()
    
    def save_start_end_ep(self):
        return self.venv.save_start_end_ep()

    def clear_history_leave_one(self):
        return self.venv.clear_history_leave_one()
    def clear_history(self):
        return self.venv.clear_history()
    @property
    def action_space(self):
        return self._action_space

    @property
    def return_end(self):
        return self.venv.return_end

    @property
    def observation_space(self):
        return self._observation_space
    def close(self):
        self.venv.close()
    @property
    def num_envs(self):
        # return self.venv.num_envs
        return 1



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.zeros(shape, 'float64')
        self.count = epsilon


    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count        
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count        

def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        assert np.allclose(ms1, ms2)
