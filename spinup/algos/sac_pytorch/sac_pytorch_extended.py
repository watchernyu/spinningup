import gym
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import time
from spinup.algos.sac_pytorch.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from torch.nn import functional as F

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

def extend_obs_with_time(obs, current_time_step, max_time_step):
    ## given the observation and current time
    ## will return an observation with remaining time added as a feature
    ## obs in the form of a list
    timestep_remain = max_time_step - current_time_step
    half_span = max_time_step/2
    time_feature = (timestep_remain-half_span)/half_span
    obs = np.concatenate((obs, np.array([time_feature])))
    return obs

def selu_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    std = 1. / np.sqrt(fan_in)
    return tensor.data.normal_(0, std)

def sac_pytorch_extended(env_fn, hidden_sizes=[256, 256], seed=0,
                steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
                polyak=0.995, lr=3e-4, alpha=0.2, batch_size=256, start_steps=10000,
                max_ep_len=1000, save_freq=1,
                exp_name='sac', data_dir='data/', logger_kwargs=dict(),
                amsgrad=False, add_time_to_obs=False, use_selu=False):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    ## seed environment always with 0 so that the series of starting obs will be same
    ## you can also use the torch,np seed for env, that will give you different starting obs for different seeds
    env.seed(seed)
    test_env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if add_time_to_obs: ## if put remaining time in observations
        obs_dim += 1

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def test_agent(n=10):
        """
        This will test the agent's performance by running n episodes
        During the runs, the agent only take deterministic action, so the
        actions are not drawn from a distribution, but just use the mean
        :param n: number of episodes to run the agent
        """

        ep_return_list = np.zeros(n)
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            if add_time_to_obs:
                o = extend_obs_with_time(o, ep_len, max_ep_len)
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a = policy_net.get_env_action(o, action_limit=act_limit, deterministic=True)
                o, r, d, _ = test_env.step(a)
                if add_time_to_obs:
                    o = extend_obs_with_time(o, ep_len, max_ep_len)
                ep_ret += r
                ep_len += 1
            ep_return_list[j] = ep_ret
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    if add_time_to_obs:
        o = extend_obs_with_time(o, ep_len, max_ep_len)

    total_steps = steps_per_epoch * epochs

    """init networks"""
    # see line 1
    if not use_selu:
        policy_net = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes)
        value_net = Mlp(obs_dim,1,hidden_sizes)
        target_value_net = Mlp(obs_dim,1,hidden_sizes)
        q1_net = Mlp(obs_dim+act_dim,1,hidden_sizes)
        q2_net = Mlp(obs_dim+act_dim,1,hidden_sizes)
    else:
        policy_net = TanhGaussianPolicy(obs_dim, act_dim, hidden_sizes, hidden_init=selu_init, hidden_activation=F.selu)
        value_net = Mlp(obs_dim,1,hidden_sizes,hidden_init=selu_init, hidden_activation=F.selu)
        target_value_net = Mlp(obs_dim,1,hidden_sizes,hidden_init=selu_init, hidden_activation=F.selu)
        q1_net = Mlp(obs_dim+act_dim,1,hidden_sizes,hidden_init=selu_init, hidden_activation=F.selu)
        q2_net = Mlp(obs_dim+act_dim,1,hidden_sizes,hidden_init=selu_init, hidden_activation=F.selu)

    # see line 2: copy parameters from value_net to target_value_net
    target_value_net.load_state_dict(value_net.state_dict())

    # set up optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(),lr=lr,amsgrad=amsgrad)
    value_optimizer = optim.Adam(value_net.parameters(),lr=lr,amsgrad=amsgrad)
    q1_optimizer = optim.Adam(q1_net.parameters(),lr=lr,amsgrad=amsgrad)
    q2_optimizer = optim.Adam(q2_net.parameters(),lr=lr,amsgrad=amsgrad)

    mse_criterion = nn.MSELoss()

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = policy_net.get_env_action(o, action_limit=act_limit)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        if add_time_to_obs:
            o2 = extend_obs_with_time(o2, t, max_ep_len)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            The original SAC paper: '. In practice, we take a single environment step
            followed by one or several gradient step' after a single environment step,
            the number of gradient steps is 1 for SAC. (see paper for reference)
            """
            print(t, ep_ret)
            for j in range(ep_len):
                # get data from replay buffer
                batch = replay_buffer.sample_batch(batch_size)
                obs_tensor =  Tensor(batch['obs1'])
                obs_next_tensor =  Tensor(batch['obs2'])
                acts_tensor =  Tensor(batch['acts'])
                # unsqueeze is to make sure rewards and done tensors are of the shape nx1, instead of n
                # to prevent problems later
                rews_tensor =  Tensor(batch['rews']).unsqueeze(1)
                done_tensor =  Tensor(batch['done']).unsqueeze(1)
                # print(obs_tensor.shape, obs_next_tensor.shape, acts_tensor.shape,rews_tensor.shape,done_tensor.shape)

                # v_value = value_net(obs_tensor)
                # q_value = q1_net(torch.cat([obs_tensor,acts_tensor],1))

                """
                now we do a SAC update, following the OpenAI spinup doc
                check the openai sac document psudocode part for reference
                line nubmers indicate lines in psudocode part
                we will first compute each of the losses
                and then update all the networks in the end
                """
                # see line 12: get a_tilda, which is newly sampled action (not action from replay buffer)
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, _ = policy_net.forward(obs_tensor)

                """get q loss"""
                # see line 12: first equation
                v_from_target_v_net = target_value_net(obs_next_tensor)
                y_q = rews_tensor + gamma*(1-done_tensor)*v_from_target_v_net
                # see line 13: compute loss for the 2 q networks, note that we want to detach the y_q value
                # since we only want to update q networks here, and don't want other gradients
                q1_prediction = q1_net(torch.cat([obs_tensor,acts_tensor], 1))
                q1_loss = mse_criterion(q1_prediction, y_q.detach())
                q2_prediction = q2_net(torch.cat([obs_tensor, acts_tensor], 1))
                q2_loss = mse_criterion(q2_prediction, y_q.detach())

                """get v loss"""
                # see line 12: second equation
                q1_a_tilda = q1_net(torch.cat([obs_tensor,a_tilda],1))
                q2_a_tilda = q2_net(torch.cat([obs_tensor,a_tilda],1))
                min_q1_q2_a_tilda = torch.min(torch.cat([q1_a_tilda,q2_a_tilda],1),1)[0].reshape(-1,1)
                y_v = min_q1_q2_a_tilda - alpha*log_prob_a_tilda

                # see line 14: compute loss for value network
                v_prediction = value_net(obs_tensor)
                v_loss = mse_criterion(v_prediction, y_v.detach())

                """policy loss"""
                # line 15: note that here we are doing gradient ascent, so we add a minus sign in the front
                policy_loss = - (q1_a_tilda - alpha*log_prob_a_tilda).mean()

                """
                policy regularization loss, this is not in openai's minimal version, but
                they are in the original sac code, see https://github.com/vitchyr/rlkit for reference
                this part is not necessary but might improve performance
                """
                policy_mean_reg_weight = 1e-3
                policy_std_reg_weight = 1e-3
                mean_reg_loss = policy_mean_reg_weight * (mean_a_tilda ** 2).mean()
                std_reg_loss = policy_std_reg_weight * (log_std_a_tilda ** 2).mean()
                policy_loss = policy_loss + mean_reg_loss + std_reg_loss

                """update networks"""
                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                value_optimizer.zero_grad()
                v_loss.backward()
                value_optimizer.step()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # see line 16: update target value network with value network
                soft_update_model1_with_model2(target_value_net, value_net, polyak)

                ## store diagnostic info to logger
                logger.store(LossPi=policy_loss.item(), LossQ1=q1_loss.item(), LossQ2=q2_loss.item(),
                             LossV=v_loss.item(),
                             Q1Vals=q1_prediction.detach().numpy(),
                             Q2Vals=q2_prediction.detach().numpy(),
                             VVals=v_prediction.detach().numpy(),
                             LogPi=log_prob_a_tilda.detach().numpy())

            ## store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ## reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            if add_time_to_obs:
                o = extend_obs_with_time(o, ep_len, max_ep_len)

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model TODO need to change code for saving pytorch model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--steps_per_epoch', type=int, default=5000)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    sac_pytorch_extended(lambda: gym.make(args.env), hidden_sizes=[args.hid] * args.l,
                gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                exp_name=args.exp_name, data_dir=args.data_dir, steps_per_epoch=args.steps_per_epoch,
                logger_kwargs=logger_kwargs)