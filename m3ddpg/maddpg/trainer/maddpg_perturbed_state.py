import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer
import tensorflow.contrib.layers as layers

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def mlp_model_a(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.tanh)
        return out

#adversary part, perturb the observated state
def a_train(make_obs_ph_n, obs_shape_n, act_space_n, a_index, a_func, p_func, q_func, optimizer, d_value, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        
        # set up placeholders
        obs_a_input = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        obs_size = int(obs_shape_n[a_index][0])
        
        a_input = obs_a_input[a_index]
        
        a = a_func(a_input, obs_size, scope="a_func", num_units=num_units) 
        a_func_vars = U.scope_vars(U.absolute_scope_name("a_func"))
        
        # wrap parameters in distribution
        dc = tf.constant(d_value, dtype=tf.float32)
        perturbed_state = tf.add(a_input, tf.multiply(a, dc))
        
        p = p_func(perturbed_state, int(act_pdtype_n[a_index].param_shape()[0]), scope="p_func", reuse=True, num_units=num_units)
        act_pd = act_pdtype_n[a_index].pdfromflat(p)
        
        act_input_n = act_ph_n + []
        act_input_n[a_index] = act_pd.sample()
        q_input = tf.concat(obs_a_input + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_a_input[a_index], act_input_n[a_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)
        
        ascent_loss = -pg_loss #+ a_reg * 1e-3
        
        optimize_expr = U.minimize_and_clip(optimizer, ascent_loss, a_func_vars, grad_norm_clipping)
        
        # Create callable functions
        train = U.function(inputs=obs_a_input + act_ph_n, outputs=ascent_loss, updates=[optimize_expr])
        perturb_obs = U.function(inputs=[obs_a_input[a_index]], outputs = perturbed_state)
        perturbed_obs = U.function([obs_a_input[a_index]], perturbed_state)

        # target network
        target_a = a_func(a_input, obs_size, scope="target_a_func", num_units=num_units)
        target_a_func_vars = U.scope_vars(U.absolute_scope_name("target_a_func"))
        update_target_a = make_update_exp(a_func_vars, target_a_func_vars)

        tdc = tf.constant(d_value, dtype=tf.float32)
        target_perturbed_state = tf.add(a_input, tf.multiply(target_a, tdc))
        target_perturbed_obs = U.function(inputs=[obs_a_input[a_index]], outputs=target_perturbed_state)

        return perturb_obs, train, update_target_a, {'perturbed_obs': perturbed_obs, 'target_perturbed_obs': target_perturbed_obs}
        

def p_train(make_obs_ph_n, obs_shape_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        perturbed_state = tf.placeholder(tf.float32, [None, int(obs_shape_n[p_index][0])], name="perturbed"+str(p_index))

        p = p_func(perturbed_state, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()
        q_input = tf.concat(make_obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([make_obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=make_obs_ph_n + act_ph_n + [perturbed_state], outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[perturbed_state], outputs=act_sample)
        p_values = U.function([perturbed_state], p)

        # target network
        target_p = p_func(perturbed_state, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[perturbed_state], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")

        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}

class MADDPGPerturbedAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, obs_space_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            obs_shape_n = obs_shape_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        
        self.obs, self.a_train, self.a_update, self.a_debug = a_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            obs_shape_n = obs_shape_n,
            act_space_n=act_space_n,
            a_index=agent_index,
            a_func = mlp_model_a,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            d_value = args.d_value,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        
    def perturb(self, obs):
        return self.obs(obs[None])

    def action(self, obs):
        return self.act(self.perturb(obs))[0]

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None
        
    def ComputeTargetPerturbedObs(self, origin_obs, agents):
        obs_array = []
        for i in range(self.n):
            perturbed_state = agents[i].a_debug['target_perturbed_obs'](origin_obs[i])
            obs_array.append(perturbed_state)
        return obs_array

    def update(self, agents, t):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        perturbed_obs = self.ComputeTargetPerturbedObs(obs_next_n, agents)
        target_act_next_n = [agents[i].p_debug['target_act'](perturbed_obs[i]) for i in range(self.n)]
        target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
        target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))
        
        # gradient descent ascent inner loop
        for i in range(self.args.gda_step):
            # train p network
            cur_perturbed_obs = [agents[self.agent_index].a_debug['perturbed_obs'](obs_n[self.agent_index])]
            p_loss = self.p_train(*(obs_n + act_n + cur_perturbed_obs))
        
            #train a network
            a_loss = self.a_train(*(obs_n + act_n))

        self.p_update()
        self.q_update()
        self.a_update()

        return [q_loss, p_loss, a_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
