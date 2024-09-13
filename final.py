import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import torch
import copy
import random
import torch.nn as nn
from torch.distributions import Normal
import math

logfile_path = '/home/wzchen/u0810774/UM180FDKMFC_C05_PB/OA/OceanScript/Transfer_ONEto28nm/output.pythonlog'

def build_net(layer_shape, hidden_activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hidden_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
		super(Actor, self).__init__()
		layers = [state_dim] + list(hid_shape)

		self.a_net = build_net(layers, hidden_activation, output_activation)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20

	def forward(self, state, deterministic, with_logprob):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
		# we learn log_std rather than std, so that exp(log_std) is always > 0
		std = torch.exp(log_std)
		dist = Normal(mu, std)
		if deterministic: u = mu
		else: u = dist.rsample()

		a = torch.tanh(u)
		if with_logprob:
			# Get probability density of logp_pi_a from probability density of u:
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a

class Double_Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]

		self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		return q1, q2

class Double_V_Critic(nn.Module):
	def __init__(self, state_dim, hidden_size):
		super(Double_V_Critic, self).__init__()

		self.V1 = nn.Sequential(
			nn.Linear(state_dim, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1),
		)
		self.V2 = nn.Sequential(
			nn.Linear(state_dim, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1),
		)

	def forward(self, state):
		v1 = self.V1(state)
		v2 = self.V2(state)
		return v1, v2


class action_decoder_network(nn.Module):
    def __init__(self, source_action_dim, target_action_dim, hidden_size):
        super(action_decoder_network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(target_action_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, source_action_dim),
            nn.Tanh()
        )
        self.target_action_dim = target_action_dim

    def forward(self, target_action):
        source_action = self.model(target_action.float())
        return source_action

class decoder_network(nn.Module):
    def __init__(self, source_state_dim, target_state_dim, hidden_size, config):
        super(decoder_network, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(target_state_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, source_state_dim),
            nn.Tanh()
        )
        self.target_state_dim = target_state_dim
        self.scaling_func = scaling_func(config=config)

    def forward(self, target_state):
        source_state = self.model(target_state.float())
        return self.scaling_func(source_state)
		
class SAC_countinuous():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,self.net_width))
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width,self.net_width))
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		self.gradFalse()

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6))

		self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True)
		self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True)
		self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

		self.state_decoder_optimizer = torch.optim.Adam(self.state_decoder.parameters(), lr=1e-3)
		self.action_decoder_optimizer = torch.optim.Adam(self.action_decoder.parameters(), lr=1e-3)

		self.update_count = 0


	def gradFalse(self):
		for p in self.q_critic_target.parameters():
			p.requires_grad = False
		for p in self.sourceQ.parameters():
			p.requires_grad = False
		for p in self.sourceV.parameters():
			p.requires_grad = False

	def select_action(self, state, deterministic):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis,:])
			a, _ = self.actor(state, deterministic, with_logprob=False)
		return a.cpu().numpy()[0]

	def train(self,):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
		self.update_count += 1

		#----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
		with torch.no_grad():
			a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
			target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffe			

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#----------------------------- ↓↓↓↓↓ Update decoder Net ↓↓↓↓↓ ------------------------------#
		source_QSA1, source_QSA2 = self.sourceQ(self.state_decoder(s),self.action_decoder(a))

		with torch.no_grad():
			source_next_s_value1, source_next_s_value2 = self.sourceV(self.state_decoder(s_next).detach())
			source_next_s_value = torch.min(source_next_s_value1, source_next_s_value2)
			target_decoder = r + (~dw) * self.gamma * source_next_s_value

		decoderLoss = F.mse_loss(source_QSA1, target_decoder) + F.mse_loss(source_QSA2, target_decoder)

		self.state_decoder_optimizer.zero_grad()
		self.action_decoder_optimizer.zero_grad()
		decoderLoss.backward()
		self.state_decoder_optimizer.step()
		self.action_decoder_optimizer.step()

		#----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
		# Freeze critic so you don't waste computational effort computing gradients for them when update actor
		for params in self.q_critic.parameters(): params.requires_grad = False

		a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
		current_Q1, current_Q2 = self.q_critic(s, a)
		min_qf_pi = torch.min(current_Q1, current_Q2)

		source_Q1, source_Q2 = self.sourceQ(self.state_decoder(s), self.action_decoder(a))
		source_min_qf_pi = torch.min(source_Q1, source_Q2)

		alpha_t = (1.0 / self.update_count)
		Q = (1-alpha_t) * min_qf_pi + alpha_t * source_min_qf_pi
		
		a_loss = (self.alpha * log_pi_a - Q).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()
		
		#----------------------------- ↓↓↓↓↓ Update alpha Net ↓↓↓↓↓ ------------------------------#
		alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
		self.alpha_optim.zero_grad()
		alpha_loss.backward()
		self.alpha_optim.step()
		self.alpha = self.log_alpha.exp()


		for params in self.q_critic.parameters(): params.requires_grad = True

		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self):
		torch.save(self.actor.state_dict(), "/home/wzchen/u0810774/UM180FDKMFC_C05_PB/OA/OceanScript/Transfer_ONE2simpleTWO/actor.pth")
		torch.save(self.q_critic.state_dict(), "/home/wzchen/u0810774/UM180FDKMFC_C05_PB/OA/OceanScript/Transfer_ONE2simpleTWO/q_critic.pth")

	def load_source(self, path_QV):
		self.sourceQ.load_state_dict(torch.load(path_QV+'/q_critic.pth'))
		self.sourceV.load_state_dict(torch.load(path_QV+'/v_critic.pth'))

class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool)

	def add(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = torch.from_numpy(s)
		self.a[self.ptr] = torch.from_numpy(a) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

	def __len__(self):
		return self.size

def convertAction(input):
    output = ''
    for i in input:
        output += str(i)+','
    output += '\n'
    return output

def compute_reward(score):
	power = score[3]
	if score[0]<= 1:
		power = 1000000
	if score[0]<= 10:
		power = power*1000
	if score[2]>=45 and score[2] <= 180:
		PM = 1.0
	elif (score[2]>0 and score[2]<45):
		PM = 0.01
	else:
		PM = 0.0001
	reward = score[0]*score[1]*PM/power
	return math.log10(reward+1e-20)
      
class scaling_func(nn.Module):
    def __init__(self, config, round=True):
        super(scaling_func, self).__init__()
        self.scale_range = config['scale']
        self.dim = config['dim']
        self.unit = config['unit']
        self.b = round


    def scale(self, input, scale):
        output_range = scale[1] - scale[0]
        output = ((input + 1) / 2) * output_range + scale[0]
        return output
    
    def specific_round(self, number, decimal):
        if self.b:
            return STE().apply(number / decimal) * decimal
        else:
            return number

    def forward(self, x):
        output = torch.ones_like(x)
        index1 = 0
        index2 = 0

        for i in range(len(self.dim)):
            for _ in range(self.dim[i]):
                output[:, index1] = self.scale(x[:, index1], self.scale_range[i])
                index1 += 1
            output[:, index2:index1] = self.specific_round(output[:, index2:index1], self.unit[i])
            index2 = index1
        return output

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scaling(x, config):
    scale_range = config['scale']
    dim = config['dim']
    unit = config['unit']

    def scale(input, scale):
        output_range = scale[1] - scale[0]
        output = ((input + 1) / 2) * output_range + scale[0]
        return output
    
    def specific_round(number, decimal):
        return np.round(number / decimal) * decimal

    output = np.ones_like(x)
    index1 = 0
    index2 = 0

    for i in range(len(dim)):
        for _ in range(dim[i]):
            output[index1] = scale(x[index1], scale_range[i])
            index1 += 1
        output[index2:index1] = specific_round(output[index2:index1], unit[i])
        index2 = index1
	
    return output


def main():
	Max_train_steps = 10000
	ep_len = 10
	gamma = 0.99
	batch_size = 256
	source_config = {'scale':[[0.18,50], [0.24,100], [0,1.8]], 'dim':[3,3,1], 'unit':[0.1,0.1,0.01]}
	config = {'scale':[[0.03,1], [0.1,2.83], [0,1]], 'dim':[3,3,1], 'unit':[0.03,0.03,0.01]}
	state_dim = 7
	action_space = 7
	a_lr = 3e-4
	c_lr = 3e-4
	alpha = 0.02
	hidden_size = 256
	max_e_steps = 1000 #random exploration

	action_decoder = action_decoder_network(sum(source_config['dim']), sum(config['dim']), hidden_size)
	state_decoder = decoder_network(sum(source_config['dim']), sum(config['dim']), hidden_size, source_config)
	sourceQ =  Double_Q_Critic(sum(source_config['dim']), sum(source_config['dim']),  (hidden_size,hidden_size))
	sourceV = Double_V_Critic(sum(source_config['dim']), hidden_size)

	agent = SAC_countinuous(state_dim=state_dim, action_dim = action_space, net_width=hidden_size,
								a_lr = a_lr, c_lr=c_lr,batch_size=batch_size,gamma=gamma,alpha=alpha,
								state_decoder=state_decoder,action_decoder=action_decoder,sourceQ=sourceQ,sourceV=sourceV,source_config=source_config) # var: transfer argparse to dictionary

	agent.load_source('/home/wzchen/u0810774/UM180FDKMFC_C05_PB/OA/OceanScript/Transfer_ONE2simpleTWO/source_model')

	total_steps = 0
	while total_steps < Max_train_steps:
		state_normal = 2 * np.random.rand(state_dim) #range[-1,1]
		state = scaling(state_normal, config)

		for l in range(10):
			if total_steps < max_e_steps:
				a = 2 * np.random.rand(action_space) - 1 # delta action
				action = state_normal + a #[-1,1] execute action
				action = action.clip(-1, 1)
				action_scale = scaling(action, config)

			else:
				a = agent.select_action(state, deterministic=False)  # delta action
				action = state_normal + a #[-1,1] execute action_
				action = action.clip(-1, 1)
				action_scale = scaling(action, config)

			action_str = convertAction(action_scale)

			logfile = open(logfile_path, "a")
			print(f"action {total_steps} = {action_str}", file = logfile) 
			logfile.close()

			sys.stdout.write(action_str)
			sys.stdout.flush()  
			output = sys.stdin.readline() 

			output = np.array(output.split(','),dtype=np.float64)
			reward = compute_reward(output[:4])

			logfile = open(logfile_path, "a")
			print(f"reward {total_steps} = {reward}", file = logfile) 
			logfile.close()

			logfile = open(logfile_path, "a")
			print(f"output {total_steps} = {output[:4]}", file = logfile) 
			logfile.close()

			if l == ep_len - 1:
				done = True
			else:
				done = False

			agent.replay_buffer.add(state, a, reward, action_scale, done)
			state_normal = action
			state = action_scale
			total_steps += 1
			
			if agent.replay_buffer.__len__() > batch_size:
				agent.train()

		logfile = open(logfile_path, "a")
		print(f"final_reward {total_steps//10} = {reward}", file = logfile) 
		logfile.close()

	agent.save()

def set_seed(seed):
	# For reproducibility, fix the random seed
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


random_seed = 23
logfile = open(logfile_path, "a")
print(f"Seed setting: {random_seed}", file = logfile) 
logfile.close()

set_seed(seed=random_seed)
logfile.close()
main()
sys.stdout.write('exit')  # stdout for Cadence Virtuoso
sys.stdout.flush()
sys.stdout.close()