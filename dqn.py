
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import env
import util
import statistics



# hyper parameters
BATCH_SIZE = 32 # batch size of network training
LEARNING_RATE = 0.01 # learning rate
EPSILON = 0.9 # greedy policy
GAMMA = 0.9 # discount factor
TARGET_REPLACE_ITER = 100 # target network update frequency
MEMORY_CAPACITY = 2000 # experience buffer size
EPISODE_NUM = 100 # number of episode used to training the DQN


# mining parameters
TOTAL_ROUNDS = 10000
ALPHA = 0.3
AGENT_POWER = 0.3
FOLLOW_FRACTION = 0.5

# If not allow stop, then the cost factor should be 0
# intuitively, set the cost factor > 0 is meaningless (agent can not stop even the cost is very large)
#
# But if allow stop, we should make the cost_factor > 0, since
# the agent may consider stop due to the mining cost
COST_FACTOR = 0.1





#N_ACTIONS = env.action_space_n # size of action space
N_ACTIONS = 3 # size of action space
#N_STATES = env.state_vector_n # dimension of state
N_STATES = 4 # dimension of state


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        # self.fc2 = nn.Linear(50, 50)
        # self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        actions_value = self.out(x)
        return actions_value


class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action



    def choose_action_by_target_net(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.target_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action




    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def flush_buffer(self):

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))


def training_op(env, model_path):
    '''
    :param env:  an environment
    :return:
    '''

    EPISODE_NUM = 100

    dqn = DQN()

    action_stop_count = []

    for i in range(EPISODE_NUM):  # episode
        print('<<<<<<<<< Episode: %s' % i)
        s = env.reset()

        dqn.flush_buffer()

        while True:

            a = dqn.choose_action(s)
            s_, r, done, info = env.step(s, a)
            dqn.store_transition(s, a, r, s_)

            s = s_


            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            if done:

                print('>>>>>>>>>>> Episode %s finish: \n' % i)

                rela_att, rela_agent, rela_other = util.rela_reward_calculate(env.total_valid_blocks, env.attacker_valid_blocks, env.agent_valid_blocks)


                break  # 该episode结束


    torch.save(dqn, model_path)




def dqn_training2():



    alphas = [0.3, 0.35, 0.4]

    betas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]


    for alpha in alphas:
        for beta in betas:

            env = env.Environment(total_rounds=TOTAL_ROUNDS, attacker_power=alpha, agent_power=beta, follower_fraction=FOLLOW_FRACTION, cost=0)
            model_path = './Model/new_test/flush_buffer/dqn_'+str(EPISODE_NUM) + 'episodes_alpha' + str(alpha) + '_beta' + str(beta)+ '_.pt'
            training_op(env, model_path)



if __name__ == '__main__':
    # training_op()

    dqn_training2()

