# The environment containing a selfish attacker and honest miners
#
# 1) the attacker follows the SM1 strategy,
# 2) attacker's mining power = \alpha,
# 3) attacker always tries to launch an attack with probability \alpha (i.e. a rush adversary)
#
#
# 4) honest miners' total mining power = 1- \alpha - \beta
# 5) honest miners follow the longest chain always
# 6) if a fork exists, honest miners follow one chain randomly
#

import numpy as np


class Environment:

    # max_hidden_block : limit the max hidden block of attacker
    # attacker_power : usually denoted as alpha, the mining power of the attacker against the whole network
    # agent_power: denoted as beta, the mining power of agent (pool)
    # follower_fraction : usually denoted as gamma, the follower's fraction against the honest miners (not containing pool)
    # rule : "longest" -- bitcoin typical rule;
    # cost : simulate the cost mining a block, such as the electric cost or others
    #      : cost represents the factor \lambda >= 0
    #      : cost == 0 means no electric cost during mining process


    def __init__(self, total_rounds, attacker_power, agent_power, follower_fraction, cost=0, max_hidden_block = 20, rule = "longest"):
        self.round = 0  # current round
        self.total_rounds = total_rounds # total rounds in one episode
        self.illegal_action_count = 0

        # self.__max_hidden_block = max_hidden_block
        self.alpha = attacker_power
        self.beta = agent_power
        self.gamma = follower_fraction

        self.total_valid_blocks = 0 # in this episode, total valid blocks
        self.agent_valid_blocks = 0 # in this episode, the blocks valid by agent
        self.attacker_valid_blocks = 0 # in this episode, the blocks valid by attacker

        self.current_alpha = self.alpha # attacker's current power (maybe adjust by attacker during mining)
        self.current_beta = self.beta # agent's current power (maybe adjust during mining)

        self.rule = rule # rule='longest' : Bitcoin longest chain rule
        self.cost = cost # electricity cost factor, denoted by \lambda in paper

        self.state_vector_n = 4  # the dimension of state


        # the global view of blockchain (used to calculate reward)
        self.current_global_state = (0, 0, 0, 0)

        self.action_space_n = 4 # dimension of action space: mine, stop, follow_a, follow_b
        self.action_space = np.arange(0,4) # 0: 'mine', 1: 'stop', 2: 'follow_a', 3: 'follow_b']

        self.act_stop_count = 0 # count the number which action 'stop' is taken



    def reset(self):
        '''
        reset the environment, especially the global state and observation
        :return:
        '''
        self.current_alpha = self.alpha
        self.current_beta = self.beta
        self.current_global_state = (0, 0, 0, 0)
        # self.__current_observed_state = (0, 0, 0, 'normal')
        self.round = 0

        self.illegal_action_count = 0
        self.total_valid_blocks = 0
        self.agent_valid_blocks = 0
        self.attacker_valid_blocks = 0
        self.act_stop_count = 0 # count the number which action 'stop' is taken

        observed_state = (0, 0, 0, 0)
        return observed_state



    def step(self, observed_state, action):
        '''
        :param observed_state: the state observed by agent
        :param action: the action taken by agent
        :return: state_, reward, done, action_
                state_ : next state
                reward : instant reward for agent
                done : episode end flag (whether this episode finish)
                action_ : actual action taken on env (if param 'action' is illegal,
                            then choose the next legal action in the action space)
        '''
        state_, reward, done, event = self.step_imp(observed_state, action)

        return state_, reward, done, event



    def step_imp(self, observed_state, action):

        # current global environment state
        a_global, h_global, pending_global, fork_global = self.current_global_state[0:4]

        # current observation of agent
        a_observed, h_observed, pending_observed, fork_observed = observed_state[0:4]

        # default value (return to agent)
        observed_state_new = (-1, -1, -1, -1)
        reward = -1 # block reward (return to agent)
        done = False
        event = -1


        # a = h = 0 (global)
        if(a_global == 0 and h_global == 0):
            # for state (0, 0, 'normal'), the global state and observation should be same

            #if(action == 'mine'):
            if(action == 0):
                prob_arr = [self.current_alpha, self.current_beta, 1 - self.current_alpha - self.current_beta]
                # mining event:
                #       0 : attacker mines a new block, and follows the SM1 strategy
                #       1 : agent mines a new block
                #       2 : other honest mine a new block
                mining_event = np.random.choice([0,1,2], p=prob_arr)
                event = mining_event

                if(mining_event == 0):
                    self.current_global_state = (1, 0, 0, 2)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0 - (self.current_beta * self.cost)

                elif(mining_event == 1):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 1 - (self.current_beta * self.cost)
                    self.agent_valid_blocks += 1
                    self.total_valid_blocks += 1

                else:
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0 - (self.current_beta * self.cost)
                    self.total_valid_blocks += 1

            #elif(action == 'stop'):
            elif(action == 1):

                self.act_stop_count += 1

                # agent do not mine, recalculate the mining power fraction
                prob_arr = [self.current_alpha/(1- self.current_beta),
                            (1 - self.current_alpha - self.current_beta)/(1- self.current_beta)]
                # mining event:
                #       0 : attacker mines a new block, and follows the SM1 strategy
                #       1 : other honest mine a new block
                mining_event = np.random.choice([0, 1], p=prob_arr)
                event = mining_event

                if (mining_event == 0):
                    self.current_global_state = (1, 0, 0, 2)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0

                else:
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0
                    self.total_valid_blocks += 1

            else:
                # if agent chooses an illegal action, the environment does not transit,
                # then return a very small reward, force agent does not choose the illegal action
                # print('Illegal action is chosen.')
                reward = -10000
                done = False
                event = -1
                self.illegal_action_count += 1
                return observed_state, reward, done, event


        # a = h > 0 (global)
        elif(a_global == h_global and h_global > 0):
            # for state (a, h, pending, 'forking'), the global state and observation should be same
            if(action == 1):
                self.act_stop_count += 1

                prob_arr = [self.current_alpha/(1- self.current_beta),
                            ((1 - self.current_alpha - self.current_beta)/(1- self.current_beta))*self.gamma,
                            ((1 - self.current_alpha - self.current_beta)/(1- self.current_beta))*(1-self.gamma)]
                # mining event:
                #       0 : attacker mines a new block, and follows the SM1 strategy
                #       1 : other honest follow the attacker's branch
                #       2 : other honest follow the honest branch
                mining_event = np.random.choice([0, 1, 2], p=prob_arr)
                event = mining_event

                if (mining_event == 0):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0
                    self.total_valid_blocks += (a_global+1)
                    self.attacker_valid_blocks += (a_global+1)


                elif(mining_event == 1):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0
                    self.total_valid_blocks += (a_global + 1)
                    self.attacker_valid_blocks += a_global


                else:
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0 + pending_global
                    self.total_valid_blocks += (h_global+1)
                    self.agent_valid_blocks += pending_global

            elif(action == 2):
                prob_arr = [self.current_alpha, self.current_beta, (1 - self.current_alpha - self.current_beta) * self.gamma,
                            (1 - self.current_alpha - self.current_beta) * (1 - self.gamma)]
                # mining event:
                #       0 : attacker mines a new block, and follows the SM1 strategy
                #       1 : agent mines a new block and follows the attacker's branch
                #       2 : other honest mine a new block and follow the attacker's branch
                #       3 : other honest mine a new block and follow the honest branch
                mining_event = np.random.choice([0, 1, 2, 3], p=prob_arr)
                event = mining_event

                if(mining_event == 0):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0 - (self.current_beta * self.cost)
                    self.total_valid_blocks += (a_global + 1)
                    self.attacker_valid_blocks += (a_global + 1)

                elif(mining_event == 1):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 1 - (self.current_beta * self.cost)
                    self.total_valid_blocks += (a_global + 1)
                    self.attacker_valid_blocks += a_global
                    self.agent_valid_blocks += 1

                elif(mining_event == 2):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0 - (self.current_beta * self.cost)
                    self.total_valid_blocks += (a_global + 1)
                    self.attacker_valid_blocks += a_global

                else:
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = pending_global - (self.current_beta * self.cost)
                    self.total_valid_blocks += (h_global + 1)
                    self.agent_valid_blocks += pending_global

            elif(action == 3):
                prob_arr = [self.current_alpha, self.current_beta,
                            (1 - self.current_alpha - self.current_beta) * self.gamma,
                            (1 - self.current_alpha - self.current_beta) * (1 - self.gamma)]
                # mining event:
                #       0 : attacker mines a new block, and follows the SM1 strategy
                #       1 : agent mines a new block and follows the honest branch
                #       2 : other honest mine a new block and follow the attacker's branch
                #       3 : other honest mine a new block and follow the honest branch
                mining_event = np.random.choice([0, 1, 2, 3], p=prob_arr)
                event = mining_event

                if (mining_event == 0):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0 - (self.current_beta * self.cost)
                    self.total_valid_blocks += (a_global + 1)
                    self.attacker_valid_blocks += (a_global + 1)

                elif (mining_event == 1):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 1 + pending_global - (self.current_beta * self.cost)
                    self.total_valid_blocks += (h_global + 1)
                    self.agent_valid_blocks += pending_global + 1

                elif (mining_event == 2):
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = 0 - (self.current_beta * self.cost)
                    self.total_valid_blocks += (a_global + 1)
                    self.attacker_valid_blocks += a_global

                else:
                    self.current_global_state = (0, 0, 0, 0)
                    observed_state_new = (0, 0, 0, 0)
                    reward = pending_global - (self.current_beta * self.cost)
                    self.total_valid_blocks += (h_global + 1)
                    self.agent_valid_blocks += pending_global

            else:
                # if agent chooses an illegal action, the environment does not transit,
                # then return a very small reward, force agent does not choose the illegal action
                # print('Illegal action is chosen.')
                reward = -10000
                done = False
                event = -1
                self.illegal_action_count +=1
                return observed_state, reward, done, event

        # a > h and h = 0 (global)
        elif(a_global > h_global and h_global == 0 ):
            if(observed_state != (0, 0, 0, 0)):
                print('error! global state is inconsistent with observation. ')
                print('global state : ' + str(self.current_global_state))
                print('observation : ' + str(observed_state))

            if(a_global == 1):
                if (action == 0):
                    prob_arr = [self.current_alpha, self.current_beta,
                                1 - self.current_alpha - self.current_beta]
                    # mining event:
                    #       0 : attacker mines a new block, and follows the SM1 strategy
                    #       1 : agent mines a new block
                    #       2 : other honest mine a new block
                    mining_event = np.random.choice([0, 1, 2], p=prob_arr)
                    event = mining_event

                    if (mining_event == 0):
                        self.current_global_state = (2, 0, 0, 2)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)

                    elif (mining_event == 1):
                        self.current_global_state = (1, 1, 1, 1)
                        observed_state_new = (1, 1, 1, 1)
                        reward = 0 - (self.current_beta * self.cost)

                    else:
                        self.current_global_state = (1, 1, 0, 1)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0 - (self.current_beta * self.cost)

                elif (action == 1):
                    self.act_stop_count += 1

                    prob_arr = [self.current_alpha/(1- self.current_beta), (1 - self.current_alpha - self.current_beta)/(1- self.current_beta)]
                    # mining event:
                    #       0 : attacker mines a new block, and follows the SM1 strategy
                    #       1 : honest mine a new block
                    mining_event = np.random.choice([0, 1], p=prob_arr)
                    event = mining_event

                    if (mining_event == 0):
                        self.current_global_state = (2, 0, 0, 2)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0

                    else:
                        self.current_global_state = (1, 1, 0, 1)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0

                else:
                    # if agent chooses an illegal action, the environment does not transit,
                    # then return a very small reward, force agent does not choose the illegal action
                    # print('Illegal action is chosen.')
                    reward = -10000
                    done = False
                    event = -1
                    self.illegal_action_count += 1
                    return observed_state, reward, done, event

            elif(a_global == 2):
                if(action == 0):
                    prob_arr = [self.current_alpha, self.current_beta,
                                1 - self.current_alpha - self.current_beta]
                    # mining event:
                    #       0 : attacker mines a new block, and follows the SM1 strategy
                    #       1 : agent mines a new block
                    #       2 : other honest mine a new block
                    mining_event = np.random.choice([0, 1, 2], p=prob_arr)
                    event = mining_event

                    if (mining_event == 0):
                        self.current_global_state = (3, 0, 0, 2)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)

                    elif (mining_event == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += 2
                        self.attacker_valid_blocks += 2

                    else:
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += 2
                        self.attacker_valid_blocks += 2

                elif(action == 1):
                    self.act_stop_count += 1

                    prob_arr = [self.current_alpha/(1- self.current_beta), (1 - self.current_alpha - self.current_beta)/(1- self.current_beta)]
                    # mining event:
                    #       0 : attacker mines a new block, and follows the SM1 strategy
                    #       1 : other honest mines a new block

                    mining_event = np.random.choice([0, 1], p=prob_arr)
                    event = mining_event

                    if (mining_event == 0):
                        self.current_global_state = (3, 0, 0, 2)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0

                    else:
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0
                        self.total_valid_blocks += 2
                        self.attacker_valid_blocks += 2

                else:
                    # if agent chooses an illegal action, the environment does not transit,
                    # then return a very small reward, force agent does not choose the illegal action
                    # print('Illegal action is chosen.')
                    reward = -10000
                    done = False
                    event = -1
                    self.illegal_action_count += 1

                    return observed_state, reward, done, event

            # a_global >= 3
            else:
                if (action == 0):
                    prob_arr = [self.current_alpha, self.current_beta,
                                1 - self.current_alpha - self.current_beta]
                    # mining event:
                    #       0 : attacker mines a new block, and follows the SM1 strategy
                    #       1 : agent mines a new block
                    #       2 : other honest mine a new block
                    mining_event = np.random.choice([0, 1, 2], p=prob_arr)
                    event = mining_event

                    if (mining_event == 0):
                        self.current_global_state = (a_global+1, 0, 0, 2)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)

                    elif (mining_event == 1):
                        self.current_global_state = (a_global, 1, 1, 2)
                        observed_state_new = (1, 1, 1, 1)
                        reward = 0 - (self.current_beta * self.cost)

                    else:
                        self.current_global_state = (a_global, 1, 1, 2)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0 - (self.current_beta * self.cost)

                elif(action == 1):
                    self.act_stop_count += 1

                    prob_arr = [self.current_alpha/(1- self.current_beta), (1 - self.current_alpha - self.current_beta)/(1- self.current_beta)]
                    # mining event:
                    #       0 : attacker mines a new block, and follows the SM1 strategy
                    #       1 : other honest mines a new block

                    mining_event = np.random.choice([0, 1], p=prob_arr)
                    event = mining_event

                    if (mining_event == 0):
                        self.current_global_state = (a_global+1, 0, 0, 2)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0

                    else:
                        self.current_global_state = (a_global, 1, 0, 2)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0

                else:
                    # if agent chooses an illegal action, the environment does not transit,
                    # then return a very small reward, force agent does not choose the illegal action
                    # print('Illegal action is chosen.')
                    reward = -10000
                    done = False
                    event = -1
                    self.illegal_action_count += 1

                    return observed_state, reward, done, event

        # a > h and h > 0 (global)
        elif(a_global > h_global and h_global > 0 ):
            if(fork_observed != 1):
                print('error! for case: a_global > h_global and h_global > 0 ')
                print('observation should be (h, h, pending, forking)')

            if(action == 1):
                self.act_stop_count += 1

                prob_arr = [self.current_alpha/(1- self.current_beta), ((1 - self.current_alpha - self.current_beta)/(1- self.current_beta)) * self.gamma,
                            ((1 - self.current_alpha - self.current_beta)/(1- self.current_beta)) * (1 - self.gamma)]
                # mining event:
                #       0 : attacker mines a new block, and follows the SM1 strategy
                #       1 : other honest follow the attacker's branch
                #       2 : other honest follow the honest branch
                mining_event = np.random.choice([0, 1, 2], p=prob_arr)
                event = mining_event

                if (mining_event == 0):
                    self.current_global_state = (a_global+1, h_global, pending_global, fork_global)
                    observed_state_new = observed_state
                    reward = 0

                elif (mining_event == 1):
                    if((a_global-h_global-1) == 0):
                        self.current_global_state = ( 1, 1, 0, 1)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0
                        self.total_valid_blocks += h_global
                        self.attacker_valid_blocks += h_global

                    elif((a_global-h_global-1) == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0
                        self.total_valid_blocks += a_global
                        self.attacker_valid_blocks += a_global
                    else:
                        self.current_global_state = (a_global-h_global, 1, 0, 2)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0
                        self.total_valid_blocks += h_global
                        self.attacker_valid_blocks += h_global

                else:
                    if ((a_global - h_global - 1) == 0):
                        self.current_global_state = (h_global+1, h_global+1, pending_global, 1)
                        observed_state_new = (h_global+1,  h_global+1, pending_global, 1)
                        reward = 0

                    elif ((a_global - h_global - 1) == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0
                        self.total_valid_blocks += a_global
                        self.attacker_valid_blocks += a_global
                    else:
                        self.current_global_state = (a_global, h_global+1, pending_global, 2)
                        observed_state_new = (h_global+1, h_global+1, pending_observed, 1)
                        reward = 0

            elif(action == 2):
                prob_arr = [self.current_alpha, self.current_beta, (1 - self.current_alpha - self.current_beta) * self.gamma,
                            (1 - self.current_alpha - self.current_beta) * (1 - self.gamma)]
                # mining event:
                #       0 : attacker mines a new block, and follows the SM1 strategy
                #       1 : agent mines a block and follows the attacker's branch
                #       2 : other honest follow the attacker's branch
                #       3 : other honest follow the honest branch
                mining_event = np.random.choice([0, 1, 2, 3], p=prob_arr)
                event = mining_event

                if(mining_event == 0):
                    self.current_global_state = (a_global + 1, h_global, pending_global, fork_global)
                    observed_state_new = observed_state
                    reward = 0 - (self.current_beta * self.cost)

                elif(mining_event == 1):
                    if ((a_global - h_global - 1) == 0):
                        self.current_global_state = (1, 1, 1, 1)
                        observed_state_new = (1, 1, 1, 1)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += h_global
                        self.attacker_valid_blocks += h_global

                    elif ((a_global - h_global - 1) == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += a_global
                        self.attacker_valid_blocks += a_global
                    else:
                        self.current_global_state = (a_global - h_global, 1, 1, 2)
                        observed_state_new = (1, 1, 1, 1)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += h_global
                        self.attacker_valid_blocks += h_global

                elif(mining_event == 2):
                    if ((a_global - h_global - 1) == 0):
                        self.current_global_state = (1, 1, 0, 1)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += h_global
                        self.attacker_valid_blocks += h_global

                    elif ((a_global - h_global - 1) == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += a_global
                        self.attacker_valid_blocks += a_global
                    else:
                        self.current_global_state = (a_global - h_global, 1, 0, 2)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += h_global
                        self.attacker_valid_blocks += h_global

                else:
                    if ((a_global - h_global - 1) == 0):
                        self.current_global_state = (h_global + 1, h_global + 1, pending_global, 1)
                        observed_state_new = (h_global + 1, h_global + 1, pending_global, 1)
                        reward = 0 - (self.current_beta * self.cost)

                    elif ((a_global - h_global - 1) == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += a_global
                        self.attacker_valid_blocks += a_global
                    else:
                        self.current_global_state = (a_global, h_global + 1, pending_global, 2)
                        observed_state_new = (h_global + 1, h_global + 1, pending_observed, 1)
                        reward = 0 - (self.current_beta * self.cost)

            elif(action == 3):
                prob_arr = [self.current_alpha, self.current_beta,
                            (1 - self.current_alpha - self.current_beta) * self.gamma,
                            (1 - self.current_alpha - self.current_beta) * (1 - self.gamma)]
                # mining event:
                #       0 : attacker mines a new block, and follows the SM1 strategy
                #       1 : agent mines a block and follows the attacker's branch
                #       2 : other honest follow the attacker's branch
                #       3 : other honest follow the honest branch
                mining_event = np.random.choice([0, 1, 2, 3], p=prob_arr)
                event = mining_event

                if(mining_event == 0):
                    self.current_global_state = (a_global + 1, h_global, pending_global, fork_global)
                    observed_state_new = observed_state
                    reward = 0 - (self.current_beta * self.cost)

                elif(mining_event == 1):
                    if ((a_global - h_global - 1) == 0):
                        self.current_global_state = (a_global, a_global, pending_global+1, 1)
                        observed_state_new = (a_global, a_global, pending_global+1, 1)
                        reward = 0 - (self.current_beta * self.cost)

                    elif ((a_global - h_global - 1) == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += a_global
                        self.attacker_valid_blocks += a_global
                    else:
                        self.current_global_state = (a_global, h_global+1, pending_global+1, 2)
                        observed_state_new = (h_global+1, h_global+1, pending_global+1, 1)
                        reward = 0 - (self.current_beta * self.cost)

                elif(mining_event == 2):
                    if ((a_global - h_global - 1) == 0):
                        self.current_global_state = (1, 1, 0, 1)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += h_global
                        self.attacker_valid_blocks += h_global

                    elif ((a_global - h_global - 1) == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += a_global
                        self.attacker_valid_blocks += a_global
                    else:
                        self.current_global_state = (a_global - h_global, 1, 0, 2)
                        observed_state_new = (1, 1, 0, 1)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += h_global
                        self.attacker_valid_blocks += h_global

                else:
                    if ((a_global - h_global - 1) == 0):
                        self.current_global_state = (h_global + 1, h_global + 1, pending_global, 1)
                        observed_state_new = (h_global + 1, h_global + 1, pending_global, 1)
                        reward = 0 - (self.current_beta * self.cost)

                    elif ((a_global - h_global - 1) == 1):
                        self.current_global_state = (0, 0, 0, 0)
                        observed_state_new = (0, 0, 0, 0)
                        reward = 0 - (self.current_beta * self.cost)
                        self.total_valid_blocks += a_global
                        self.attacker_valid_blocks += a_global
                    else:
                        self.current_global_state = (a_global, h_global + 1, pending_global, 2)
                        observed_state_new = (h_global + 1, h_global + 1, pending_observed, 1)
                        reward = 0 - (self.current_beta * self.cost)

            else:
                # if agent chooses an illegal action, the environment does not transit,
                # then return a very small reward, force agent does not choose the illegal action
                # print('Illegal action is chosen.')
                reward = -10000
                done = False
                event = -1
                self.illegal_action_count +=1

                return observed_state, reward, done, event


        self.round += 1
        if(self.round == self.total_rounds):
            done = True


        return observed_state_new, reward, done, event


