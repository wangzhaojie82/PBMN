
import numpy as np



# attacker's theoretical relative reward in SM1
def SM1_theoretical_gain(alpha, gamma):
    '''
    :param a: attacker's mining power fraction
    :param gamma: follow fraction
    :return:
    '''
    a = alpha
    rate = (a * (1 - a) * (1 - a) * (4.0 * a + gamma * (1 - 2 * a)) - np.power(a, 3)) / (1 - a * (1 + (2 - a) * a))
    # rate = a
    return round(rate, 4)


# Agent's theoretical relative reward if he becomes honestly
def Agent_theoreticle_gain(alpha, beta, gamma):
    '''
    :param alpha: attacker's mining power
    :param beta: agent's mining power
    :param gamma: follow fraction
    :return:
    '''
    SM1_gain = SM1_theoretical_gain(alpha, gamma)
    fraction = beta / (1 - alpha)
    reward = fraction * (1 - SM1_gain)
    return round(reward, 4)


# other honest miners relative reward
def Other_theoreticle_gain(alpha, beta, gamma):

    SM1_gain = SM1_theoretical_gain(alpha, gamma)
    fraction = (1 - alpha - beta) / (1 - alpha)
    reward = fraction * (1 - SM1_gain)
    return round(reward, 4)


def rela_reward_calculate(total_valid_b, attacker_valid_b, agent_valid_b):
    '''
    :param total_valid_b: total valid blocks
    :param attacker_valid_b: attacker's valid blocks
    :param agent_valid_b:  agent's valid blocks
    :return: attacker_reward, agent_reward, other_reward
    '''

    att_reward = round(attacker_valid_b / total_valid_b, 4)
    agent_reward = round(agent_valid_b / total_valid_b, 4)
    other_reward = round( (total_valid_b - attacker_valid_b - agent_valid_b) / total_valid_b, 4)

    return att_reward, agent_reward, other_reward

