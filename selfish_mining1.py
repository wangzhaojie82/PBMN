
'''
The implementation of SM1

'''
import random
import time
import sys
import statistics
import numpy as np

class Selfish_Mining:

    def __init__(self, **d):
        self.__nb_simulations = d['nb_simulations']
        self.__delta = 0 # advance of selfish miners on honests'ones
        self.__privateChain = 0 # length of private chain RESET at each validation
        self.__publicChain = 0 # length of public chain RESET at each validation
        self.__honestsValidBlocks = 0 # the number of honest pool's  valid blocks
        self.__selfishValidBlocks = 0 # the number of selfish miner's valid blocks
        self.__counter = 1

        # Setted Parameters
        self.__alpha = d['alpha']
        self.__gamma = d['gamma']

        # self.__publishParam = 0.5

        # For results
        self.__revenue = None
        self.__orphanBlocks = 0
        self.__totalMinedBlocks = 0 # total valid blocks mined in current round simulation

        self.__block_num = d['block_num']



    def Simulate(self):
        '''
        to implement one round simulation
        '''
        while(self.__counter <= self.__nb_simulations):
            # Mining power does not mean the block is actually found
            # there is a probability p to find it
            r = random.uniform(0, 1) # random number for each simulation

            self.__delta = self.__privateChain - self.__publicChain

            if r <= float(self.__alpha):
                self.On_Selfish_Miners() # selfish miner mines a new block
            else:
                self.On_Honest_Miners() # honest pool mine a new block

            ### COPY-PASTE THE 3 LINES BELOW IN THE IF/ELSE TO GET EACH ITERATION RESULTS ###
            #self.actualize_results()
            #print(self)
            #time.sleep(1)
            self.__counter += 1

            self.__totalMinedBlocks = self.__honestsValidBlocks + self.__selfishValidBlocks

            if self.__totalMinedBlocks >= self.__block_num:

                return round((self.__honestsValidBlocks / self.__totalMinedBlocks), 4)



        # Publishing private chain if not empty when total nb of simulations reached
        self.__delta = self.__privateChain - self.__publicChain
        if self.__delta > 0:
            self.__selfishValidBlocks += self.__privateChain
            self.__publicChain, self.__privateChain = 0,0

        print(self)

    def On_Selfish_Miners(self):
        self.__privateChain += 1
        if self.__delta == 0 and self.__privateChain == 2:
            self.__privateChain, self.__publicChain = 0,0
            self.__selfishValidBlocks += 2
            # Publishing private chain reset both public and private chains lengths to 0

    def On_Honest_Miners(self):
        self.__publicChain += 1
        if self.__delta == 0:
            # if 1 block is found => 1 block validated as honest miners take advance
            self.__honestsValidBlocks += 1
            # If there is a competition though (1-1) considering gamma,
            # (Reminder: gamma = ratio of honest miners who choose to mine on pool's block)
            # --> either it appends the private chain => 1 block for each competitor in revenue
            # --> either it appends the honnest chain => 2 blocks for honnest miners (1 more then)
            s = random.uniform(0, 1)
            if self.__privateChain > 0 and s <= float(self.__gamma):
                self.__selfishValidBlocks += 1
            elif self.__privateChain > 0 and s > float(self.__gamma):
                self.__honestsValidBlocks += 1
            #in all cases (append private or public chain) all is reset to 0
            self.__privateChain, self.__publicChain = 0,0

        elif self.__delta == 2:
            self.__selfishValidBlocks += self.__privateChain
            self.__publicChain, self.__privateChain = 0,0



    # Show message
    def __str__(self):
        if self.__counter <= self.__nb_simulations:
            simulation_message = '\nSimulation ' + str(self.__counter) + ' out of ' + str(self.__nb_simulations) + '\n'
            current_stats = 'Private chain : ' + '+ '*int(self.__privateChain) + '\n'\
            'public chain : ' + '+ '*int(self.__publicChain) + '\n'
        else:
            simulation_message = '\n\n' + str(self.__nb_simulations) + ' Simulations Done // publishing private chain if non-empty\n'
            current_stats = ''

        choosen_parameters = 'Alpha : ' + str(self.__alpha) + '\t||\t' +'Gamma : ' + str(self.__gamma) +'\n'


        return simulation_message + current_stats + choosen_parameters



if len(sys.argv) == 1:

    alpha = 0.4

    chain_quality_for_block_num = []
    for block_num in np.arange(1000, 10100, 100):

        chain_quality_list = []
        trails = 100  # number of trials
        for i in range(trails):

            new = Selfish_Mining(**{'nb_simulations': 2000000, 'alpha': alpha, 'gamma': 0.5, 'block_num': block_num})

            chain_quality = new.Simulate()

            chain_quality_list.append(chain_quality)

        average_chain_quality = round(statistics.mean(chain_quality_list), 4)
        print('block num: ' + str(block_num))
        print('Average chain quality: ' + str(average_chain_quality) + '\n')

        chain_quality_for_block_num.append(average_chain_quality)

    print('Params: alpha = ' + str(alpha) )
    print(chain_quality_for_block_num)
