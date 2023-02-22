## The Implementation of ErMine



**Scripts list:**

- env.py: To simulate a Bitcoin mining environment containing selfish attacker and honest miners.

- dqn.py: This script is responsible for instantiating and training a deep Q-network based on the given environment.

- selfish_mining.py: To simulate the selfish mining attack.

- util.py: To claculate the theoretical result of selfish mining attack.



### Basic Usage

Run `dqn.py` to train a batch of deep Q-network models with the given environment and  parameters. Specifically, you should first initialize an environment according to the preset mining parameters, and then use the environment to train a DQN model. When the training process finishes, the model will be saved to the specified path.



You may want to try out different configurations or other blockchain mining strategies. If so, you can rework the function `step_imp()` in `env.py`, and then re-initialize an environment.




