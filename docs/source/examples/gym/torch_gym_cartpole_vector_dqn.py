import gym

# import the skrl components to build the RL system
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import Shape, deterministic_model


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# load and wrap the gym environment.
# note: the environment version may change depending on the gym version
try:
    env = gym.vector.make("CartPole-v0", num_envs=5, asynchronous=False)
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("CartPole-v")][0]
    print("CartPole-v0 not found. Trying {}".format(env_id))
    env = gym.vector.make(env_id, num_envs=5, asynchronous=False)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=200000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators) using the model instantiator utility.
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#models
models = {}
models["q_network"] = deterministic_model(observation_space=env.observation_space,
                                          action_space=env.action_space,
                                          device=device,
                                          clip_actions=False,
                                          network=[{
                                              "name": "net",
                                              "input": "STATES",
                                              "layers": [64, 64],
                                              "activations": "relu",
                                          }],
                                          output="ACTIONS")
models["target_q_network"] = deterministic_model(observation_space=env.observation_space,
                                                 action_space=env.action_space,
                                                 device=device,
                                                 clip_actions=False,
                                                 network=[{
                                                     "name": "net",
                                                     "input": "STATES",
                                                     "layers": [64, 64],
                                                     "activations": "relu",
                                                 }],
                                                 output="ACTIONS")

# initialize models' lazy modules
for role, model in models.items():
    model.init_state_dict(role)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 100
cfg["exploration"]["final_epsilon"] = 0.04
cfg["exploration"]["timesteps"] = 1500
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/CartPole"

agent = DQN(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
