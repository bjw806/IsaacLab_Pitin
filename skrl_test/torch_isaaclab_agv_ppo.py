import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # (512x512) -> (127x127)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (127x127) -> (62x62)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (62x62) -> (60x60)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 512),  # 9216 / 230400
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return (
            self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)),
            self.log_std_parameter,
            {},
        )


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # (512x512) -> (127x127)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (127x127) -> (62x62)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (62x62) -> (60x60)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 512),  # 9216 / 230400
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def compute(self, inputs, role):
        x = self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2))
        return x, {}


# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-AGV-Direct")
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 20000
cfg["learning_epochs"] = 100
cfg["mini_batches"] = 100
cfg["discount_factor"] = 0.9995
cfg["lambda"] = 0.95
cfg["policy_learning_rate"] = 2.5e-4
cfg["value_learning_rate"] = 2.5e-4
# cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
# cfg["random_timesteps"] = 1000
cfg["learning_rate_scheduler"] = torch.optim.lr_scheduler.StepLR
cfg["learning_rate_scheduler_kwargs"] = {"step_size": 10000, "gamma": 0.5}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 300
cfg["experiment"]["checkpoint_interval"] = 100000
cfg["experiment"]["directory"] = "runs/torch/AGV"

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
