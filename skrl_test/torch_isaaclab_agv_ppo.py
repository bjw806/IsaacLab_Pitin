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
from torch.cuda.amp import autocast

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


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

        self.net_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.Mish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Mish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.net_mlp = nn.Sequential(
            nn.Linear(39, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
        )
        self.net_hide = nn.Sequential(
            nn.Linear(7744 + 16, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        cnn = self.net_cnn(
            inputs["states"]["image"].view(-1, *self.observation_space["image"].shape).permute(0, 3, 1, 2)
        )
        mlp = self.net_mlp(inputs["states"]["value"])
        hide = self.net_hide(torch.cat([cnn, mlp], dim=1))

        if torch.isnan(hide).any():
            raise ValueError("hide")
        # print(hide)
        """
        LeakyRelu
        tensor([[-8.8203,  7.1914,  0.9116]], device='cuda:0', dtype=torch.float16)
        tensor([[-9.0391,  7.1602,  0.8994]], device='cuda:0', dtype=torch.float16)
        tensor([[-9.0391,  7.1133,  0.8794]], device='cuda:0', dtype=torch.float16)
        tensor([[-9.0703,  6.9336,  0.8677]], device='cuda:0', dtype=torch.float16)
        tensor([[-8.9844,  7.1289,  0.9390]], device='cuda:0', dtype=torch.float16)
        tensor([[-9.1484,  7.0156,  0.8950]], device='cuda:0', dtype=torch.float16)
        tensor([[-9.0469,  7.0820,  0.9312]], device='cuda:0', dtype=torch.float16)
        tensor([[-9.0547,  7.0898,  0.9067]], device='cuda:0', dtype=torch.float16)
        """

        return (
            hide,
            self.log_std_parameter,
            {},
        )


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_mlp = nn.Sequential(
            nn.Linear(63, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def compute(self, inputs, role):
        mlp = self.net_mlp(inputs["states"]["critic"])
        if torch.isnan(mlp).any():
            raise ValueError("mlp")
        return (
            mlp,
            {},
        )


# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-AGV-Direct")
env = wrap_env(env, wrapper="isaaclab-single-agent")

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
rollouts = 2048
memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)


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
cfg["rollouts"] = rollouts
cfg["learning_epochs"] = 1024
cfg["mini_batches"] = 512
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
cfg["experiment"]["write_interval"] = 1000
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

# agent.load("./runs/torch/AGV/24-09-25_17-12-11-556727_PPO/checkpoints/agent_100000.pt")

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
with autocast():
    trainer.train()
