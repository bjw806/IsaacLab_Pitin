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
from skrl.utils.spaces.torch import unflatten_tensorized_space
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
            nn.Conv2d(3, 16, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        self.net_local = nn.Sequential(
            nn.Linear(3200, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10), # x, y, z, width, height x2
        )

        self.net_hide = nn.Sequential(
            nn.Linear(42, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, self.num_actions),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        # cnn = self.net_cnn(states["image"].view(-1, *self.observation_space["image"].shape).permute(0, 3, 1, 2))
        # hide = self.net_hide(torch.cat([cnn, states["value"], self.net_local(cnn)], dim=1))
        hide = self.net_hide(states["value"])

        return (
            hide,
            self.log_std_parameter,
            {},
        )
    
    def mask(self):
        from ultralytics import SAM
        model = SAM("models/sam2_t.pt")
        mask = model("sam_test.jpg")[0].masks


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_mlp = nn.Sequential(
            nn.Linear(85, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        mlp = self.net_mlp(states["critic"])

        return (
            mlp,
            {},
        )


# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-AGV-Direct")
env = wrap_env(env, wrapper="isaaclab-single-agent")

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
replay_buffer_size = 1024 * 3
memory_size = int(replay_buffer_size / env.num_envs)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)


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
cfg["rollouts"] = memory_size
cfg["learning_epochs"] = 16
cfg["mini_batches"] = 256
# cfg["discount_factor"] = 0.9995
# cfg["lambda"] = 0.95
cfg["policy_learning_rate"] = 2.5e-4
cfg["value_learning_rate"] = 2.5e-4
cfg["grad_norm_clip"] = 1.0
# cfg["ratio_clip"] = 0.2
# cfg["value_clip"] = 0.2
# cfg["clip_predicted_values"] = False
# cfg["entropy_loss_scale"] = 0.0
# cfg["value_loss_scale"] = 0.5
# cfg["kl_threshold"] = 0
# cfg["random_timesteps"] = 3000
# cfg["learning_rate_scheduler"] = torch.optim.lr_scheduler.StepLR
# cfg["learning_rate_scheduler_kwargs"] = {"step_size": 10000, "gamma": 0.5}

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
cfg_trainer = {"timesteps": 10000000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
with autocast():
    trainer.train()
