import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import torch.nn.functional as F
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC  # _RNN as SAC

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


class Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-5,
        max_log_std=2,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net_mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.net_hide = nn.Sequential(
            nn.Linear(36992 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Tanh(),
            nn.Linear(32, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # print(torch.isinf(inputs["states"]["image"]).any())
        # print("value:", inputs["states"]["value"])
        cnn = self.net_cnn(inputs["states"]["image"].view(-1, *self.observation_space["image"].shape).permute(0, 3, 1, 2))
        mlp = self.net_mlp(inputs["states"]["value"])
        hide = self.net_hide(torch.cat([cnn, mlp], dim=1))

        self.log_std_parameter.data = torch.clamp(
            self.log_std_parameter.data, min=-5, max=2
        )

        # print(hide)
        # print(3, torch.isnan(mlp).any())
        # if torch.isnan(hide).any():
        #     print(cnn, mlp)
        #     print(inputs["states"])
        #     raise ValueError("nan")

        return (
            # self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)),
            # self.net(inputs["states"]["image"]),
            hide,
            self.log_std_parameter,
            {},
        )


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_fc = nn.Sequential(
            nn.Linear(57 + self.num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        # print(inputs["states"]["critic"].shape)
        # print(inputs["taken_actions"].shape)
        return self.net_fc(torch.cat([inputs["states"]["critic"], inputs["taken_actions"]], dim=1)), {}


# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-AGV-Direct")
env = wrap_env(env, wrapper="isaaclab-single-agent")

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory_size = 1024
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)


models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


# initialize models' parameters (weights and biases)
# for model in models.values():
#     model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    # model.init_weights(method_name="uniform_", a=-0.1, b=0.1)
    # model.init_parameters("orthogonal_", gain=0.5)
    # model.init_weights(method_name="normal_", mean=0.0, std=0.25)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 128
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 100
cfg["actor_learning_rate"] = 1e-5  # actor learning rate
cfg["critic_learning_rate"] = 1e-5   # critic learning rate
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 300
cfg["experiment"]["checkpoint_interval"] = 100000
cfg["experiment"]["directory"] = "runs/torch/AGV"

agent = SAC(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# agent.load("./runs/torch/AGV/24-09-20_11-23-45-384181_PPO_RNN/checkpoints/agent_200000.pt")

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
with torch.autograd.set_detect_anomaly(True):
    trainer.train()
