import torch
import torch.nn as nn

from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC
from skrl.utils.spaces.torch import unflatten_tensorized_space

set_seed(42)


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

        self.net_features = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.net_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, self.num_actions),
            nn.Tanh()
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        image = states["image"].view(-1, *self.observation_space["image"].shape)
        features = self.net_features(image)
        fc = self.net_fc(features)

        return (
            fc,
            self.log_std_parameter,
            {},
        )


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_features = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.net_mlp = nn.Sequential(
            nn.Linear(512 + 12 + self.num_actions, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
        )

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        image = states["image"].view(-1, *self.observation_space["image"].shape)
        taken_actions = inputs["taken_actions"]
        i = torch.cat([self.net_features(image), states["critic"], taken_actions], dim=1)
        mlp = self.net_mlp(i)

        return (
            mlp,
            {},
        )

# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-AGV-Direct")
env = wrap_env(env, wrapper="isaaclab-single-agent")

device = env.device

replay_buffer_size = 1024 * 512
memory_size = int(replay_buffer_size / env.num_envs)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)


models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 1024
cfg["discount_factor"] = 0.98
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-5
cfg["critic_learning_rate"] = 1e-6
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1024 * env.num_envs
cfg["grad_norm_clip"] = 1.0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-5
cfg["initial_entropy_value"] = 0.9
# cfg["target_entropy"] = 0.98 * np.array(-np.log(1.0 / 3), dtype=np.float32)

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

# agent.load("./runs/torch/AGV/24-10-11_17-43-00-133994_SAC/checkpoints/agent_400000.pt")

cfg_trainer = {"timesteps": 1000000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
from torch.cuda.amp import autocast

with autocast():
    trainer.train()
