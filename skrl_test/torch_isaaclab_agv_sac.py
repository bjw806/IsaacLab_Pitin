import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import timm

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
from skrl.utils.spaces.torch import unflatten_tensorized_space

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

#     'vit_small_patch14_dinov2.lvd142m',
model = timm.create_model(
    "facebook/sam2.1-hiera-tiny", #'mobilenetv4_hybrid_medium.ix_e550_r384_in1k',
    pretrained=True,
    num_classes=0,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)


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

        # self.net_cnn = timm.models.eva.eva02_base_patch14_448(pretrained=True)
        self.net_cnn = model
        
        self.net_mlp = nn.Sequential(
            nn.Linear(39, 16),
            nn.ELU(),
            nn.Linear(16, 4),
            nn.ELU(),
        )
        self.net_hide = nn.Sequential(
            nn.Linear(384 + 39, 128),
            nn.ELU(),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, self.num_actions),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        image = states["image"].view(-1, *self.observation_space["image"].shape).permute(0, 3, 1, 2)
        cnn = self.net_cnn(transforms(image))

        # print(cnn.shape)
        # self.plot_layer_outputs(cnn.to(torch.device('cpu')))
        # mlp = self.net_mlp(states["value"])
        hide = self.net_hide(torch.cat([cnn, states["value"]], dim=1))

        return (
            hide,
            self.log_std_parameter,
            {},
        )
    
    def plot_layer_outputs(self, image, num_channels_to_show=5):
        unflattened = image.view(-1, 128, 24, 24).detach().numpy()[0] 
        mean_output = np.mean(unflattened, axis=0)
        
        fig, axes = plt.subplots(1, num_channels_to_show + 1, figsize=(20, 5))
        axes[0].imshow(mean_output, cmap='gray')
        axes[0].set_title('Mean of Channels')

        for i in range(1, num_channels_to_show + 1):
            axes[i].imshow(unflattened[i-1], cmap='gray')
            axes[i].set_title(f'Channel {i}')
        
        plt.tight_layout()
        plt.savefig('cnn_layer_outputs.png')


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_mlp = nn.Sequential(
            nn.Linear(63 + self.num_actions, 16),
            nn.ELU(),
            nn.Linear(16, 1),
        )

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        taken_actions = inputs["taken_actions"]
        i = torch.cat([states["critic"], taken_actions], dim=1)
        mlp = self.net_mlp(i)

        return (
            mlp,
            {},
        )

# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-AGV-Direct")
env = wrap_env(env, wrapper="isaaclab-single-agent")

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
gb = 2
memory_size = 1024 * gb
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=torch.device('cpu'))


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
cfg["batch_size"] = 32
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-4
cfg["critic_learning_rate"] = 1e-5
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 3000
cfg["grad_norm_clip"] = 1.0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-4
cfg["initial_entropy_value"] = 0.9
# cfg["target_entropy"] = 0.98 * np.array(-np.log(1.0 / 3), dtype=np.float32)

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
from torch.cuda.amp import autocast

# start training
with autocast():
    trainer.train()
