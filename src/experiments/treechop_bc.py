"""
Reinforcement Learning agent that trains on MineRLTreechop environment. It is then evaluated on MineRLObtainDiamond by
running it for a certain number of ticks and then switching to the scripted part that crafts a wooden_pickaxe and digs
down to get some cobblestone.
With default parameters it trains in about 8 hours on a machine with a GeForce RTX 2080 Ti GPU.
It uses less than 8GB RAM and achieves an average reward of 8.3.
"""

import time
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from torch.utils.tensorboard import SummaryWriter
import minerl  # it's important to import minerl after SB3, otherwise model.save doesn't work...
import torch as th
import numpy as np
from tqdm import tqdm
import torch as th
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

DATA_DIR = ""
EPOCHS = 5  # How many times we train over the dataset.
LEARNING_RATE = 0.0001  # Learning rate for the neural network.

print(th.version.cuda)
# If you want to try out wandb integration, scroll to the bottom an uncomment line regarding `track_exp`
try:
    wandb = None
    import wandb
except ImportError:
    pass

# Parameters:
config = {
    "TRAIN_TIMESTEPS": 2000000,  # number of steps to train the agent for. At 70 FPS 2m steps take about 8 hours.
    "TRAIN_ENV": 'MineRLTreechop-v0',  # training environment for the RL agent. Could use MineRLObtainDiamondDense-v0 here.
    "TEST_ENV": 'MineRLTreechop-v0',  # evaluation environment for the RL agent.
    "TRAIN_MODEL_NAME": '',  # name to use when saving the trained agent.
    "TEST_MODEL_NAME": 'ppo_bc_treechop',  # name to use when loading the trained agent.
    "STUDENT_POLICY_NAME": 'student_ppo_policy_treechop', # name to use when loading the behavioral cloning policy
    "STUDENT_MODEL_NAME": 'student_ppo_treechop', # name to use when loading the model from which the above policy was taken (needed for SB3)
    "TEST_EPISODES": 100,  # number of episodes to test the agent for.
    "MAX_TEST_EPISODE_LEN": 18000,  # 18k is the default for MineRLObtainDiamond.
    "TREECHOP_STEPS": 4000,  # number of steps to run RL lumberjack for in evaluations.
    "RECORD_TRAINING_VIDEOS": False,  # if True, records videos of all episodes done during training.
    "RECORD_TEST_VIDEOS": False,  # if True, records videos of all episodes done during evaluation.
}
experiment_name = f"ppo_bc_{int(time.time())}"


def make_env(idx):
    def thunk():
        env = gym.make(config["TRAIN_ENV"])
        if idx == 0 and config["RECORD_TRAINING_VIDEOS"]:
            env = gym.wrappers.Monitor(env, f"train_videos/{experiment_name}")
        env = PovOnlyObservation(env)
        env = ActionShaping(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
        return env
    return thunk


def track_exp(project_name=None):
    wandb.init(
        anonymous="allow",
        project=project_name,
        config=config,
        sync_tensorboard=True,
        name=experiment_name,
        monitor_gym=True,
        save_code=True,
    )


class PovOnlyObservation(gym.ObservationWrapper):
    """
    Turns the observation space into POV only, ignoring the inventory. This is needed for stable_baselines3 RL agents,
    as they don't yet support dict observations. The support should be coming soon (as of April 2021).
    See following PR for details:
    https://github.com/DLR-RM/stable-baselines3/pull/243
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']

class ActionShaping(gym.ActionWrapper):
    """
    The default MineRL action space is the following dict:

    Dict(attack:Discrete(2),
         back:Discrete(2),
         camera:Box(low=-180.0, high=180.0, shape=(2,)),
         craft:Enum(crafting_table,none,planks,stick,torch),
         equip:Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         forward:Discrete(2),
         jump:Discrete(2),
         left:Discrete(2),
         nearbyCraft:Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         nearbySmelt:Enum(coal,iron_ingot,none),
         place:Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch),
         right:Discrete(2),
         sneak:Discrete(2),
         sprint:Discrete(2))

    It can be viewed as:
         - buttons, like attack, back, forward, sprint that are either pressed or not.
         - mouse, i.e. the continuous camera action in degrees. The two values are pitch (up/down), where up is
           negative, down is positive, and yaw (left/right), where left is negative, right is positive.
         - craft/equip/place actions for items specified above.
    So an example action could be sprint + forward + jump + attack + turn camera, all in one action.

    This wrapper makes the action space much smaller by selecting a few common actions and making the camera actions
    discrete. You can change these actions by changing self._actions below. That should just work with the RL agent,
    but would require some further tinkering below with the BC one.
    """
    def __init__(self, env, camera_angle=10, always_attack=True):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            # [('back', 1)],
            # [('left', 1)],
            # [('right', 1)],
            # [('jump', 1)],
            # [('forward', 1), ('attack', 1)],
            # [('craft', 'planks')],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
            #[('place', 'dirt')]
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.no_op()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]


def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size,), dtype=np.int)

    for i in range(len(camera_actions)):
        # Moving camera is most important (horizontal first)
        if camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = -1
    return actions

def train_bc():
    data = minerl.data.make("MineRLTreechop-v0",  data_dir=DATA_DIR, num_workers=4)
    env = DummyVecEnv([make_env(i) for i in range(1)])
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print(f"**********Using device: {device} for training***********")

    student_ppo = PPO('CnnPolicy', env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                      learning_rate=0.001, ent_coef=0.5, gamma=0.9999)
    model = student_ppo.policy.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    iter_count = 0
    losses = []
    logged_losses = []
    
    for dataset_obs, dataset_actions, _, _, _ in tqdm(data.batch_iter(num_epochs = EPOCHS, batch_size=32, seq_len=1)):
    # Process observations and actions
        obs = dataset_obs["pov"].squeeze().astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations
        obs = th.tensor(obs, dtype=th.float32).to(device) / 255.0

        # Actions need bit more work
        actions = dataset_action_batch_to_actions(dataset_actions)

        mask = actions != -1
        obs = obs[mask]
        actions = th.tensor(actions[mask], dtype=th.long).to(device)

        dist = model.get_distribution(obs)
        action_prediction = dist.distribution.logits

        loss = criterion(action_prediction, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 500) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            logged_losses.append((iter_count, mean_loss)) 
            losses.clear()
       
    student_ppo.policy = model
    iterations, mean_losses = zip(*logged_losses)

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, mean_losses, marker='o', linestyle='-')
    plt.title('Loss vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('loss_plot.png')
    plt.show()

    model.save(config["STUDENT_POLICY_NAME"])
    student_ppo.save(config["STUDENT_MODEL_NAME"])

def train_rl(rl_bc):
    env = DummyVecEnv([make_env(i) for i in range(1)])
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print(f"**********Using device: {device} for training***********")
    # For all the PPO hyperparameters you could tune see this:
    # https://github.com/DLR-RM/stable-baselines3/blob/6f822b9ed7d6e8f57e5a58059923a5b24e8db283/stable_baselines3/ppo/ppo.py#L16
    
    if(rl_bc):
        model = PPO.load(config["STUDENT_MODEL_NAME"], verbose=1)
        model.set_env(env)
        policy = model.policy.load(config["STUDENT_POLICY_NAME"])
        model.policy = policy
    
    if(not rl_bc):
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=f"runs/{experiment_name}", device=device,
                learning_rate=0.001, n_steps=1024, batch_size=64, ent_coef=0.25, gamma=0.9999)
        model.set_env(env)

    model.learn(total_timesteps=config["TRAIN_TIMESTEPS"])   # 2m steps is about 8h at 70 FPS

    model.save(config["TRAIN_MODEL_NAME"])

    # MineRL might throw an exception when closing on Windows, but it can be ignored (the environment does close).
    try:
        print('*********Closing environment...')
        env.close()
    except Exception:
        pass


def test(bc_only):
    writer = SummaryWriter(f"runs/{experiment_name}")
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"

    env = gym.make(config['TEST_ENV']).env
    time_limit = min(config["MAX_TEST_EPISODE_LEN"], config["TREECHOP_STEPS"])
    env = gym.wrappers.TimeLimit(env, time_limit)

    # optional interactive mode, where you can connect to your agent and play together (see link for details):
    # https://minerl.io/docs/tutorials/minerl_tools.html#interactive-mode-minerl-interactor
    # env.make_interactive(port=6666, realtime=True)

    if config["RECORD_TEST_VIDEOS"]:
        env = gym.wrappers.Monitor(env, f"test_videos/{experiment_name}")
    env = PovOnlyObservation(env)
    env = ActionShaping(env)

    if(not bc_only):
        print(config["TEST_MODEL_NAME"])
        model = PPO.load(config["TEST_MODEL_NAME"], verbose=1)
        model.set_env(env)

    if(bc_only):
        print(config["STUDENT_MODEL_NAME"])
        model = PPO.load(config["STUDENT_MODEL_NAME"], verbose=1)
        model.set_env(env)
        policy = model.policy.load(config["STUDENT_POLICY_NAME"])
        model.policy = policy


    for episode in range(config["TEST_EPISODES"]):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # RL part to get some logs:
        for i in range(config["TREECHOP_STEPS"]):
            if(bc_only):
                # For testing policy only (just BC) 
                obs = th.from_numpy(obs.transpose(2, 0, 1)[None].astype(np.float32) / 255).to(device)
                action = policy(obs)

            if(not bc_only):
                action = model.predict(obs.copy())

            #env.render()
            obs, reward, done, _ = env.step(action[0])
            total_reward += reward
            steps += 1
            if done:
                break
                   
        print(f'Episode #{episode + 1} return: {total_reward}\t\t episode length: {steps}')
        writer.add_scalar("return", total_reward, global_step=episode)

    env.close()


def main():

    #train_bc()
    #train_rl(rl_bc=True)
    test(bc_only=False)


if __name__ == '__main__':
    main()
