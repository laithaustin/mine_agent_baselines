import gym
import numpy as np
import argparse

def initParser():
    # initialize arg parser
    parser = argparse.ArgumentParser(description="A2C with Behavioral Cloning")
    parser.add_argument("--task", type=str, default="MineRLTreechop-v0", help="Task to train on")
    parser.add_argument("--max_timesteps", type=int, default=2000000, help="Maximum timesteps to train")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default="a2c_bc", help="Name of experiment")
    parser.add_argument("--method", type=str, default="a2c", help="Method to use (a2c or a2c_bc)")
    parser.add_argument("--test", type=bool, default=False, help="Test model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for behavioral cloning")
    parser.add_argument('--wandb', type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--load_model", type=bool, default=False, help="Load model")
    parser.add_argument("--entropy_start", type=float, default=0.5, help="Starting entropy value")
    parser.add_argument('--actor_path', type=str, default="a2c_bc_actor.pth", help="Path to actor model")
    parser.add_argument('--critic_path', type=str, default="a2c_bc_critic.pth", help="Path to critic model")
    parser.add_argument('--bc_path', type=str, default="bc_model.pth", help="Path to behavioral cloning model")
    parser.add_argument('--annealing', type=bool, default=False, help="Anneal entropy")
    parser.add_argument('--model_path', type=str, default="model.pth", help="Path to retrieve model for testing")
    parser.add_argument('--episodes', type=int, default=100, help="Number of episodes to test")
    return parser

def make_env(task, camera_angle=10, always_attack=False, simple_test=False):
    if simple_test:
        # test algo on cartpole
        env = gym.make("CartPole-v1")
        env = gym.wrappers.TransformReward(env, lambda r: r / 100.0)
        env = gym.wrappers.FlattenObservation(env)
        return env
    env = gym.make(task)
    env = PovOnlyObservation(env)
    env = ActionShaping(env, camera_angle, always_attack)
    return env

# Observation Wrapper
class PovOnlyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']

# Action Wrapper
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
    def __init__(self, env, camera_angle=10, always_attack=False):
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
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
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

def compute_gae(next_value, rewards, masks, values, gamma=1.0, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def get_entropy_linear(start, end, current, total_steps):
    """
    Linear entropy annealing function.
    """
    if current >= total_steps:
        return end
    return start + (end - start) * current / total_steps

def update_timesteps(initial_timesteps, max_timesteps, current_step, total_steps, growth_factor=1.1):
    """
    Updates the number of timesteps per episode based on the current training step.
    
    :param initial_timesteps: The initial timesteps for each episode.
    :param max_timesteps: The maximum timesteps an episode can have.
    :param current_step: The current training step number.
    :param total_steps: The total number of training steps for the entire training.
    :param growth_factor: The factor by which timesteps will increase.
    :return: Updated timesteps for the current episode.
    """
    # Calculate the growth rate based on the current step and total steps
    growth_rate = np.minimum(growth_factor ** (current_step / total_steps), max_timesteps)
    # Calculate new timesteps
    new_timesteps = int(initial_timesteps * growth_rate)
    # Make sure it does not exceed max_timesteps
    new_timesteps = np.minimum(new_timesteps, max_timesteps)
    return new_timesteps