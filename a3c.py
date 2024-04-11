import minerl
import torch as th
import torch.nn as nn
import multiprocessing as mp
import tqdm

class A3C_Orchestrator():
    """
    Orchestrator class that will manage the training of the A3C model. 
    Should also manage the workers and the global network.
    """

    def __init__(
        self, 
        env_name, 
        num_episodes, 
        num_workers, 
        T_max, 
        gamma, 
        lr
    ):
        # define shared counter
        self.T = 0
        # define device
        self.device = "cpu"
        # define environment
        self.env = minerl.env.make(env_name)
        self.num_episodes = num_episodes
        self.num_workers = num_workers
        self.T_max = T_max
        self.gamma = gamma
        self.lr = lr
        # setup models
        self._setup_models()

    def _setup_models(self):
        """
        Setup the global and local models for the A3C algorithm.
        """
        in_channels = self.env.observation_space.shape[0]
        out_channels = 16
        num_actions = self.env.action_space.n
        self.global_value = ValueCNN(in_channels, out_channels, num_actions)
        self.global_policy = PolicyCNN(in_channels, out_channels, num_actions)
        # share memory between processes
        self.global_value.share_memory()
        self.global_policy.share_memory()
        # define optimizer using RMSProp with shared memory
        self.optimizer_val = th.optim.RMSprop(self.global_value.parameters(), lr=0.0001)
        self.optimizer_policy = th.optim.RMSprop(self.global_policy.parameters(), lr=0.0001)

    def train(self):
        """
        Train the model using the A3C algorithm. Should also output graphs and the 
        total reward gathered by the agent across multiple episodes.
        """
        # number of processes
        num_processes = mp.cpu_count() if self.num_workers == -1 else self.num_workers

        # define workers
        workers = []
        for _ in range(num_processes):
            worker = A3C_Worker(self, T_max=1000)
            workers.append(worker)

        # start workers
        for worker in workers:
            worker.start()

        # join workers
        for worker in workers:
            worker.join()

        # save model
        th.save(self.global_policy.state_dict(), "policy_model.pth")
        th.save(self.global_value.state_dict(), "value_model.pth")

class A3C_Worker(mp.Process):
    """
    Each worker will have its own instance of environment and accumulate 
    gradients that will update the global network.
    """
    def __init__(self,
                glbl: A3C_Orchestrator,
                worker_id,
                t_max,
        ):
        # initialize local network
        self.local_value = ValueCNN(glbl.env.observation_space.shape[0], 16, glbl.env.action_space.n)
        self.local_policy = PolicyCNN(glbl.env.observation_space.shape[0], 16, glbl.env.action_space.n)
        self.worker_id = worker_id
        self.env = glbl.env
        self.device = glbl.device
        self.T_max = glbl.T_max
        self.glbl = glbl
        # define optimizers
        self.glbl_optim_val = glbl.optimizer
        self.glbl_optim_policy = glbl.optimizer
        self.optim_val = glbl.optimizer_val
        self.optim_policy = glbl.optimizer_policy
        # set local counters
        self.t = 0
        self.t_max = 10

    def run(self):
        while self.glbl.T < self.glbl.T_max:
            # reset gradients of the global model
            self.optim_val.zero_grad()
            self.optim_policy.zero_grad()

            # reset gradients
            d_policy = 0
            d_value = 0

            # sync with global network
            self.local_policy.load_state_dict(self.glbl.global_policy.state_dict())
            self.local_value.load_state_dict(self.glbl.global_value.state_dict())

            # reset environment
            obs = self.env.reset()
            done = False
            t_start = self.t

            traj = []
            while not done and not self.t - t_start == self.t_max:
                action, log_prob = self.local_policy(obs, return_prob=True)
                obs, reward, done, _ = self.env.step(action)
                self.t += 1
                self.glbl.T += 1
                traj.append((obs, action, reward, log_prob))
            
            self.glbl.env.close()
            R = 0 if done else self.local_value(obs)

            for i in range(len(traj) - 1, -1, -1):
                obs, action, reward, log_prob = traj[i]
                R = reward + self.glbl.gamma * R
                # calculate loss
                adv = R - self.local_value(obs)
                p_loss = -log_prob * adv
                v_loss = adv ** 2
                # accumulate gradients
                d_policy += th.autograd.grad(p_loss, self.local_policy.parameters())
                d_value += th.autograd.grad(v_loss, self.local_value.parameters())
                print(f"Worker {self.worker_id}, episode {self.glbl.T}, policy_loss: {p_loss}, value_loss: {v_loss}")

            # update global network
            for p, gp in zip(self.glbl.global_policy.parameters(), d_policy):
                p.grad = gp if p.grad is None else p.grad + gp
            for p, gv in zip(self.glbl.global_value.parameters(), d_value):
                p.grad = gv if p.grad is None else p.grad + gv

            self.glbl.optimizer_policy.step()
            self.glbl.optimizer_val.step()

class ValueCNN(nn.Module):
    """
    Convolutional Neural Network that will be used by the Actor-Critic model. 
    Should have at least two convolutional layers and two fully connected layers.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size = 3, 
        stride = 1, 
        padding = 0
    ):
        """
        Give the model 2 convolutional layers and 2 fully connected layers.
        """
        super(ValueCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.fc1 = nn.Linear(out_channels, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, input):
        """
        Forward pass through the model.
        """
        x = self.conv1(input)
        x = self.conv2(th.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(th.relu(x))
        x = self.fc2(th.relu(x))
        return x
    
class PolicyCNN(nn.Module):
    """
    Convolutional Neural Network that will be used by the Actor-Critic model. 
    Should have some CNN layers and output logits for the action space. Also
    should have a method that returns the action given a state.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_actions,
        kernel_size = 3, 
        stride = 1, 
        padding = 0
    ):
        super(PolicyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.fc1 = nn.Linear(out_channels, 128)
        self.fc2 = nn.Linear(128, num_actions)

        
    def forward(self, input, return_prob=False):
        """
        Forward pass through the model.
        """
        x = self.conv1(input)
        x = self.conv2(th.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(th.relu(x))
        x = self.fc2(th.relu(x))
        if return_prob:
            # return both action and its log probability
            return th.argmax(th.softmax(x, dim=-1)).item(), th.log_softmax(x, dim=-1)
        else:
            return th.argmax(th.softmax(x, dim=-1)).item()


def test(env, policy_model_path, num_episodes=10):
    """
    Test the trained model on the environment. Should also record the agent's actions and the environment's state. Also should
    output graphs and the total reward gathered by the agent across multiple episodes.
    """
    # load model from path
    policy_model = PolicyCNN(env.observation_space.shape[0], 16, env.action_space.n)
    policy_model.load_state_dict(th.load(policy_model_path))
    policy_model.eval()

    env = minerl.env.make('MineRLTreechop-v0')

    for i in tqdm.tqdm(range(num_episodes)):
        done = False
        total_reward = 0
        obs = env.reset()
        steps = 0
        env.render()

        while not done:
            # get action from model
            action = policy_model(obs)

            # take action
            obs, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            steps += 1

        print(f"Episode {i}, total reward: {total_reward} in {steps} steps.")

if __name__ == "__main__":
    # train the model
    a3c = A3C_Orchestrator("MineRLTreechop-v0", 100, 4, 1000, 0.99, 0.0001)
    a3c.train()

    # test results on environment
    test()
