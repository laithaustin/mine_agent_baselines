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
        # define global network
        self.global_network = ActorCNNCritic()
        self.global_network.share_memory()
        # define shared counter
        self.T = 0
        # define optimizer using RMSProp with shared memory
        self.optimizer = th.optim.RMSprop(self.global_network.parameters(), lr=0.0001)
        # define device
        self.device = "cpu"
        # define environment
        self.env = minerl.env.make(env_name)
        self.num_episodes = num_episodes
        self.num_workers = num_workers
        self.T_max = T_max
        self.gamma = gamma
        self.lr = lr

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
        th.save(self.global_network.state_dict(), "a3c_model.pth")

class A3C_Worker(mp.Process):
    """
    Each worker will have its own instance of environment and accumulate 
    gradients that will update the global network.
    """
    def __init__(self,
                global_net: A3C_Orchestrator,
                worker_id,
                t_max,
        ):
        # initialize local network
        self.local_net = ActorCNNCritic()
        self.worker_id = worker_id
        self.env = global_net.env
        self.device = global_net.device
        self.T_max = global_net.T_max
        self.global_net = global_net
        self.global_optim = global_net.optimizer
        self.optim = th.optim.RMSprop(self.local_net.parameters(), lr=self.global_net.lr)
        self.t = 0
        self.t_max = 10

    def run(self):
        while self.global_net.T < self.global_net.T_max:
            # reset gradients
            d_policy = 0
            d_value = 0

            # sync with global network
            self.local_net.load_state_dict(self.global_net.state_dict())

            # reset environment
            obs = self.env.reset()
            done = False
            t_start = self.t

            traj = []
            while not done and not self.t - t_start == self.t_max:
                action, log_prob = self.local_net(obs)
                obs, reward, done, _ = self.env.step(action)
                self.t += 1
                self.global_net.T += 1
                traj.append((obs, action, reward, log_prob))
            
            R = 0 if done else self.local_net(obs)

            for i in range(len(traj) - 1, -1, -1):
                obs, action, reward, log_prob = traj[i]
                R = reward + self.global_net.gamma * R
                # calculate loss
                adv = R - self.local_net(obs)
                p_loss = -log_prob * adv
                v_loss = adv ** 2
                # accumulate gradients
                d_policy += th.autograd.grad(p_loss, self.local_net.parameters())
                d_value += th.autograd.grad(v_loss, self.local_net.parameters())


            # update global network
            for p, gp in zip(self.global_net.parameters(), d_policy):
                p.grad = gp if p.grad is None else p.grad + gp
            for p, gv in zip(self.global_net.parameters(), d_value):
                p.grad = gv if p.grad is None else p.grad + gv

            self.global_optim.step()
                
class ActorCNNCritic():
    """
    Actor-Critic with value and policy networks that will be used by both the global network and the workers. 
    Should have two CNN
    """
    def __init__(self):
        pass

    def forward(self):
        pass

class CNN(nn.Module):
    """
    Convolutional Neural Network that will be used by the Actor-Critic model. 
    Should have at least two convolutional layers and two fully connected layers.
    """
    def __init__(self):
        pass

    def forward(self):
        pass

def test(env, model_path, num_episodes=10):
    """
    Test the trained model on the environment. Should also record the agent's actions and the environment's state. Also should
    output graphs and the total reward gathered by the agent across multiple episodes.
    """
    # load model
    model = ActorCNNCritic()
    model.load_state_dict(th.load(model_path))
    model.eval()

    for i in tqdm.tqdm(range(num_episodes)):
        done = False
        total_reward = 0
        env = minerl.env.make('MineRLTreechop-v0')
        obs = env.reset()
        steps = 0

        while not done:
            # get action from model
            action = model(obs)

            # take action
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Episode {i}, total reward: {total_reward} in {steps} steps.")

if __name__ == "__main__":
    # train the model
    train()

    # test results on environment
    test()



