import minerl
import torch as th
import torch.nn as nn
import multiprocessing as mp
import tqdm

class A3C_Worker(mp.Process):
    """
    Each worker will have its own instance of environment and accumulate gradients that will update the global network.
    """
    def __init__():
        pass

    def run():
        pass

class ActorCNNCritic(nn.Module):
    """
    Actor-Critic network that will be used by both the global network and the workers. 
    Should have two outputs: one for the policy and one for the value function.
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


def train():
    """
    Train the model using the A3C algorithm. Should also output graphs and the total reward gathered by the agent across multiple episodes.
    """
    # define global network
    global_network = ActorCNNCritic()
    global_network.share_memory()

    # define optimizer using RMSProp with shared memory
    optimizer = th.optim.RMSprop(global_network.parameters(), lr=0.0001)

    # define workers
    workers = []
    for _ in range(mp.cpu_count()):
        worker = A3C_Worker(global_network, optimizer)
        workers.append(worker)

    # start workers
    for worker in workers:
        worker.start()

    # join workers
    for worker in workers:
        worker.join()

    # save model
    th.save(global_network.state_dict(), "a3c_model.pth")

if __name__ == "__main__":
    # train the model
    train()

    # test results on environment
    test()



