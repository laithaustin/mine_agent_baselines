import torch as th
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, in_channels, height, width, device="cpu", num_actions=None):
        super(Critic, self).__init__()
        self.device = device
        self.num_actions = num_actions
        
        # DQN Nature paper architecture
        self.cnn1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.flatten(self.cnn3(self.cnn2(self.cnn1(th.zeros(1, in_channels, height, width))))).shape[1]

        self.fc1 = nn.Linear(n_flatten, 512)  # Adjust the input features to match your input size
        self.value = nn.Linear(512, 1)  # Output a single value for the state value

        # use q functin for IQL
        if num_actions is not None:
            self.fc2 = nn.Linear(n_flatten + num_actions, 512) 
            self.q = nn.Linear(512, 1)

    def getV(self, x):
        x = th.relu(self.fc1(x))
        value = self.value(x)
        return value
    
    def getQ(self, x, action):
        # append action to the input
        x = self.prepare_input(x)
        action = th.zeros((x.shape[0], self.num_actions), device=self.device).scatter(1, action.unsqueeze(1).to(th.int64), 1).to(th.float32)
        x = th.cat([x, action], dim=1)
        x = th.relu(self.fc2(x))
        q = self.q(x)
        return q

    def forward(self, x, action=None):
        x = self.prepare_input(x)
        if action is None:
            return self.getV(x)
        else: # TODO
            pass


    def prepare_input(self, x):
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x.copy()).float().unsqueeze(0).to(self.device)
      
        x = th.relu(self.cnn1(x))
        x = th.relu(self.cnn2(x))
        x = th.relu(self.cnn3(x))
        x = self.flatten(x)
        return x