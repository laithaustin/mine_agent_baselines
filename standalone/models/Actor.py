import torch as th
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, in_channels, height, width, num_actions, device="cpu", bc=False):
        super(Actor, self).__init__()
        self.device = device
        self.bc = bc
        
        # DQN Nature paper architecture
        self.cnn1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.flatten(self.cnn3(self.cnn2(self.cnn1(th.zeros(1, in_channels, height, width))))).shape[1]

        self.fc = nn.Linear(n_flatten, 512)  # Adjust the input features to match your input size
        self.policy = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.prepare_input(x)
        x = th.relu(self.fc(x))
        probs = self.policy(x)
        probs = th.softmax(probs, dim=-1)
        action = th.multinomial(probs, num_samples=1)
        # if there is only one action, we need to just return the item
        action = action.item() if len(action.shape) == 0 else action
        log_prob = th.log(probs[0, action])
        return action, log_prob, probs

    def prepare_input(self, x):
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x.copy()).float().unsqueeze(0).to(self.device)

        x = th.relu(self.cnn1(x))
        x = th.relu(self.cnn2(x))
        x = th.relu(self.cnn3(x))
        x = self.flatten(x)
        return x