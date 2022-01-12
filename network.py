import torch
import torch.nn as nn
import torch.nn.functional


class ActorNetwork(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, output_size, device=torch.device('cpu')):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_size)
        self.output = nn.Softmax(dim=1)

    def __call__(self, state):
        batch_sz = state.shape[0]
        input_device = state.device
        state = state.to(self.device)
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        out = nn.functional.relu(self.fc4(out))
        out = nn.functional.relu(self.fc5(out))
        out = nn.functional.relu(self.fc6(out))
        out = self.output(out)

        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, output_size=1, device=torch.device('cpu')):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64 + action_dim, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, output_size)

    def __call__(self, state, action):
        batch_sz = state.shape[0]
        input_device = state.device
        state = state.to(self.device)
        action = action.to(self.device)
        out = nn.functional.relu(self.fc1(state))
        out = torch.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        out = nn.functional.relu(self.fc4(out))
        out = self.fc5(out)
        return out
