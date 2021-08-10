
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from collections import namedtuple, deque
import random
from torch.nn.modules.flatten import Flatten
from torch.utils.tensorboard import SummaryWriter
from kaggle_environments.envs.hungry_geese.hungry_geese import GreedyAgent, Configuration, Observation
import tracemalloc
import sys

ACTIONS = ['NORTH', 'SOUTH', 'WEST', 'EAST']


class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        # Add padding on columns by edge_size on either side
        h = torch.cat([x[:, :, :, -self.edge_size[1]:], x,
                       x[:, :, :, :self.edge_size[1]]], dim=3)

        # Add padding on rows by edge_size on either side
        h = torch.cat([h[:, :, -self.edge_size[0]:], h,
                       h[:, :, :self.edge_size[0]]], dim=2)

        h = self.conv(h)

        # batch normalization
        h = self.bn(h) if self.bn is not None else h
        return h


class ToroidalNet(nn.Module):
    """ Neural ToroidalNet to learn Q function
    """

    def __init__(self, num_states, num_actions, num_convolutions):
        super(ToroidalNet, self).__init__()
        filters = 32
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList(
            [TorusConv2d(filters, filters, (3, 3), True) for _ in range(num_convolutions)])

        self.afc1 = nn.Linear(filters, 128)
        self.afc2 = nn.Linear(128, num_actions)

        self.vfc1 = nn.Linear(filters, 128)
        self.vfc2 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))

        h_head = (h * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)

        a = F.leaky_relu(self.afc1(h_avg))
        a = F.leaky_relu(self.afc2(a))

        v = F.leaky_relu(self.vfc1(h_avg))
        v = F.leaky_relu(self.vfc2(v))

        q = torch.add(v, torch.sub(a, torch.mean(a)))
        return q


class GeeseNet(nn.Module):
    def __init__(self):
        super(GeeseNet, self).__init__()
        self.conv1 = nn.Conv2d(17, 128, kernel_size=(3, 5))
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x


class DuelingGeeseNet(nn.Module):
    def __init__(self):
        super(DuelingGeeseNet, self).__init__()
        self.conv1 = nn.Conv2d(17, 128, kernel_size=(3, 5))
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten(start_dim=1)

        self.afc1 = nn.Linear(384, 128)
        self.afc2 = nn.Linear(128, 4)

        self.vfc1 = nn.Linear(384, 128)
        self.vfc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)

        a = F.leaky_relu(self.afc1(x))
        a = F.leaky_relu(self.afc2(a))

        v = F.leaky_relu(self.vfc1(x))
        v = F.leaky_relu(self.vfc2(v))

        q = torch.add(v, torch.sub(a, torch.mean(a)))
        return q


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """ Experience replay taken
    from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDQN():

    def __init__(
            self,
            num_states,
            num_actions,
            num_convolutions,
            epsilon,
            opt):
        super(DDQN, self).__init__()
        # State and action space sizes
        self.num_states = num_states
        self.num_actions = num_actions

        # Options passed to DDQN
        self.opt = opt

        # Two nets for Double DQN
        # self.eval_net, self.target_net = GeeseNet(), GeeseNet()

        if opt.net_type == 'geese':
            self.eval_net, self.target_net = GeeseNet(), GeeseNet()
        elif opt.net_type == 'dueling':
            self.eval_net, self.target_net = DuelingGeeseNet(), DuelingGeeseNet()
        elif opt.net_type == 'toroidal':
            self.eval_net, self.target_net = ToroidalNet(
                num_states, num_actions, num_convolutions), ToroidalNet(
                num_states, num_actions, num_convolutions)
        else:
            raise ValueError("Undefined net type")

        self.learn_step_counter = 0
        self.memory_counter = 0

        # Setup Experience Replay
        self.replay_memory = ReplayMemory(opt.mem_cap)
        print(sys.getsizeof(self.replay_memory))

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=opt.lr)
        # update lr 4 times
        # self.lr_scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=opt.num_episodes // 4)
        # Huber Loss
        self.loss_func = nn.SmoothL1Loss()

        # Setup starting and ending points for epsilon decay
        self.epsilon_start = epsilon
        self.epsilon = epsilon

        # Setup Tensorboard
        self.writer = SummaryWriter(
            f"runs/{opt.net_type}-net-{opt.num_conv}",
            filename_suffix=f"{opt.num_conv}")

        # Use gpu if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Shove our neural nets onto gpu if available else cpu
        self.eval_net = self.eval_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

    def save(self, path):
        """ Save eval_net state dictionary to the inputted path
        :param path: str of a path to the desired save location
        """
        torch.save(self.eval_net.state_dict(), path)

    def load(self, path):
        """ Load eval_net state dictionary from the inputted path
        :param path: str of a path to the location from where to load saved state_dictionary
        """
        self.eval_net.load_state_dict(
            torch.load(path, map_location=self.device))

    def choose_action(self, state, observation, epsilon=None) -> int:
        """ Epsilon greedy algorithm, if no epsilon is inputted directly use epsilon
        from options
        :param state: current state vector to choose action from created by create_state_vector
        :param epsilon: optional arg for choosing epsilon other than decaying epsilon from options
        """
        state = torch.as_tensor(state, dtype=torch.float, device=self.device)

        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() > epsilon or self.opt.testing:
            # Exploit
            with torch.no_grad():
                action_value = self.eval_net(state).cpu().numpy()
            action = action_value.argmax()
        else:
            g_agent = GreedyAgent(
                Configuration({'rows': 7, 'columns': 11}))
            action = ACTIONS.index(g_agent(Observation(observation)))
            # # Explore
            # if np.random.rand() < epsilon:
            #     # Will always to never
            #     g_agent = GreedyAgent(
            #         Configuration({'rows': 7, 'columns': 11}))
            #     action = ACTIONS.index(g_agent(Observation(observation)))

            # else:
            #     # Will never to rarely
            #     action = np.random.randint(0, 4)

        return action

    def store_transition(self, state, action, reward, next_state):
        action = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        reward = torch.tensor([reward], device=self.device)

        self.replay_memory.push(state, action, next_state, reward)
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < self.opt.mem_cap:
            return

        # update the parameters
        if self.learn_step_counter % self.opt.q_net_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        transitions = self.replay_memory.sample(self.opt.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None]).to(self.device)

        # sample batch from memory
        batch_state = torch.cat(batch.state).to(self.device)
        batch_action = torch.cat(batch.action)
        batch_reward = torch.cat(batch.reward)
        # batch_next_state = torch.cat(batch.next_state).to(self.device)

        q_eval = self.eval_net(batch_state)
        q_eval = q_eval.gather(1, batch_action)

        next_state_values = torch.zeros(self.opt.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()


        # with torch.no_grad():
        #     q_next = self.target_net(batch_next_state)
        # q_target = batch_reward + self.opt.gamma * \
        #     q_next.max(1, keepdim=True)[0]
        # print(q_target.shape)
        #   print(q_eval.shape)

        q_target = batch_reward + (next_state_values * self.opt.gamma)
        q_target = q_target.unsqueeze(1)

        assert(q_eval.shape == q_target.shape)
        loss = self.loss_func(q_eval, q_target)

        self.writer.add_scalar(
            'loss?',
            loss.detach().item(),
            self.learn_step_counter)
        # self.writer.add_graph(self.eval_net, input_to_model=batch_state)
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # self.lr_scheduler.step()

    def ep_decay(self, EPS_DECAY, steps_done):
        self.writer.add_scalar(
            'epsilon',
            self.epsilon,
            self.learn_step_counter)
        EPS_END = 0.05
        EPS_START = self.epsilon_start
        self.epsilon = EPS_END + (EPS_START - EPS_END) * \
            (1 - steps_done / EPS_DECAY)
