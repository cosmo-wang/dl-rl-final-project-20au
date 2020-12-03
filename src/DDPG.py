import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from Renderer.model import *
from DRL.replay_buffer import replay_buffer
from ResNet import *
from DRL.wgan import *
from utils.util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

class Painter():
    def __init__(self, path):
        self.randerder = FCN()
        self.randerder.load_state_dict(torch.load(path))

    def paint(self, action, state):
        action = action.view(-1, 10 + 3)
        stroke = 1 - self.randerder(action[:, :10])
        stroke = stroke.view(-1, 128, 128, 1)
        color_stroke = stroke * action[:, -3:].view(-1, 1, 1, 3)
        stroke = stroke.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        stroke = stroke.view(-1, 5, 1, 128, 128)
        color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
        for i in range(5):
            state = state * (1 - stroke[:, i]) + color_stroke[:, i]
        return state

class DDPG_copy():
    def __init__(self, batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, buf_size=800, \
                 writer=None, resume_path=None):

        self.painter = Painter('./renderer.pkl')

        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size        

        self.actor = ResNet("actor", 7, 18, 65) # target, canvas, stepnum 3 + 3 + 1
        self.actor_target = ResNet("actor", 7, 18, 65)
        self.critic = ResNet("critic", 3 + 7, 18, 1) # add the last canvas for better prediction
        self.critic_target = ResNet("critic", 3 + 7, 18, 1)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-2)

        if resume_path != None:
            self.load_weights(resume_path)
        
        # Create replay buffer
        self.buffer = replay_buffer(buf_size * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0
        
        # most state and action
        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action

        self.choose_device()

    def get_action(self, state, target=False):
        state = torch.cat((state[:, :6].float() / 255, state[:, 6:7].float() / self.max_step), 1)
        return self.actor_target(state) if target else self.actor(state)

    def update_gan(self, state):
        canvas = state[:, :3]
        gt = state[:, 3 : 6]
        fake, real, penal = update(canvas.float() / 255, gt.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar('train/gan_fake', fake, self.log)
            self.writer.add_scalar('train/gan_real', real, self.log)
            self.writer.add_scalar('train/gan_penal', penal, self.log)       
        
    def evaluate(self, state, action, target=False):
        # step
        T = state[:, 6 : 7]
        # target image
        gt = state[:, 3 : 6].float() / 255
        # current canvas
        canvas0 = state[:, :3].float() / 255
        # apply the action to get the updated canvas
        canvas1 = self.painter.paint(action, canvas0)
        # gan_reward is the additional reward by applying the current action
        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        # merge two canvases to pass to critic
        merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step], 1)
        if target:
            V = self.critic_target(merged_state)
            return (V + gan_reward), gan_reward
        else:
            V = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
                self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
            return (V + gan_reward), gan_reward

    def update_policy(self, lr):
        self.log += 1

        # set learning rates for optimizers
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr[1]

        # sample a batch from replay buffer
        state, action, reward, next_state, terminal = self.buffer.sample_batch(self.batch_size, device)

        self.update_gan(next_state)

        # calculate discounted V(s_t+1)
        with torch.no_grad():
            next_action = self.get_action(next_state, True)
            expected_V, _ = self.evaluate(next_state, next_action, True)
            expected_V = self.discount * ((1 - terminal.float()).view(-1, 1)) * expected_V

        cur_V, r = self.evaluate(state, action)
        expected_V += r.detach()

        # update critic
        loss = nn.MSELoss()
        V_loss = loss(cur_V, expected_V)
        self.critic.zero_grad()
        V_loss.backward()
        self.critic_optimizer.step()

        # update actor
        action = self.get_action(state)
        V, _ = self.evaluate(state, action)
        policy_loss = V.mean()
        self.actor.zero_grad()
        (-policy_loss).backward()
        self.actor_optimizer.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss, V_loss

    def observe(self, reward, state, done, step):
        s0 = self.state.clone().detach().to("cpu")
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = state.clone().detach().to("cpu")
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.buffer.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def reset(self, obs):
        self.state = obs

    def select_action(self, state):
        self.eval()
        with torch.no_grad():
            action = self.get_action(state)
            action = to_numpy(action)
        self.train()
        self.action = action
        return self.action

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def load_weights(self, path):
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        load_gan(path)

    def save_model(self, path):
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(path))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(path))
        save_gan(path)
        self.choose_device()

    def choose_device(self):
        # self.painter.to(device)
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)