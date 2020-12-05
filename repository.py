import torch
from collections import deque
from collections import namedtuple
import numpy as np
import random
from skimage.transform import rescale
from skimage.transform import resize
import copy

class memoryDataset(object):
    def __init__(self, maxlen, n_ensemble=1, bernoulli_prob=0.9):
        self.memory = deque(maxlen=maxlen)
        self.n_ensemble = n_ensemble
        self.bernoulli_prob = bernoulli_prob

        ## if ensemble is 0 then no need to apply mask
        if n_ensemble==1:
            self.bernoulli_prob = 1

        self.subset = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'life', 'terminal', 'mask'))


    def push(self, state, action, next_state, reward, done, life, terminal):

        state = np.array(state)
        action = np.array([action])
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array([done])
        life = np.array([life])
        terminal = np.array([terminal])
        mask = np.random.binomial(1, self.bernoulli_prob, self.n_ensemble)

        self.memory.append(self.subset(state, action, next_state, reward, done, life, terminal, mask))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        batch = self.subset(*zip(*batch))

        state = torch.tensor(np.stack(batch.state), dtype=torch.float)
        action = torch.tensor(np.stack(batch.action), dtype=torch.long)
        reward = torch.tensor(np.stack(batch.reward), dtype=torch.float)
        next_state = torch.tensor(np.stack(batch.next_state), dtype=torch.float)

        done = torch.tensor(np.stack(batch.done), dtype=torch.long)
        ##Life : 0,1,2,3,4,5
        life = torch.tensor(np.stack(batch.life), dtype=torch.float)
        terminal = torch.tensor(np.stack(batch.terminal), dtype=torch.long)
        mask = torch.tensor(np.stack(batch.mask), dtype=torch.float)
        batch = self.subset(state, action, next_state, reward, done, life, terminal, mask)

        return batch

class historyDataset(object):
    def __init__(self, history_size, img, crop_flag=False):
        self.history_size = history_size
        self.crop_flag = crop_flag

        state = self.convert_channel(img)
        self.height, self.width = state.shape

        temp = []
        for _ in range(history_size):
            temp.append(state)
        self.history = temp

    def convert_channel(self, img):
        # input type : |img| = (Height, Width, channel)
        # remove useless item

        if self.crop_flag:
            img = img[31:193, 8:152]

        #img = rescale(img, 1.0 / 2.0, anti_aliasing=False, multichannel=False)
        img = resize(img, output_shape=(84, 84))

        # conver channel(3) -> channel(1)
        img = np.any(img, axis=2)
        # |img| = (Height, Width)  boolean
        return img

    def push(self, img):
        temp = self.history
        state = self.convert_channel(img)
        temp.append(state)
        self.history = temp[1:]

    def get_state(self):
        #return self.history
        return copy.deepcopy(self.history)
