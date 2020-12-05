import gym
from torch import nn, optim
import torch.nn.functional as F
import torch
from collections import deque
import numpy as np
import os
from tqdm import tqdm
import logging
from model import EnsembleNet
from properties import build_parser, CONSOLE_LEVEL, LOG_FILE, LOGFILE_LEVEL
from repository import historyDataset, memoryDataset
import sys
import traceback
from PIL import Image
from collections import Counter


CHECKPOINT_NAME = 'pytorch_model.bin'
CONFIG_NAME = "training_args.bin"

## 모델 불러오기
def load_saved_model(model, path):
    checkpoint = torch.load(os.path.join(path, CHECKPOINT_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
    return model

## 모델 저장
def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(path, CHECKPOINT_NAME))

def save_config(config, path):
    torch.save(config, os.path.join(path, CONFIG_NAME))

def load_saved_config(path):
    return torch.load(os.path.join(path, CONFIG_NAME))

def save_numpy(temp_data, path, name):
    temp_path = os.path.join(path, name)
    np.save(temp_path, temp_data)

def build_model(config):
    return EnsembleNet(n_ensemble=config.n_ensemble, n_actions=config.class_num, h=config.resize_unit[0], w=config.resize_unit[1],
                num_channels=config.history_size)


class DQNSolver():

    def __init__(self, config):
        self.device = config.device
        self.env = gym.make(config.env)
        self.valid_env = gym.make(config.env)
        self.memory_size = config.memory_size
        self.update_freq = config.update_freq
        self.learn_start = config.learn_start
        self.history_size = config.history_size

        self.batch_size = config.batch_size
        self.ep = config.ep
        self.eps_end = config.eps_end
        self.eps_endt = config.eps_endt
        self.eps_start = self.learn_start

        self.lr = config.lr
        self.discount = config.discount

        self.agent_type = config.agent_type
        self.max_steps = config.max_steps
        self.eval_freq = config.eval_freq
        self.eval_steps = config.eval_steps
        self.target_update = config.target_update
        self.max_eval_iter = config.max_eval_iter

        ##Breakout Setting
        if config.pretrained_dir is not None:
            pretrained_config = load_saved_config(config.pretrained_dir)
            config.n_ensemble = pretrained_config.n_ensemble
            config.class_num = pretrained_config.class_num
            config.resize_unit = pretrained_config.resize_unit

            policy_model = build_model(config)
            target_model = build_model(config)
            self.policy_model = load_saved_model(policy_model, config.pretrained_dir)
            self.target_model = load_saved_model(target_model, config.pretrained_dir)

        else:
            config.resize_unit = (84, 84)
            config.class_num = self.env.action_space.n
            self.policy_model = build_model(config)
            self.target_model = build_model(config)

        self.resize_unit = config.resize_unit
        self.class_num = config.class_num
        self.n_ensemble = config.n_ensemble

        self.policy_model.to(config.device)
        self.target_model.to(config.device)

        self.optimizer = optim.Adam(params=self.policy_model.parameters(), lr=self.lr)

        ##INIT Memory SETTING
        self.memory = memoryDataset(maxlen=config.memory_size, n_ensemble=config.n_ensemble,
                                    bernoulli_prob=config.bernoulli_prob)

        ##INIT LOGGER
        if not logging.getLogger() == None:
            for handler in logging.getLogger().handlers[:]:  # make a copy of the list
                logging.getLogger().removeHandler(handler)
        logging.basicConfig(filename=LOG_FILE, level=LOGFILE_LEVEL) ## set log config
        console = logging.StreamHandler() # console out
        console.setLevel(CONSOLE_LEVEL) # set log level
        logging.getLogger().addHandler(console)

        ##save options
        self.out_dir = config.out_dir
        if not os.path.isdir(config.out_dir):
            os.mkdir(config.out_dir)

        self.test_score_memory = []
        self.test_length_memory = []
        self.train_score_memory = []
        self.train_length_memory = []

        ##중간시작
        self.start_steps = config.start_steps
        self.learn_start = self.learn_start + self.start_steps
        self.eval_steps = self.eval_steps + self.start_steps

        self.config = config
        save_config(config, self.out_dir)

        self.refer_img = config.refer_img
        if self.refer_img is not None:
            assert os.path.isdir(self.refer_img), 'there is no reference image folder'

        self.crop_flag = False
        if 'breakout' in config.env.lower():
            self.crop_flag = True




    def choose_action(self, history, header_number:int=None, epsilon=None):
        if epsilon is not None:
            if np.random.random() <= epsilon:
                return self.env.action_space.sample()
            else:
                with torch.no_grad():
                    state = torch.tensor(history.get_state(), dtype=torch.float).unsqueeze(0).to(self.device)
                    if header_number is not None:
                        action = self.target_model(state, header_number).cpu()
                        return int(action.max(1).indices.numpy())
                    else:
                        # vote
                        actions = self.target_model(state)
                        actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
                        actions = Counter(actions)
                        action = actions.most_common(1)[0][0]
                        return action
        else:
            with torch.no_grad():
                state = torch.tensor(history.get_state(), dtype=torch.float).unsqueeze(0).to(self.device)
                if header_number is not None:
                    action = self.policy_model(state, header_number).cpu()
                    return int(action.max(1).indices.numpy())
                else:
                    # vote
                    actions = self.policy_model(state)
                    actions = [int(action.cpu().max(1).indices.numpy()) for action in actions]
                    actions = Counter(actions)
                    action = actions.most_common(1)[0][0]
                    return action



    def get_epsilon(self, t):
        epsilon =  self.eps_end + max(0, (self.ep - self.eps_end)*(self.eps_endt - max(0, t - self.eps_start)) /self.eps_endt )
        return epsilon

    def replay(self, batch_size):
        self.optimizer.zero_grad()

        batch = self.memory.sample(batch_size)

        state = batch.state.to(self.device)
        action = batch.action.to(self.device)
        next_state = batch.next_state.to(self.device)
        reward = batch.reward
        reward = reward.type(torch.bool).type(torch.float).to(self.device)

        done = batch.done.to(self.device)
        life = batch.life.to(self.device)
        terminal = batch.terminal.to(self.device)
        mask = batch.mask.to(self.device)

        with torch.no_grad():
            next_state_action_values = self.policy_model(next_state)
        state_action_values = self.policy_model(state)

        total_loss = []
        for head_num in range(self.n_ensemble):
            total_used = torch.sum(mask[:, head_num])
            if total_used > 0.0:
                next_state_value = torch.max(next_state_action_values[head_num], dim=1).values.view(-1, 1)
                reward = reward.view(-1, 1)
                target_state_value = torch.stack([reward + (self.discount * next_state_value), reward], dim=1).squeeze().gather(1, terminal)
                state_action_value = state_action_values[head_num].gather(1, action)
                loss = F.smooth_l1_loss(state_action_value, target_state_value, reduction='none')
                loss = mask[:, head_num] * loss
                loss = torch.sum(loss / total_used)
                total_loss.append(loss)

        if len(total_loss) > 0:
            total_loss = sum(total_loss)/self.n_ensemble
            total_loss.backward()
            self.optimizer.step()


    def valid_run(self):

        state = self.valid_env.reset()
        valid_history = historyDataset(self.history_size, state, self.crop_flag)
        score = 0
        count = 0
        terminal = True
        done = False
        last_life = 0

        ## put valid time limits to make it fast evaluation
        while not done and count < self.max_eval_iter:
            action = self.choose_action(valid_history)
            if terminal: ## There is error when it is just started. So do action = 1 at first
               action = 1
            next_state, reward, done, life = self.valid_env.step(action)
            valid_history.push(next_state)
            score += reward
            life = life['ale.lives']
            count = count + 1

            ## Terminal options
            if life < last_life:
                terminal = True
            else:
                terminal = False
            last_life = life

        return score, count

    def render_policy_net(self):

        def get_concat_h(im1, im2):
            baseheight = im1.size[1]
            wpercent = (baseheight / float(im2.size[1]))
            wsize = int((float(im2.size[0]) * float(wpercent)))
            im2_modified = im2.resize((wsize, baseheight), Image.ANTIALIAS)

            dst = Image.new('RGB', (im1.width + im2_modified.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2_modified, (im1.width, 0))
            return dst

        def get_concat_v(im1, im2):
            if im1 is None:
                return im2

            dst = Image.new('RGB', (im1.width, im1.height + im2.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (0, im1.height))
            return dst

        arrow_images = []
        if self.refer_img is not None and 'breakout' in self.config.env.lower():
            arrow_files = [filename for filename in os.listdir(self.refer_img) if '.png' in filename]
            if len(arrow_files) >= self.class_num:
                arrow_images = [Image.open(os.path.join(self.refer_img, filename)) for filename in arrow_files]

        state = self.env.reset()
        history = historyDataset(self.history_size, state, self.crop_flag)
        score = 0
        count = 0
        raw_frames = []
        frames = []
        done = False
        terminal = True
        last_life = 0
        while not done:
            action = self.choose_action(history)
            actions = []
            for head_idx in range(self.config.n_ensemble):
                actions.append(self.choose_action(history, head_idx))

            # img = Image.fromarray(next_state)
            # frames.append(img)
            img = self.env.render(mode='rgb_array')
            img = Image.fromarray(img)
            if len(arrow_images) > 0:
                img2 = arrow_images[action]
                img = get_concat_h(img, img2)

                if self.config.n_ensemble > 1:
                    merge_img = None
                    for head_idx in range(self.config.n_ensemble):
                        action_img = arrow_images[self.choose_action(history, head_idx)]
                        merge_img = get_concat_v(merge_img, action_img)
                    img = get_concat_h(img, merge_img)

            frames.append(img)


            if terminal: ## There is error when it is just started. So do action = 1 at first
               action = 1
            next_state, reward, done, life = self.env.step(action)
            history.push(next_state)

            score += reward
            life = life['ale.lives']
            count = count + 1

            ## Terminal options
            if life < last_life:
                terminal = True
            else:
                terminal = False
            last_life = life
        self.env.close()
        frames[0].save(os.path.join(self.out_dir, 'Breakout_result.gif'), format='GIF', append_images=frames[1:], save_all=True, duration=0.0001)
        print("save picture -- Breakout_result.gif")
        print("score", score)
        print("count", count)


    def train(self):
        progress_bar = tqdm(range(self.start_steps, self.max_steps))
        state = self.env.reset()
        history = historyDataset(self.history_size, state, self.crop_flag)
        done = False

        ##Report
        train_scores = deque(maxlen=10)
        train_lengths = deque(maxlen=10)
        episode = 0
        max_score = 0

        ##If it is done everytime init value
        train_score = 0
        train_length = 0
        last_life = 0
        terminal = True

        ## number of ensemble
        heads = list(range(self.n_ensemble))
        active_head = heads[0]

        try:
            for step in progress_bar:

                ## model update
                if step > self.learn_start and step % self.target_update == 0:
                    self.target_model.load_state_dict(self.policy_model.state_dict())

                ## game is over
                if done:

                    np.random.shuffle(heads)
                    active_head = heads[0]

                    state = self.env.reset()
                    history = historyDataset(self.history_size, state, self.crop_flag)
                    train_scores.append(train_score)
                    train_lengths.append(train_length)
                    episode += 1

                    ##If it is done everytime init value
                    train_score = 0
                    train_length = 0
                    last_life = 0
                    terminal = True

                action = self.choose_action(history, active_head, self.get_epsilon(step))
                if terminal: ## There is error when it is just started. So do action = 1 at first
                    action = 1
                next_state, reward, done, life = self.env.step(action)
                state = history.get_state()
                history.push(next_state)
                next_state = history.get_state()
                life = life['ale.lives']
                train_length = train_length + 1

                ## Terminal options
                if life < last_life:
                    terminal = True
                else :
                    terminal = False
                last_life = life

                self.memory.push(state, action, next_state, reward, done, life, terminal)
                if step > self.learn_start and step % self.update_freq == 0:
                    self.replay(self.batch_size)

                train_score = train_score + reward

                if step > self.eval_steps and step % self.eval_freq == 0:
                    train_mean_score = np.mean(train_scores)
                    train_mean_length = np.mean(train_lengths)
                    self.train_score_memory.append(train_mean_score)
                    self.train_length_memory.append(train_mean_length)

                    save_numpy(self.train_score_memory, self.out_dir, 'train_score')
                    save_numpy(self.train_length_memory, self.out_dir, 'train_length_memory')

                    valid_score, valid_length = self.valid_run()
                    self.test_score_memory.append(valid_score)
                    self.test_length_memory.append(valid_length)

                    save_numpy(self.test_score_memory, self.out_dir, 'test_score')
                    save_numpy(self.test_length_memory, self.out_dir, 'test_length_memory')

                    if valid_score >= max_score:
                        max_score = valid_score
                        save_model(self.policy_model, self.out_dir)

                    progress_bar.set_postfix_str(
                        '[Episode %s] - train_score : %.2f, test_score : %.2f, max_score : %.2f, epsilon : %.2f' % (episode,
                                                                                                                    train_mean_score,
                                                                                                                    valid_score,
                                                                                                                    max_score,
                                                                                                                    self.get_epsilon(step)))
                    logging.debug(
                        '[Episode %s] - train_score : %.2f, test_score : %.2f, max_score : %.2f, epsilon : %.2f' % (episode,
                                                                                                                    train_mean_score,
                                                                                                                    valid_score,
                                                                                                                    max_score,
                                                                                                                    self.get_epsilon(step)))
        except Exception as e:
            # Get current system exception
            ex_type, ex_value, ex_traceback = sys.exc_info()

            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            logging.warning("Exception type : %s " % ex_type.__name__)
            logging.warning("Exception message : %s" % ex_value)
            for trace in trace_back:
                logging.warning("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                trace[0], trace[1], trace[2], trace[3]))


if __name__ == '__main__':
    parser = build_parser()
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and config.device in ["gpu",'cuda'] else "cpu")
    config.device = device
    agent = DQNSolver(config)
    if config.mode == "train":
        agent.train()
    if config.mode =="test":
        if config.pretrained_dir is None:
            raise ValueError(
                "평가를 하려면 pretrained_dir 에 저장된 모델을 넣어야 합니다. {}".format(
                    config.pretrained_dir
                )
            )
        agent.render_policy_net()