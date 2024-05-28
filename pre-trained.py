import math

import numpy as np
import torch
from torch import nn
import random

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(1200, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(1, 40, 30)
        x = self.flatten(x)
        return self.network(x)

def pretrain_play_game(model):
    # 初始化游戏状态，蛇长为1，并在地图中间，随机生成一个食物位置
    state = np.zeros(shape=(40, 30), dtype=np.float32)
    snake = [(19, 14), (18, 14), (17, 14), (16, 14), (15, 14), (14, 14), (13, 14), (12, 14), (11, 14), (10, 14), (9, 14), (8, 14), (7, 14)]
    for body in snake: state[body[0]][body[1]] = 1
    food = random.randint(0, 39), random.randint(0, 29)
    while food in snake:
        food = random.randint(0, 39), random.randint(0, 29)
    state[food[0], food[1]] = 2

    # 初始化数据集和reward数组,并设置衰减率为0.9
    data = []
    reward = []
    decay = 0.9

    # 初始化移动方向
    direction = np.array([1, 0])
    left_rotate = np.array([[0, -1], [1, 0]])
    right_rotate = np.array([[0, 1], [-1, 0]])
    rotate = (left_rotate, right_rotate)
    move_cnt = 0

    # 开始玩游戏
    while True and len(data) < 10000:
        # 根据模型输出的动作概率分布，随机sample一个动作
        action_pos = model(torch.tensor(state))
        action_cumsum = action_pos.flatten().cumsum(dim=0)
        rd = random.random()
        action = torch.argmax(torch.gt(action_cumsum, rd).int()).item()

        # 如果action为0或1，则对应左转或右转
        if action < 2:
            direction = np.dot(rotate[action], direction)

        # 将状态、动作概率分布和动作的三元组存储进数据集中
        data.append((state, action_pos, action))

        # 根据选择的动作得到新状态
        head_x, head_y = snake[0]
        head_x += direction[0]
        head_y += direction[1]

        # 处理穿越边界
        head_x = head_x % 40
        head_y = head_y % 30

        # 检查是否吃到食物
        if snake[0] == food:
            while True:
                food = (random.randint(0, 39), random.randint(0, 29))
                if food not in snake: break
            # reward.append(0)
            state[food[0], food[1]] = 2
            # move_cnt = 0
        else:
            state[snake[-1][0], snake[-1][1]] = 0
            snake.pop()
            # move_cnt += 1
            # if move_cnt > 1000:
            #     reward.append(0)
            #     move_cnt = 0
            # else:
            #     reward.append(0)

        # 添加新蛇头
        snake.insert(0, (head_x, head_y))
        state[head_x, head_y] = 1

        if(len(data) % 100 == 0):
            reward.append(10)
        else:
            reward.append(0)

        # 检查是否撞到自身
        if len(snake) != len(set(snake)):
            print("Crashed! length of snake:", len(snake))
            reward[-1] = -10
            break

    if len(data) == 10000:
        print("Timeout! length of snake:", len(snake))

    tmp_reward = reward[-1]
    for i in range(len(reward) - 2, -1, -1):
        tmp_reward = tmp_reward * decay + reward[i]
        reward[i] = tmp_reward

    return data, reward, len(snake)


def pre_train(episodes, optimizer):
    best_len = 0
    avg_len = 0
    avg_step = 0
    avg_efficiency = 0
    crashed_cnt = 0
    for i in range(episodes):
        model.train()
        data, reward, len_snake = pretrain_play_game(model)
        if len_snake >= best_len:
            best_len = len_snake
        torch.save(model.state_dict(), 'model_start_with_11.pth')
        loss = torch.tensor(0.0)
        for j in range(len(data)):
            loss += nn.CrossEntropyLoss()(data[j][1], torch.tensor([data[j][2]])) * reward[j]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("episode {} loss:{}".format(i + 1, loss.item()))
        avg_len += len_snake / episodes
        avg_step += len(data) / episodes
        avg_efficiency += (len(data) / len_snake) / episodes
        crashed_cnt += (len(data) == 10000)
    print("crashed times:", crashed_cnt)
    print("avg step:", avg_step)
    print("avg efficiency:", avg_efficiency)
    print("avg len:", avg_len)
    print("best len:", best_len)

model = PolicyNetwork()
model.load_state_dict(torch.load('model.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
episodes = 100
pre_train(episodes, optimizer)