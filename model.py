import math

import numpy as np
import torch
from torch import nn
import random
import pygame

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

def play_game(model, isVisible):
    # 初始化游戏状态，蛇长为1，并在地图中间，随机生成一个食物位置
    state = np.zeros(shape=(40, 30), dtype=np.float32)
    state[19, 14] = 1
    snake = [(19, 14)]
    food = random.randint(0, 39), random.randint(0, 29)
    while food[0] == 19 and food[1] == 14:
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

    if isVisible:
        # 初始化Pygame
        pygame.init()

        # 定义常量
        WIDTH, HEIGHT = 800, 600
        GRID_SIZE = 20
        FPS = 240

        # 定义颜色
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)

        # 创建游戏窗口
        window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("贪吃蛇游戏")

        # 游戏循环
        clock = pygame.time.Clock()
        running = True
        game_over = False
        move_cnt = 0
        while running:
            move_cnt += 1
            if move_cnt == 10000: break
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not game_over:
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

                tmp_reward = 0
                # 检查是否吃到食物
                if snake[0] == food:
                    while True:
                        food = (random.randint(0, 39), random.randint(0, 29))
                        if food not in snake: break
                    state[food[0], food[1]] = 2
                    tmp_reward += 5
                    print("eat a food, now the length of snake:", len(snake) + 1)
                else:
                    state[snake[-1][0], snake[-1][1]] = 0
                    snake.pop()

                # 添加新蛇头
                snake.insert(0, (head_x, head_y))
                state[head_x, head_y] = 1

                # 检查是否撞到自身
                if len(snake) != len(set(snake)):
                    game_over = True
                    running = False
                    print("Crashed!, the length of snake:", len(snake))
                    reward.append(0)
                else:
                    reward.append(tmp_reward)

            # 绘制游戏界面
            window.fill(WHITE)
            pygame.draw.rect(window, RED, (food[0] * GRID_SIZE, food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            for segment in snake:
                pygame.draw.rect(window, BLACK, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

            # 显示游戏结束文字
            if game_over:
                font = pygame.font.Font(None, 36)
                text = font.render("Game Over!", True, BLACK)
                text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                window.blit(text, text_rect)

            pygame.display.flip()

            # 控制游戏帧率
            clock.tick(FPS)

        # 退出游戏
        pygame.quit()
    else:
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
                reward.append(5)
                state[food[0], food[1]] = 2
                move_cnt = 0
            else:
                state[snake[-1][0], snake[-1][1]] = 0
                snake.pop()
                move_cnt += 1
                if move_cnt > 1000:
                    reward.append(-5)
                    move_cnt = 0
                else:
                    reward.append(0)

            # 添加新蛇头
            snake.insert(0, (head_x, head_y))
            state[head_x, head_y] = 1

            # 检查是否撞到自身
            if len(snake) != len(set(snake)):
                print("length of snake:", len(snake))
                break

        if len(data) == 10000:
            print("length of snake:", len(snake))

    tmp_reward = reward[-1]
    for i in range(len(reward) - 2, -1, -1):
        tmp_reward = tmp_reward * decay + reward[i]
        reward[i] = tmp_reward

    return data, reward, len(snake)

def train(model, optimizer, episodes):
    best_len = 0
    for i in range(episodes):
        model.train()
        data, reward, len_snake = play_game(model, False)
        if len_snake >= best_len:
            best_len = len_snake
            torch.save(model.state_dict(), 'model.pth')
        loss = torch.tensor(0.0)
        for j in range(len(data)):
            loss += nn.CrossEntropyLoss()(data[j][1], torch.tensor([data[j][2]])) * reward[j]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print("episode {} loss:{}".format(i + 1, loss.item()))


if __name__ == '__main__':
    model = PolicyNetwork()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    play_game(model, True)
    # model = PolicyNetwork()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # episodes = 1000
    # train(model, optimizer, episodes)