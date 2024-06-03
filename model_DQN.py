import math
import time

import numpy as np
import torch
from torch import nn
import random
import pygame
from datetime import datetime

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.network(x.view(1, 11))
def get_state(snake, food, direction: np.ndarray):
    state = np.zeros(shape=(11, ), dtype=np.float32)
    # 首先确定方向
    if tuple(direction.tolist()) == (1, 0):  # 向右
        state[0] = 1
    elif tuple(direction.tolist()) == (0, -1):  # 向上
        state[1] = 1
    elif tuple(direction.tolist()) == (-1, 0):  # 向左
        state[2] = 1
    else:
        state[3] = 1

    # 然后确定食物位置
    if food[0] > snake[0][0]:  # 在右边
        state[4] = 1
    if food[1] < snake[0][1]:  # 在上边
        state[5] = 1
    if food[0] < snake[0][0]:  # 在左边
        state[6] = 1
    if food[1] > snake[0][1]:  # 在下边
        state[7] = 1

    # 最后确定危险方向
    right_rotate_direction = np.array([[0, -1], [1, 0]]).dot(direction)
    left_rotate_direction = np.array([[0, 1], [-1, 0]]).dot(direction)
    if ((snake[0][0] + direction[0]) % 10, (snake[0][1] + direction[1]) % 10) in snake:
        state[8] = 1  # 直走有危险
    if ((snake[0][0] + right_rotate_direction[0]) % 10, (snake[0][1] + right_rotate_direction[1]) % 10) in snake:
        state[9] = 1  # 右转有危险
    if ((snake[0][0] + left_rotate_direction[0]) % 10, (snake[0][1] + left_rotate_direction[1]) % 10) in snake:
        state[10] = 1  # 左转有危险

    return state

def play_game(model, optimizer, loss_function, epsilon, training):
    if training:
        model.train()
    else:
        model.eval()

    avg_loss = 0.0
    step = 0

    # 初始化游戏状态，蛇长为1，并在地图中间，随机生成一个食物位置
    snake = [(4, 4)]
    food = random.randint(0, 9), random.randint(0, 9)
    while food[0] == 4 and food[1] == 4:
        food = random.randint(0, 9), random.randint(0, 9)

    # 初始化超参数
    decay = 0.8  # 衰减率
    alpha = 0.1  # 学习率

    # 初始化移动方向
    direction = np.array([1, 0])
    right_rotate = np.array([[0, -1], [1, 0]])
    left_rotate = np.array([[0, 1], [-1, 0]])
    rotate = (left_rotate, right_rotate)

    # 获取游戏状态
    state = get_state(snake, food, direction)
    game_over = False

    # 开始玩游戏
    while not game_over:
        step += 1
        # 根据模型输出的动作概率分布，随机sample一个动作
        action_pos = model(torch.tensor(state))
        # 有1-epsilon的概率进行exploration
        if training and random.random() > epsilon:
            action = random.randint(0, 2)
        else:
            action = torch.argmax(action_pos).item()

        # 如果action为0或1，则对应左转或右转
        if action < 2:
            direction = np.dot(rotate[action], direction)

        # 根据选择的动作得到新状态
        head_x, head_y = snake[0]
        head_x += direction[0]
        head_y += direction[1]

        # 处理穿越边界
        head_x = head_x % 10
        head_y = head_y % 10

        # 检查是否吃到食物
        if snake[0] == food:
            while True:
                food = (random.randint(0, 9), random.randint(0, 9))
                if food not in snake: break
            reward = 5
        else:
            snake.pop()
            reward = 0

        # 添加新蛇头
        snake.insert(0, (head_x, head_y))

        # 获取新状态
        state = get_state(snake, food, direction)

        # 检查是否撞到自身
        if len(snake) != len(set(snake)):
            print("Crashed! length of snake:", len(snake))
            reward = -5
            game_over = True

        # 训练模式下，对参数进行更新
        if training:
            old_q = action_pos[0, action]
            new_q = old_q + alpha * (reward + decay * model(torch.tensor(state)).max() - old_q)
            loss = loss_function(new_q, old_q)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return step, avg_loss / step, len(snake)

def train(model, optimizer, loss_function, episodes):
    for i in range(episodes):
        step, avg_loss, length = play_game(model, optimizer, loss_function, 0.9, True)
        print("Episode {} / {}".format(i + 1, episodes))
        print("avg loss: {}, length of snake: {}".format(avg_loss, length))
        print("------------------------------")
        if (i + 1) % 200 == 0:
            evaluate_by_playing(model)
            evaluate_by_statistic(model, 100)

def evaluate_by_playing(model):
    model.eval()

    # 初始化Pygame
    pygame.init()

    # 定义常量
    WIDTH, HEIGHT = 600, 600
    GRID_SIZE = 60
    FPS = 10
    MARGIN = 5

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

    # 初始化游戏状态，蛇长为1，并在地图中间，随机生成一个食物位置
    step = 0
    snake = [(4, 4)]
    food = random.randint(0, 9), random.randint(0, 9)
    while food[0] == 4 and food[1] == 4:
        food = random.randint(0, 9), random.randint(0, 9)

    # 初始化移动方向
    direction = np.array([1, 0])
    right_rotate = np.array([[0, -1], [1, 0]])
    left_rotate = np.array([[0, 1], [-1, 0]])
    rotate = (left_rotate, right_rotate)

    # 获取游戏状态
    state = get_state(snake, food, direction)

    while running and step < 1000:
        step += 1

        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            # 根据模型输出的动作概率分布，随机sample一个动作
            action_pos = model(torch.tensor(state))
            action = action_pos.argmax().item()

            # 如果action为0或1，则对应左转或右转
            if action < 2:
                direction = np.dot(rotate[action], direction)

            # 根据选择的动作得到新状态
            head_x, head_y = snake[0]
            head_x += direction[0]
            head_y += direction[1]

            # 处理穿越边界
            head_x = head_x % 10
            head_y = head_y % 10

            # 检查是否吃到食物
            if snake[0] == food:
                while True:
                    food = (random.randint(0, 9), random.randint(0, 9))
                    if food not in snake: break
            else:
                snake.pop()

            # 添加新蛇头
            snake.insert(0, (head_x, head_y))

            # 获取新状态
            state = get_state(snake, food, direction)

            # 检查是否撞到自身
            if len(snake) != len(set(snake)):
                game_over = True
                running = False
                print("Crashed!, the length of snake:", len(snake))

        # 绘制游戏界面
        window.fill(WHITE)
        pygame.draw.rect(window, RED, (food[0] * GRID_SIZE + MARGIN, food[1] * GRID_SIZE + MARGIN, GRID_SIZE - MARGIN, GRID_SIZE - MARGIN))
        for segment in snake:
            pygame.draw.rect(window, BLACK, (segment[0] * GRID_SIZE + MARGIN, segment[1] * GRID_SIZE + MARGIN, GRID_SIZE - MARGIN, GRID_SIZE - MARGIN))

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


def evaluate_by_statistic(model, cnt):
    model.eval()
    print('--------------------------')
    print("Evaluating...")

    best_len = 0
    avg_len = 0
    avg_step = 0
    avg_efficiency = 0

    for i in range(cnt):
        step, loss, length = play_game(model, None, None, None, False)
        best_len = max(best_len, length)
        avg_len += length
        avg_step += step
        avg_efficiency += (step / length)

    avg_len /= cnt
    avg_step /= cnt
    avg_efficiency /= cnt

    torch.save(model.state_dict(), './checkpoint/model_' + datetime.now().strftime("%Y-%m-%d-%H%M") + '.pth')
    print('--------------------------')
    print('Model saved!')
    print("avg step:", avg_step)
    print("avg efficiency:", avg_efficiency)
    print("avg len:", avg_len)
    print("best len:", best_len)
    print('--------------------------')
    with open('./checkpoint/description.md', 'a') as file:
        file.write('\n|{}|{}|{:.2f}|{:.2f}|{:.2f}|'.format(
            datetime.now().strftime("%Y-%m-%d-%H%M"),
            best_len,
            avg_len,
            avg_efficiency,
            avg_step,
        ))

    print("Evaluation finished...")
    print('--------------------------')


if __name__ == '__main__':
    model = QNetwork()
    model.load_state_dict(torch.load('model.pth'))
    # evaluate_by_playing(model)
    # play_game(model, True, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    episodes = 600
    train(model, optimizer, loss_function, episodes)