import pygame
import random
import torch
import numpy as np
from model import PolicyNetwork

# 初始化Pygame
pygame.init()

# 加载模型
# model = PolicyNetwork()
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

# 存储数据集
data = []

# 定义常量
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
FPS = 10

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# 创建游戏窗口
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("贪吃蛇游戏")

# 初始化游戏状态，蛇长为1，并在地图中间，随机生成一个食物位置
state = np.zeros(shape=(40, 30), dtype=np.float32)
state[19, 14] = 1
snake = [(19, 14)]
food = random.randint(0, 39), random.randint(0, 29)
while food[0] == 19 and food[1] == 14:
    food = random.randint(0, 39), random.randint(0, 29)
state[food[0], food[1]] = 2

# 初始化移动方向
snake_direction = 'RIGHT'
direction = np.array([1, 0])
left_rotate = np.array([[0, -1], [1, 0]])
right_rotate = np.array([[0, 1], [-1, 0]])
rotate = (left_rotate, right_rotate)

# 游戏循环
clock = pygame.time.Clock()
running = True
game_over = False
while running:
    action = 2
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and snake_direction != 'DOWN':
                action = snake_direction == 'RIGHT'
                snake_direction = 'UP'
            elif event.key == pygame.K_DOWN and snake_direction != 'UP':
                action = snake_direction == 'LEFT'
                snake_direction = 'DOWN'
            elif event.key == pygame.K_LEFT and snake_direction != 'RIGHT':
                action = snake_direction == 'UP'
                snake_direction = 'LEFT'
            elif event.key == pygame.K_RIGHT and snake_direction != 'LEFT':
                action = snake_direction == 'DOWN'
                snake_direction = 'RIGHT'

    data.append(np.append(state.flatten(), action))
    print(data[-1].shape)

    if not game_over:

        # 根据选择的动作得到新状态
        head_x, head_y = snake[0]
        if snake_direction == 'UP':
            head_y -= 1
        elif snake_direction == 'DOWN':
            head_y += 1
        elif snake_direction == 'LEFT':
            head_x -= 1
        elif snake_direction == 'RIGHT':
            head_x += 1


        # 处理穿越边界
        head_x = head_x % 40
        head_y = head_y % 30

        # 检查是否吃到食物
        if snake[0] == food:
            while True:
                food = (random.randint(0, 39), random.randint(0, 29))
                if food not in snake: break
            state[food[0], food[1]] = 2
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

    print("step: ",len(data))
    # 控制游戏帧率
    clock.tick(FPS)

# 退出游戏
pygame.quit()
np.save('data5.npy', np.array(data))

