import pygame
from pygame.locals import *
import random
import numpy as np
import pickle

# Configurações da tela e pixel
screen_size = (600, 600)
pixel_size = 10

# Funções auxiliares
def random_on_grid():
    x = random.randint(0, (screen_size[0] // pixel_size) - 1) * pixel_size
    y = random.randint(0, (screen_size[1] // pixel_size) - 1) * pixel_size
    return x, y

def collision(pos1, pos2):
    return pos1 == pos2

def off_limits(position):
    return not (0 <= position[0] < screen_size[0] and 0 <= position[1] < screen_size[1])

# Inicialização do Pygame
pygame.init()
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Snake 2 Player')

# Jogador 1
player1_position = [(250, 50)]
player1_direction = K_RIGHT

# Jogador 2 (Agente)
player2_position = [(100, 100)]
player2_direction = K_LEFT

# Maças
apple_surface = pygame.Surface((pixel_size, pixel_size))
apple_surface.fill((255, 0, 0))
apple_position = random_on_grid()

# Função para reiniciar o jogo
def restart_game():
    global player1_position, player1_direction
    global player2_position, player2_direction
    global apple_position
    
    player1_position = [(250, 50)]
    player1_direction = K_RIGHT
    
    player2_position = [(100, 100)]
    player2_direction = K_LEFT
    
    apple_position = random_on_grid()

# Função para obter o estado do agente
def get_agent_state():
    head_x = player2_position[0][0] // pixel_size
    head_y = player2_position[0][1] // pixel_size
    return (head_x, head_y)

# Inicialização do agente
class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((60, 60, 4))  # Q-table com estados e ações
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.rewards_per_episode = []
        self.penalties_per_episode = []

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def end_episode(self, total_reward, total_penalty):
        self.rewards_per_episode.append(total_reward)
        self.penalties_per_episode.append(total_penalty)

    def save_q_table(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename='q_table.pkl'):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

    def save_rewards(self, filename='rewards.txt'):
        np.savetxt(filename, self.rewards_per_episode)

    def load_rewards(self, filename='rewards.txt'):
        self.rewards_per_episode = np.loadtxt(filename).tolist()

    def save_penalties(self, filename='penalties.txt'):
        np.savetxt(filename, self.penalties_per_episode)

    def load_penalties(self, filename='penalties.txt'):
        self.penalties_per_episode = np.loadtxt(filename).tolist()
        
agent = QLearningAgent()

# Loop principal do jogo
total_reward = 0
total_penalty = 0
done = False

while True:
    pygame.time.Clock().tick(10)  # Controle de FPS
    screen.fill((0, 0, 0))
    
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            quit()
        elif event.type == KEYDOWN:
            if event.key in [K_UP, K_DOWN, K_LEFT, K_RIGHT]:
                player1_direction = event.key

    # Atualizar posições dos jogadores
    if player1_direction == K_UP:
        new_head = (player1_position[0][0], player1_position[0][1] - pixel_size)
    elif player1_direction == K_DOWN:
        new_head = (player1_position[0][0], player1_position[0][1] + pixel_size)
    elif player1_direction == K_LEFT:
        new_head = (player1_position[0][0] - pixel_size, player1_position[0][1])
    elif player1_direction == K_RIGHT:
        new_head = (player1_position[0][0] + pixel_size, player1_position[0][1])
    
    player1_position.insert(0, new_head)

    if collision(apple_position, player1_position[0]):
        apple_position = random_on_grid()
        total_reward += 1  # Recompensa por comer maçã vermelha
    else:
        player1_position.pop()

    if player1_position[0] in player1_position[1:] or off_limits(player1_position[0]):
        restart_game()

    # Atualização do Jogador 2 (Agente)
    agent_state = get_agent_state()
    player2_action = agent.choose_action(agent_state)

    if player2_action == 0:  # UP
        new_head2 = (player2_position[0][0], player2_position[0][1] - pixel_size)
    elif player2_action == 1:  # DOWN
        new_head2 = (player2_position[0][0], player2_position[0][1] + pixel_size)
    elif player2_action == 2:  # LEFT
        new_head2 = (player2_position[0][0] - pixel_size, player2_position[0][1])
    elif player2_action == 3:  # RIGHT
        new_head2 = (player2_position[0][0] + pixel_size, player2_position[0][1])

    # Verifique se o agente colidiu
    if new_head2 in player2_position[1:] or off_limits(new_head2):
        reward = -1  # Penalidade por colidir
        total_penalty += 1  # Contabiliza a penalidade
        done = True
        restart_game()
    else:
        player2_position.insert(0, new_head2)
        player2_position.pop()  # O agente não cresce com a movimentação

        reward = 0  # Sem recompensa ou penalidade por movimento

    # Atualiza a Q-table do agente
    next_state = get_agent_state()
    agent.update_q_table(agent_state, player2_action, reward, next_state)

    # Se o episódio terminou
    if done:
        agent.end_episode(total_reward, total_penalty)  # Armazena recompensas e penalidades
        agent.save_q_table()  # Salva a tabela Q após o episódio
        agent.save_rewards()   # Salva as recompensas após o episódio
        agent.save_penalties()  # Salva as penalidades após o episódio
        total_reward = 0  # Reseta a recompensa total para o próximo episódio
        total_penalty = 0  # Reseta a penalidade total para o próximo episódio
        done = False  # Reseta a condição de término

    # Desenhar tudo na tela
    for pos in player1_position:
        pygame.draw.rect(screen, (255, 255, 255), (pos[0], pos[1], pixel_size, pixel_size))

    for pos in player2_position:
        pygame.draw.rect(screen, (0, 255, 0), (pos[0], pos[1], pixel_size, pixel_size))

    screen.blit(apple_surface, apple_position)
    pygame.display.update()
