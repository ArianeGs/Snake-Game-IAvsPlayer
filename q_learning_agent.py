import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((60, 60, 4))  # Q-table com estados e ações
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.rewards_per_episode = []  # Para armazenar recompensas
        self.penalties_per_episode = []  # Para armazenar penalidades

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
        self.penalties_per_episode.append(total_penalty)  # Armazena penalidades

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
