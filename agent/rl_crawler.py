import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import logging
from env.crawler_env import CrawlerEnv
from env.state_extractor import StateExtractor
from agent.dqn_model import DQNModel
from agent.replay_buffer import ReplayBuffer

logger = logging.getLogger("RLCrawler")

class RLCrawler:
    def __init__(self, target_domain, max_depth=5, gamma=0.99, lr=1e-3, batch_size=32, max_features=100, max_pages=100):
        self.env = CrawlerEnv(target_domain, max_depth)
        self.state_extractor = StateExtractor(max_features)
        self.max_pages = max_pages
        
        self.state_dim = max_features + 2 # text features + queue_size + current_depth
        self.q_net = DQNModel(self.state_dim)
        self.target_net = DQNModel(self.state_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer()
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        self.frontier = [] # List of URLs to visit
        
    async def initialize(self):
        await self.env.init_browser()
        
    async def train_episode(self, start_url):
        self.env.reset()
        self.frontier = [start_url]
        current_depth = 0
        total_reward = 0
        steps = 0
        
        # We need an initial state. Let's just create a dummy zero state
        current_state = np.zeros(self.state_dim)
        
        while self.frontier and current_depth <= self.env.max_depth and steps < self.max_pages:
            # Epsilon-greedy action selection from frontier
            if random.random() < self.epsilon:
                # Explore: random URL
                action_idx = random.randint(0, len(self.frontier) - 1)
            else:
                # Exploit: For each URL in frontier, we need to estimate Q-value.
                # In a real scenario, we'd extract text from the anchor tag or URL name.
                # Here, we just use dummy states since we haven't downloaded them yet.
                # To simplify, we randomly select if we don't have a good representation before visiting.
                action_idx = random.randint(0, len(self.frontier) - 1) # simplified
                
            selected_url = self.frontier.pop(action_idx)
            # Khởi tạo state bằng 0 nếu chưa có (simplified)
            # action là index của URL được chọn từ frontier
            
            html, reward, new_links, done = await self.env.step(selected_url, current_depth)
            
            logger.info(f"RL Queue size: {len(self.frontier)} | Epsilon: {self.epsilon:.2f} | Cào: {selected_url}")
            
            # Extract state
            next_state = self.state_extractor.get_state(html, len(self.frontier), current_depth)
            
            # Assuming action is 0 for the selected URL (simplified action space)
            action = 0 
            
            self.memory.push(current_state, action, reward, next_state, done)
            current_state = next_state
            total_reward += reward
            steps += 1
            
            for link in new_links:
                if link not in self.env.visited_urls and link not in self.frontier:
                    self.frontier.append(link)
                    
            self.replay()
            
            if done:
                pass
            
            current_depth += 1
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target net occasionally
        if steps % 10 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return total_reward, steps

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        q_values = self.q_net(state)
        next_q_values = self.target_net(next_state).detach()
        target_q_values = reward + (1 - done) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    async def close(self):
        await self.env.close()
