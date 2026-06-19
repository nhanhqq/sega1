import asyncio
import urllib.parse
from env.crawler_env import CrawlerEnv

class BFSCrawler:
    def __init__(self, target_domain, max_depth=5):
        self.env = CrawlerEnv(target_domain, max_depth)
        self.frontier = []
        
    async def initialize(self):
        await self.env.init_browser()
        
    async def crawl(self, start_url):
        self.env.reset()
        self.frontier = [(start_url, 0)]
        total_reward = 0
        steps = 0
        
        while self.frontier:
            current_url, depth = self.frontier.pop(0) # BFS: pop from front
            
            if depth > self.env.max_depth:
                continue
                
            html, reward, new_links, done = await self.env.step(current_url, depth)
            total_reward += reward
            steps += 1
            
            for link in new_links:
                if link not in self.env.visited_urls and not any(link == q_url for q_url, _ in self.frontier):
                    self.frontier.append((link, depth + 1))
                    
        return total_reward, steps
        
    async def close(self):
        await self.env.close()
