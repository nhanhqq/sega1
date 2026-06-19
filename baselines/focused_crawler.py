import asyncio
import logging
import urllib.parse
from env.crawler_env import CrawlerEnv

logger = logging.getLogger("FocusedCrawler")

class FocusedCrawler:
    def __init__(self, target_domain, max_depth=5, target_keywords=None, max_pages=100):
        self.env = CrawlerEnv(target_domain, max_depth)
        self.max_pages = max_pages
        self.frontier = []
        self.target_keywords = target_keywords or []
        
    async def initialize(self):
        await self.env.init_browser()
        
    def heuristic_score(self, url):
        # A simple heuristic: check if url contains target keywords
        score = 0
        url_lower = url.lower()
        for kw in self.target_keywords:
            if kw.lower() in url_lower:
                score += 1
        return score
        
    async def crawl(self, start_url):
        self.env.reset()
        # queue stores tuples of (score, depth, url)
        # We want to pop the highest score first, so we'll sort the frontier
        self.frontier = [(self.heuristic_score(start_url), 0, start_url)]
        total_reward = 0
        steps = 0
        
        while self.frontier and steps < self.max_pages:
            # Sort to simulate priority queue (highest score first)
            self.frontier.sort(key=lambda x: x[0], reverse=True)
            score, depth, current_url = self.frontier.pop(0)
            
            if depth > self.env.max_depth:
                continue
                
            html, reward, new_links, done = await self.env.step(current_url, depth)
            total_reward += reward
            steps += 1
            
            logger.info(f"Focused Queue size: {len(self.frontier)} | Điểm heuristic: {score} | Cào: {current_url}")
            
            for link in new_links:
                if link not in self.env.visited_urls and not any(link == q_url for _, _, q_url in self.frontier):
                    self.frontier.append((self.heuristic_score(link), depth + 1, link))
                    
        return total_reward, steps
        
    async def close(self):
        await self.env.close()
