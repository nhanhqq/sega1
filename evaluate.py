import asyncio
import time
import matplotlib.pyplot as plt
import urllib.parse
from agent.rl_crawler import RLCrawler
from baselines.bfs_crawler import BFSCrawler
from baselines.focused_crawler import FocusedCrawler

async def run_evaluation():
    target_url = "https://quotes.toscrape.com" # A safe dummy site to crawl
    parsed_url = urllib.parse.urlparse(target_url)
    target_domain = parsed_url.netloc
    max_depth = 2 # Small depth for quick evaluation
    
    print(f"Starting evaluation on {target_domain} (Max Depth: {max_depth})")
    
    results = {}
    
    # 1. Evaluate BFS Crawler
    print("\n--- Running BFS Crawler ---")
    bfs = BFSCrawler(target_domain, max_depth)
    await bfs.initialize()
    start_time = time.time()
    bfs_reward, bfs_steps = await bfs.crawl(target_url)
    bfs_time = time.time() - start_time
    await bfs.close()
    
    results['BFS'] = {'reward': bfs_reward, 'steps': bfs_steps, 'time': bfs_time}
    print(f"BFS finished in {bfs_time:.2f}s | Reward: {bfs_reward} | Steps: {bfs_steps}")
    
    # 2. Evaluate Focused Crawler
    print("\n--- Running Focused Crawler ---")
    focused = FocusedCrawler(target_domain, max_depth, target_keywords=['login', 'tag'])
    await focused.initialize()
    start_time = time.time()
    focused_reward, focused_steps = await focused.crawl(target_url)
    focused_time = time.time() - start_time
    await focused.close()
    
    results['Focused'] = {'reward': focused_reward, 'steps': focused_steps, 'time': focused_time}
    print(f"Focused finished in {focused_time:.2f}s | Reward: {focused_reward} | Steps: {focused_steps}")
    
    # 3. Evaluate RL Crawler
    print("\n--- Running RL Crawler ---")
    rl = RLCrawler(target_domain, max_depth)
    await rl.initialize()
    start_time = time.time()
    # In a real scenario we'd train for many episodes.
    # Here we just run 1 episode to compare it.
    rl_reward, rl_steps = await rl.train_episode(target_url)
    rl_time = time.time() - start_time
    await rl.close()
    
    results['RL (1 Episode)'] = {'reward': rl_reward, 'steps': rl_steps, 'time': rl_time}
    print(f"RL finished in {rl_time:.2f}s | Reward: {rl_reward} | Steps: {rl_steps}")
    
    # Plotting
    names = list(results.keys())
    rewards = [results[n]['reward'] for n in names]
    steps = [results[n]['steps'] for n in names]
    times = [results[n]['time'] for n in names]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].bar(names, rewards, color=['blue', 'orange', 'green'])
    axs[0].set_title('Total Reward (Coverage)')
    axs[0].set_ylabel('Reward')
    
    axs[1].bar(names, steps, color=['blue', 'orange', 'green'])
    axs[1].set_title('Total Steps (Cost)')
    axs[1].set_ylabel('Steps')
    
    axs[2].bar(names, times, color=['blue', 'orange', 'green'])
    axs[2].set_title('Execution Time (Cost)')
    axs[2].set_ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print("\nResults plotted to evaluation_results.png")

if __name__ == "__main__":
    # Workaround for playwright asyncio issue
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(run_evaluation())
