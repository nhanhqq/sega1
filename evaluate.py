import asyncio
import time
import json
import logging
import urllib.parse
import matplotlib.pyplot as plt

# Cấu hình logging toàn cục
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler_run.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Evaluate")

from agent.rl_crawler import RLCrawler
from baselines.bfs_crawler import BFSCrawler
from baselines.focused_crawler import FocusedCrawler

async def run_evaluation():
    target_url = "https://vnexpress.net" # Vietnamese news site
    parsed_url = urllib.parse.urlparse(target_url)
    target_domain = parsed_url.netloc
    max_depth = 5 # Tăng độ sâu vì đã có giới hạn số trang
    max_pages = 100 # Chạy tối đa 100 mẫu
    
    logger.info(f"Bắt đầu quá trình Evaluation trên domain: {target_domain} (Max Depth: {max_depth}, Max Pages: {max_pages})")
    
    results = {}
    
    # 1. Evaluate BFS Crawler
    logger.info("\n" + "="*50 + "\n--- Chạy BFS Crawler ---\n" + "="*50)
    bfs = BFSCrawler(target_domain, max_depth, max_pages)
    await bfs.initialize()
    start_time = time.time()
    bfs_reward, bfs_steps = await bfs.crawl(target_url)
    bfs_bandwidth = bfs.env.total_bandwidth / (1024 * 1024) # MB
    bfs_time = time.time() - start_time
    await bfs.close()
    
    results['BFS'] = {'reward': bfs_reward, 'steps': bfs_steps, 'time': bfs_time, 'bandwidth_MB': bfs_bandwidth}
    logger.info(f"BFS hoàn thành trong {bfs_time:.2f}s | Băng thông: {bfs_bandwidth:.2f} MB | Tổng Reward: {bfs_reward} | Tổng số trang cào: {bfs_steps}")
    
    # 2. Evaluate Focused Crawler
    logger.info("\n" + "="*50 + "\n--- Chạy Focused Crawler ---\n" + "="*50)
    # Từ khóa dành riêng cho VnExpress
    focused = FocusedCrawler(target_domain, max_depth, target_keywords=['thể thao', 'kinh doanh', 'công nghệ', 'thế giới', 'thời sự'], max_pages=max_pages)
    await focused.initialize()
    start_time = time.time()
    focused_reward, focused_steps = await focused.crawl(target_url)
    focused_bandwidth = focused.env.total_bandwidth / (1024 * 1024) # MB
    focused_time = time.time() - start_time
    await focused.close()
    
    results['Focused'] = {'reward': focused_reward, 'steps': focused_steps, 'time': focused_time, 'bandwidth_MB': focused_bandwidth}
    logger.info(f"Focused hoàn thành trong {focused_time:.2f}s | Băng thông: {focused_bandwidth:.2f} MB | Tổng Reward: {focused_reward} | Tổng số trang cào: {focused_steps}")
    
    # 3. Evaluate RL Crawler
    logger.info("\n" + "="*50 + "\n--- Chạy RL Crawler (1 Episode) ---\n" + "="*50)
    rl = RLCrawler(target_domain, max_depth, max_pages=max_pages)
    await rl.initialize()
    start_time = time.time()
    
    # Mô phỏng tính năng "Deep Web Search" bằng cách tương tác với form tìm kiếm
    logger.info("RL Crawler thực hiện Deep Web Action: Tìm kiếm form với từ khóa 'thể thao'")
    await rl.env.simulate_deep_web_search("thể thao")
    
    # Chạy 1 episode test
    rl_reward, rl_steps = await rl.train_episode(target_url)
    rl_bandwidth = rl.env.total_bandwidth / (1024 * 1024) # MB
    rl_time = time.time() - start_time
    await rl.close()
    
    results['RL (1 Ep)'] = {'reward': rl_reward, 'steps': rl_steps, 'time': rl_time, 'bandwidth_MB': rl_bandwidth}
    logger.info(f"RL hoàn thành trong {rl_time:.2f}s | Băng thông: {rl_bandwidth:.2f} MB | Tổng Reward: {rl_reward} | Tổng số trang cào: {rl_steps}")
    
    # Plotting
    logger.info("Đang tạo biểu đồ so sánh kết quả...")
    names = list(results.keys())
    rewards = [results[n]['reward'] for n in names]
    bandwidths = [results[n]['bandwidth_MB'] for n in names]
    times = [results[n]['time'] for n in names]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].bar(names, rewards, color=['blue', 'orange', 'green'])
    axs[0].set_title('Độ Phủ (Total Reward / Quality Content)')
    axs[0].set_ylabel('Reward')
    
    axs[1].bar(names, bandwidths, color=['blue', 'orange', 'green'])
    axs[1].set_title('Chi phí tính toán (Băng thông - MB)')
    axs[1].set_ylabel('MB')
    
    axs[2].bar(names, times, color=['blue', 'orange', 'green'])
    axs[2].set_title('Chi phí tính toán (Thời gian - s)')
    axs[2].set_ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    
    # Lưu kết quả tính toán thô ra JSON
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    logger.info("Kết quả tính toán đã được lưu vào 'evaluation_results.json'.")
    logger.info("Kết quả biểu đồ đã được lưu thành tệp 'evaluation_results.png'. Hãy mở tệp này để xem.")
    logger.info("Hoàn tất quá trình cào dữ liệu! Text của các bài báo đã được lưu trong folder 'data/vnexpress_net/'.")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    try:
        asyncio.run(run_evaluation())
    except KeyboardInterrupt:
        logger.info("Người dùng đã buộc dừng script (KeyboardInterrupt).")
