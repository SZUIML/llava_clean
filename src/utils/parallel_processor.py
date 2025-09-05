import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable
import logging
from threading import Lock
import time
from tqdm import tqdm
import queue

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """多线程并行处理器"""
    
    def __init__(
        self,
        max_workers: int = 4,
        use_threading: bool = True,
        timeout: Optional[int] = None,
        retry_failed: bool = True,
        max_retries: int = 2
    ):
        """
        Args:
            max_workers: 最大线程数
            use_threading: 是否使用多线程（False则串行处理）
            timeout: 单个任务超时时间（秒）
            retry_failed: 是否重试失败的任务
            max_retries: 最大重试次数
        """
        self.max_workers = max_workers
        self.use_threading = use_threading
        self.timeout = timeout
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        
        # 线程安全的计数器和锁
        self.progress_lock = Lock()
        self.stats_lock = Lock()
        self.completed_count = 0
        self.failed_count = 0
        self.retry_count = 0
        
        # 结果收集
        self.results = []
        self.failed_items = []
        
    def process_batch_parallel(
        self,
        items: List[Any],
        process_func: Callable,
        batch_id: int = 0,
        desc: str = "Processing"
    ) -> tuple[List[Any], List[Any]]:
        """
        并行处理一批数据
        
        Args:
            items: 待处理的数据列表
            process_func: 处理单个数据的函数
            batch_id: 批次ID
            desc: 进度条描述
            
        Returns:
            (successful_results, failed_items)
        """
        
        successful_results = []
        failed_items = []
        
        if not self.use_threading or self.max_workers == 1:
            # 串行处理
            logger.info(f"Processing batch {batch_id} in serial mode")
            for item in tqdm(items, desc=desc):
                try:
                    result = process_func(item)
                    if result is not None:
                        successful_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process item {item.get('id', 'unknown')}: {str(e)}")
                    failed_items.append(item)
            
            return successful_results, failed_items
        
        # 并行处理
        logger.info(f"Processing batch {batch_id} with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {}
            for item in items:
                future = executor.submit(self._process_with_retry, process_func, item)
                future_to_item[future] = item
            
            # 使用tqdm显示进度
            with tqdm(total=len(items), desc=desc) as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    
                    try:
                        result = future.result(timeout=self.timeout)
                        if result is not None:
                            successful_results.append(result)
                            with self.stats_lock:
                                self.completed_count += 1
                        else:
                            failed_items.append(item)
                            with self.stats_lock:
                                self.failed_count += 1
                                
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Timeout processing item {item.get('id', 'unknown')}")
                        failed_items.append(item)
                        with self.stats_lock:
                            self.failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing item {item.get('id', 'unknown')}: {str(e)}")
                        failed_items.append(item)
                        with self.stats_lock:
                            self.failed_count += 1
                    
                    pbar.update(1)
        
        logger.info(f"Batch {batch_id} completed: {len(successful_results)} successful, {len(failed_items)} failed")
        return successful_results, failed_items
    
    def _process_with_retry(self, process_func: Callable, item: Any) -> Optional[Any]:
        """处理单个项目，支持重试"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = process_func(item)
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries and self.retry_failed:
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} for item {item.get('id', 'unknown')}")
                    with self.stats_lock:
                        self.retry_count += 1
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    break
        
        # All retries failed
        logger.error(f"Failed after {self.max_retries + 1} attempts: {str(last_exception)}")
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """获取处理统计"""
        with self.stats_lock:
            return {
                'completed': self.completed_count,
                'failed': self.failed_count,
                'retries': self.retry_count
            }
    
    def reset_stats(self):
        """重置统计计数器"""
        with self.stats_lock:
            self.completed_count = 0
            self.failed_count = 0
            self.retry_count = 0


class RateLimitedProcessor(ParallelProcessor):
    """带速率限制的并行处理器（用于API调用）"""
    
    def __init__(
        self,
        max_workers: int = 4,
        requests_per_minute: int = 60,
        **kwargs
    ):
        super().__init__(max_workers=max_workers, **kwargs)
        self.requests_per_minute = requests_per_minute
        self.request_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0
        self.request_lock = Lock()
    
    def _rate_limited_process(self, process_func: Callable, item: Any) -> Optional[Any]:
        """带速率限制的处理"""
        with self.request_lock:
            # 计算需要等待的时间
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.request_interval:
                wait_time = self.request_interval - time_since_last
                time.sleep(wait_time)
            
            self.last_request_time = time.time()
        
        # 执行实际处理
        return self._process_with_retry(process_func, item)
    
    def process_batch_parallel(
        self,
        items: List[Any],
        process_func: Callable,
        batch_id: int = 0,
        desc: str = "Processing"
    ) -> tuple[List[Any], List[Any]]:
        """重写以使用速率限制"""
        
        # 如果有速率限制，包装处理函数
        if self.request_interval > 0:
            wrapped_func = lambda item: self._rate_limited_process(process_func, item)
        else:
            wrapped_func = lambda item: self._process_with_retry(process_func, item)
        
        successful_results = []
        failed_items = []
        
        logger.info(f"Processing batch {batch_id} with rate limit: {self.requests_per_minute} req/min")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {}
            for item in items:
                future = executor.submit(wrapped_func, item)
                future_to_item[future] = item
            
            with tqdm(total=len(items), desc=desc) as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    
                    try:
                        result = future.result(timeout=self.timeout)
                        if result is not None:
                            successful_results.append(result)
                        else:
                            failed_items.append(item)
                            
                    except Exception as e:
                        logger.error(f"Error: {str(e)}")
                        failed_items.append(item)
                    
                    pbar.update(1)
        
        return successful_results, failed_items