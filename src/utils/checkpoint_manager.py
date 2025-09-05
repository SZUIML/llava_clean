import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class CheckpointManager:
    """管理断点续传功能的检查点"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_checkpoint_file = self.checkpoint_dir / "current_checkpoint.json"
        self.processed_ids_file = self.checkpoint_dir / "processed_ids.pkl"
        self.failed_ids_file = self.checkpoint_dir / "failed_ids.pkl"
        
    def save_checkpoint(
        self,
        processed_ids: set,
        failed_ids: set,
        current_batch: int,
        total_batches: int,
        stats: Dict[str, Any],
        config: Dict[str, Any]
    ):
        """保存检查点"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "current_batch": current_batch,
            "total_batches": total_batches,
            "processed_count": len(processed_ids),
            "failed_count": len(failed_ids),
            "stats": stats,
            "config_hash": hash(json.dumps(config, sort_keys=True))
        }
        
        # 保存主检查点信息
        with open(self.current_checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # 保存已处理的ID集合
        with open(self.processed_ids_file, 'wb') as f:
            pickle.dump(processed_ids, f)
        
        # 保存失败的ID集合
        with open(self.failed_ids_file, 'wb') as f:
            pickle.dump(failed_ids, f)
        
        logger.info(f"Checkpoint saved: batch {current_batch}/{total_batches}, "
                   f"processed: {len(processed_ids)}, failed: {len(failed_ids)}")
    
    def load_checkpoint(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        if not self.current_checkpoint_file.exists():
            logger.info("No checkpoint found, starting fresh")
            return None
        
        try:
            # 加载主检查点信息
            with open(self.current_checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # 检查配置是否改变
            current_config_hash = hash(json.dumps(config, sort_keys=True))
            if checkpoint.get('config_hash') != current_config_hash:
                logger.warning("Configuration has changed since last checkpoint. Starting fresh.")
                return None
            
            # 加载已处理的ID集合
            processed_ids = set()
            if self.processed_ids_file.exists():
                with open(self.processed_ids_file, 'rb') as f:
                    processed_ids = pickle.load(f)
            
            # 加载失败的ID集合
            failed_ids = set()
            if self.failed_ids_file.exists():
                with open(self.failed_ids_file, 'rb') as f:
                    failed_ids = pickle.load(f)
            
            checkpoint['processed_ids'] = processed_ids
            checkpoint['failed_ids'] = failed_ids
            
            logger.info(f"Checkpoint loaded from {checkpoint['timestamp']}")
            logger.info(f"Resuming from batch {checkpoint['current_batch']}/{checkpoint['total_batches']}")
            logger.info(f"Already processed: {len(processed_ids)}, failed: {len(failed_ids)}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def clear_checkpoint(self):
        """清除检查点"""
        files_to_remove = [
            self.current_checkpoint_file,
            self.processed_ids_file,
            self.failed_ids_file
        ]
        
        for file_path in files_to_remove:
            if file_path.exists():
                os.remove(file_path)
                logger.info(f"Removed checkpoint file: {file_path}")
    
    def save_intermediate_results(
        self,
        batch_num: int,
        results: List[Dict[str, Any]],
        output_dir: str
    ):
        """保存中间结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSONL格式，便于追加
        intermediate_file = output_path / f"batch_{batch_num:04d}.jsonl"
        
        with open(intermediate_file, 'w') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved intermediate results to {intermediate_file}")
        return str(intermediate_file)
    
    def merge_intermediate_results(
        self,
        output_dir: str,
        final_output_path: str,
        clean_intermediate: bool = False
    ):
        """合并所有中间结果"""
        output_path = Path(output_dir)
        intermediate_files = sorted(output_path.glob("batch_*.jsonl"))
        
        if not intermediate_files:
            logger.warning("No intermediate files found to merge")
            return
        
        all_results = []
        for file_path in intermediate_files:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
        
        # 保存最终结果
        final_path = Path(final_output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        
        if final_path.suffix == '.jsonl':
            with open(final_path, 'w') as f:
                for item in all_results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(final_path, 'w') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Merged {len(intermediate_files)} files into {final_path}")
        logger.info(f"Total samples: {len(all_results)}")
        
        # 清理中间文件
        if clean_intermediate:
            for file_path in intermediate_files:
                os.remove(file_path)
                logger.info(f"Removed intermediate file: {file_path}")