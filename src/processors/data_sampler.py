from typing import Dict, List, Any, Optional, Tuple
import logging
import random
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSampler:
    """
    智能数据采样器，用于从大规模数据集中选择高质量的子集
    """
    
    def __init__(
        self,
        target_size: int = 50000,
        sampling_strategy: str = "quality_first",
        diversity_weight: float = 0.3,
        seed: Optional[int] = 42
    ):
        """
        Args:
            target_size: 目标采样数量
            sampling_strategy: 采样策略 
                - "quality_first": 优先选择高质量样本
                - "stratified": 分层采样，保持类别平衡
                - "diversity": 最大化多样性
                - "hybrid": 混合策略
            diversity_weight: 多样性权重 (0-1)
            seed: 随机种子
        """
        self.target_size = target_size
        self.sampling_strategy = sampling_strategy
        self.diversity_weight = diversity_weight
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.statistics = {
            'total_candidates': 0,
            'selected_samples': 0,
            'rejected_samples': 0,
            'avg_quality_score': 0,
            'category_distribution': {}
        }
    
    def sample_data(
        self,
        data: List[Dict[str, Any]],
        quality_scores: Optional[List[float]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        对数据进行采样
        
        Args:
            data: 所有候选数据
            quality_scores: 每个样本的质量分数
            
        Returns:
            (selected_samples, sampling_stats)
        """
        self.statistics['total_candidates'] = len(data)
        
        if len(data) <= self.target_size:
            logger.info(f"Data size {len(data)} <= target size {self.target_size}, returning all data")
            self.statistics['selected_samples'] = len(data)
            return data, self.statistics
        
        # 根据策略选择采样方法
        if self.sampling_strategy == "quality_first":
            selected = self._quality_first_sampling(data, quality_scores)
        elif self.sampling_strategy == "stratified":
            selected = self._stratified_sampling(data, quality_scores)
        elif self.sampling_strategy == "diversity":
            selected = self._diversity_sampling(data, quality_scores)
        elif self.sampling_strategy == "hybrid":
            selected = self._hybrid_sampling(data, quality_scores)
        else:
            # 默认随机采样
            selected = random.sample(data, self.target_size)
        
        self.statistics['selected_samples'] = len(selected)
        self.statistics['rejected_samples'] = len(data) - len(selected)
        
        # 计算统计信息
        if quality_scores:
            selected_indices = [data.index(s) for s in selected]
            selected_scores = [quality_scores[i] for i in selected_indices]
            self.statistics['avg_quality_score'] = np.mean(selected_scores)
        
        return selected, self.statistics
    
    def _quality_first_sampling(
        self,
        data: List[Dict[str, Any]],
        quality_scores: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        质量优先采样：选择质量分数最高的样本
        """
        if not quality_scores:
            # 如果没有质量分数，从数据中提取
            quality_scores = []
            for item in data:
                if 'quality_metrics' in item and 'overall_score' in item['quality_metrics']:
                    quality_scores.append(item['quality_metrics']['overall_score'])
                else:
                    quality_scores.append(0.5)  # 默认分数
        
        # 按质量分数排序
        indexed_data = list(enumerate(data))
        indexed_data.sort(key=lambda x: quality_scores[x[0]], reverse=True)
        
        # 选择前N个高质量样本
        selected_indices = [idx for idx, _ in indexed_data[:self.target_size]]
        selected = [data[idx] for idx in selected_indices]
        
        logger.info(f"Selected top {len(selected)} samples by quality")
        return selected
    
    def _stratified_sampling(
        self,
        data: List[Dict[str, Any]],
        quality_scores: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        分层采样：根据数据类别进行均衡采样
        """
        # 按类别分组
        categories = defaultdict(list)
        
        for item in data:
            # 尝试多种方式获取类别
            category = None
            
            # 1. 从metadata中获取image_type
            if 'metadata' in item and 'image_type' in item['metadata']:
                category = item['metadata']['image_type']
            # 2. 从质量指标中获取
            elif 'quality_metrics' in item and 'category' in item['quality_metrics']:
                category = item['quality_metrics']['category']
            # 3. 根据问题类型推断
            elif 'question' in item:
                question = item['question'].lower()
                if any(word in question for word in ['calculate', 'solve', 'compute']):
                    category = 'mathematical'
                elif any(word in question for word in ['chart', 'graph', 'plot']):
                    category = 'chart'
                elif any(word in question for word in ['diagram', 'circuit', 'flow']):
                    category = 'diagram'
                else:
                    category = 'general'
            else:
                category = 'unknown'
            
            categories[category].append(item)
        
        # 计算每个类别应该采样的数量
        num_categories = len(categories)
        samples_per_category = self.target_size // num_categories
        remainder = self.target_size % num_categories
        
        selected = []
        category_counts = {}
        
        for i, (category, items) in enumerate(categories.items()):
            # 为某些类别分配额外的样本
            n_samples = samples_per_category + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(items))
            
            # 如果该类别有质量分数，按质量采样
            if quality_scores:
                category_indices = [data.index(item) for item in items]
                category_scores = [quality_scores[idx] for idx in category_indices]
                
                # 按质量排序后采样
                sorted_items = sorted(zip(items, category_scores), 
                                    key=lambda x: x[1], reverse=True)
                category_selected = [item for item, _ in sorted_items[:n_samples]]
            else:
                # 随机采样
                category_selected = random.sample(items, n_samples)
            
            selected.extend(category_selected)
            category_counts[category] = len(category_selected)
        
        self.statistics['category_distribution'] = category_counts
        logger.info(f"Stratified sampling completed: {category_counts}")
        
        return selected
    
    def _diversity_sampling(
        self,
        data: List[Dict[str, Any]],
        quality_scores: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        多样性采样：最大化选中样本的多样性
        """
        selected = []
        remaining = data.copy()
        
        # 如果有质量分数，先选择一些高质量样本作为种子
        if quality_scores:
            seed_size = int(self.target_size * 0.2)  # 20%作为种子
            indexed_data = list(zip(data, quality_scores))
            indexed_data.sort(key=lambda x: x[1], reverse=True)
            
            for item, _ in indexed_data[:seed_size]:
                selected.append(item)
                remaining.remove(item)
        
        # 使用贪婪算法选择最大化多样性的样本
        while len(selected) < self.target_size and remaining:
            # 计算每个候选样本与已选样本的差异度
            max_diversity_score = -1
            best_candidate = None
            
            for candidate in remaining[:min(100, len(remaining))]:  # 限制搜索范围以提高效率
                diversity_score = self._calculate_diversity(candidate, selected)
                
                # 如果有质量分数，结合质量和多样性
                if quality_scores:
                    idx = data.index(candidate)
                    quality = quality_scores[idx]
                    combined_score = (1 - self.diversity_weight) * quality + \
                                   self.diversity_weight * diversity_score
                else:
                    combined_score = diversity_score
                
                if combined_score > max_diversity_score:
                    max_diversity_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # 如果没有找到合适的候选，随机选择
                random_candidate = random.choice(remaining)
                selected.append(random_candidate)
                remaining.remove(random_candidate)
        
        logger.info(f"Diversity sampling selected {len(selected)} samples")
        return selected
    
    def _hybrid_sampling(
        self,
        data: List[Dict[str, Any]],
        quality_scores: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        混合采样策略：结合质量、多样性和分层
        """
        # 分配比例
        quality_portion = int(self.target_size * 0.5)  # 50%高质量
        stratified_portion = int(self.target_size * 0.3)  # 30%分层
        diversity_portion = self.target_size - quality_portion - stratified_portion  # 20%多样性
        
        selected = []
        used_indices = set()
        
        # 1. 质量优先部分
        if quality_scores:
            indexed_data = list(enumerate(data))
            indexed_data.sort(key=lambda x: quality_scores[x[0]], reverse=True)
            
            for idx, item in indexed_data[:quality_portion]:
                selected.append(item)
                used_indices.add(idx)
        
        # 2. 分层采样部分（从剩余数据中）
        remaining_data = [item for i, item in enumerate(data) if i not in used_indices]
        remaining_scores = [quality_scores[i] for i in range(len(data)) if i not in used_indices] if quality_scores else None
        
        # 临时修改target_size进行分层采样
        original_target = self.target_size
        self.target_size = stratified_portion
        stratified_samples = self._stratified_sampling(remaining_data, remaining_scores)
        self.target_size = original_target
        
        selected.extend(stratified_samples)
        
        # 3. 多样性部分（从剩余数据中）
        for item in stratified_samples:
            if item in remaining_data:
                remaining_data.remove(item)
        
        # 临时修改target_size进行多样性采样
        self.target_size = diversity_portion
        diversity_samples = self._diversity_sampling(remaining_data, None)
        self.target_size = original_target
        
        selected.extend(diversity_samples)
        
        # 确保不超过目标大小
        if len(selected) > self.target_size:
            selected = selected[:self.target_size]
        
        logger.info(f"Hybrid sampling completed: {len(selected)} samples")
        return selected
    
    def _calculate_diversity(
        self,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]]
    ) -> float:
        """
        计算候选样本与已选样本集的多样性分数
        """
        if not selected:
            return 1.0
        
        # 简单的多样性度量：基于问题文本的差异
        candidate_question = candidate.get('question', '').lower()
        
        # 计算与已选样本的平均相似度
        similarities = []
        for item in selected[-10:]:  # 只比较最近的10个样本以提高效率
            selected_question = item.get('question', '').lower()
            
            # 简单的词汇重叠度
            candidate_words = set(candidate_question.split())
            selected_words = set(selected_question.split())
            
            if candidate_words or selected_words:
                overlap = len(candidate_words & selected_words)
                total = len(candidate_words | selected_words)
                similarity = overlap / total if total > 0 else 0
            else:
                similarity = 0
            
            similarities.append(similarity)
        
        # 多样性是相似度的反向
        avg_similarity = np.mean(similarities) if similarities else 0
        diversity = 1 - avg_similarity
        
        return diversity
    
    def save_sampling_report(self, output_path: str):
        """
        保存采样报告
        """
        report = {
            'sampling_config': {
                'target_size': self.target_size,
                'strategy': self.sampling_strategy,
                'diversity_weight': self.diversity_weight
            },
            'statistics': self.statistics
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Sampling report saved to {output_path}")