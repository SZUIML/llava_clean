#!/usr/bin/env python3
"""
LLaVA-CoT Dataset Cleaning Pipeline
Based on R1-Onevision methodology
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.data_loader import LLaVACoTDataLoader, ImageProcessor
from utils.checkpoint_manager import CheckpointManager
from utils.parallel_processor import RateLimitedProcessor
from models.image_description import ImageFormalDescriptionGenerator
from processors.cot_restructure import CoTRestructurer, AnswerExtractor
from processors.quality_check import QualityChecker, DataFilter
from processors.data_sampler import DataSampler


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


class LLaVACoTCleaner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Statistics and locks
        self.stats = {
            'total_samples': 0,
            'processed_samples': 0,
            'passed_samples': 0,
            'failed_samples': 0,
            'errors': 0,
            'processing_time': 0,
            'retries': 0
        }
        self.stats_lock = Lock()
    
    def _initialize_components(self):
        """Initialize all processing components"""
        self.logger.info("Initializing components...")
        
        # Data loader
        self.data_loader = LLaVACoTDataLoader(
            data_path=self.config['data']['input_path'],
            image_dir=self.config['data']['image_dir'],
            batch_size=self.config['processing']['batch_size'],
            max_samples=self.config['data'].get('max_samples'),
            limit_to_first_half=self.config['data'].get('limit_to_first_half', False)
        )
        
        # Image processor
        self.image_processor = ImageProcessor()
        
        # Formal description generator
        self.description_generator = ImageFormalDescriptionGenerator(
            vlm_api_key=self.config['api_keys']['openai'],
            use_object_detection=self.config['models'].get('use_object_detection', True),
            use_ocr=self.config['models'].get('use_ocr', True),
            vlm_model=self.config['models'].get('vlm_model', 'gpt-4o'),
            device=self.config['processing'].get('device', 'cuda'),
            ocr_fallback_on_error=self.config['models'].get('ocr_fallback_on_error', True),
            grounding_dino_config=self.config['models'].get('grounding_dino_config')
        )
        
        # CoT restructurer
        self.cot_restructurer = CoTRestructurer(
            api_key=self.config['api_keys']['openai'],
            model_name=self.config['models'].get('cot_model', 'gpt-4o')
        )
        
        # Answer extractor
        self.answer_extractor = AnswerExtractor()
        
        # Quality checker and filter
        self.quality_checker = QualityChecker(
            api_key=self.config['api_keys']['openai'],
            model_name=self.config['models'].get('quality_model', 'gpt-4o'),
            quality_threshold=self.config['quality'].get('min_score', 0.7)
        )
        
        self.data_filter = DataFilter(
            quality_checker=self.quality_checker,
            min_quality_score=self.config['quality'].get('min_score', 0.7),
            use_llm_verification=self.config['quality'].get('use_llm_verification', True),
            sample_rate_for_llm=self.config['quality'].get('llm_sample_rate', 0.1)
        )
        
        # Data sampler (if enabled)
        self.data_sampler = None
        if self.config.get('sampling', {}).get('enabled', False):
            self.data_sampler = DataSampler(
                target_size=self.config['sampling'].get('target_size', 50000),
                sampling_strategy=self.config['sampling'].get('strategy', 'hybrid'),
                diversity_weight=self.config['sampling'].get('diversity_weight', 0.3),
                seed=self.config['sampling'].get('seed', 42)
            )
        
        # Checkpoint manager
        self.checkpoint_manager = None
        if self.config['checkpoint'].get('enabled', False):
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.config['checkpoint'].get('dir', './checkpoints')
            )

        # Parallel Processor (if enabled)
        self.parallel_processor = None
        if self.config['processing'].get('use_threading', False):
            self.parallel_processor = RateLimitedProcessor(
                max_workers=self.config['processing'].get('max_workers', 4),
                requests_per_minute=self.config['processing'].get('requests_per_minute', 60),
                retry_failed=self.config['processing'].get('retry_failed', True),
                max_retries=self.config['processing'].get('max_retries', 2)
            )

        self.logger.info("Components initialized successfully")

    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample"""
        try:
            sample_id = sample.get('id', 'unknown')
            self.logger.debug(f"Processing sample: {sample_id}")
            
            # Load image
            image = self.image_processor.load_image(sample['image_path'])
            if image is None:
                self.logger.warning(f"Failed to load image for sample {sample_id}")
                return None
            
            # Preprocess image
            image = self.image_processor.preprocess_image(image)
            
            # Generate formal description
            self.logger.debug(f"Generating formal description for {sample_id}")
            description_result = self.description_generator.generate_formal_description(
                image=image,
                image_path=sample['image_path'],
                question=sample.get('question'),
                existing_cot=sample.get('original_cot')
            )
            
            # Restructure CoT
            self.logger.debug(f"Restructuring CoT for {sample_id}")
            restructured_cot = self.cot_restructurer.restructure_cot(
                question=sample['question'],
                formal_description=description_result['image_formal_description'],
                original_cot=sample['original_cot'],
                image_type=description_result.get('image_type')
            )
            
            # Extract and format answer
            final_answer = self.answer_extractor.extract_answer(
                cot_text=restructured_cot,
                original_answer=sample.get('original_answer')
            )
            
            # Use LLM for better answer extraction if needed
            if self.config['processing'].get('use_llm_answer_extraction', False):
                final_answer = self.answer_extractor.refine_answer_with_llm(
                    cot_text=restructured_cot,
                    question=sample['question'],
                    api_key=self.config['api_keys']['openai']
                )
            
            # Create processed sample
            processed_sample = {
                'id': sample_id,
                'image_path': sample['image_path'],
                'question': sample['question'],
                'image_formal_description': description_result['image_formal_description'],
                'cot_thinking': restructured_cot,
                'final_answer': final_answer,
                'metadata': {
                    'image_type': description_result.get('image_type'),
                    'original_cot': sample['original_cot'],
                    'original_answer': sample.get('original_answer'),
                    'processing_details': {
                        'dense_caption': description_result.get('dense_caption'),
                        'has_object_detection': bool(description_result.get('object_detections')),
                        'has_ocr': bool(description_result.get('ocr_results'))
                    }
                }
            }
            
            return processed_sample
            
        except Exception as e:
            self.logger.error(f"Error processing sample {sample.get('id')}: {str(e)}")
            self.stats['errors'] += 1
            return None
    
    def _process_and_filter_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Wraps processing and filtering for a single sample."""
        processed_sample = self.process_sample(sample)
        if not processed_sample:
            return None

        passed, filtered_sample = self.data_filter.filter_sample(processed_sample)
        
        if passed or self.config['quality'].get('keep_failed_samples', False):
            filtered_sample['passed_quality_check'] = passed
            
            # Use lock for thread-safe statistics update
            with self.stats_lock:
                if passed:
                    self.stats['passed_samples'] += 1
                else:
                    self.stats['failed_samples'] += 1
                self.stats['processed_samples'] += 1
            
            return filtered_sample
        return None
    
    def run(self):
        """Run the complete cleaning pipeline"""
        start_time = time.time()
        self.logger.info("Starting LLaVA-CoT cleaning pipeline")

        # Load data
        data = self.data_loader.load_data()
        self.stats['total_samples'] = len(data)
        self.logger.info(f"Loaded {len(data)} samples to process")

        # Load checkpoint if enabled
        processed_ids = set()
        start_batch = 0
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint(self.config)
            if checkpoint:
                processed_ids = checkpoint.get('processed_ids', set())
                start_batch = checkpoint.get('current_batch', 0)
                self.stats = checkpoint.get('stats', self.stats)
        
        # Filter out already processed samples
        if processed_ids:
            data = [d for d in data if d.get('id') not in processed_ids]
            self.logger.info(f"{len(data)} samples remaining after loading checkpoint")

        # Process in batches
        batches = self.data_loader.get_batches() # Re-batch after filtering
        total_batches = len(batches)
        all_processed = []

        for i, batch in enumerate(batches):
            current_batch_num = start_batch + i
            if current_batch_num < start_batch: # Should not happen with current logic
                continue

            self.logger.info(f"Processing batch {current_batch_num + 1}/{total_batches}")
            
            if self.parallel_processor:
                successful, failed = self.parallel_processor.process_batch_parallel(
                    items=batch,
                    process_func=self._process_and_filter_sample,
                    batch_id=current_batch_num + 1
                )
                all_processed.extend(successful)
            else:
                # Serial processing
                for sample in tqdm(batch, desc=f"Batch {current_batch_num + 1}"):
                    result = self._process_and_filter_sample(sample)
                    if result:
                        all_processed.append(result)

            # Save checkpoint after each batch
            if self.checkpoint_manager:
                current_processed_ids = {item.get('id') for item in batch if item}
                processed_ids.update(current_processed_ids)
                self.checkpoint_manager.save_checkpoint(
                    processed_ids=processed_ids,
                    failed_ids=set(), # Assuming failed are retried or handled inside processor
                    current_batch=current_batch_num + 1,
                    total_batches=total_batches,
                    stats=self.stats,
                    config=self.config
                )
                # Save intermediate results
                if self.config['checkpoint'].get('save_intermediate', True):
                    self.checkpoint_manager.save_intermediate_results(
                        batch_num=current_batch_num + 1,
                        results=all_processed[-len(batch):], # Save only current batch results
                        output_dir=self.config['checkpoint'].get('intermediate_dir', './intermediate')
                    )

        # Merge intermediate results if checkpointing was used
        if self.checkpoint_manager and self.config['checkpoint'].get('save_intermediate', True):
            self.logger.info("Merging intermediate results...")
            self.checkpoint_manager.merge_intermediate_results(
                output_dir=self.config['checkpoint'].get('intermediate_dir', './intermediate'),
                final_output_path=self.config['output']['output_path'],
                clean_intermediate=self.config['checkpoint'].get('clean_on_finish', True)
            )
        else:
            # Save all at once if not using intermediate saves
            self.data_loader.save_processed_data(self.config['output']['output_path'], all_processed)
        
        # Apply sampling if enabled
        if self.data_sampler and len(all_processed) > self.data_sampler.target_size:
            self.logger.info(f"Applying sampling to select {self.data_sampler.target_size} from {len(all_processed)} samples...")
            quality_scores = [s.get('quality_metrics', {}).get('overall_score', 0.5) for s in all_processed]
            sampled_data, sampling_stats = self.data_sampler.sample_data(all_processed, quality_scores)
            
            self.stats['sampling_stats'] = sampling_stats
            self.stats['pre_sampling_count'] = len(all_processed)
            self.stats['post_sampling_count'] = len(sampled_data)
            
            final_data_to_save = sampled_data
            self.logger.info(f"Sampling complete: {len(final_data_to_save)} samples selected.")
        else:
            final_data_to_save = all_processed

        # Save final results (if not using intermediate merge)
        if not (self.checkpoint_manager and self.config['checkpoint'].get('save_intermediate', True)):
            self.data_loader.save_processed_data(self.config['output']['output_path'], final_data_to_save)


        self.stats['processing_time'] = time.time() - start_time
        self.save_statistics()
        self.print_summary()

        # Clear checkpoint on successful completion
        if self.checkpoint_manager and self.config['checkpoint'].get('clean_on_finish', True):
            self.checkpoint_manager.clear_checkpoint()
    
    def save_statistics(self):
        """Save processing statistics"""
        # Add filter statistics
        self.stats['filter_stats'] = self.data_filter.get_statistics()
        
        stats_path = Path(self.config['output']['output_path']).parent / "processing_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved statistics to {stats_path}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*50)
        print("LLaVA-CoT Cleaning Pipeline - Summary")
        print("="*50)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Processed samples: {self.stats['processed_samples']}")
        print(f"Passed quality check: {self.stats['passed_samples']}")
        print(f"Failed quality check: {self.stats['failed_samples']}")
        print(f"Errors: {self.stats['errors']}")
        
        # Add sampling information if applicable
        if 'sampling_stats' in self.stats:
            print("\n--- Sampling Results ---")
            print(f"Pre-sampling count: {self.stats['pre_sampling_count']}")
            print(f"Post-sampling count: {self.stats['post_sampling_count']}")
            print(f"Average quality score: {self.stats['sampling_stats'].get('avg_quality_score', 0):.3f}")
            
            if 'category_distribution' in self.stats['sampling_stats']:
                print("\nCategory distribution:")
                for category, count in self.stats['sampling_stats']['category_distribution'].items():
                    print(f"  {category}: {count}")
        
        print(f"\nProcessing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['processed_samples'] > 0:
            pass_rate = self.stats['passed_samples'] / self.stats['processed_samples'] * 100
            print(f"Pass rate: {pass_rate:.2f}%")
        
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="LLaVA-CoT Dataset Cleaning Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create and run cleaner
    cleaner = LLaVACoTCleaner(config)
    cleaner.run()


if __name__ == "__main__":
    main()