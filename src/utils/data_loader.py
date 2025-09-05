from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import jsonlines
import pandas as pd
from PIL import Image
import logging
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


class LLaVACoTDataLoader:
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
        limit_to_first_half: bool = False
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.limit_to_first_half = limit_to_first_half
        self.data = []
        
    def load_data(self) -> List[Dict[str, Any]]:
        logger.info(f"Loading data from {self.data_path}")
        
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        elif self.data_path.suffix == '.jsonl':
            raw_data = []
            with jsonlines.open(self.data_path) as reader:
                for obj in reader:
                    raw_data.append(obj)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Apply data limiting options
        if self.limit_to_first_half:
            half_size = len(raw_data) // 2
            raw_data = raw_data[:half_size]
            logger.info(f"Limited to first half: {len(raw_data)} samples")
        
        # Limit samples if specified
        if self.max_samples:
            raw_data = raw_data[:self.max_samples]
            
        logger.info(f"Loaded {len(raw_data)} samples")
        
        # Process and validate data
        for idx, item in enumerate(tqdm(raw_data, desc="Validating data")):
            processed_item = self._process_item(item, idx)
            if processed_item:
                self.data.append(processed_item)
                
        logger.info(f"Successfully processed {len(self.data)} samples")
        return self.data
    
    def _process_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        try:
            # Extract required fields from LLaVA-CoT format
            processed = {
                'id': item.get('id', f'sample_{idx}'),
                'image_path': None,
                'question': None,
                'original_cot': None,
                'original_answer': None,
                'metadata': {}
            }
            
            # Handle different possible field names
            if 'image' in item:
                image_name = item['image']
                processed['image_path'] = str(self.image_dir / image_name)
            elif 'image_path' in item:
                processed['image_path'] = item['image_path']
            elif 'image_id' in item:
                # Try common image file extensions
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    potential_path = self.image_dir / f"{item['image_id']}{ext}"
                    if potential_path.exists():
                        processed['image_path'] = str(potential_path)
                        break
            
            # Extract question
            if 'question' in item:
                processed['question'] = item['question']
            elif 'conversations' in item:
                # Handle conversation format
                for conv in item['conversations']:
                    if conv.get('from') == 'human':
                        processed['question'] = conv.get('value', '')
                        break
            
            # Extract CoT and answer
            if 'rationale' in item:
                processed['original_cot'] = item['rationale']
                processed['original_answer'] = item.get('answer', '')
            elif 'conversations' in item:
                # Extract from conversation format
                for conv in item['conversations']:
                    if conv.get('from') == 'gpt' or conv.get('from') == 'assistant':
                        response = conv.get('value', '')
                        # Try to separate CoT from answer
                        if '<think>' in response and '</think>' in response:
                            cot_start = response.index('<think>') + 7
                            cot_end = response.index('</think>')
                            processed['original_cot'] = response[cot_start:cot_end].strip()
                            processed['original_answer'] = response[cot_end + 8:].strip()
                        else:
                            # Assume entire response is CoT + answer
                            processed['original_cot'] = response
                            # Try to extract answer from the end
                            lines = response.split('\n')
                            if lines:
                                processed['original_answer'] = lines[-1].strip()
            
            # Store any additional metadata
            for key, value in item.items():
                if key not in ['id', 'image', 'image_path', 'image_id', 'question', 
                              'conversations', 'rationale', 'answer']:
                    processed['metadata'][key] = value
            
            # Validate required fields
            if not processed['image_path']:
                logger.warning(f"Sample {idx}: No image path found")
                return None
            if not processed['question']:
                logger.warning(f"Sample {idx}: No question found")
                return None
            if not processed['original_cot']:
                logger.warning(f"Sample {idx}: No CoT found")
                return None
                
            # Check if image exists
            if not Path(processed['image_path']).exists():
                logger.warning(f"Sample {idx}: Image not found at {processed['image_path']}")
                return None
                
            return processed
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            return None
    
    def get_batches(self) -> List[List[Dict[str, Any]]]:
        if not self.data:
            self.load_data()
            
        batches = []
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i + self.batch_size]
            batches.append(batch)
            
        return batches
    
    def save_processed_data(self, output_path: str, data: List[Dict[str, Any]]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif output_path.suffix == '.jsonl':
            with jsonlines.open(output_path, 'w') as writer:
                writer.write_all(data)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
            
        logger.info(f"Saved {len(data)} samples to {output_path}")


class ImageProcessor:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return None
                
            if image_path.suffix.lower() not in self.supported_formats:
                logger.error(f"Unsupported image format: {image_path.suffix}")
                return None
                
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def preprocess_image(self, image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        # Resize image if too large
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    
    def save_image(self, image: Image.Image, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        logger.info(f"Saved image to {output_path}")