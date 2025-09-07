#!/usr/bin/env python3
"""
Test script for Grounding DINO API functionality
"""

import json
from PIL import Image
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.image_description import ObjectDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_api_detection():
    """Test the Grounding DINO API detection"""
    
    # Load config
    config_path = "configs/config_api_example.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get Grounding DINO config
    grounding_dino_config = config['models']['grounding_dino_config']
    
    print(f"Testing Grounding DINO API with config:")
    print(f"  - Use API: {grounding_dino_config.get('use_api', False)}")
    print(f"  - API Endpoint: {grounding_dino_config.get('api_config', {}).get('endpoint')}")
    print(f"  - Box Threshold: {grounding_dino_config.get('box_threshold', 0.35)}")
    print(f"  - Text Threshold: {grounding_dino_config.get('text_threshold', 0.25)}")
    
    # Initialize detector with API
    detector = ObjectDetector(
        use_real_model=grounding_dino_config.get('use_real_model', False),
        use_api=grounding_dino_config.get('use_api', False),
        api_config=grounding_dino_config.get('api_config', {}),
        box_threshold=grounding_dino_config.get('box_threshold', 0.35),
        text_threshold=grounding_dino_config.get('text_threshold', 0.25),
        device='cpu'
    )
    
    # Create a test image (simple white image for testing)
    test_image = Image.new('RGB', (640, 480), color='white')
    
    # Test prompts
    test_prompts = [
        "all objects in the image",
        "person . car . dog . cat",
        "text and numbers"
    ]
    
    print("\n" + "="*50)
    for prompt in test_prompts:
        print(f"\nTesting with prompt: '{prompt}'")
        print("-" * 30)
        
        try:
            # Run detection
            result = detector.generate(test_image, prompt)
            
            print(f"Detection result:")
            print(f"  - Objects found: {len(result.get('objects', []))}")
            print(f"  - Objects: {result.get('objects', [])}")
            print(f"  - Number of boxes: {len(result.get('boxes', []))}")
            print(f"  - Scores: {result.get('scores', [])}")
            
        except Exception as e:
            print(f"Error during detection: {e}")
    
    print("\n" + "="*50)
    print("Test completed!")

if __name__ == "__main__":
    test_api_detection()