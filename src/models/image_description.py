from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import numpy as np
import logging
import json
import base64
import time
from io import BytesIO
from abc import ABC, abstractmethod

# Optional imports
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    sv = None
    SUPERVISION_AVAILABLE = False

import torch

# Optional import for API functionality
try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        pass


class VLMDescriptionGenerator(BaseModel):
    def __init__(self, api_key: str, model_name: str = "gpt-4o", max_retries: int = 3):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="http://35.220.164.252:3888/v1/"
            )
        except ImportError:
            logger.error("OpenAI library not installed. Please install with: pip install openai")
            raise
    
    def encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def generate(self, image: Image.Image, prompt_template: Optional[str] = None) -> str:
        if prompt_template is None:
            prompt_template = """You are an expert at creating formal image descriptions.
Analyze this image and provide a concise, structured description focusing on:
1. Main objects and their properties
2. Spatial relationships between objects
3. Any text or numbers visible
4. Overall scene context

Be precise and factual. Avoid subjective interpretations."""
        
        base64_image = self.encode_image(image)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a vision expert that creates precise, formal descriptions of images."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_template},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to generate description after {self.max_retries} attempts")
                    raise


class ObjectDetector(BaseModel):
    def __init__(
        self,
        use_real_model: bool = False,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        use_api: bool = False,
        api_config: Optional[Dict] = None
    ):
        self.use_real_model = use_real_model
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_api = use_api
        self.api_config = api_config or {}
        self.model = None
        self.model_initialized = False
        self.api_client = None
        
        if self.use_api:
            self.initialize_api_client()
        elif self.use_real_model:
            self.initialize_model()

    def initialize_api_client(self):
        """Initialize API client for Grounding DINO service"""
        if not self.use_api:
            return
            
        if requests is None:
            logger.error("requests library not installed. Please install with: pip install requests")
            self.use_api = False
            return
            
        try:
            # Set default API configuration
            self.api_endpoint = self.api_config.get('endpoint', 'http://localhost:8000/detect')
            self.api_key = self.api_config.get('api_key', '')
            self.api_timeout = self.api_config.get('timeout', 30)
            self.max_retries = self.api_config.get('max_retries', 3)
            
            # Test connection
            headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
            test_response = requests.get(
                self.api_endpoint.replace('/detect', '/health'),
                headers=headers,
                timeout=5
            )
            
            if test_response.status_code == 200:
                logger.info("Grounding DINO API client initialized successfully")
                self.api_client = requests.Session()
                self.api_client.headers.update(headers)
            else:
                logger.warning(f"API health check failed with status {test_response.status_code}")
                self.use_api = False
                
        except Exception as e:
            logger.warning(f"Failed to initialize API client: {str(e)}. Falling back to placeholder mode.")
            self.use_api = False

    def initialize_model(self):
        if not self.use_real_model or self.model_initialized:
            return
        
        logger.info(f"Initializing Grounding DINO model on {self.device}")
        if not self.config_path or not self.checkpoint_path:
            logger.error("Real model enabled, but config_path or checkpoint_path is missing.")
            self.use_real_model = False
            return
        
        try:
            from groundingdino.util.inference import load_model
            self.model = load_model(self.config_path, self.checkpoint_path, device=self.device)
            self.model_initialized = True
            logger.info("Grounding DINO model initialized successfully.")
        except ImportError:
            logger.error("Grounding DINO dependencies not found. Disabling real model.")
            self.use_real_model = False
        except Exception as e:
            logger.error(f"Failed to initialize Grounding DINO: {str(e)}")
            self.use_real_model = False

    def generate(
        self, 
        image_path: str, 
        text_prompt: str = "all objects in the image"
    ) -> Dict[str, List]:
        # Priority: API > Real Model > Placeholder
        if self.use_api:
            # For API mode, we need to load PIL image from path
            try:
                from PIL import Image
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                return self._detect_with_api(pil_image, text_prompt)
            except Exception as e:
                logger.error(f"Failed to load image for API: {str(e)}")
                return self._generate_placeholder(text_prompt)
        
        if not self.use_real_model:
            return self._generate_placeholder(text_prompt)
        
        if not self.model_initialized:
            logger.warning("Grounding DINO model not initialized. Using placeholder.")
            return self._generate_placeholder(text_prompt)
        
        try:
            from groundingdino.util.inference import load_image, predict
            import torch.nn.functional as F

            # Load and transform image
            image_source, image_tensor = load_image(image_path)

            # Run inference with a retry mechanism for tensor size mismatch
            for attempt in range(2):
                try:
                    boxes, logits, phrases = predict(
                        model=self.model,
                        image=image_tensor,
                        caption=text_prompt,
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        device=self.device
                    )
                    break  # Success
                
                except Exception as e:
                    if "The size of tensor a" in str(e) and "must match the size of tensor b" in str(e) and attempt == 0:
                        logger.warning(f"Tensor size mismatch error for {image_path}. Retrying with padding.")
                        
                        # Pad the image tensor to be divisible by 32
                        h, w = image_tensor.shape[-2:]
                        pad_h = (32 - h % 32) % 32
                        pad_w = (32 - w % 32) % 32
                        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))
                        
                        continue # Retry with padded tensor
                    else:
                        raise e # Re-raise other errors or on second attempt
            
            # Annotate image (optional, for debugging)
            # annotated_frame = self._annotate_image(image_source, boxes, logits, phrases)
            
            detections = {
                "objects": phrases,
                "boxes": boxes.tolist(),
                "scores": logits.tolist()
            }
            
            return detections

        except Exception as e:
            logger.error(f"Grounding DINO prediction failed: {str(e)}")
            return self._generate_placeholder(text_prompt)

    def _detect_with_api(self, image: Image.Image, text_prompt: str) -> Dict[str, List]:
        """Detect objects using Grounding DINO API service"""
        if not self.api_client:
            logger.warning("API client not initialized, falling back to placeholder")
            return self._generate_placeholder(text_prompt)
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare API request payload
        payload = {
            "image": img_base64,
            "text_prompt": text_prompt,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.api_client.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=self.api_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Ensure the response has the expected structure
                    detections = {
                        "objects": result.get("objects", result.get("phrases", [])),
                        "boxes": result.get("boxes", []),
                        "scores": result.get("scores", result.get("logits", []))
                    }
                    
                    logger.info(f"API detection successful: found {len(detections['objects'])} objects")
                    return detections
                else:
                    logger.warning(f"API request failed with status {response.status_code}: {response.text}")
                    
            except requests.Timeout:
                logger.warning(f"API request timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in API detection: {str(e)}")
                
            # Exponential backoff
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)
        
        logger.error(f"API detection failed after {self.max_retries} attempts, using placeholder")
        return self._generate_placeholder(text_prompt)
    
    def _generate_placeholder(self, text_prompt: str) -> Dict[str, List]:
        logger.info(f"Using placeholder for object detection with prompt: '{text_prompt}'")
        return {
            "objects": [],
            "boxes": [],
            "scores": []
        }
    
    def _annotate_image(self, image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
        """Annotate image with detections for visualization/debugging"""
        if not SUPERVISION_AVAILABLE:
            logger.warning("Supervision library not available for annotation")
            return image_source
            
        detections = sv.Detections(
            xyxy=sv.BoxAnnotator.sv_to_sa(boxes.cpu().numpy()),
            confidence=logits.cpu().numpy(),
            data={"class_name": phrases}
        )

        box_annotator = sv.BoxAnnotator()
        labels = [f"{phrases[i]} {logits[i]:.2f}" for i in range(len(phrases))]
        annotated_frame = box_annotator.annotate(scene=image_source, detections=detections, labels=labels)
        return annotated_frame


class OCRProcessor(BaseModel):
    def __init__(self, languages: List[str] = ['en', 'ch_sim'], max_retries: int = 3):
        self.languages = languages
        self.max_retries = max_retries
        self.reader = None
        self.ocr_available = False
        
    def initialize_reader(self):
        try:
            import easyocr
            
            # Try to initialize with retries
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Initializing EasyOCR (attempt {attempt + 1}/{self.max_retries})")
                    self.reader = easyocr.Reader(
                        self.languages, 
                        gpu=True,
                        download_enabled=True,
                        model_storage_directory=None,
                        user_network_directory=None
                    )
                    self.ocr_available = True
                    logger.info(f"EasyOCR initialized successfully with languages: {self.languages}")
                    break
                    
                except Exception as e:
                    logger.warning(f"EasyOCR initialization attempt {attempt + 1} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error("Failed to initialize EasyOCR after all retries. OCR will be disabled.")
                        self.ocr_available = False
                        
        except ImportError:
            logger.error("EasyOCR not installed. Please install with: pip install easyocr")
            self.ocr_available = False
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {str(e)}")
            self.ocr_available = False
    
    def generate(self, image: Image.Image) -> Dict[str, Any]:
        if not self.ocr_available:
            if self.reader is None:
                self.initialize_reader()
            
            if not self.ocr_available:
                logger.warning("OCR not available, returning empty results")
                return {"texts": [], "boxes": [], "confidences": []}
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        try:
            results = self.reader.readtext(img_array)
            
            ocr_output = {
                "texts": [],
                "boxes": [],
                "confidences": []
            }
            
            for (bbox, text, confidence) in results:
                ocr_output["texts"].append(text)
                ocr_output["boxes"].append(bbox)
                ocr_output["confidences"].append(confidence)
            
            return ocr_output
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return {"texts": [], "boxes": [], "confidences": []}


class ImageFormalDescriptionGenerator:
    def __init__(
        self,
        vlm_api_key: str,
        use_object_detection: bool = True,
        use_ocr: bool = True,
        vlm_model: str = "gpt-4o",
        device: str = "cuda",
        ocr_fallback_on_error: bool = True,
        grounding_dino_config: Optional[Dict] = None
    ):
        self.vlm_generator = VLMDescriptionGenerator(vlm_api_key, vlm_model)
        self.use_object_detection = use_object_detection
        self.use_ocr = use_ocr
        self.device = device
        self.ocr_fallback_on_error = ocr_fallback_on_error
        
        if use_object_detection:
            dino_config = grounding_dino_config or {}
            self.object_detector = ObjectDetector(
                use_real_model=dino_config.get('use_real_model', False),
                config_path=dino_config.get('config_path'),
                checkpoint_path=dino_config.get('checkpoint_path'),
                device=device,
                box_threshold=dino_config.get('box_threshold', 0.35),
                text_threshold=dino_config.get('text_threshold', 0.25),
                use_api=dino_config.get('use_api', False),
                api_config=dino_config.get('api_config', {})
            )
        
        if use_ocr:
            self.ocr_processor = OCRProcessor()
    
    def identify_image_type(self, image: Image.Image) -> str:
        # Use VLM to identify image type
        prompt = """Classify this image into one of the following categories:
1. natural_scene: Natural photographs or real-world scenes
2. diagram: Technical diagrams, flowcharts, circuit diagrams
3. chart: Statistical charts, graphs, plots
4. text_heavy: Documents, screenshots with lots of text
5. mathematical: Mathematical equations, geometric figures
6. mixed: Combination of above

Return only the category name."""
        
        try:
            image_type = self.vlm_generator.generate(image, prompt)
            image_type = image_type.strip().lower()
            
            valid_types = ['natural_scene', 'diagram', 'chart', 'text_heavy', 'mathematical', 'mixed']
            if image_type not in valid_types:
                image_type = 'mixed'
                
            return image_type
        except:
            return 'mixed'
    
    def generate_formal_description(
        self,
        image: Image.Image,
        image_path: Optional[str] = None,
        question: Optional[str] = None,
        existing_cot: Optional[str] = None
    ) -> Dict[str, Any]:
        
        logger.info("Generating formal image description")
        
        # Step 1: Identify image type
        image_type = self.identify_image_type(image)
        logger.info(f"Image type identified as: {image_type}")
        
        # Step 2: Generate dense caption
        caption_prompt = f"""Create a detailed, formal description of this {image_type} image.
Focus on factual information that would be relevant for answering questions about it.
Be precise about:
- Objects and their properties
- Spatial relationships
- Any text, numbers, or symbols
- Overall context and purpose"""
        
        if question:
            caption_prompt += f"\n\nContext: This image is related to the question: '{question}'"
        
        dense_caption = self.vlm_generator.generate(image, caption_prompt)
        
        # Step 3: Extract objects if applicable
        object_info = {}
        if self.use_object_detection and image_type in ['natural_scene', 'diagram', 'mixed']:
            try:
                if image_path:
                    # Use image path for Grounding DINO
                    detections = self.object_detector.generate(image_path)
                else:
                    # Fallback: save image temporarily if no path provided
                    import tempfile
                    import os
                    temp_path = tempfile.mktemp(suffix='.jpg')
                    image.save(temp_path)
                    try:
                        detections = self.object_detector.generate(temp_path)
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                object_info = {
                    "detected_objects": detections.get("objects", []),
                    "bounding_boxes": detections.get("boxes", [])
                }
            except Exception as e:
                logger.warning(f"Object detection failed: {str(e)}")
        
        # Step 4: Extract text if applicable
        ocr_info = {}
        if self.use_ocr and image_type in ['text_heavy', 'diagram', 'chart', 'mathematical', 'mixed']:
            try:
                ocr_results = self.ocr_processor.generate(image)
                ocr_info = {
                    "extracted_texts": ocr_results.get("texts", []),
                    "text_locations": ocr_results.get("boxes", [])
                }
            except Exception as e:
                logger.warning(f"OCR failed: {str(e)}")
                if not self.ocr_fallback_on_error:
                    raise
                # Continue without OCR if fallback is enabled
        
        # Step 5: Synthesize all information
        synthesis_prompt = f"""Synthesize the following information into a single, formal image description:

Dense Caption: {dense_caption}

"""
        if object_info:
            synthesis_prompt += f"Detected Objects: {object_info.get('detected_objects', [])}\n\n"
        
        if ocr_info:
            synthesis_prompt += f"Extracted Text: {ocr_info.get('extracted_texts', [])}\n\n"
        
        synthesis_prompt += """Create a concise, structured description that:
1. Integrates all the above information
2. Focuses on facts and relationships
3. Is suitable for reasoning tasks
4. Avoids redundancy

Format: A single paragraph of formal description."""
        
        final_description = self.vlm_generator.generate(image, synthesis_prompt)
        
        return {
            "image_formal_description": final_description,
            "image_type": image_type,
            "dense_caption": dense_caption,
            "object_detections": object_info,
            "ocr_results": ocr_info
        }