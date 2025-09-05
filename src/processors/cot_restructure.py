from typing import Dict, Optional, List, Any
import logging
import re
import time
from openai import OpenAI

logger = logging.getLogger(__name__)


class CoTRestructurer:
    def __init__(self, api_key: str, model_name: str = "gpt-4o", max_retries: int = 3):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://35.220.164.252:3888/v1/"
        )
    
    def restructure_cot(
        self,
        question: str,
        formal_description: str,
        original_cot: str,
        image_type: Optional[str] = None
    ) -> str:
        
        # Build the restructuring prompt based on image type
        base_prompt = self._build_restructure_prompt(
            question, formal_description, original_cot, image_type
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert at restructuring Chain-of-Thought reasoning.
Your task is to create clear, logical reasoning that directly references the formal image description.
Always wrap your reasoning in <think></think> tags."""
                        },
                        {
                            "role": "user",
                            "content": base_prompt
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.2
                )
                
                restructured_cot = response.choices[0].message.content.strip()
                
                # Ensure the CoT is wrapped in think tags
                if not restructured_cot.startswith('<think>'):
                    restructured_cot = f"<think>{restructured_cot}</think>"
                elif '<think>' not in restructured_cot:
                    restructured_cot = f"<think>{restructured_cot}</think>"
                
                return restructured_cot
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to restructure CoT after {self.max_retries} attempts")
                    # Return original CoT wrapped in think tags as fallback
                    return f"<think>{original_cot}</think>"
    
    def _build_restructure_prompt(
        self,
        question: str,
        formal_description: str,
        original_cot: str,
        image_type: Optional[str] = None
    ) -> str:
        
        prompt = f"""Restructure the Chain-of-Thought reasoning based on the formal image description.

Question: {question}

Formal Image Description: {formal_description}

Original Chain-of-Thought: {original_cot}

Instructions:
1. Create a new reasoning chain that directly references the formal description
2. Use phrases like "According to the image description...", "The image shows...", "As described..."
3. Maintain logical flow and mathematical/scientific accuracy
4. Be precise and reference specific details from the formal description
5. Wrap your entire reasoning in <think></think> tags
"""

        # Add image-type specific instructions
        if image_type == 'mathematical':
            prompt += """
6. For mathematical problems:
   - Clearly identify given values from the description
   - Show step-by-step calculations
   - Reference any formulas or equations visible"""
        
        elif image_type == 'diagram':
            prompt += """
6. For diagrams:
   - Reference specific components and their relationships
   - Use spatial information from the description
   - Explain how the diagram structure relates to the solution"""
        
        elif image_type == 'chart':
            prompt += """
6. For charts/graphs:
   - Reference specific data points mentioned in the description
   - Explain trends or patterns
   - Use quantitative information precisely"""
        
        prompt += """

Example Output:
<think>
According to the formal description, the image shows a pulley system with mass m=6kg hanging freely and mass M=14kg on a horizontal surface. Since the system is in equilibrium (stationary), the forces must be balanced. 

For the hanging mass m=6kg, the downward gravitational force is F = mg = 6kg × 9.8m/s² = 58.8N. 

In equilibrium, the tension T in the string must equal this gravitational force to maintain balance. Therefore, T = 58.8N ≈ 60N when rounded to the nearest 10N.

The mass M=14kg on the table doesn't affect the tension calculation since it's on a horizontal surface and the pulley redirects the force horizontally.
</think>"""
        
        return prompt


class AnswerExtractor:
    def __init__(self):
        self.answer_patterns = [
            r'(?:answer|solution|result)(?:\s+is)?[:：]\s*([^\n.。]+)',
            r'(?:therefore|thus|hence|so),?\s+([^\n.。]+)',
            r'=\s*([^\n.。]+)(?:\s*$|\n)',
            r'(?:选择|选|答案是?)\s*[:：]?\s*([A-Z])',
            r'<answer>(.*?)</answer>',
            r'\*\*([^\*]+)\*\*\s*$'
        ]
    
    def extract_answer(self, cot_text: str, original_answer: Optional[str] = None) -> str:
        # First, try to extract from CoT using patterns
        cot_text_clean = cot_text.replace('<think>', '').replace('</think>', '')
        
        for pattern in self.answer_patterns:
            matches = re.findall(pattern, cot_text_clean, re.IGNORECASE | re.MULTILINE)
            if matches:
                answer = matches[-1].strip()  # Take the last match
                if answer:
                    return self.format_answer(answer)
        
        # If no answer found in CoT, try to extract from the last line
        lines = cot_text_clean.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if last_line and len(last_line) < 200:  # Reasonable answer length
                return self.format_answer(last_line)
        
        # Fall back to original answer if provided
        if original_answer:
            return self.format_answer(original_answer)
        
        return "<answer>Unable to extract answer</answer>"
    
    def format_answer(self, answer: str) -> str:
        # Clean up the answer
        answer = answer.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            'the answer is', 'answer is', 'answer:', 
            'solution:', 'result:', 'therefore',
            'thus', 'hence', 'so', '所以', '答案是', '答案'
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Remove trailing punctuation except for units
        answer = re.sub(r'[.,。，;；:：]$', '', answer)
        
        # Wrap in answer tags
        return f"<answer>{answer}</answer>"
    
    def refine_answer_with_llm(
        self,
        cot_text: str,
        question: str,
        api_key: str,
        model_name: str = "gpt-4o"
    ) -> str:

        client = OpenAI(
            api_key=api_key,
            base_url="http://35.220.164.252:3888/v1/"
        )

        prompt = f"""Extract the final answer from this Chain-of-Thought reasoning.

Question: {question}

Chain-of-Thought: {cot_text}

Instructions:
1. Extract ONLY the final answer
2. Keep it concise and direct
3. Include units if applicable
4. For multiple choice, give just the letter
5. Wrap the answer in <answer></answer> tags

Example outputs:
- <answer>60N</answer>
- <answer>A</answer>
- <answer>3.14</answer>
- <answer>The momentum is conserved</answer>"""
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting answers from reasoning text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to refine answer with LLM: {str(e)}")
            return self.extract_answer(cot_text)