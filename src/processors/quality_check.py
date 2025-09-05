from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import time
from openai import OpenAI
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    description_quality: float  # 0-1
    cot_coherence: float  # 0-1
    answer_correctness: float  # 0-1
    overall_score: float  # 0-1
    issues: List[str]
    passed: bool


class QualityChecker:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        quality_threshold: float = 0.7,
        max_retries: int = 2
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://35.220.164.252:3888/v1/"
        )
    
    def check_quality(
        self,
        question: str,
        formal_description: str,
        cot_thinking: str,
        final_answer: str,
        original_data: Optional[Dict] = None
    ) -> QualityScore:
        
        issues = []
        
        # Check formal description quality
        desc_score, desc_issues = self._check_description_quality(formal_description)
        issues.extend(desc_issues)
        
        # Check CoT coherence
        cot_score, cot_issues = self._check_cot_coherence(
            question, formal_description, cot_thinking
        )
        issues.extend(cot_issues)
        
        # Check answer correctness/format
        answer_score, answer_issues = self._check_answer_quality(
            question, cot_thinking, final_answer
        )
        issues.extend(answer_issues)
        
        # Calculate overall score
        overall_score = (desc_score * 0.3 + cot_score * 0.4 + answer_score * 0.3)
        
        # Determine if passed
        passed = overall_score >= self.quality_threshold and len(issues) < 3
        
        return QualityScore(
            description_quality=desc_score,
            cot_coherence=cot_score,
            answer_correctness=answer_score,
            overall_score=overall_score,
            issues=issues,
            passed=passed
        )
    
    def _check_description_quality(self, description: str) -> Tuple[float, List[str]]:
        issues = []
        score = 1.0
        
        # Check length
        if len(description) < 50:
            issues.append("Description too short")
            score -= 0.3
        elif len(description) > 2000:
            issues.append("Description too long")
            score -= 0.1
        
        # Check for key indicators of quality
        quality_indicators = [
            r'\d+',  # Contains numbers
            r'[A-Z]',  # Contains proper nouns or labels
            r'(shows|displays|contains|includes)',  # Descriptive verbs
        ]
        
        indicator_count = sum(1 for pattern in quality_indicators 
                             if re.search(pattern, description))
        
        if indicator_count < 2:
            issues.append("Description lacks specific details")
            score -= 0.2
        
        # Check for problematic patterns
        problematic_patterns = [
            r'I can\'t|I cannot|unable to',
            r'error|mistake|wrong',
            r'blurry|unclear|illegible'
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                issues.append(f"Description contains problematic content: {pattern}")
                score -= 0.2
        
        return max(0, score), issues
    
    def _check_cot_coherence(
        self,
        question: str,
        formal_description: str,
        cot_thinking: str
    ) -> Tuple[float, List[str]]:
        
        issues = []
        score = 1.0
        
        # Remove think tags for analysis
        cot_clean = cot_thinking.replace('<think>', '').replace('</think>', '').strip()
        
        # Check if CoT references the formal description
        reference_patterns = [
            r'(according to|based on|from).*description',
            r'image shows',
            r'as (described|shown|stated)',
            r'the formal description'
        ]
        
        has_reference = any(re.search(pattern, cot_clean, re.IGNORECASE) 
                           for pattern in reference_patterns)
        
        if not has_reference:
            issues.append("CoT doesn't reference the formal description")
            score -= 0.3
        
        # Check logical flow (presence of reasoning indicators)
        reasoning_indicators = [
            r'(therefore|thus|hence|so)',
            r'(because|since|as)',
            r'(first|second|then|next|finally)',
            r'(step \d+|equation|formula)'
        ]
        
        reasoning_count = sum(1 for pattern in reasoning_indicators 
                             if re.search(pattern, cot_clean, re.IGNORECASE))
        
        if reasoning_count < 1:
            issues.append("CoT lacks clear reasoning flow")
            score -= 0.2
        
        # Check if CoT is wrapped in think tags
        if not cot_thinking.startswith('<think>') or not cot_thinking.endswith('</think>'):
            issues.append("CoT not properly wrapped in <think> tags")
            score -= 0.1
        
        # Check length
        if len(cot_clean) < 100:
            issues.append("CoT too short")
            score -= 0.2
        elif len(cot_clean) > 3000:
            issues.append("CoT too long")
            score -= 0.1
        
        return max(0, score), issues
    
    def _check_answer_quality(
        self,
        question: str,
        cot_thinking: str,
        final_answer: str
    ) -> Tuple[float, List[str]]:
        
        issues = []
        score = 1.0
        
        # Check if answer is wrapped in tags
        if not final_answer.startswith('<answer>') or not final_answer.endswith('</answer>'):
            issues.append("Answer not properly wrapped in <answer> tags")
            score -= 0.2
        
        # Extract answer content
        answer_content = re.sub(r'</?answer>', '', final_answer).strip()
        
        # Check if answer exists
        if not answer_content or answer_content.lower() in ['unable to extract answer', 'none', 'n/a']:
            issues.append("No valid answer extracted")
            score -= 0.5
        
        # Check answer length
        if len(answer_content) > 500:
            issues.append("Answer too long")
            score -= 0.2
        
        # Check if answer appears in CoT
        cot_clean = cot_thinking.replace('<think>', '').replace('</think>', '')
        if answer_content and answer_content not in cot_clean:
            # Allow for minor variations (e.g., "60N" vs "60 N")
            answer_normalized = re.sub(r'\s+', '', answer_content.lower())
            cot_normalized = re.sub(r'\s+', '', cot_clean.lower())
            if answer_normalized not in cot_normalized:
                issues.append("Answer doesn't appear in the CoT reasoning")
                score -= 0.2
        
        return max(0, score), issues
    
    def verify_with_llm(
        self,
        sample: Dict[str, Any],
        quick_check: bool = False
    ) -> Tuple[bool, str]:
        
        prompt = f"""Evaluate the quality of this processed sample:

Question: {sample.get('question', '')}

Formal Description: {sample.get('image_formal_description', '')}

Chain-of-Thought: {sample.get('cot_thinking', '')}

Final Answer: {sample.get('final_answer', '')}

Evaluate:
1. Does the formal description accurately capture relevant image information?
2. Is the CoT reasoning logical and based on the description?
3. Is the final answer correct and properly extracted?

Respond with:
- PASS: If all three criteria are met
- FAIL: If any criteria is not met
- Briefly explain the main issue if FAIL"""

        if quick_check:
            prompt += "\n\nProvide a quick assessment in 1-2 sentences."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a quality control expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            passed = 'PASS' in result.upper()[:20]  # Check in first 20 chars
            
            return passed, result
            
        except Exception as e:
            logger.error(f"LLM verification failed: {str(e)}")
            return False, f"Verification error: {str(e)}"


class DataFilter:
    def __init__(
        self,
        quality_checker: QualityChecker,
        min_quality_score: float = 0.7,
        use_llm_verification: bool = True,
        sample_rate_for_llm: float = 0.1
    ):
        self.quality_checker = quality_checker
        self.min_quality_score = min_quality_score
        self.use_llm_verification = use_llm_verification
        self.sample_rate_for_llm = sample_rate_for_llm
        self.statistics = {
            'total_processed': 0,
            'passed': 0,
            'failed': 0,
            'failed_reasons': {}
        }
    
    def filter_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        self.statistics['total_processed'] += 1
        
        # Perform quality check
        quality_score = self.quality_checker.check_quality(
            question=sample.get('question', ''),
            formal_description=sample.get('image_formal_description', ''),
            cot_thinking=sample.get('cot_thinking', ''),
            final_answer=sample.get('final_answer', ''),
            original_data=sample
        )
        
        # Add quality metrics to sample
        sample['quality_metrics'] = {
            'description_quality': quality_score.description_quality,
            'cot_coherence': quality_score.cot_coherence,
            'answer_correctness': quality_score.answer_correctness,
            'overall_score': quality_score.overall_score,
            'issues': quality_score.issues
        }
        
        # Initial pass/fail based on quality score
        passed = quality_score.passed
        
        # Additional LLM verification for a sample
        if self.use_llm_verification and passed:
            import random
            if random.random() < self.sample_rate_for_llm:
                llm_passed, llm_reason = self.quality_checker.verify_with_llm(
                    sample, quick_check=True
                )
                if not llm_passed:
                    passed = False
                    sample['quality_metrics']['llm_verification'] = llm_reason
                    quality_score.issues.append(f"LLM verification failed: {llm_reason}")
        
        # Update statistics
        if passed:
            self.statistics['passed'] += 1
        else:
            self.statistics['failed'] += 1
            # Track failure reasons
            for issue in quality_score.issues:
                if issue not in self.statistics['failed_reasons']:
                    self.statistics['failed_reasons'][issue] = 0
                self.statistics['failed_reasons'][issue] += 1
        
        return passed, sample
    
    def filter_batch(
        self,
        samples: List[Dict[str, Any]],
        keep_failed: bool = False
    ) -> List[Dict[str, Any]]:
        
        filtered_samples = []
        
        for sample in samples:
            passed, processed_sample = self.filter_sample(sample)
            
            if passed or keep_failed:
                processed_sample['passed_quality_check'] = passed
                filtered_samples.append(processed_sample)
        
        return filtered_samples
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = self.statistics.copy()
        if stats['total_processed'] > 0:
            stats['pass_rate'] = stats['passed'] / stats['total_processed']
            stats['fail_rate'] = stats['failed'] / stats['total_processed']
        else:
            stats['pass_rate'] = 0
            stats['fail_rate'] = 0
        
        return stats
    
    def save_statistics(self, output_path: str):
        stats = self.get_statistics()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved statistics to {output_path}")