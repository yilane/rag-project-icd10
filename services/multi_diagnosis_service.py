#!/usr/bin/env python3
"""
多诊断匹配服务
处理包含多个诊断的文本，分别进行向量匹配
"""

import numpy as np
from typing import List, Dict, Any
from loguru import logger

from tools.text_processor import DiagnosisTextProcessor
from services.embedding_service import EmbeddingService
from services.milvus_service import MilvusService
from models.icd_models import DiagnosisMatch, Candidate


class MultiDiagnosisService:
    """多诊断匹配服务"""
    
    def __init__(self, embedding_service: EmbeddingService, milvus_service: MilvusService):
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.text_processor = DiagnosisTextProcessor()
    
    def match_multiple_diagnoses(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        匹配多个诊断
        
        Args:
            text: 包含多个诊断的文本
            top_k: 每个诊断返回的候选数量
            
        Returns:
            匹配结果字典
        """
        logger.info(f"开始多诊断匹配: {text}")
        
        try:
            # 1. 提取诊断词
            diagnoses = self.text_processor.extract_diagnoses(text)
            
            if not diagnoses:
                logger.warning("未能提取到有效的诊断词")
                return {
                    "original_text": text,
                    "extracted_diagnoses": [],
                    "matches": [],
                    "total_matches": 0
                }
            
            logger.info(f"提取到 {len(diagnoses)} 个诊断词: {diagnoses}")
            
            # 2. 对每个诊断词进行向量匹配
            matches = []
            total_candidates = 0
            
            for diagnosis in diagnoses:
                diagnosis_match = self._match_single_diagnosis(diagnosis, top_k)
                matches.append(diagnosis_match)
                total_candidates += len(diagnosis_match.candidates)
            
            result = {
                "original_text": text,
                "extracted_diagnoses": diagnoses,
                "matches": matches,
                "total_matches": total_candidates
            }
            
            logger.info(f"多诊断匹配完成，共找到 {total_candidates} 个候选结果")
            return result
            
        except Exception as e:
            logger.error(f"多诊断匹配失败: {e}")
            raise
    
    def _match_single_diagnosis(self, diagnosis: str, top_k: int) -> DiagnosisMatch:
        """
        匹配单个诊断
        
        Args:
            diagnosis: 单个诊断文本
            top_k: 返回候选数量
            
        Returns:
            诊断匹配结果
        """
        logger.info(f"匹配单个诊断: {diagnosis}")
        
        try:
            # 1. 生成查询向量
            query_vector = self.embedding_service.encode_query(diagnosis)
            
            # 2. 执行向量搜索
            search_results = self.milvus_service.search(query_vector, top_k)
            
            # 3. 转换为标准格式
            candidates = []
            for result in search_results:
                candidate = Candidate(
                    code=result.get("code", ""),
                    title=result.get("title", ""),
                    score=result.get("score", 0.0)
                )
                candidates.append(candidate)
            
            # 4. 计算整体匹配置信度
            match_confidence = self._calculate_match_confidence(candidates)
            
            diagnosis_match = DiagnosisMatch(
                diagnosis_text=diagnosis,
                candidates=candidates,
                match_confidence=match_confidence
            )
            
            logger.info(f"诊断 '{diagnosis}' 找到 {len(candidates)} 个候选结果，匹配置信度: {match_confidence:.3f}")
            return diagnosis_match
            
        except Exception as e:
            logger.error(f"单个诊断匹配失败: {diagnosis} - {e}")
            # 返回空的匹配结果
            return DiagnosisMatch(
                diagnosis_text=diagnosis,
                candidates=[],
                match_confidence=0.0
            )
    
    def _calculate_match_confidence(self, candidates: List[Candidate]) -> float:
        """
        计算匹配置信度
        
        Args:
            candidates: 候选结果列表
            
        Returns:
            置信度分数 (0.0-1.0)
        """
        if not candidates:
            return 0.0
        
        # 使用最高分数作为基础置信度
        max_score = max(candidate.score for candidate in candidates)
        
        # 考虑候选结果的数量和分数分布
        scores = [candidate.score for candidate in candidates]
        
        # 如果最高分很高（>0.9），置信度较高
        if max_score > 0.9:
            confidence = min(max_score, 0.95)
        # 如果有多个高分候选，置信度中等
        elif len([s for s in scores if s > 0.7]) >= 2:
            confidence = max_score * 0.8
        # 如果只有低分候选，置信度较低
        else:
            confidence = max_score * 0.6
        
        return round(confidence, 3)
    
    def get_diagnosis_suggestions(self, text: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        获取诊断建议（过滤低置信度结果）
        
        Args:
            text: 输入文本
            min_confidence: 最低置信度阈值
            
        Returns:
            过滤后的诊断建议列表
        """
        result = self.match_multiple_diagnoses(text)
        suggestions = []
        
        for match in result["matches"]:
            if match.match_confidence >= min_confidence and match.candidates:
                suggestions.append({
                    "diagnosis": match.diagnosis_text,
                    "confidence": match.match_confidence,
                    "best_match": {
                        "code": match.candidates[0].code,
                        "title": match.candidates[0].title,
                        "score": match.candidates[0].score
                    },
                    "alternative_matches": [
                        {
                            "code": candidate.code,
                            "title": candidate.title,
                            "score": candidate.score
                        }
                        for candidate in match.candidates[1:]
                    ]
                })
        
        return suggestions 