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
from services.hierarchical_similarity_service import HierarchicalSimilarityService
from services.medical_ner_service import MedicalNERService
from services.multidimensional_confidence_service import MultiDimensionalConfidenceService
from models.icd_models import DiagnosisMatch, Candidate


class MultiDiagnosisService:
    """多诊断匹配服务"""
    
    def __init__(self, embedding_service: EmbeddingService, milvus_service: MilvusService):
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        
        # 初始化医学NER服务（启用非诊断实体过滤）
        self.ner_service = MedicalNERService(use_model=True)
        
        # 初始化层级相似度服务
        self.hierarchical_similarity = HierarchicalSimilarityService(
            embedding_service=embedding_service,
            ner_service=self.ner_service
        )
        
        # 初始化多维度置信度服务
        self.confidence_service = MultiDimensionalConfidenceService(
            embedding_service=embedding_service,
            ner_service=self.ner_service,
            hierarchical_similarity_service=self.hierarchical_similarity
        )
        
        # 使用增强的文本处理器，传入嵌入服务以支持语义边界检测
        self.text_processor = DiagnosisTextProcessor(
            embedding_service=embedding_service,
            use_enhanced_processing=True
        )
        
        logger.info("多诊断服务初始化完成，已集成层级相似度计算和多维度置信度评分")
    
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
            # 1. 提取诊断词（使用增强方法获取更多信息，启用非诊断实体过滤）
            enhanced_diagnoses = self.text_processor.extract_diagnoses_enhanced(text)
            
            # 非诊断实体过滤默认启用
            filter_drugs = True
            logger.info(f"启用非诊断实体过滤模式")
            diagnoses = [d['text'] for d in enhanced_diagnoses]
            
            if not diagnoses:
                logger.warning("未能提取到有效的诊断词")
                return {
                    "original_text": text,
                    "extracted_diagnoses": [],
                    "matches": [],
                    "total_matches": 0,
                    "processing_mode": self.text_processor.get_processing_mode(),
                    "extraction_metadata": {
                        "enhanced_results_count": len(enhanced_diagnoses),
                        "avg_extraction_confidence": 0.0
                    }
                }
            
            # 计算提取置信度统计
            extraction_confidences = [d.get('diagnosis_confidence', 0.5) for d in enhanced_diagnoses]
            avg_extraction_confidence = sum(extraction_confidences) / len(extraction_confidences)
            
            logger.info(f"提取到 {len(diagnoses)} 个诊断词: {diagnoses}")
            logger.info(f"平均提取置信度: {avg_extraction_confidence:.3f}")
            
            # 2. 对每个诊断词进行向量匹配，传递增强信息
            matches = []
            total_candidates = 0
            
            for i, diagnosis in enumerate(diagnoses):
                # 获取对应的增强诊断信息
                enhanced_info = enhanced_diagnoses[i] if i < len(enhanced_diagnoses) else {}
                diagnosis_match = self._match_single_diagnosis_enhanced(diagnosis, top_k, enhanced_info)
                matches.append(diagnosis_match)
                total_candidates += len(diagnosis_match.candidates)
            
            result = {
                "original_text": text,
                "extracted_diagnoses": diagnoses,
                "matches": matches,
                "total_matches": total_candidates,
                "processing_mode": self.text_processor.get_processing_mode(),
                "extraction_metadata": {
                    "enhanced_results_count": len(enhanced_diagnoses),
                    "avg_extraction_confidence": avg_extraction_confidence,
                    "extraction_method": "enhanced" if enhanced_diagnoses else "simple",
                    "drug_filtering_enabled": True
                }
            }
            
            logger.info(f"多诊断匹配完成，共找到 {total_candidates} 个候选结果")
            logger.info(f"处理模式: {result['processing_mode']}")
            return result
            
        except Exception as e:
            logger.error(f"多诊断匹配失败: {e}")
            raise
    
    def _match_single_diagnosis_enhanced(self, diagnosis: str, top_k: int, enhanced_info: Dict[str, Any] = None) -> DiagnosisMatch:
        """
        增强的单个诊断匹配（集成层级相似度计算，启用非诊断实体过滤）
        
        Args:
            diagnosis: 单个诊断文本
            top_k: 返回候选数量
            enhanced_info: 增强的诊断信息
            
        Returns:
            诊断匹配结果
        """
        logger.info(f"增强匹配单个诊断: {diagnosis}")
        
        if enhanced_info is None:
            enhanced_info = {}
        
        try:
            # 1. 提取查询实体信息（默认启用非诊断实体过滤）
            filter_drugs = True
            query_entities = self.ner_service.extract_medical_entities(diagnosis, filter_drugs=filter_drugs)
            entity_count = sum(len(v) for v in query_entities.values())
            logger.debug(f"诊断 '{diagnosis}' 提取到 {entity_count} 个实体（非诊断实体过滤: 开启）")
            
            # 2. 生成查询向量并执行基础搜索
            query_vector = self.embedding_service.encode_query(diagnosis)
            base_search_results = self.milvus_service.search(query_vector, top_k * 2)  # 获取更多候选进行筛选
            
            # 3. 使用层级相似度服务进行增强计算
            enhanced_results = self.hierarchical_similarity.batch_calculate_similarities(
                diagnosis, query_entities, base_search_results
            )
            
            # 4. 转换为标准格式并取top_k
            candidates = []
            for enhanced_record, enhanced_score, similarity_factors in enhanced_results[:top_k]:
                candidate = Candidate(
                    code=enhanced_record.get("code", ""),
                    title=enhanced_record.get("title", ""),
                    score=float(enhanced_score)  # 使用增强后的分数
                )
                # 添加层级信息到候选结果
                candidate.level = enhanced_record.get("level", 1)
                candidate.parent_code = enhanced_record.get("parent_code", "")
                candidate.enhanced_score = float(enhanced_score) if enhanced_score is not None else None
                candidate.original_score = float(enhanced_record.get("original_score", 0.0))
                candidate.similarity_factors = similarity_factors
                
                candidates.append(candidate)
            
            # 5. 使用多维度置信度评分系统
            candidate_records = [
                {
                    'code': c.code, 
                    'title': c.title, 
                    'score': c.enhanced_score if hasattr(c, 'enhanced_score') else c.score,
                    'level': getattr(c, 'level', 1)
                } 
                for c in candidates
            ]
            
            # 计算多维度综合置信度
            confidence_metrics, confidence_factors = self.confidence_service.calculate_comprehensive_confidence(
                diagnosis, candidate_records, 
                similarity_factors={
                    'vector_similarity': enhanced_results[0][2].vector_similarity if enhanced_results else 0.0,
                    'hierarchy_boost': enhanced_results[0][2].hierarchy_boost if enhanced_results else 0.0,
                    'entity_match_score': enhanced_results[0][2].entity_match_score if enhanced_results else 0.0
                } if enhanced_results else None
            )
            
            match_confidence = float(confidence_metrics.overall_confidence)
            
            diagnosis_match = DiagnosisMatch(
                diagnosis_text=diagnosis,
                candidates=candidates,
                match_confidence=match_confidence
            )
            
            # 添加置信度详情到诊断匹配结果
            diagnosis_match.confidence_metrics = confidence_metrics
            diagnosis_match.confidence_factors = confidence_factors
            diagnosis_match.confidence_level = self.confidence_service.get_confidence_level(match_confidence)
            
            logger.info(f"诊断 '{diagnosis}' 找到 {len(candidates)} 个候选结果，增强匹配置信度: {match_confidence:.3f}")
            return diagnosis_match
            
        except Exception as e:
            logger.error(f"增强单个诊断匹配失败: {diagnosis} - {e}")
            # 回退到原始方法
            return self._match_single_diagnosis(diagnosis, top_k)
    
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
                metadata = result.get("metadata", {})
                candidate = Candidate(
                    code=result.get("code", ""),
                    title=result.get("title", ""),
                    score=result.get("score", 0.0),
                    level=metadata.get("level", 1),
                    parent_code=metadata.get("parent_code", ""),
                    enhanced_score=result.get("score", 0.0),
                    original_score=result.get("original_score", result.get("score", 0.0))
                )
                candidates.append(candidate)
            
            # 4. 计算整体匹配置信度（原始方法）
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
    
    def _calculate_enhanced_match_confidence(self, candidates: List[Candidate], enhanced_info: Dict[str, Any], query_entities: Dict[str, List[Dict]] = None) -> float:
        """
        计算增强的匹配置信度
        
        Args:
            candidates: 候选结果列表
            enhanced_info: 增强的诊断信息
            
        Returns:
            增强的置信度分数 (0.0-1.0)
        """
        if not candidates:
            return 0.0
        
        # 基础置信度（原始方法）
        base_confidence = self._calculate_match_confidence(candidates)
        
        # 增强因子
        enhancement_factor = 1.0
        
        # 1. 提取置信度因子
        extraction_confidence = enhanced_info.get('diagnosis_confidence', 0.5)
        if extraction_confidence > 0.7:
            enhancement_factor += 0.1
        elif extraction_confidence < 0.4:
            enhancement_factor -= 0.1
        
        # 2. 实体密度因子
        entity_density = enhanced_info.get('entity_density', 0.0)
        if entity_density > 0.1:  # 高实体密度
            enhancement_factor += 0.05
        
        # 3. 实体类型因子
        metadata = enhanced_info.get('metadata', {})
        if metadata.get('has_disease_entity', False):
            enhancement_factor += 0.1  # 包含疾病实体
        if metadata.get('has_symptom_entity', False):
            enhancement_factor += 0.05  # 包含症状实体
        
        # 4. 边界置信度因子
        boundary_confidence = enhanced_info.get('boundary_confidence', 0.5)
        if boundary_confidence > 0.8:
            enhancement_factor += 0.05
        
        # 5. 查询实体质量因子
        if query_entities:
            total_entities = sum(len(v) for v in query_entities.values())
            high_conf_entities = 0
            
            for entity_type, entities in query_entities.items():
                for entity in entities:
                    if entity.get('confidence', 0.0) > 0.8:
                        high_conf_entities += 1
            
            if total_entities > 0:
                entity_quality_ratio = high_conf_entities / total_entities
                if entity_quality_ratio > 0.6:
                    enhancement_factor += 0.08  # 高质量实体占比高
                elif entity_quality_ratio < 0.3:
                    enhancement_factor -= 0.05  # 低质量实体占比高
        
        # 6. 层级增强分数考虑
        if candidates:
            avg_hierarchy_boost = 0.0
            hierarchy_count = 0
            
            for candidate in candidates:
                if hasattr(candidate, 'similarity_factors') and candidate.similarity_factors:
                    avg_hierarchy_boost += candidate.similarity_factors.hierarchy_boost
                    hierarchy_count += 1
            
            if hierarchy_count > 0:
                avg_hierarchy_boost /= hierarchy_count
                if avg_hierarchy_boost > 0.2:
                    enhancement_factor += 0.06  # 强层级增强
        
        # 应用增强因子
        enhanced_confidence = base_confidence * enhancement_factor
        
        return round(min(enhanced_confidence, 1.0), 3)
    
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
    
    def get_detailed_confidence_report(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        获取详细的置信度报告
        
        Args:
            text: 输入文本
            top_k: 候选数量
            
        Returns:
            详细的置信度分析报告
        """
        logger.info(f"生成详细置信度报告: {text}")
        
        try:
            # 执行多诊断匹配
            match_results = self.match_multiple_diagnoses(text, top_k)
            
            report = {
                "original_text": text,
                "processing_summary": {
                    "total_diagnoses": len(match_results.get("matches", [])),
                    "processing_mode": match_results.get("processing_mode", "standard"),
                    "extraction_metadata": match_results.get("extraction_metadata", {})
                },
                "diagnosis_reports": [],
                "overall_assessment": {}
            }
            
            # 为每个诊断生成详细报告
            total_confidence = 0.0
            high_confidence_count = 0
            
            for match in match_results.get("matches", []):
                if hasattr(match, 'confidence_metrics') and hasattr(match, 'confidence_factors'):
                    # 获取置信度解释
                    explanation = self.confidence_service.get_confidence_explanation(
                        match.confidence_metrics, match.confidence_factors
                    )
                    
                    diagnosis_report = {
                        "diagnosis": match.diagnosis_text,
                        "match_confidence": match.match_confidence,
                        "confidence_level": getattr(match, 'confidence_level', 'unknown'),
                        "confidence_interval": match.confidence_metrics.confidence_interval,
                        "reliability_score": match.confidence_metrics.reliability_score,
                        "top_candidates": [
                            {
                                "code": c.code,
                                "title": c.title,
                                "score": c.score,
                                "enhanced_score": getattr(c, 'enhanced_score', c.score)
                            }
                            for c in match.candidates[:3]
                        ],
                        "factor_analysis": explanation['factor_contributions'],
                        "top_contributing_factors": explanation['top_contributing_factors'],
                        "improvement_suggestions": explanation.get('improvement_suggestions', [])
                    }
                    
                    report["diagnosis_reports"].append(diagnosis_report)
                    total_confidence += match.match_confidence
                    
                    if match.match_confidence >= self.confidence_service.confidence_thresholds['high_confidence']:
                        high_confidence_count += 1
            
            # 生成整体评估
            if report["diagnosis_reports"]:
                avg_confidence = total_confidence / len(report["diagnosis_reports"])
                
                report["overall_assessment"] = {
                    "average_confidence": avg_confidence,
                    "high_confidence_ratio": high_confidence_count / len(report["diagnosis_reports"]),
                    "overall_quality": self._assess_overall_quality(avg_confidence, high_confidence_count, len(report["diagnosis_reports"])),
                    "recommendations": self._generate_recommendations(report["diagnosis_reports"])
                }
            
            logger.info(f"置信度报告生成完成，平均置信度: {report['overall_assessment'].get('average_confidence', 0.0):.3f}")
            return report
            
        except Exception as e:
            logger.error(f"置信度报告生成失败: {e}")
            return {
                "original_text": text,
                "error": str(e),
                "processing_summary": {},
                "diagnosis_reports": [],
                "overall_assessment": {}
            }
    
    def _assess_overall_quality(self, avg_confidence: float, high_conf_count: int, total_count: int) -> str:
        """评估整体质量"""
        if avg_confidence >= 0.8 and high_conf_count / total_count >= 0.7:
            return "优秀"
        elif avg_confidence >= 0.6 and high_conf_count / total_count >= 0.5:
            return "良好" 
        elif avg_confidence >= 0.4:
            return "一般"
        else:
            return "待改进"
    
    def _generate_recommendations(self, diagnosis_reports: List[Dict]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        low_confidence_count = sum(1 for r in diagnosis_reports if r['match_confidence'] < 0.6)
        total_count = len(diagnosis_reports)
        
        if low_confidence_count / total_count > 0.5:
            recommendations.append("建议补充更详细的临床症状描述")
            recommendations.append("考虑使用更准确的医学术语")
        
        # 检查是否有改进建议
        all_suggestions = []
        for report in diagnosis_reports:
            all_suggestions.extend(report.get('improvement_suggestions', []))
        
        if all_suggestions:
            # 去重并添加最常见的建议
            unique_suggestions = list(set(all_suggestions))
            recommendations.extend(unique_suggestions[:2])
        
        return recommendations 