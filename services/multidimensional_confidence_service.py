#!/usr/bin/env python3
"""
多维度置信度评分系统
基于多个维度的综合评估来计算最终的诊断置信度
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ConfidenceFactors:
    """置信度评分因子"""
    # 基础因子
    vector_similarity: float = 0.0          # 向量相似度
    hierarchy_boost: float = 0.0            # 层级增强分数
    entity_match_score: float = 0.0         # 实体匹配分数
    
    # 语义因子
    semantic_coherence: float = 0.0         # 语义一致性
    context_consistency: float = 0.0        # 上下文一致性
    terminology_accuracy: float = 0.0       # 术语准确性
    
    # 复杂度因子
    diagnosis_complexity: float = 0.0       # 诊断复杂度
    professional_specificity: float = 0.0  # 专业特异性
    clinical_relevance: float = 0.0         # 临床相关性
    
    # 质量因子
    data_quality: float = 0.0               # 数据质量
    model_uncertainty: float = 0.0          # 模型不确定性
    cross_validation_score: float = 0.0     # 交叉验证分数
    
    def __post_init__(self):
        """确保所有值都是Python原生float类型"""
        self.vector_similarity = float(self.vector_similarity)
        self.hierarchy_boost = float(self.hierarchy_boost)
        self.entity_match_score = float(self.entity_match_score)
        self.semantic_coherence = float(self.semantic_coherence)
        self.context_consistency = float(self.context_consistency)
        self.terminology_accuracy = float(self.terminology_accuracy)
        self.diagnosis_complexity = float(self.diagnosis_complexity)
        self.professional_specificity = float(self.professional_specificity)
        self.clinical_relevance = float(self.clinical_relevance)
        self.data_quality = float(self.data_quality)
        self.model_uncertainty = float(self.model_uncertainty)
        self.cross_validation_score = float(self.cross_validation_score)


@dataclass
class ConfidenceMetrics:
    """置信度评估指标"""
    overall_confidence: float = 0.0         # 总体置信度
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 置信区间
    reliability_score: float = 0.0          # 可靠性分数
    prediction_variance: float = 0.0        # 预测方差
    calibration_score: float = 0.0          # 校准分数
    
    def __post_init__(self):
        """确保所有值都是Python原生float类型"""
        self.overall_confidence = float(self.overall_confidence)
        self.reliability_score = float(self.reliability_score)
        self.prediction_variance = float(self.prediction_variance)
        self.calibration_score = float(self.calibration_score)
        # 处理置信区间元组
        if self.confidence_interval:
            self.confidence_interval = (
                float(self.confidence_interval[0]),
                float(self.confidence_interval[1])
            )


class MultiDimensionalConfidenceService:
    """多维度置信度评分服务"""
    
    def __init__(self, 
                 embedding_service=None,
                 ner_service=None,
                 hierarchical_similarity_service=None):
        """
        初始化多维度置信度服务
        
        Args:
            embedding_service: 嵌入服务
            ner_service: NER服务  
            hierarchical_similarity_service: 层级相似度服务
        """
        self.embedding_service = embedding_service
        self.ner_service = ner_service
        self.hierarchical_similarity_service = hierarchical_similarity_service
        
        # 置信度因子权重
        self.factor_weights = {
            # 基础因子 (50%)
            'vector_similarity': 0.20,
            'hierarchy_boost': 0.15,
            'entity_match_score': 0.15,
            
            # 语义因子 (30%)
            'semantic_coherence': 0.12,
            'context_consistency': 0.10,
            'terminology_accuracy': 0.08,
            
            # 复杂度因子 (15%)
            'diagnosis_complexity': 0.05,
            'professional_specificity': 0.05,
            'clinical_relevance': 0.05,
            
            # 质量因子 (5%)
            'data_quality': 0.02,
            'model_uncertainty': 0.02,
            'cross_validation_score': 0.01
        }
        
        # 动态阈值配置
        self.confidence_thresholds = {
            'high_confidence': 0.80,      # 高置信度阈值
            'medium_confidence': 0.60,    # 中等置信度阈值
            'low_confidence': 0.40,       # 低置信度阈值
            'reject_threshold': 0.20      # 拒绝阈值
        }
        
        # ICD数据缓存（用于动态术语专业度计算）
        self.icd_terminology_cache = {}
        self.icd_data_loaded = False
        
        # 诊断复杂度分类器
        self.complexity_classifier = self._init_complexity_classifier()
        
        logger.info("多维度置信度评分系统初始化完成")
        logger.info(f"权重配置: {self.factor_weights}")
    
    
    def _init_complexity_classifier(self) -> Dict[str, Any]:
        """初始化复杂度分类器"""
        return {
            'simple_patterns': [
                r'^[^，。；]{2,8}病$',        # 简单疾病名
                r'^[^，。；]{2,6}[痛|热|肿]$'   # 简单症状
            ],
            'moderate_patterns': [
                r'伴[^，。；]{2,10}',         # 伴随症状
                r'[^，。；]{3,12}综合征',      # 综合征
                r'[急性|慢性][^，。；]{2,10}'   # 急慢性疾病
            ],
            'complex_patterns': [
                r'[^，。；]{5,}并[^，。；]{5,}',  # 并发症
                r'[^，。；]{3,}伴[^，。；]{3,}伴[^，。；]{3,}',  # 多重并发
                r'[^，。；]{8,}酸中毒',        # 复杂代谢疾病
                r'多发性[^，。；]{3,}',        # 多发性疾病
            ]
        }
    
    def calculate_comprehensive_confidence(self,
                                         query_text: str,
                                         candidate_records: List[Dict[str, Any]],
                                         similarity_factors: Optional[Dict] = None) -> Tuple[ConfidenceMetrics, ConfidenceFactors]:
        """
        计算综合置信度评分
        
        Args:
            query_text: 查询文本
            candidate_records: 候选记录列表
            similarity_factors: 相似度因子（来自Stage 2）
            
        Returns:
            置信度指标和详细因子
        """
        logger.info(f"开始计算综合置信度: {query_text}")
        
        try:
            # 1. 计算基础置信度因子
            factors = self._calculate_base_factors(
                query_text, candidate_records, similarity_factors
            )
            
            # 2. 计算语义置信度因子
            semantic_factors = self._calculate_semantic_factors(
                query_text, candidate_records
            )
            
            # 3. 计算复杂度置信度因子
            complexity_factors = self._calculate_complexity_factors(
                query_text, candidate_records
            )
            
            # 4. 计算质量置信度因子  
            quality_factors = self._calculate_quality_factors(
                query_text, candidate_records
            )
            
            # 5. 合并所有因子
            all_factors = self._merge_factors(
                factors, semantic_factors, complexity_factors, quality_factors
            )
            
            # 6. 计算最终置信度指标
            metrics = self._calculate_final_metrics(all_factors, candidate_records)
            
            logger.info(f"综合置信度计算完成: {metrics.overall_confidence:.4f}")
            return metrics, all_factors
            
        except Exception as e:
            logger.error(f"综合置信度计算失败: {e}")
            # 返回默认值
            return (
                ConfidenceMetrics(overall_confidence=0.5),
                ConfidenceFactors()
            )
    
    def _calculate_base_factors(self,
                              query_text: str,
                              candidate_records: List[Dict[str, Any]],
                              similarity_factors: Optional[Dict] = None) -> ConfidenceFactors:
        """计算基础置信度因子"""
        factors = ConfidenceFactors()
        
        if not candidate_records:
            return factors
        
        try:
            # 使用最佳候选记录
            best_candidate = candidate_records[0]
            
            # 1. 向量相似度（直接使用或重新计算）
            if similarity_factors and 'vector_similarity' in similarity_factors:
                factors.vector_similarity = similarity_factors['vector_similarity']
            else:
                factors.vector_similarity = best_candidate.get('score', 0.0)
            
            # 2. 层级增强分数
            if similarity_factors and 'hierarchy_boost' in similarity_factors:
                factors.hierarchy_boost = similarity_factors['hierarchy_boost']
            else:
                factors.hierarchy_boost = self._calculate_hierarchy_score(
                    best_candidate
                )
            
            # 3. 实体匹配分数
            if similarity_factors and 'entity_match_score' in similarity_factors:
                factors.entity_match_score = similarity_factors['entity_match_score']
            else:
                factors.entity_match_score = self._calculate_entity_match(
                    query_text, best_candidate
                )
            
            return factors
            
        except Exception as e:
            logger.warning(f"基础因子计算失败: {e}")
            return factors
    
    def _calculate_semantic_factors(self,
                                  query_text: str, 
                                  candidate_records: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算语义置信度因子"""
        semantic_factors = {
            'semantic_coherence': 0.0,
            'context_consistency': 0.0,
            'terminology_accuracy': 0.0
        }
        
        if not candidate_records:
            return semantic_factors
        
        try:
            best_candidate = candidate_records[0]
            candidate_text = best_candidate.get('preferred_zh', '')
            
            # 1. 语义一致性（基于嵌入相似度）
            if self.embedding_service:
                query_vector = self.embedding_service.encode_query(query_text)
                candidate_vector = self.embedding_service.encode_query(candidate_text)
                semantic_factors['semantic_coherence'] = cosine_similarity(
                    [query_vector], [candidate_vector]
                )[0][0]
            
            # 2. 上下文一致性（基于词汇重叠和语义距离）
            semantic_factors['context_consistency'] = self._calculate_context_consistency(
                query_text, candidate_text
            )
            
            # 3. 术语准确性（基于医学术语匹配）
            semantic_factors['terminology_accuracy'] = self._calculate_terminology_accuracy(
                query_text, candidate_text
            )
            
            return semantic_factors
            
        except Exception as e:
            logger.warning(f"语义因子计算失败: {e}")
            return semantic_factors
    
    def _calculate_complexity_factors(self,
                                    query_text: str,
                                    candidate_records: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算复杂度置信度因子"""
        complexity_factors = {
            'diagnosis_complexity': 0.0,
            'professional_specificity': 0.0,
            'clinical_relevance': 0.0
        }
        
        try:
            # 1. 诊断复杂度评估
            complexity_factors['diagnosis_complexity'] = self._assess_diagnosis_complexity(
                query_text
            )
            
            # 2. 专业特异性评估
            complexity_factors['professional_specificity'] = self._assess_professional_specificity(
                query_text
            )
            
            # 3. 临床相关性评估
            if candidate_records:
                complexity_factors['clinical_relevance'] = self._assess_clinical_relevance(
                    query_text, candidate_records[0]
                )
            
            return complexity_factors
            
        except Exception as e:
            logger.warning(f"复杂度因子计算失败: {e}")
            return complexity_factors
    
    def _calculate_quality_factors(self,
                                 query_text: str,
                                 candidate_records: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算质量置信度因子"""
        quality_factors = {
            'data_quality': 0.0,
            'model_uncertainty': 0.0,
            'cross_validation_score': 0.0
        }
        
        try:
            # 1. 数据质量评估
            quality_factors['data_quality'] = self._assess_data_quality(
                candidate_records
            )
            
            # 2. 模型不确定性评估
            quality_factors['model_uncertainty'] = self._assess_model_uncertainty(
                candidate_records
            )
            
            # 3. 交叉验证分数（基于候选结果的一致性）
            quality_factors['cross_validation_score'] = self._calculate_cross_validation(
                candidate_records
            )
            
            return quality_factors
            
        except Exception as e:
            logger.warning(f"质量因子计算失败: {e}")
            return quality_factors
    
    def _calculate_context_consistency(self, query_text: str, candidate_text: str) -> float:
        """计算上下文一致性"""
        try:
            # 简化的上下文一致性计算
            query_words = set(query_text.replace(' ', ''))
            candidate_words = set(candidate_text.replace(' ', ''))
            
            if not query_words or not candidate_words:
                return 0.0
            
            # Jaccard相似度
            intersection = len(query_words & candidate_words)
            union = len(query_words | candidate_words)
            
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # 考虑长度相似性
            length_similarity = 1.0 - abs(len(query_text) - len(candidate_text)) / max(len(query_text), len(candidate_text), 1)
            
            # 综合分数
            consistency = (jaccard_score * 0.7 + length_similarity * 0.3)
            return min(consistency, 1.0)
            
        except Exception as e:
            logger.warning(f"上下文一致性计算失败: {e}")
            return 0.5
    
    def _calculate_terminology_accuracy(self, query_text: str, candidate_text: str) -> float:
        """计算术语准确性（基于NER实体匹配）"""
        try:
            # 如果有NER服务，使用基于实体的匹配
            if self.ner_service:
                return self._calculate_terminology_accuracy_with_ner(query_text, candidate_text)
            else:
                # 回退到基于ICD数据的动态评估
                return self._calculate_terminology_accuracy_fallback(query_text, candidate_text)
                
        except Exception as e:
            logger.warning(f"术语准确性计算失败: {e}")
            return 0.5
    
    def _calculate_terminology_accuracy_with_ner(self, query_text: str, candidate_text: str) -> float:
        """基于NER结果计算术语准确性"""
        try:
            # 提取查询文本和候选文本的医学实体
            query_entities = self.ner_service.extract_medical_entities(query_text)
            candidate_entities = self.ner_service.extract_medical_entities(candidate_text)
            
            total_weight = 0.0
            matched_weight = 0.0
            
            # 遍历查询文本中的实体
            for entity_type, entities in query_entities.items():
                # 给不同类型的实体分配不同权重
                type_weight = self._get_entity_type_weight(entity_type)
                
                for entity in entities:
                    # 使用NER置信度 * 实体类型权重作为总权重
                    entity_weight = entity['confidence'] * type_weight
                    total_weight += entity_weight
                    
                    # 检查是否在候选文本的实体中匹配
                    if self._entity_matches_in_candidate(entity, candidate_entities):
                        matched_weight += entity_weight
            
            # 计算匹配准确性
            if total_weight > 0:
                accuracy = matched_weight / total_weight
            else:
                # 如果没有提取到实体，使用字符级匹配作为回退
                accuracy = self._calculate_char_level_similarity(query_text, candidate_text)
            
            return min(accuracy, 1.0)
            
        except Exception as e:
            logger.warning(f"基于NER的术语准确性计算失败: {e}")
            return self._calculate_terminology_accuracy_fallback(query_text, candidate_text)
    
    def _calculate_terminology_accuracy_fallback(self, query_text: str, candidate_text: str) -> float:
        """术语准确性计算的回退方法（基于动态术语评估）"""
        try:
            # 基于文本特征的动态评估
            query_terms = self._extract_medical_terms_from_text(query_text)
            candidate_terms = self._extract_medical_terms_from_text(candidate_text)
            
            if not query_terms:
                return self._calculate_char_level_similarity(query_text, candidate_text)
            
            total_score = 0.0
            matched_score = 0.0
            
            for term, weight in query_terms.items():
                total_score += weight
                if term in candidate_terms:
                    matched_score += weight
            
            return matched_score / total_score if total_score > 0 else 0.5
            
        except Exception as e:
            logger.warning(f"回退术语准确性计算失败: {e}")
            return 0.5
    
    def _get_entity_type_weight(self, entity_type: str) -> float:
        """获取不同实体类型的权重"""
        type_weights = {
            'disease': 1.0,        # 疾病实体最重要
            'symptom': 0.8,        # 症状次之
            'anatomy': 0.6,        # 解剖部位
            'pathology': 0.9,      # 病理状态
            'treatment': 0.5,      # 治疗方法
            'drug': 0.3,           # 药物（诊断相关性较低）
            'equipment': 0.2       # 设备（诊断相关性最低）
        }
        return type_weights.get(entity_type, 0.5)
    
    def _entity_matches_in_candidate(self, query_entity: Dict[str, Any], candidate_entities: Dict[str, List[Dict[str, Any]]]) -> bool:
        """检查查询实体是否在候选实体中匹配"""
        query_text = query_entity['text']
        
        # 遍历候选文本中的所有实体类型
        for entity_type, entities in candidate_entities.items():
            for entity in entities:
                candidate_text = entity['text']
                
                # 精确匹配
                if query_text == candidate_text:
                    return True
                
                # 包含匹配（考虑中文医学术语的部分匹配）
                if query_text in candidate_text or candidate_text in query_text:
                    # 确保匹配长度合理（避免过短的误匹配）
                    if len(query_text) >= 2 and len(candidate_text) >= 2:
                        return True
        
        return False
    
    def _calculate_char_level_similarity(self, text1: str, text2: str) -> float:
        """计算字符级相似度（回退方法）"""
        if not text1 or not text2:
            return 0.0
        
        chars1 = set(text1.replace(' ', ''))
        chars2 = set(text2.replace(' ', ''))
        
        if not chars1 or not chars2:
            return 0.0
        
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_medical_terms_from_text(self, text: str) -> Dict[str, float]:
        """从文本中提取医学术语及其权重（动态方法）"""
        terms = {}
        
        # 基于常见医学模式提取术语
        import re
        
        # 疾病模式
        disease_patterns = [
            r'[^，。；\s]{2,10}病',
            r'[^，。；\s]{2,10}症',
            r'[^，。；\s]{2,10}炎',
            r'[^，。；\s]{2,10}综合征',
            r'急性[^，。；\s]{2,10}',
            r'慢性[^，。；\s]{2,10}',
        ]
        
        for pattern in disease_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 基于术语特征计算权重
                weight = self._calculate_term_weight(match)
                terms[match] = weight
        
        return terms
    
    def _calculate_term_weight(self, term: str) -> float:
        """基于术语特征计算权重（集成ICD数据）"""
        # 首先尝试从ICD缓存获取专业度
        icd_weight = self._get_term_specificity_from_icd(term)
        if icd_weight > 0.5:  # 如果ICD中找到了较好的匹配
            return icd_weight
        
        # 回退到基于特征的计算
        weight = 0.5  # 基础权重
        
        # 长度因子（较长的术语通常更专业）
        if len(term) >= 6:
            weight += 0.3
        elif len(term) >= 4:
            weight += 0.2
        
        # 专业性关键词
        professional_keywords = ['急性', '慢性', '综合征', '功能不全', '梗死', '出血', '肿瘤', '癌']
        for keyword in professional_keywords:
            if keyword in term:
                weight += 0.2
                break
        
        # 结合ICD权重（如果有部分匹配）
        if icd_weight != 0.5:
            weight = (weight + icd_weight) / 2
        
        # 确保权重在合理范围内
        return min(weight, 1.0)
    
    def _load_icd_terminology_if_needed(self):
        """按需加载ICD术语数据"""
        if self.icd_data_loaded:
            return
        
        try:
            import pandas as pd
            import os
            
            # 尝试加载ICD数据文件
            icd_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ICD_10v601.csv')
            if os.path.exists(icd_file_path):
                df = pd.read_csv(icd_file_path)
                
                for _, row in df.iterrows():
                    code = row.get('code', '')
                    disease = row.get('disease', '')
                    
                    if disease and len(disease.strip()) > 1:
                        # 基于ICD层级计算专业度
                        level = self._parse_icd_level(code)
                        base_score = self._calculate_icd_base_score(level, disease)
                        
                        # 基于疾病类别调整分数
                        category_score = self._calculate_category_score(code)
                        
                        # 最终专业度分数
                        final_score = (base_score + category_score) / 2
                        self.icd_terminology_cache[disease.strip()] = final_score
                
                self.icd_data_loaded = True
                logger.info(f"已加载 {len(self.icd_terminology_cache)} 个ICD术语到缓存")
            else:
                logger.warning(f"ICD数据文件不存在: {icd_file_path}")
                
        except Exception as e:
            logger.warning(f"加载ICD术语数据失败: {e}")
    
    def _parse_icd_level(self, code: str) -> int:
        """解析ICD代码的层级"""
        if not code:
            return 1
        
        if '.' not in code:
            return 1  # 主类别
        
        dot_parts = code.split('.')
        if len(dot_parts) == 2:
            after_dot = dot_parts[1]
            if len(after_dot) == 1:
                return 2  # 亚类别
            else:
                return 3  # 详细类别
        
        return 1
    
    def _calculate_icd_base_score(self, level: int, disease_name: str) -> float:
        """基于ICD层级和疾病名称计算基础专业度分数"""
        # 层级分数：越详细越专业
        level_scores = {1: 0.6, 2: 0.75, 3: 0.9}
        level_score = level_scores.get(level, 0.6)
        
        # 名称复杂度分数
        name_complexity = min(len(disease_name) / 15.0, 0.3)  # 最多0.3分
        
        # 专业术语检测
        professional_bonus = 0.0
        professional_terms = ['急性', '慢性', '综合征', '功能不全', '梗死', '出血', '肿瘤', '癌', '病毒', '细菌']
        for term in professional_terms:
            if term in disease_name:
                professional_bonus = 0.1
                break
        
        return min(level_score + name_complexity + professional_bonus, 1.0)
    
    def _calculate_category_score(self, code: str) -> float:
        """基于ICD主类别计算专业度分数"""
        if not code:
            return 0.5
        
        main_category = code[0].upper()
        category_scores = {
            'A': 0.8,  # 传染病 - 较高专业度
            'B': 0.8,  # 传染病
            'C': 0.95, # 肿瘤 - 最高专业度
            'D': 0.9,  # 血液病 - 高专业度
            'E': 0.85, # 内分泌 - 高专业度
            'F': 0.8,  # 精神障碍 - 较高专业度
            'G': 0.9,  # 神经系统 - 高专业度
            'H': 0.75, # 眼耳疾病 - 中高专业度
            'I': 0.9,  # 循环系统 - 高专业度
            'J': 0.75, # 呼吸系统 - 中高专业度
            'K': 0.8,  # 消化系统 - 较高专业度
            'L': 0.7,  # 皮肤病 - 中等专业度
            'M': 0.75, # 肌肉骨骼 - 中高专业度
            'N': 0.8,  # 泌尿生殖 - 较高专业度
            'O': 0.85, # 妊娠分娩 - 高专业度
            'P': 0.9,  # 围产期 - 高专业度
            'Q': 0.85, # 先天畸形 - 高专业度
            'R': 0.6,  # 症状体征 - 中等专业度
            'S': 0.7,  # 损伤 - 中等专业度
            'T': 0.75, # 中毒 - 中高专业度
            'Z': 0.5   # 影响健康状态 - 较低专业度
        }
        return category_scores.get(main_category, 0.6)
    
    def _get_term_specificity_from_icd(self, term: str) -> float:
        """从ICD缓存中获取术语专业度"""
        self._load_icd_terminology_if_needed()
        
        # 精确匹配
        if term in self.icd_terminology_cache:
            return self.icd_terminology_cache[term]
        
        # 部分匹配
        for icd_term, score in self.icd_terminology_cache.items():
            if term in icd_term or icd_term in term:
                if len(term) >= 2 and len(icd_term) >= 2:
                    # 基于匹配长度调整分数
                    match_ratio = min(len(term), len(icd_term)) / max(len(term), len(icd_term))
                    return score * match_ratio
        
        # 如果没有找到匹配，返回默认值
        return 0.5
    
    def _assess_diagnosis_complexity(self, query_text: str) -> float:
        """评估诊断复杂度"""
        try:
            import re
            
            # 复杂度评分（越复杂分数越高）
            complexity_score = 0.0
            
            # 检查复杂度模式
            for pattern in self.complexity_classifier['complex_patterns']:
                if re.search(pattern, query_text):
                    complexity_score += 0.8
            
            for pattern in self.complexity_classifier['moderate_patterns']:
                if re.search(pattern, query_text):
                    complexity_score += 0.5
                    
            for pattern in self.complexity_classifier['simple_patterns']:
                if re.search(pattern, query_text):
                    complexity_score += 0.2
            
            # 基于文本长度和复杂度
            length_factor = min(len(query_text) / 50.0, 1.0)  # 标准化到0-1
            complexity_score += length_factor * 0.3
            
            # 基于分隔符数量（多诊断复杂度）
            separator_count = query_text.count('，') + query_text.count('；') + query_text.count('伴')
            if separator_count > 0:
                complexity_score += min(separator_count * 0.2, 0.6)
            
            return min(complexity_score, 1.0)
            
        except Exception as e:
            logger.warning(f"诊断复杂度评估失败: {e}")
            return 0.5
    
    def _assess_professional_specificity(self, query_text: str) -> float:
        """评估专业特异性（基于NER实体类型和置信度）"""
        try:
            # 如果有NER服务，使用基于实体的评估
            if self.ner_service:
                return self._assess_professional_specificity_with_ner(query_text)
            else:
                # 回退到基于文本特征的评估
                return self._assess_professional_specificity_fallback(query_text)
                
        except Exception as e:
            logger.warning(f"专业特异性评估失败: {e}")
            return 0.5
    
    def _assess_professional_specificity_with_ner(self, query_text: str) -> float:
        """基于NER实体评估专业特异性"""
        try:
            entities = self.ner_service.extract_medical_entities(query_text)
            
            if not any(entities.values()):
                # 没有识别到医学实体，使用回退方法
                return self._assess_professional_specificity_fallback(query_text)
            
            total_weight = 0.0
            specificity_sum = 0.0
            
            for entity_type, entity_list in entities.items():
                # 获取实体类型的专业度权重
                type_specificity = self._get_entity_type_specificity(entity_type)
                
                for entity in entity_list:
                    # 结合NER置信度和实体类型的专业度
                    entity_confidence = entity['confidence']
                    entity_weight = entity_confidence
                    
                    # 基于实体内容进一步调整专业度
                    content_specificity = self._assess_entity_content_specificity(entity['text'])
                    final_specificity = (type_specificity + content_specificity) / 2
                    
                    total_weight += entity_weight
                    specificity_sum += entity_weight * final_specificity
            
            if total_weight > 0:
                return min(specificity_sum / total_weight, 1.0)
            else:
                return self._assess_professional_specificity_fallback(query_text)
                
        except Exception as e:
            logger.warning(f"基于NER的专业特异性评估失败: {e}")
            return self._assess_professional_specificity_fallback(query_text)
    
    def _assess_professional_specificity_fallback(self, query_text: str) -> float:
        """专业特异性评估的回退方法"""
        try:
            # 基于动态术语提取的专业度评估
            terms = self._extract_medical_terms_from_text(query_text)
            
            if not terms:
                # 如果没有专业术语，基于文本特征判断
                if any(keyword in query_text for keyword in ['急性', '慢性', '并发', '综合征']):
                    return 0.6
                elif any(keyword in query_text for keyword in ['病', '症', '炎']):
                    return 0.4
                else:
                    return 0.2
            
            # 计算平均专业度
            total_weight = sum(terms.values())
            if total_weight > 0:
                return min(total_weight / len(terms), 1.0)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"回退专业特异性评估失败: {e}")
            return 0.5
    
    def _get_entity_type_specificity(self, entity_type: str) -> float:
        """获取不同实体类型的专业特异性分数"""
        type_specificity = {
            'disease': 0.9,        # 疾病名称专业度最高
            'pathology': 0.85,     # 病理状态专业度很高  
            'symptom': 0.6,        # 症状专业度中等
            'anatomy': 0.5,        # 解剖部位专业度中等
            'treatment': 0.8,      # 治疗方法专业度较高
            'drug': 0.7,           # 药物专业度较高
            'equipment': 0.4       # 设备专业度较低
        }
        return type_specificity.get(entity_type, 0.5)
    
    def _assess_entity_content_specificity(self, entity_text: str) -> float:
        """基于实体内容评估专业特异性"""
        specificity = 0.5  # 基础分数
        
        # 长度因子：更长的术语通常更专业
        if len(entity_text) >= 6:
            specificity += 0.2
        elif len(entity_text) >= 4:
            specificity += 0.1
        
        # 专业医学前缀/后缀
        professional_prefixes = ['急性', '慢性', '原发性', '继发性', '复发性']
        professional_suffixes = ['综合征', '功能不全', '功能障碍', '梗死', '出血', '肿瘤', '癌症']
        
        for prefix in professional_prefixes:
            if entity_text.startswith(prefix):
                specificity += 0.15
                break
        
        for suffix in professional_suffixes:
            if entity_text.endswith(suffix):
                specificity += 0.15
                break
        
        # 复杂医学词汇
        complex_terms = ['酸中毒', '综合征', '功能不全', '动脉硬化', '心肌梗死']
        for term in complex_terms:
            if term in entity_text:
                specificity += 0.1
                break
        
        return min(specificity, 1.0)
    
    def _assess_clinical_relevance(self, query_text: str, candidate: Dict[str, Any]) -> float:
        """评估临床相关性"""
        try:
            relevance_score = 0.0
            
            # 基于ICD层级的相关性
            code = candidate.get('code', '')
            level = candidate.get('level', 1)
            
            # 层级越深，越具体，临床相关性可能越高
            if level == 3:
                relevance_score += 0.4  # 细分类
            elif level == 2:
                relevance_score += 0.3  # 亚类
            else:
                relevance_score += 0.2  # 主类
            
            # 基于ICD主类别的临床重要性
            if code:
                main_category = code[0]
                category_relevance = {
                    'I': 0.9,  # 循环系统 - 高临床重要性
                    'C': 0.9,  # 肿瘤 - 高临床重要性
                    'E': 0.8,  # 内分泌 - 较高临床重要性
                    'J': 0.7,  # 呼吸系统 - 中等重要性
                    'K': 0.7,  # 消化系统 - 中等重要性
                    'N': 0.7,  # 泌尿生殖系统 - 中等重要性
                    'S': 0.6,  # 损伤中毒 - 中等重要性
                }.get(main_category, 0.5)
                
                relevance_score += category_relevance * 0.4
            
            # 基于候选文本与查询的临床上下文匹配
            candidate_text = candidate.get('preferred_zh', '')
            context_match = self._calculate_context_consistency(query_text, candidate_text)
            relevance_score += context_match * 0.2
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.warning(f"临床相关性评估失败: {e}")
            return 0.5
    
    def _assess_data_quality(self, candidate_records: List[Dict[str, Any]]) -> float:
        """评估数据质量"""
        try:
            if not candidate_records:
                return 0.0
            
            quality_score = 0.0
            
            # 检查候选结果的完整性
            complete_records = 0
            for record in candidate_records:
                if (record.get('code') and 
                    record.get('preferred_zh') and 
                    record.get('score', 0) > 0):
                    complete_records += 1
            
            completeness = complete_records / len(candidate_records)
            quality_score += completeness * 0.4
            
            # 检查分数分布的合理性
            scores = [r.get('score', 0) for r in candidate_records]
            if scores:
                max_score = max(scores)
                min_score = min(scores)
                score_range = max_score - min_score
                
                # 合理的分数分布应该有一定差异
                if score_range > 0.1:
                    quality_score += 0.3
                if max_score > 0.7:  # 有高质量匹配
                    quality_score += 0.3
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"数据质量评估失败: {e}")
            return 0.5
    
    def _assess_model_uncertainty(self, candidate_records: List[Dict[str, Any]]) -> float:
        """评估模型不确定性（分数越高表示不确定性越低）"""
        try:
            if not candidate_records:
                return 0.0
            
            scores = [r.get('score', 0) for r in candidate_records]
            if not scores:
                return 0.0
            
            # 计算分数的标准差（不确定性指标）
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            
            # 标准差越小，不确定性越低，置信度越高
            uncertainty_score = 1.0 - min(std_score, 0.5) / 0.5  # 归一化到0-1
            
            # 考虑最高分数（最高分数越高，不确定性越低）
            max_score = max(scores)
            score_confidence = max_score
            
            # 综合评估
            final_uncertainty = (uncertainty_score * 0.6 + score_confidence * 0.4)
            return min(final_uncertainty, 1.0)
            
        except Exception as e:
            logger.warning(f"模型不确定性评估失败: {e}")
            return 0.5
    
    def _calculate_cross_validation(self, candidate_records: List[Dict[str, Any]]) -> float:
        """计算交叉验证分数（基于候选结果的一致性）"""
        try:
            if len(candidate_records) < 2:
                return 0.5
            
            # 检查top-k结果的一致性
            top_scores = [r.get('score', 0) for r in candidate_records[:min(3, len(candidate_records))]]
            
            if not top_scores:
                return 0.0
            
            # 计算top结果的分数差异
            max_score = max(top_scores)
            min_score = min(top_scores)
            
            # 如果最高分远高于其他分数，说明结果一致性好
            if max_score > 0.8 and (max_score - min_score) > 0.2:
                return 0.8
            elif max_score > 0.6 and (max_score - min_score) > 0.1:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            logger.warning(f"交叉验证分数计算失败: {e}")
            return 0.5
    
    def _calculate_hierarchy_score(self, candidate: Dict[str, Any]) -> float:
        """计算层级分数（简化版，用于回退）"""
        try:
            level = candidate.get('level', 1)
            # 基于层级的简单分数
            level_scores = {1: 0.6, 2: 0.8, 3: 1.0}
            return level_scores.get(level, 0.5)
        except:
            return 0.5
    
    def _calculate_entity_match(self, query_text: str, candidate: Dict[str, Any]) -> float:
        """计算实体匹配分数（简化版，用于回退）"""
        try:
            candidate_text = candidate.get('preferred_zh', '')
            # 简单的字符重叠检查
            query_chars = set(query_text)
            candidate_chars = set(candidate_text)
            
            if not query_chars or not candidate_chars:
                return 0.0
            
            overlap = len(query_chars & candidate_chars)
            union = len(query_chars | candidate_chars)
            
            return overlap / union if union > 0 else 0.0
        except:
            return 0.0
    
    def _merge_factors(self, *factor_dicts) -> ConfidenceFactors:
        """合并所有因子"""
        factors = ConfidenceFactors()
        
        for factor_dict in factor_dicts:
            if isinstance(factor_dict, dict):
                for key, value in factor_dict.items():
                    if hasattr(factors, key):
                        setattr(factors, key, value)
            elif isinstance(factor_dict, ConfidenceFactors):
                # 直接使用基础因子
                for field in ['vector_similarity', 'hierarchy_boost', 'entity_match_score']:
                    if hasattr(factor_dict, field):
                        setattr(factors, field, getattr(factor_dict, field))
        
        return factors
    
    def _calculate_final_metrics(self, 
                               factors: ConfidenceFactors, 
                               candidate_records: List[Dict[str, Any]]) -> ConfidenceMetrics:
        """计算最终置信度指标"""
        try:
            # 计算加权总体置信度
            overall_confidence = 0.0
            
            factor_dict = {
                'vector_similarity': factors.vector_similarity,
                'hierarchy_boost': factors.hierarchy_boost,
                'entity_match_score': factors.entity_match_score,
                'semantic_coherence': factors.semantic_coherence,
                'context_consistency': factors.context_consistency,
                'terminology_accuracy': factors.terminology_accuracy,
                'diagnosis_complexity': factors.diagnosis_complexity,
                'professional_specificity': factors.professional_specificity,
                'clinical_relevance': factors.clinical_relevance,
                'data_quality': factors.data_quality,
                'model_uncertainty': factors.model_uncertainty,
                'cross_validation_score': factors.cross_validation_score
            }
            
            for factor_name, factor_value in factor_dict.items():
                weight = self.factor_weights.get(factor_name, 0.0)
                overall_confidence += factor_value * weight
            
            # 计算置信区间
            variance = self._calculate_prediction_variance(factors, candidate_records)
            confidence_interval = self._calculate_confidence_interval(overall_confidence, variance)
            
            # 计算可靠性分数
            reliability_score = self._calculate_reliability_score(factors)
            
            # 计算校准分数
            calibration_score = self._calculate_calibration_score(overall_confidence, factors)
            
            return ConfidenceMetrics(
                overall_confidence=min(overall_confidence, 1.0),
                confidence_interval=confidence_interval,
                reliability_score=reliability_score,
                prediction_variance=variance,
                calibration_score=calibration_score
            )
            
        except Exception as e:
            logger.error(f"最终指标计算失败: {e}")
            return ConfidenceMetrics(overall_confidence=0.5)
    
    def _calculate_prediction_variance(self, 
                                     factors: ConfidenceFactors, 
                                     candidate_records: List[Dict[str, Any]]) -> float:
        """计算预测方差"""
        try:
            # 基于候选结果分数的方差
            scores = [r.get('score', 0) for r in candidate_records]
            if len(scores) > 1:
                return float(np.var(scores))
            else:
                return 0.1  # 默认方差
        except:
            return 0.1
    
    def _calculate_confidence_interval(self, 
                                     confidence: float, 
                                     variance: float) -> Tuple[float, float]:
        """计算置信区间"""
        try:
            std_dev = float(np.sqrt(variance))
            margin = 1.96 * std_dev  # 95%置信区间
            
            lower = max(0.0, confidence - margin)
            upper = min(1.0, confidence + margin)
            
            return (lower, upper)
        except:
            return (max(0.0, confidence - 0.1), min(1.0, confidence + 0.1))
    
    def _calculate_reliability_score(self, factors: ConfidenceFactors) -> float:
        """计算可靠性分数"""
        try:
            # 基于多个因子的一致性
            key_factors = [
                factors.vector_similarity,
                factors.entity_match_score,
                factors.semantic_coherence,
                factors.terminology_accuracy
            ]
            
            if key_factors:
                mean_factor = float(np.mean(key_factors))
                std_factor = float(np.std(key_factors))
                
                # 标准差越小，可靠性越高
                reliability = 1.0 - min(std_factor, 0.5) / 0.5
                return reliability
            else:
                return 0.5
        except:
            return 0.5
    
    def _calculate_calibration_score(self, confidence: float, factors: ConfidenceFactors) -> float:
        """计算校准分数"""
        try:
            # 简化的校准评估：检查置信度与各因子的一致性
            factor_values = [
                factors.vector_similarity,
                factors.semantic_coherence,
                factors.terminology_accuracy
            ]
            
            if factor_values:
                avg_factor = float(np.mean(factor_values))
                # 置信度与平均因子值越接近，校准越好
                calibration = 1.0 - abs(confidence - avg_factor)
                return max(calibration, 0.0)
            else:
                return 0.5
        except:
            return 0.5
    
    def get_confidence_level(self, confidence: float) -> str:
        """获取置信度等级"""
        if confidence >= self.confidence_thresholds['high_confidence']:
            return "高置信度"
        elif confidence >= self.confidence_thresholds['medium_confidence']:
            return "中等置信度"
        elif confidence >= self.confidence_thresholds['low_confidence']:
            return "低置信度"
        else:
            return "极低置信度"
    
    def should_reject_prediction(self, confidence: float) -> bool:
        """判断是否应该拒绝预测"""
        return confidence < self.confidence_thresholds['reject_threshold']
    
    def adjust_thresholds(self, new_thresholds: Dict[str, float]):
        """动态调整置信度阈值"""
        for threshold_name, value in new_thresholds.items():
            if threshold_name in self.confidence_thresholds:
                self.confidence_thresholds[threshold_name] = value
                logger.info(f"阈值调整: {threshold_name} = {value}")
    
    def get_confidence_explanation(self, 
                                 metrics: ConfidenceMetrics, 
                                 factors: ConfidenceFactors) -> Dict[str, Any]:
        """获取置信度评分的详细解释"""
        explanation = {
            'overall_confidence': metrics.overall_confidence,
            'confidence_level': self.get_confidence_level(metrics.overall_confidence),
            'confidence_interval': metrics.confidence_interval,
            'reliability_score': metrics.reliability_score,
            'should_reject': self.should_reject_prediction(metrics.overall_confidence),
            
            'factor_contributions': {},
            'top_contributing_factors': [],
            'improvement_suggestions': []
        }
        
        # 计算各因子贡献
        factor_dict = {
            '向量相似度': factors.vector_similarity,
            '层级增强': factors.hierarchy_boost,
            '实体匹配': factors.entity_match_score,
            '语义一致性': factors.semantic_coherence,
            '上下文一致性': factors.context_consistency,
            '术语准确性': factors.terminology_accuracy,
            '诊断复杂度': factors.diagnosis_complexity,
            '专业特异性': factors.professional_specificity,
            '临床相关性': factors.clinical_relevance,
            '数据质量': factors.data_quality,
            '模型不确定性': factors.model_uncertainty,
            '交叉验证': factors.cross_validation_score
        }
        
        weight_mapping = {
            '向量相似度': 'vector_similarity',
            '层级增强': 'hierarchy_boost',
            '实体匹配': 'entity_match_score',
            '语义一致性': 'semantic_coherence',
            '上下文一致性': 'context_consistency',
            '术语准确性': 'terminology_accuracy',
            '诊断复杂度': 'diagnosis_complexity',
            '专业特异性': 'professional_specificity',
            '临床相关性': 'clinical_relevance',
            '数据质量': 'data_quality',
            '模型不确定性': 'model_uncertainty',
            '交叉验证': 'cross_validation_score'
        }
        
        for factor_name_zh, factor_value in factor_dict.items():
            factor_name_en = weight_mapping[factor_name_zh]
            weight = self.factor_weights.get(factor_name_en, 0.0)
            contribution = factor_value * weight
            
            explanation['factor_contributions'][factor_name_zh] = {
                'value': factor_value,
                'weight': weight,
                'contribution': contribution
            }
        
        # 找出贡献最大的因子
        sorted_contributions = sorted(
            explanation['factor_contributions'].items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )
        explanation['top_contributing_factors'] = [
            f"{name}: {info['contribution']:.4f}" 
            for name, info in sorted_contributions[:3]
        ]
        
        # 提供改进建议
        if metrics.overall_confidence < 0.6:
            explanation['improvement_suggestions'] = [
                "考虑补充更多医学术语信息",
                "检查查询文本的完整性和准确性",
                "增加上下文信息以提高匹配精度"
            ]
        
        return explanation


def main():
    """测试函数"""
    print("=== 多维度置信度评分系统测试 ===")
    
    # 模拟服务
    class MockEmbeddingService:
        def encode_query(self, text):
            return np.random.rand(10)
    
    mock_embedding = MockEmbeddingService()
    confidence_service = MultiDimensionalConfidenceService(mock_embedding)
    
    # 测试数据
    query_text = "急性心肌梗死伴心律失常"
    candidate_records = [
        {
            'code': 'I21.9',
            'preferred_zh': '急性心肌梗死，未特指',
            'level': 3,
            'score': 0.95
        },
        {
            'code': 'I47.9', 
            'preferred_zh': '阵发性心动过速，未特指',
            'level': 3,
            'score': 0.72
        }
    ]
    
    # 计算综合置信度
    metrics, factors = confidence_service.calculate_comprehensive_confidence(
        query_text, candidate_records
    )
    
    print(f"查询文本: {query_text}")
    print(f"综合置信度: {metrics.overall_confidence:.4f}")
    print(f"置信度等级: {confidence_service.get_confidence_level(metrics.overall_confidence)}")
    print(f"置信区间: [{metrics.confidence_interval[0]:.3f}, {metrics.confidence_interval[1]:.3f}]")
    print(f"可靠性分数: {metrics.reliability_score:.4f}")
    
    # 获取详细解释
    explanation = confidence_service.get_confidence_explanation(metrics, factors)
    
    print(f"\n主要贡献因子:")
    for factor in explanation['top_contributing_factors']:
        print(f"  {factor}")
    
    if explanation['improvement_suggestions']:
        print(f"\n改进建议:")
        for suggestion in explanation['improvement_suggestions']:
            print(f"  - {suggestion}")


if __name__ == "__main__":
    main()