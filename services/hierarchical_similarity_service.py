#!/usr/bin/env python3
"""
层级相似度计算服务
基于ICD-10层级结构的多维度相似度计算和评分增强
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityFactors:
    """相似度计算因子"""
    vector_similarity: float = 0.0      # 向量相似度
    hierarchy_boost: float = 0.0        # 层级增强分数
    entity_match_score: float = 0.0     # 实体匹配分数  
    semantic_coherence: float = 0.0     # 语义一致性
    category_alignment: float = 0.0     # 类别对齐度
    context_relevance: float = 0.0      # 上下文相关性
    
    def __post_init__(self):
        """确保所有值都是Python原生float类型"""
        self.vector_similarity = float(self.vector_similarity)
        self.hierarchy_boost = float(self.hierarchy_boost)
        self.entity_match_score = float(self.entity_match_score)
        self.semantic_coherence = float(self.semantic_coherence)
        self.category_alignment = float(self.category_alignment)
        self.context_relevance = float(self.context_relevance)


@dataclass
class HierarchyInfo:
    """ICD-10层级信息"""
    level: int = 1
    parent_code: str = ""
    category_path: str = ""
    main_category: str = ""
    sub_category: str = ""
    semantic_keywords: List[str] = None
    
    def __post_init__(self):
        if self.semantic_keywords is None:
            self.semantic_keywords = []


class HierarchicalSimilarityService:
    """层级相似度计算服务"""
    
    def __init__(self, embedding_service=None, ner_service=None):
        """
        初始化层级相似度服务
        
        Args:
            embedding_service: 嵌入服务实例
            ner_service: 命名实体识别服务实例 
        """
        self.embedding_service = embedding_service
        self.ner_service = ner_service
        
        # 层级权重配置（基于现有系统）
        self.level_weights = {
            1: 1.2,  # 主类别 - 权重更高
            2: 1.0,  # 亚类别 - 标准权重  
            3: 0.8   # 细分类 - 权重较低
        }
        
        # 相似度因子权重配置（优化版）
        self.factor_weights = {
            'vector_similarity': 0.50,      # 基础向量相似度 - 提高权重
            'hierarchy_boost': 0.20,        # 层级增强分数 - 适中权重
            'entity_match_score': 0.15,     # 实体匹配分数 - 适中权重
            'semantic_coherence': 0.08,     # 语义一致性 - 降低权重
            'category_alignment': 0.04,     # 类别对齐度 - 降低权重
            'context_relevance': 0.03       # 上下文相关性 - 降低权重
        }
        
        # ICD-10主类别映射（用于语义增强）
        self.main_categories = self._load_main_categories()
        
        # 相似度计算缓存
        self.similarity_cache = {}
        
        logger.info(f"层级相似度服务初始化完成，权重配置: {self.factor_weights}")
    
    def _load_main_categories(self) -> Dict[str, Dict[str, Any]]:
        """加载ICD-10主类别信息"""
        return {
            'A': {
                'name': '某些传染病和寄生虫病',
                'keywords': ['感染', '传染', '病毒', '细菌', '寄生虫', '真菌'],
                'semantic_weight': 1.1
            },
            'B': {
                'name': '肿瘤', 
                'keywords': ['癌', '瘤', '肿瘤', '恶性', '良性', '转移'],
                'semantic_weight': 1.2
            },
            'C': {
                'name': '血液及造血器官疾病',
                'keywords': ['血液', '贫血', '白血病', '出血', '凝血'],
                'semantic_weight': 1.0
            },
            'E': {
                'name': '内分泌、营养和代谢疾病',
                'keywords': ['糖尿病', '甲状腺', '代谢', '内分泌', '营养'],
                'semantic_weight': 1.1
            },
            'I': {
                'name': '循环系统疾病',
                'keywords': ['心脏', '血管', '高血压', '心肌', '循环'],
                'semantic_weight': 1.2
            },
            'J': {
                'name': '呼吸系统疾病',
                'keywords': ['肺', '呼吸', '咳嗽', '气管', '支气管'],
                'semantic_weight': 1.1
            },
            'K': {
                'name': '消化系统疾病',
                'keywords': ['胃', '肠', '肝', '消化', '腹泻'],
                'semantic_weight': 1.0
            },
            'N': {
                'name': '泌尿生殖系统疾病',
                'keywords': ['肾', '膀胱', '泌尿', '生殖', '尿'],
                'semantic_weight': 1.0
            },
            'S': {
                'name': '损伤、中毒和外因的某些其他后果',
                'keywords': ['损伤', '外伤', '骨折', '中毒', '烧伤'],
                'semantic_weight': 0.9
            }
        }
    
    def calculate_enhanced_similarity(self, 
                                    query_text: str,
                                    query_entities: Dict[str, List[Dict]], 
                                    candidate_record: Dict[str, Any]) -> Tuple[float, SimilarityFactors]:
        """
        计算增强的层级相似度
        
        Args:
            query_text: 查询文本
            query_entities: 查询文本的实体信息
            candidate_record: 候选ICD记录
            
        Returns:
            增强相似度分数和详细因子
        """
        factors = SimilarityFactors()
        
        try:
            # 1. 基础向量相似度
            factors.vector_similarity = self._calculate_vector_similarity(
                query_text, candidate_record
            )
            
            # 2. 层级增强分数
            factors.hierarchy_boost = self._calculate_hierarchy_boost(
                query_text, query_entities, candidate_record
            )
            
            # 3. 实体匹配分数
            factors.entity_match_score = self._calculate_entity_match_score(
                query_entities, candidate_record
            )
            
            # 4. 语义一致性
            factors.semantic_coherence = self._calculate_semantic_coherence(
                query_text, candidate_record
            )
            
            # 5. 类别对齐度
            factors.category_alignment = self._calculate_category_alignment(
                query_entities, candidate_record
            )
            
            # 6. 上下文相关性
            factors.context_relevance = self._calculate_context_relevance(
                query_text, candidate_record
            )
            
            # 计算加权总分
            enhanced_score = self._calculate_weighted_score(factors)
            
            logger.debug(f"增强相似度计算完成: {candidate_record.get('code', 'unknown')} = {enhanced_score:.4f}")
            
            return float(enhanced_score), factors
            
        except Exception as e:
            logger.error(f"增强相似度计算失败: {e}")
            # 返回基础相似度作为回退
            base_score = candidate_record.get('score', 0.0)
            return float(base_score), factors
    
    def _calculate_vector_similarity(self, query_text: str, candidate_record: Dict[str, Any]) -> float:
        """计算基础向量相似度"""
        try:
            if not self.embedding_service:
                return candidate_record.get('score', 0.0)
            
            # 使用已有的向量相似度分数，或重新计算
            if 'score' in candidate_record:
                return float(candidate_record['score'])
            
            # 如果需要重新计算
            query_vector = self.embedding_service.encode_query(query_text)
            candidate_text = candidate_record.get('semantic_text', candidate_record.get('preferred_zh', ''))
            candidate_vector = self.embedding_service.encode_query(candidate_text)
            
            similarity = cosine_similarity([query_vector], [candidate_vector])[0][0]
            return float(max(similarity, 0.0))
            
        except Exception as e:
            logger.warning(f"向量相似度计算失败: {e}")
            return candidate_record.get('score', 0.0)
    
    def _calculate_hierarchy_boost(self, 
                                 query_text: str,
                                 query_entities: Dict[str, List[Dict]],
                                 candidate_record: Dict[str, Any]) -> float:
        """计算层级增强分数"""
        boost_score = 0.0
        
        try:
            # 获取候选记录的层级信息
            level = candidate_record.get('level', 1)
            code = candidate_record.get('code', '')
            parent_code = candidate_record.get('parent_code', '')
            category_path = candidate_record.get('category_path', '')
            
            # 基于层级的基础增强
            level_boost = self._get_level_boost_factor(level)
            boost_score += level_boost * 0.3
            
            # 主类别语义匹配增强
            main_category_code = code[0] if code else ''
            if main_category_code in self.main_categories:
                category_info = self.main_categories[main_category_code]
                category_boost = self._calculate_category_semantic_boost(
                    query_text, query_entities, category_info
                )
                boost_score += category_boost * 0.4
            
            # 父子关系增强
            if parent_code:
                parent_boost = self._calculate_parent_child_boost(
                    query_entities, code, parent_code
                )
                boost_score += parent_boost * 0.3
            
            return float(min(boost_score, 0.3))  # 限制最大增强分数，避免过度增强
            
        except Exception as e:
            logger.warning(f"层级增强分数计算失败: {e}")
            return 0.0
    
    def _get_level_boost_factor(self, level: int) -> float:
        """获取层级增强因子"""
        # 不同层级的增强策略
        level_boost_factors = {
            1: 0.15,  # 主类别 - 适中增强
            2: 0.20,  # 亚类别 - 最大增强（平衡点）
            3: 0.10   # 细分类 - 较小增强  
        }
        return float(level_boost_factors.get(level, 0.10))
    
    def _calculate_category_semantic_boost(self,
                                         query_text: str,
                                         query_entities: Dict[str, List[Dict]], 
                                         category_info: Dict[str, Any]) -> float:
        """计算类别语义增强分数"""
        boost = 0.0
        
        try:
            category_keywords = category_info.get('keywords', [])
            semantic_weight = category_info.get('semantic_weight', 1.0)
            
            # 检查查询文本中的类别关键词匹配
            query_lower = query_text.lower()
            matched_keywords = 0
            for keyword in category_keywords:
                if keyword in query_lower:
                    matched_keywords += 1
            
            if matched_keywords > 0:
                keyword_boost = (matched_keywords / len(category_keywords)) * 0.3
                boost += keyword_boost * semantic_weight
            
            # 检查疾病实体与类别的匹配度
            disease_entities = query_entities.get('disease', [])
            for entity in disease_entities:
                entity_text = entity.get('text', '').lower()
                entity_matches = sum(1 for kw in category_keywords if kw in entity_text)
                if entity_matches > 0:
                    entity_boost = (entity_matches / len(category_keywords)) * 0.2
                    boost += entity_boost * entity.get('confidence', 0.5)
            
            return float(min(boost, 0.4))
            
        except Exception as e:
            logger.warning(f"类别语义增强计算失败: {e}")
            return 0.0
    
    def _calculate_parent_child_boost(self,
                                    query_entities: Dict[str, List[Dict]],
                                    code: str,
                                    parent_code: str) -> float:
        """计算父子关系增强分数"""
        # 简化的父子关系增强
        # 实际实现中可以基于ICD-10的具体层级关系进行更复杂的计算
        if len(code) > len(parent_code) and code.startswith(parent_code):
            return 0.1  # 确实是父子关系
        return 0.0
    
    def _calculate_entity_match_score(self,
                                    query_entities: Dict[str, List[Dict]],
                                    candidate_record: Dict[str, Any]) -> float:
        """计算实体匹配分数"""
        match_score = 0.0
        
        try:
            candidate_text = candidate_record.get('preferred_zh', '').lower()
            semantic_text = candidate_record.get('semantic_text', '').lower()
            combined_text = f"{candidate_text} {semantic_text}"
            
            # 疾病实体匹配（权重最高）
            disease_entities = query_entities.get('disease', [])
            for entity in disease_entities:
                entity_text = entity.get('text', '').lower()
                confidence = entity.get('confidence', 0.5)
                
                if entity_text in combined_text:
                    match_score += confidence * 0.4
                elif any(word in combined_text for word in entity_text.split()):
                    match_score += confidence * 0.2
            
            # 症状实体匹配
            symptom_entities = query_entities.get('symptom', [])
            for entity in symptom_entities:
                entity_text = entity.get('text', '').lower()
                confidence = entity.get('confidence', 0.5)
                
                if entity_text in combined_text:
                    match_score += confidence * 0.2
            
            # 解剖部位匹配
            anatomy_entities = query_entities.get('anatomy', [])
            for entity in anatomy_entities:
                entity_text = entity.get('text', '').lower()
                confidence = entity.get('confidence', 0.5)
                
                if entity_text in combined_text:
                    match_score += confidence * 0.1
            
            return float(min(match_score, 1.0))
            
        except Exception as e:
            logger.warning(f"实体匹配分数计算失败: {e}")
            return 0.0
    
    def _calculate_semantic_coherence(self,
                                    query_text: str,
                                    candidate_record: Dict[str, Any]) -> float:
        """计算语义一致性"""
        try:
            if not self.embedding_service:
                return 0.5
            
            # 使用语义文本进行更精确的一致性计算
            candidate_semantic = candidate_record.get('semantic_text', '')
            if not candidate_semantic:
                return 0.3
            
            # 计算语义向量的一致性
            query_vector = self.embedding_service.encode_query(query_text)
            semantic_vector = self.embedding_service.encode_query(candidate_semantic)
            
            coherence = cosine_similarity([query_vector], [semantic_vector])[0][0]
            return max(coherence, 0.0)
            
        except Exception as e:
            logger.warning(f"语义一致性计算失败: {e}")
            return 0.5
    
    def _calculate_category_alignment(self,
                                    query_entities: Dict[str, List[Dict]],
                                    candidate_record: Dict[str, Any]) -> float:
        """计算类别对齐度"""
        try:
            code = candidate_record.get('code', '')
            if not code:
                return 0.0
            
            main_category = code[0]
            if main_category not in self.main_categories:
                return 0.0
            
            category_info = self.main_categories[main_category]
            category_keywords = category_info.get('keywords', [])
            
            # 检查查询实体与类别的对齐程度
            alignment_score = 0.0
            total_entities = 0
            
            for entity_type, entities in query_entities.items():
                for entity in entities:
                    total_entities += 1
                    entity_text = entity.get('text', '').lower()
                    
                    # 检查实体是否与类别关键词对齐
                    for keyword in category_keywords:
                        if keyword in entity_text:
                            alignment_score += entity.get('confidence', 0.5)
                            break
            
            return float(alignment_score / total_entities) if total_entities > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"类别对齐度计算失败: {e}")
            return 0.0
    
    def _calculate_context_relevance(self,
                                   query_text: str,
                                   candidate_record: Dict[str, Any]) -> float:
        """计算上下文相关性"""
        try:
            # 简化的上下文相关性计算
            # 基于文本长度、复杂度等因素
            
            query_length = len(query_text)
            candidate_text = candidate_record.get('preferred_zh', '')
            candidate_length = len(candidate_text)
            
            # 长度相似性
            length_similarity = 1.0 - abs(query_length - candidate_length) / max(query_length, candidate_length, 1)
            
            # 复杂度匹配（基于字符多样性）
            query_chars = set(query_text)
            candidate_chars = set(candidate_text)
            char_overlap = len(query_chars & candidate_chars) / len(query_chars | candidate_chars) if (query_chars | candidate_chars) else 0
            
            relevance = (length_similarity * 0.3 + char_overlap * 0.7)
            return max(relevance, 0.0)
            
        except Exception as e:
            logger.warning(f"上下文相关性计算失败: {e}")
            return 0.5
    
    def _calculate_weighted_score(self, factors: SimilarityFactors) -> float:
        """计算加权总分（采用加法增强模式）"""
        try:
            # 基础向量相似度作为起点
            base_score = factors.vector_similarity
            
            # 各种增强因子的加法贡献
            enhancements = 0.0
            
            # 层级增强（直接加到基础分数上）
            enhancements += factors.hierarchy_boost * self.factor_weights['hierarchy_boost'] / 0.2  # 归一化到权重
            
            # 实体匹配增强
            enhancements += factors.entity_match_score * self.factor_weights['entity_match_score'] / 0.15
            
            # 语义一致性增强（如果高于基础分数）
            if factors.semantic_coherence > base_score:
                semantic_boost = (factors.semantic_coherence - base_score) * self.factor_weights['semantic_coherence'] / 0.08
                enhancements += semantic_boost
            
            # 类别对齐增强
            enhancements += factors.category_alignment * self.factor_weights['category_alignment'] / 0.04
            
            # 上下文相关性增强
            enhancements += factors.context_relevance * self.factor_weights['context_relevance'] / 0.03
            
            # 最终分数 = 基础分数 + 增强分数
            final_score = base_score + enhancements
            
            return float(min(final_score, 1.8))  # 允许显著增强，但设置合理上限
            
        except Exception as e:
            logger.error(f"加权分数计算失败: {e}")
            return float(factors.vector_similarity)  # 回退到基础向量相似度
    
    def batch_calculate_similarities(self,
                                   query_text: str,
                                   query_entities: Dict[str, List[Dict]],
                                   candidate_records: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float, SimilarityFactors]]:
        """
        批量计算增强相似度
        
        Args:
            query_text: 查询文本
            query_entities: 查询实体
            candidate_records: 候选记录列表
            
        Returns:
            增强后的候选记录列表，包含分数和因子信息
        """
        enhanced_results = []
        
        logger.info(f"开始批量计算 {len(candidate_records)} 个候选记录的增强相似度")
        
        for record in candidate_records:
            try:
                enhanced_score, factors = self.calculate_enhanced_similarity(
                    query_text, query_entities, record
                )
                
                # 更新记录的分数
                enhanced_record = record.copy()
                enhanced_record['enhanced_score'] = enhanced_score
                enhanced_record['original_score'] = record.get('score', 0.0)
                enhanced_record['similarity_factors'] = factors
                
                enhanced_results.append((enhanced_record, enhanced_score, factors))
                
            except Exception as e:
                logger.error(f"记录 {record.get('code', 'unknown')} 的相似度计算失败: {e}")
                # 使用原始分数作为回退
                original_score = record.get('score', 0.0)
                enhanced_results.append((record, original_score, SimilarityFactors()))
        
        # 按增强分数排序
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"批量相似度计算完成，平均增强分数: {float(np.mean([r[1] for r in enhanced_results])):.4f}")
        
        return enhanced_results
    
    def get_similarity_explanation(self, factors: SimilarityFactors) -> Dict[str, Any]:
        """获取相似度计算的详细解释"""
        explanation = {
            'total_score': self._calculate_weighted_score(factors),
            'factors': {
                'vector_similarity': {
                    'score': factors.vector_similarity,
                    'weight': self.factor_weights['vector_similarity'],
                    'contribution': factors.vector_similarity * self.factor_weights['vector_similarity'],
                    'description': '基础向量相似度'
                },
                'hierarchy_boost': {
                    'score': factors.hierarchy_boost,
                    'weight': self.factor_weights['hierarchy_boost'],
                    'contribution': factors.hierarchy_boost * self.factor_weights['hierarchy_boost'],
                    'description': 'ICD-10层级增强分数'
                },
                'entity_match_score': {
                    'score': factors.entity_match_score,
                    'weight': self.factor_weights['entity_match_score'],
                    'contribution': factors.entity_match_score * self.factor_weights['entity_match_score'],
                    'description': '医学实体匹配分数'
                },
                'semantic_coherence': {
                    'score': factors.semantic_coherence,
                    'weight': self.factor_weights['semantic_coherence'],
                    'contribution': factors.semantic_coherence * self.factor_weights['semantic_coherence'],
                    'description': '语义一致性分数'
                },
                'category_alignment': {
                    'score': factors.category_alignment,
                    'weight': self.factor_weights['category_alignment'],
                    'contribution': factors.category_alignment * self.factor_weights['category_alignment'],
                    'description': 'ICD类别对齐分数'
                },
                'context_relevance': {
                    'score': factors.context_relevance,
                    'weight': self.factor_weights['context_relevance'],
                    'contribution': factors.context_relevance * self.factor_weights['context_relevance'],
                    'description': '上下文相关性分数'
                }
            }
        }
        
        return explanation
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新权重配置"""
        for factor, weight in new_weights.items():
            if factor in self.factor_weights:
                self.factor_weights[factor] = weight
                logger.info(f"权重更新: {factor} = {weight}")
        
        # 确保权重和为1
        total_weight = sum(self.factor_weights.values())
        if total_weight != 1.0:
            logger.warning(f"权重总和不为1.0: {total_weight}，自动归一化")
            for factor in self.factor_weights:
                self.factor_weights[factor] /= total_weight


def main():
    """测试函数"""
    print("=== 层级相似度计算服务测试 ===")
    
    # 模拟服务
    class MockEmbeddingService:
        def encode_query(self, text):
            import numpy as np
            # 简单的基于长度和字符的向量模拟
            vector = np.random.rand(10)
            vector[0] = len(text) / 50.0  # 长度特征
            vector[1] = len(set(text)) / 20.0  # 字符多样性
            return vector
    
    mock_embedding = MockEmbeddingService()
    similarity_service = HierarchicalSimilarityService(mock_embedding)
    
    # 测试数据
    query_text = "急性心肌梗死伴心律失常"
    query_entities = {
        'disease': [
            {'text': '急性心肌梗死', 'confidence': 0.95, 'start': 0, 'end': 6},
            {'text': '心律失常', 'confidence': 0.88, 'start': 7, 'end': 11}
        ],
        'anatomy': [
            {'text': '心肌', 'confidence': 0.85, 'start': 2, 'end': 4}
        ]
    }
    
    candidate_records = [
        {
            'code': 'I21.9',
            'preferred_zh': '急性心肌梗死，未特指',
            'level': 3,
            'parent_code': 'I21',
            'category_path': 'I > I21 > I21.9',
            'semantic_text': '急性心肌梗死 | 循环系统疾病 | ICD-10: I21.9',
            'score': 0.85
        },
        {
            'code': 'I47.9',
            'preferred_zh': '阵发性心动过速，未特指',
            'level': 3,
            'parent_code': 'I47',
            'category_path': 'I > I47 > I47.9',
            'semantic_text': '阵发性心动过速 | 心律失常 | ICD-10: I47.9',
            'score': 0.72
        },
        {
            'code': 'I25.9',
            'preferred_zh': '慢性缺血性心脏病，未特指',
            'level': 3,
            'parent_code': 'I25',
            'category_path': 'I > I25 > I25.9',
            'semantic_text': '慢性缺血性心脏病 | 循环系统疾病 | ICD-10: I25.9',
            'score': 0.68
        }
    ]
    
    # 批量计算增强相似度
    enhanced_results = similarity_service.batch_calculate_similarities(
        query_text, query_entities, candidate_records
    )
    
    print(f"\n查询文本: {query_text}")
    print(f"查询实体: {len(query_entities.get('disease', []))} 个疾病, {len(query_entities.get('anatomy', []))} 个解剖部位")
    
    print(f"\n增强相似度结果 (Top {len(enhanced_results)}):")
    for i, (record, enhanced_score, factors) in enumerate(enhanced_results, 1):
        original_score = record.get('original_score', 0.0)
        improvement = enhanced_score - original_score
        
        print(f"\n{i}. {record['code']}: {record['preferred_zh']}")
        print(f"   原始分数: {original_score:.4f}")
        print(f"   增强分数: {enhanced_score:.4f} ({improvement:+.4f})")
        print(f"   层级: Level {record['level']}")
        print(f"   因子贡献:")
        print(f"     向量相似度: {factors.vector_similarity:.3f}")
        print(f"     层级增强: {factors.hierarchy_boost:.3f}")
        print(f"     实体匹配: {factors.entity_match_score:.3f}")
        print(f"     语义一致性: {factors.semantic_coherence:.3f}")
    
    # 获取详细解释
    if enhanced_results:
        top_result = enhanced_results[0]
        explanation = similarity_service.get_similarity_explanation(top_result[2])
        
        print(f"\n📊 最佳匹配的详细分析:")
        print(f"总分: {explanation['total_score']:.4f}")
        for factor_name, factor_info in explanation['factors'].items():
            print(f"  {factor_info['description']}: {factor_info['score']:.3f} × {factor_info['weight']:.2f} = {factor_info['contribution']:.4f}")


if __name__ == "__main__":
    main()