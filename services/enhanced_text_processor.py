#!/usr/bin/env python3
"""
增强的文本处理器
整合医学实体识别和语义边界检测功能
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from services.medical_ner_service import MedicalNERService
from services.semantic_boundary_service import SemanticBoundaryDetector


class EnhancedTextProcessor:
    """增强的文本处理器"""
    
    def __init__(self, embedding_service=None, use_model_ner=None):
        """
        初始化增强文本处理器
        
        Args:
            embedding_service: 嵌入服务实例
            use_model_ner: 是否使用大模型NER，默认从环境变量读取
        """
        # 初始化医学NER服务（优先使用大模型）
        self.ner_service = MedicalNERService(use_model=use_model_ner)
        self.boundary_detector = SemanticBoundaryDetector(embedding_service)
        self.embedding_service = embedding_service
        
        # 处理配置
        self.config = {
            'min_diagnosis_length': 2,
            'max_diagnosis_length': 50,
            'min_entity_confidence': 0.6,
            'use_semantic_boundary': True,
            'fallback_to_simple_split': True
        }
    
    def extract_diagnoses_enhanced(self, text: str, filter_drugs: bool = True) -> List[Dict[str, Any]]:
        """
        增强的诊断提取方法
        
        Args:
            text: 输入的医疗文本
            filter_drugs: 是否过滤非诊断实体（药品、设备、科室等），默认True
            
        Returns:
            增强的诊断结果列表，包含更多元数据
        """
        if not text or not text.strip():
            return []
        
        logger.info(f"开始增强诊断提取: {text}")
        
        try:
            # 1. 医学实体识别（带过滤）
            entities = self.ner_service.extract_medical_entities(text, filter_drugs=filter_drugs)
            
            # 2. 语义边界检测
            if self.config['use_semantic_boundary'] and self.embedding_service:
                boundaries = self.boundary_detector.detect_diagnosis_boundaries(text)
                boundary_confidences = self.boundary_detector.get_boundary_confidence(boundaries)
            else:
                # 回退到简单分割
                boundaries = self._simple_boundary_detection(text)
                boundary_confidences = [0.5] * len(boundaries)
            
            # 3. 融合实体和边界信息  
            enhanced_diagnoses = self._fuse_entity_boundary_info(
                text, entities, boundaries, boundary_confidences
            )
            
            # 4. 质量过滤和排序
            filtered_diagnoses = self._filter_and_rank_diagnoses(enhanced_diagnoses)
            
            logger.info(f"增强提取完成，得到 {len(filtered_diagnoses)} 个高质量诊断")
            
            # 记录提取的诊断详情
            for i, diagnosis in enumerate(filtered_diagnoses):
                entity_count = diagnosis['metadata'].get('entity_count', 0)
                logger.debug(f"  诊断 {i+1}: {diagnosis['text']} (置信度: {diagnosis['diagnosis_confidence']:.3f}, 实体数: {entity_count})")
            
            return filtered_diagnoses
            
        except Exception as e:
            logger.error(f"增强诊断提取失败: {e}")
            # 回退到简单方法
            return self._fallback_extraction(text)
    
    def _simple_boundary_detection(self, text: str) -> List[Tuple[int, int, str]]:
        """简单的边界检测（回退方案）"""
        import re
        
        # 优化的分隔符顺序，优先使用分号
        separators = [
            r'[；;]',        # 分号 - 医学文本中最强的诊断分隔符
            r'[，,](?![^（]*）)', # 逗号 - 但排除括号内的逗号
            r'[+＋]',        # 加号
        ]
        
        for separator_pattern in separators:
            parts = re.split(separator_pattern, text)
            if len(parts) > 1:
                boundaries = []
                pos = 0
                for part in parts:
                    part = part.strip()
                    if part and len(part) >= self.config['min_diagnosis_length']:
                        start = text.find(part, pos)
                        if start != -1:
                            end = start + len(part)
                            boundaries.append((start, end, part))
                            pos = end
                
                if boundaries and len(boundaries) > 1:  # 确保真正分割了
                    logger.debug(f"简单边界检测使用分隔符 '{separator_pattern}' 得到 {len(boundaries)} 个边界")
                    return boundaries
        
        # 尝试基于医学关键词的智能分割
        medical_keywords = ['既往', '病史', '术后', '治疗', '保守', '规律', '控制']
        for keyword in medical_keywords:
            if keyword in text:
                parts = text.split(keyword)
                if len(parts) > 1:
                    boundaries = []
                    pos = 0
                    for i, part in enumerate(parts):
                        if i == 0:
                            segment = part.strip()
                        else:
                            segment = (keyword + part).strip()
                        
                        if segment and len(segment) >= self.config['min_diagnosis_length']:
                            start = text.find(segment, pos)
                            if start != -1:
                                end = start + len(segment)
                                boundaries.append((start, end, segment))
                                pos = end
                    
                    if boundaries and len(boundaries) > 1:
                        logger.debug(f"基于关键词 '{keyword}' 分割得到 {len(boundaries)} 个边界")
                        return boundaries
        
        # 如果没有找到分割点，返回整个文本
        logger.debug("未找到合适的分割点，返回整个文本")
        return [(0, len(text), text.strip())]
    
    def _fuse_entity_boundary_info(self, 
                                  text: str,
                                  entities: Dict[str, List[Dict]], 
                                  boundaries: List[Tuple[int, int, str]],
                                  boundary_confidences: List[float]) -> List[Dict[str, Any]]:
        """融合实体和边界信息"""
        enhanced_diagnoses = []
        
        for i, (start, end, boundary_text) in enumerate(boundaries):
            boundary_confidence = boundary_confidences[i] if i < len(boundary_confidences) else 0.5
            
            # 先尝试进一步分割这个边界段落（如果包含多个疾病实体）
            boundary_sub_diagnoses = self._extract_sub_diagnoses_from_boundary(boundary_text, entities, start, end)
            
            for sub_diagnosis in boundary_sub_diagnoses:
                # 为每个子诊断创建增强诊断信息
                diagnosis_info = {
                    'text': sub_diagnosis['text'].strip(),
                    'start_pos': sub_diagnosis['start'],
                    'end_pos': sub_diagnosis['end'],
                    'boundary_confidence': boundary_confidence,
                    'entities': sub_diagnosis['entities'],
                    'entity_density': 0.0,
                    'primary_entity_types': [],
                    'diagnosis_confidence': 0.0,
                    'metadata': {
                        'length': len(sub_diagnosis['text'].strip()),
                        'has_disease_entity': False,
                        'has_symptom_entity': False,
                        'entity_count': 0,
                        'ner_method': self.ner_service.get_model_info().get('extraction_method', 'unknown')
                    }
                }
                
                # 计算实体密度和主要实体类型
                boundary_entities = diagnosis_info['entities']
                total_entities = sum(len(boundary_entities.get(entity_type, [])) for entity_type in boundary_entities)
                
                if total_entities > 0:
                    diagnosis_info['entity_density'] = total_entities / len(sub_diagnosis['text']) if sub_diagnosis['text'] else 0
                    diagnosis_info['metadata']['entity_count'] = total_entities
                    
                    # 确定主要实体类型
                    for entity_type, entity_list in boundary_entities.items():
                        if entity_list:
                            diagnosis_info['primary_entity_types'].append(entity_type)
                            if entity_type == 'disease':
                                diagnosis_info['metadata']['has_disease_entity'] = True
                            elif entity_type == 'symptom':
                                diagnosis_info['metadata']['has_symptom_entity'] = True
                
                # 计算综合诊断置信度
                diagnosis_info['diagnosis_confidence'] = self._calculate_diagnosis_confidence(diagnosis_info)
                
                enhanced_diagnoses.append(diagnosis_info)
        
        return enhanced_diagnoses
    
    def _extract_sub_diagnoses_from_boundary(self, boundary_text: str, entities: Dict[str, List[Dict]], boundary_start: int, boundary_end: int) -> List[Dict]:
        """从边界段落中提取子诊断"""
        # 首先获取边界内的疾病实体
        disease_entities = []
        for entity in entities.get('disease', []):
            entity_start = entity.get('start', 0)
            entity_end = entity.get('end', 0)
            if boundary_start <= entity_start < boundary_end:
                disease_entities.append(entity)
        
        # 如果只有0-1个疾病实体，直接返回整个边界
        if len(disease_entities) <= 1:
            return [{
                'text': boundary_text,
                'start': boundary_start,
                'end': boundary_end,
                'entities': self._extract_entities_in_boundary(entities, boundary_start, boundary_end)
            }]
        
        # 如果有多个疾病实体，尝试基于疾病实体分割
        sub_diagnoses = []
        sorted_diseases = sorted(disease_entities, key=lambda x: x.get('start', 0))
        
        prev_entity_end = boundary_start
        for i, disease_entity in enumerate(sorted_diseases):
            entity_start = disease_entity.get('start', boundary_start)
            entity_end = disease_entity.get('end', entity_start + len(disease_entity.get('text', '')))
            
            # 找到以这个疾病实体为中心的诊断片段
            if i < len(sorted_diseases) - 1:
                next_entity_start = sorted_diseases[i + 1].get('start', boundary_end)
                segment_end = min(next_entity_start, boundary_end)
            else:
                segment_end = boundary_end
            
            # 构建子诊断文本
            segment_start = max(prev_entity_end, entity_start - 10)  # 包含一些前文
            segment_text = boundary_text[segment_start - boundary_start:segment_end - boundary_start].strip()
            
            if segment_text and len(segment_text) >= 2:
                sub_diagnoses.append({
                    'text': segment_text,
                    'start': segment_start,
                    'end': segment_end,
                    'entities': self._extract_entities_in_boundary(entities, segment_start, segment_end)
                })
            
            prev_entity_end = entity_end
        
        return sub_diagnoses if sub_diagnoses else [{
            'text': boundary_text,
            'start': boundary_start,
            'end': boundary_end,
            'entities': self._extract_entities_in_boundary(entities, boundary_start, boundary_end)
        }]
    
    def _extract_entities_in_boundary(self, entities: Dict[str, List[Dict]], start: int, end: int) -> Dict[str, List[Dict]]:
        """提取边界内的实体"""
        boundary_entities = {}
        
        for entity_type, entity_list in entities.items():
            boundary_entities[entity_type] = []
            
            for entity in entity_list:
                entity_start = entity.get('start', 0)
                entity_end = entity.get('end', 0)
                
                # 检查实体是否在边界内
                if (entity_start >= start and entity_end <= end) or \
                   (entity_start < end and entity_end > start):  # 重叠情况
                    boundary_entities[entity_type].append(entity)
        
        return boundary_entities
    
    def _calculate_diagnosis_confidence(self, diagnosis_info: Dict[str, Any]) -> float:
        """计算诊断置信度"""
        confidence = 0.3  # 基础置信度
        
        # 边界置信度权重
        confidence += diagnosis_info['boundary_confidence'] * 0.3
        
        # 实体质量权重
        entities = diagnosis_info['entities']
        entity_scores = []
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_confidence = entity.get('confidence', 0.5)
                
                # 疾病实体权重更高
                if entity_type == 'disease':
                    entity_scores.append(entity_confidence * 1.2)
                elif entity_type == 'symptom':
                    entity_scores.append(entity_confidence * 0.8)
                else:
                    entity_scores.append(entity_confidence * 0.6)
        
        if entity_scores:
            avg_entity_score = sum(entity_scores) / len(entity_scores)
            confidence += avg_entity_score * 0.4
        
        # 长度合理性
        text_length = len(diagnosis_info['text'])
        if 4 <= text_length <= 20:
            confidence += 0.1
        elif text_length < 2:
            confidence -= 0.2
        
        # 实体密度
        if diagnosis_info['entity_density'] > 0.1:  # 每10个字符至少1个实体
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _filter_and_rank_diagnoses(self, diagnoses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤和排序诊断结果"""
        # 1. 长度过滤
        filtered = [
            d for d in diagnoses 
            if self.config['min_diagnosis_length'] <= len(d['text']) <= self.config['max_diagnosis_length']
        ]
        
        # 2. 置信度过滤（使用较低的阈值，因为这是增强的置信度）
        min_confidence = max(0.4, self.config.get('min_diagnosis_confidence', 0.4))
        filtered = [d for d in filtered if d['diagnosis_confidence'] >= min_confidence]
        
        # 3. 去重（基于文本相似性）
        deduplicated = self._deduplicate_diagnoses(filtered)
        
        # 4. 按置信度排序
        ranked = sorted(deduplicated, key=lambda x: x['diagnosis_confidence'], reverse=True)
        
        return ranked
    
    def _deduplicate_diagnoses(self, diagnoses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重诊断结果"""
        if len(diagnoses) <= 1:
            return diagnoses
        
        deduplicated = []
        
        for diagnosis in diagnoses:
            is_duplicate = False
            
            for existing in deduplicated:
                # 简单的文本相似性检查
                similarity = self._text_similarity(diagnosis['text'], existing['text'])
                
                if similarity > 0.8:  # 80%相似度认为是重复
                    # 保留置信度更高的
                    if diagnosis['diagnosis_confidence'] > existing['diagnosis_confidence']:
                        deduplicated.remove(existing)
                        deduplicated.append(diagnosis)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(diagnosis)
        
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单版本）"""
        if not text1 or not text2:
            return 0.0
        
        # 使用字符集合计算Jaccard相似度
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _fallback_extraction(self, text: str) -> List[Dict[str, Any]]:
        """回退提取方法"""
        logger.warning("使用回退提取方法")
        
        # 使用简单分割
        boundaries = self._simple_boundary_detection(text)
        
        fallback_diagnoses = []
        for start, end, boundary_text in boundaries:
            diagnosis_info = {
                'text': boundary_text.strip(),
                'start_pos': start,
                'end_pos': end,
                'boundary_confidence': 0.5,
                'entities': {},
                'entity_density': 0.0,
                'primary_entity_types': [],
                'diagnosis_confidence': 0.5,
                'metadata': {
                    'length': len(boundary_text.strip()),
                    'has_disease_entity': False,
                    'has_symptom_entity': False,
                    'entity_count': 0,
                    'is_fallback': True
                }
            }
            fallback_diagnoses.append(diagnosis_info)
        
        return fallback_diagnoses
    
    def extract_diagnoses_simple(self, text: str) -> List[str]:
        """
        简单的诊断提取（向后兼容）
        
        Args:
            text: 输入文本
            
        Returns:
            诊断文本列表
        """
        enhanced_results = self.extract_diagnoses_enhanced(text)
        return [result['text'] for result in enhanced_results]
    
    def get_processing_summary(self, text: str) -> Dict[str, Any]:
        """
        获取处理摘要
        
        Args:
            text: 输入文本
            
        Returns:
            处理摘要信息
        """
        enhanced_results = self.extract_diagnoses_enhanced(text)
        
        summary = {
            'original_text': text,
            'total_diagnoses': len(enhanced_results),
            'avg_confidence': sum(r['diagnosis_confidence'] for r in enhanced_results) / len(enhanced_results) if enhanced_results else 0,
            'entity_types_found': set(),
            'high_confidence_count': 0,
            'processing_method': 'enhanced' if self.config['use_semantic_boundary'] and self.embedding_service else 'simple',
            'ner_info': self.ner_service.get_model_info()
        }
        
        for result in enhanced_results:
            summary['entity_types_found'].update(result['primary_entity_types'])
            if result['diagnosis_confidence'] > 0.7:
                summary['high_confidence_count'] += 1
        
        summary['entity_types_found'] = list(summary['entity_types_found'])
        
        return summary


def main():
    """测试函数"""
    # 模拟嵌入服务
    class MockEmbeddingService:
        def encode_query(self, text):
            import numpy as np
            return np.random.rand(10)
    
    # 测试不同配置
    print("=== 测试增强文本处理器 ===")
    
    # 带嵌入服务和大模型NER的处理器
    mock_embedding = MockEmbeddingService()
    enhanced_processor = EnhancedTextProcessor(mock_embedding, use_model_ner=True)
    
    # 无嵌入服务，使用规则NER的处理器
    simple_processor = EnhancedTextProcessor(use_model_ner=False)
    
    test_cases = [
        "急性心肌梗死伴心律失常",
        "慢性肾功能不全 高血压病3级 糖尿病",
        "疑似急性胃肠炎，伴发热腹泻症状",
        "左肺上叶肺癌 胸腔积液 呼吸困难",
        "2型糖尿病伴血糖控制不佳，蛋白尿待查，肾功能不全",
    ]
    
    for test_text in test_cases:
        print(f"\n测试文本: {test_text}")
        
        # 增强处理
        enhanced_results = enhanced_processor.extract_diagnoses_enhanced(test_text)
        print(f"增强处理结果 ({len(enhanced_results)} 个):")
        for i, result in enumerate(enhanced_results):
            print(f"  {i+1}. {result['text']} (置信度: {result['diagnosis_confidence']:.3f})")
        
        # 处理摘要
        summary = enhanced_processor.get_processing_summary(test_text)
        print(f"摘要: 平均置信度 {summary['avg_confidence']:.3f}, 高置信度数量 {summary['high_confidence_count']}")


if __name__ == "__main__":
    main()