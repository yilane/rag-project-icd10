#!/usr/bin/env python3
"""
诊断实体过滤器
用于从医学NER结果中过滤出诊断相关实体，排除药品等非诊断内容
"""

import re
import os
from typing import Dict, List, Any, Set
from loguru import logger


class DiagnosisEntityFilter:
    """诊断实体过滤器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化过滤器
        
        Args:
            config: 过滤配置参数
        """
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        
        # 药品实体的诊断相关关键词
        self.drug_diagnosis_keywords = {
            '过敏', '中毒', '不良反应', '副作用', '依赖', '滥用', 
            '耐药', '抗药性', '药物性', '中毒性', '戒断', '成瘾',
            '肝毒性', '肾毒性', '心脏毒性', '神经毒性'
        }
        
        # 明确的药品名称模式（需要过滤）
        self.drug_name_patterns = [
            r'.*片$', r'.*胶囊$', r'.*注射液$', r'.*口服液$',
            r'.*颗粒$', r'.*软膏$', r'.*滴眼液$', r'.*喷雾剂$',
            r'.*素$', r'.*霉素$', r'.*西林$', r'.*沙星$',
            r'.*洛尔$', r'.*普利$', r'.*沙坦$', r'.*司汀$',
            r'^阿.*', r'^氨.*', r'^左.*', r'^右.*',  # 常见药品前缀
            r'.*缓释.*', r'.*控释.*', r'.*肠溶.*'
        ]
        
        # 治疗程序模式（通常需要过滤，除非是疾病相关）
        self.treatment_patterns = [
            r'.*手术$', r'.*切除术$', r'.*造影$', r'.*穿刺$',
            r'.*化疗$', r'.*放疗$', r'.*康复$', r'.*训练$',
            r'.*护理$', r'.*检查$', r'.*监测$'
        ]
        
        # 疾病相关后缀（即使在治疗程序中也要保留）
        self.disease_suffixes = {
            '病', '症', '炎', '癌', '瘤', '综合征', '性疾病',
            '功能不全', '功能障碍', '衰竭', '梗死', '出血',
            '破裂', '穿孔', '狭窄', '扩张', '增生', '萎缩'
        }
        
        logger.info(f"诊断实体过滤器初始化完成，配置: {self.config}")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'strict_mode': os.getenv('DIAGNOSIS_FILTER_STRICT_MODE', 'false').lower() == 'true',
            'keep_drug_diseases': os.getenv('KEEP_DRUG_DISEASES', 'true').lower() == 'true',
            'keep_lab_indicators': os.getenv('KEEP_LAB_INDICATORS', 'false').lower() == 'true',
            'context_window': int(os.getenv('FILTER_CONTEXT_WINDOW', '20')),
            'confidence_threshold': float(os.getenv('FILTER_CONFIDENCE_THRESHOLD', '0.6')),
            'enable_context_analysis': os.getenv('ENABLE_CONTEXT_ANALYSIS', 'true').lower() == 'true'
        }
    
    def filter_entities(self, entities: Dict[str, List[Dict]], original_text: str) -> Dict[str, List[Dict]]:
        """
        主过滤方法
        
        Args:
            entities: 原始实体字典
            original_text: 原始文本
            
        Returns:
            过滤后的实体字典
        """
        if not entities:
            return {}
        
        logger.debug(f"开始过滤实体，原始实体类型: {list(entities.keys())}")
        
        if self.config['strict_mode']:
            filtered_entities = self._strict_filter(entities)
        else:
            filtered_entities = self._smart_filter(entities, original_text)
        
        # 统计过滤结果
        original_count = sum(len(v) for v in entities.values())
        filtered_count = sum(len(v) for v in filtered_entities.values())
        
        logger.info(f"实体过滤完成: {original_count} -> {filtered_count} "
                   f"(过滤掉 {original_count - filtered_count} 个)")
        
        return filtered_entities
    
    def _strict_filter(self, entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        严格过滤：完全排除非诊断实体
        
        Args:
            entities: 原始实体字典
            
        Returns:
            严格过滤后的实体字典
        """
        logger.debug("使用严格过滤模式")
        
        # 明确的诊断相关实体类型
        diagnosis_types = {
            'disease',      # 疾病
            'symptom',      # 症状  
            'anatomy',      # 解剖部位
            'pathology',    # 病理损伤
            'injury',       # 外伤中毒
            'sign',         # 体征
            'microbiology', # 微生物学
        }
        
        # 可选的实验室指标
        if self.config['keep_lab_indicators']:
            diagnosis_types.add('lab_indicator')
        
        filtered_entities = {}
        for entity_type, entity_list in entities.items():
            if entity_type in diagnosis_types:
                # 进一步过滤低置信度实体
                high_confidence_entities = [
                    entity for entity in entity_list
                    if entity.get('confidence', 0) >= self.config['confidence_threshold']
                ]
                if high_confidence_entities:
                    filtered_entities[entity_type] = high_confidence_entities
        
        return filtered_entities
    
    def _smart_filter(self, entities: Dict[str, List[Dict]], text: str) -> Dict[str, List[Dict]]:
        """
        智能过滤：基于上下文判断
        
        Args:
            entities: 原始实体字典
            text: 原始文本
            
        Returns:
            智能过滤后的实体字典
        """
        logger.debug("使用智能过滤模式")
        
        filtered_entities = {}
        
        for entity_type, entity_list in entities.items():
            if entity_type == 'drug':
                # 药品实体需要特殊处理
                filtered_list = self._filter_drug_entities(entity_list, text)
                if filtered_list:
                    filtered_entities['drug_related_disease'] = filtered_list
                    
            elif entity_type in ['treatment', 'procedure']:
                # 治疗程序实体
                filtered_list = self._filter_treatment_entities(entity_list, text)
                if filtered_list:
                    filtered_entities[f'{entity_type}_related_disease'] = filtered_list
                    
            elif entity_type == 'equipment' or entity_type == 'inspect_equipment':
                # 设备类实体通常不是诊断相关，但可能有例外
                filtered_list = self._filter_equipment_entities(entity_list, text)
                if filtered_list:
                    filtered_entities[f'{entity_type}_related'] = filtered_list
                    
            elif entity_type == 'department':
                # 科室通常不是诊断，完全过滤
                logger.debug(f"过滤科室实体: {[e['text'] for e in entity_list]}")
                continue
                
            elif entity_type == 'lab_indicator':
                # 实验室指标根据配置决定是否保留
                if self.config['keep_lab_indicators']:
                    filtered_entities[entity_type] = self._filter_by_confidence(entity_list)
                else:
                    logger.debug(f"配置禁用实验室指标，过滤: {[e['text'] for e in entity_list]}")
                    
            else:
                # 其他诊断相关实体（disease, symptom, anatomy等）直接保留
                filtered_list = self._filter_by_confidence(entity_list)
                if filtered_list:
                    filtered_entities[entity_type] = filtered_list
        
        return filtered_entities
    
    def _filter_drug_entities(self, entity_list: List[Dict], text: str) -> List[Dict]:
        """过滤药品实体，保留药物相关疾病"""
        if not self.config['keep_drug_diseases']:
            logger.debug("配置禁用药物相关疾病，完全过滤药品实体")
            return []
        
        filtered_entities = []
        
        for entity in entity_list:
            entity_text = entity['text']
            
            # 检查是否为明确的药品名称
            is_drug_name = any(re.match(pattern, entity_text) for pattern in self.drug_name_patterns)
            
            if is_drug_name:
                logger.debug(f"过滤药品名称: {entity_text}")
                continue
            
            # 检查上下文是否为诊断相关
            if self.config['enable_context_analysis']:
                has_diagnosis_context = self._has_diagnosis_context(entity, text)
                
                if has_diagnosis_context:
                    logger.debug(f"保留药物相关疾病: {entity_text}")
                    filtered_entities.append(entity)
                else:
                    logger.debug(f"药品实体无诊断上下文，过滤: {entity_text}")
            else:
                # 不进行上下文分析时，检查实体本身是否包含疾病特征
                if self._has_disease_characteristics(entity_text):
                    filtered_entities.append(entity)
        
        return filtered_entities
    
    def _filter_treatment_entities(self, entity_list: List[Dict], text: str) -> List[Dict]:
        """过滤治疗程序实体，保留疾病相关的"""
        filtered_entities = []
        
        for entity in entity_list:
            entity_text = entity['text']
            
            # 检查是否包含疾病后缀
            has_disease_suffix = any(suffix in entity_text for suffix in self.disease_suffixes)
            
            if has_disease_suffix:
                logger.debug(f"保留疾病相关治疗: {entity_text}")
                filtered_entities.append(entity)
            else:
                # 检查是否为纯治疗程序
                is_pure_treatment = any(re.match(pattern, entity_text) for pattern in self.treatment_patterns)
                
                if not is_pure_treatment:
                    # 不确定的情况下，检查上下文
                    if self.config['enable_context_analysis']:
                        has_diagnosis_context = self._has_diagnosis_context(entity, text)
                        if has_diagnosis_context:
                            logger.debug(f"保留有诊断上下文的治疗: {entity_text}")
                            filtered_entities.append(entity)
                        else:
                            logger.debug(f"过滤纯治疗程序: {entity_text}")
                    else:
                        # 不确定时保守处理，过滤掉
                        logger.debug(f"过滤不确定的治疗程序: {entity_text}")
                else:
                    logger.debug(f"过滤纯治疗程序: {entity_text}")
        
        return filtered_entities
    
    def _filter_equipment_entities(self, entity_list: List[Dict], text: str) -> List[Dict]:
        """过滤设备实体，极少数情况下可能相关"""
        # 设备通常不是诊断相关，但可能有特殊情况（如"起搏器综合征"）
        filtered_entities = []
        
        for entity in entity_list:
            entity_text = entity['text']
            
            # 检查是否包含疾病特征
            if self._has_disease_characteristics(entity_text):
                logger.debug(f"保留设备相关疾病: {entity_text}")
                filtered_entities.append(entity)
            else:
                logger.debug(f"过滤设备实体: {entity_text}")
        
        return filtered_entities
    
    def _filter_by_confidence(self, entity_list: List[Dict]) -> List[Dict]:
        """按置信度过滤实体"""
        return [
            entity for entity in entity_list
            if entity.get('confidence', 0) >= self.config['confidence_threshold']
        ]
    
    def _has_diagnosis_context(self, entity: Dict, text: str) -> bool:
        """检查实体是否具有诊断相关的上下文"""
        entity_start = entity.get('start', 0)
        entity_end = entity.get('end', len(entity['text']))
        
        # 获取上下文窗口
        context_start = max(0, entity_start - self.config['context_window'])
        context_end = min(len(text), entity_end + self.config['context_window'])
        context = text[context_start:context_end]
        
        # 检查诊断相关关键词
        diagnosis_keywords = self.drug_diagnosis_keywords | {
            '诊断', '疑似', '考虑', '排除', '病史', '既往史',
            '症状', '表现', '发作', '急性', '慢性', '复发',
            '并发症', '合并症', '继发', '原发'
        }
        
        return any(keyword in context for keyword in diagnosis_keywords)
    
    def _has_disease_characteristics(self, entity_text: str) -> bool:
        """检查实体文本是否具有疾病特征"""
        return any(suffix in entity_text for suffix in self.disease_suffixes)
    
    def get_filter_stats(self, original_entities: Dict, filtered_entities: Dict) -> Dict[str, Any]:
        """获取过滤统计信息"""
        original_count = sum(len(v) for v in original_entities.values())
        filtered_count = sum(len(v) for v in filtered_entities.values())
        
        filtered_out_by_type = {}
        for entity_type, entity_list in original_entities.items():
            original_type_count = len(entity_list)
            filtered_type_count = len(filtered_entities.get(entity_type, []))
            
            # 考虑类型重命名的情况
            renamed_types = [k for k in filtered_entities.keys() if k.startswith(entity_type)]
            if renamed_types:
                filtered_type_count += sum(len(filtered_entities[k]) for k in renamed_types)
            
            filtered_out_count = original_type_count - filtered_type_count
            if filtered_out_count > 0:
                filtered_out_by_type[entity_type] = filtered_out_count
        
        return {
            'original_total': original_count,
            'filtered_total': filtered_count,
            'filtered_out_total': original_count - filtered_count,
            'filtered_out_by_type': filtered_out_by_type,
            'filter_config': self.config,
            'filter_efficiency': (original_count - filtered_count) / original_count if original_count > 0 else 0
        }


def main():
    """测试函数"""
    print("=== 诊断实体过滤器测试 ===")
    
    # 创建测试过滤器
    strict_filter = DiagnosisEntityFilter({'strict_mode': True})
    smart_filter = DiagnosisEntityFilter({'strict_mode': False})
    
    # 模拟NER结果
    test_entities = {
        'disease': [
            {'text': '急性心肌梗死', 'start': 0, 'end': 6, 'confidence': 0.95},
            {'text': '高血压病', 'start': 7, 'end': 11, 'confidence': 0.88}
        ],
        'drug': [
            {'text': '阿司匹林片', 'start': 12, 'end': 17, 'confidence': 0.92},
            {'text': '药物过敏', 'start': 18, 'end': 22, 'confidence': 0.85},
            {'text': '青霉素中毒', 'start': 23, 'end': 28, 'confidence': 0.78}
        ],
        'treatment': [
            {'text': '心脏手术', 'start': 29, 'end': 33, 'confidence': 0.80},
            {'text': '手术后综合征', 'start': 34, 'end': 40, 'confidence': 0.75}
        ],
        'department': [
            {'text': '心内科', 'start': 41, 'end': 44, 'confidence': 0.90}
        ],
        'symptom': [
            {'text': '胸痛', 'start': 45, 'end': 47, 'confidence': 0.85}
        ]
    }
    
    test_text = "急性心肌梗死 高血压病 阿司匹林片 药物过敏 青霉素中毒 心脏手术 手术后综合征 心内科 胸痛"
    
    print(f"\n原始实体: {sum(len(v) for v in test_entities.values())} 个")
    for entity_type, entities in test_entities.items():
        entity_texts = [e['text'] for e in entities]
        print(f"  {entity_type}: {entity_texts}")
    
    # 严格过滤测试
    print("\n=== 严格过滤模式 ===")
    strict_result = strict_filter.filter_entities(test_entities, test_text)
    print(f"过滤后实体: {sum(len(v) for v in strict_result.values())} 个")
    for entity_type, entities in strict_result.items():
        entity_texts = [e['text'] for e in entities]
        print(f"  {entity_type}: {entity_texts}")
    
    strict_stats = strict_filter.get_filter_stats(test_entities, strict_result)
    print(f"过滤统计: {strict_stats}")
    
    # 智能过滤测试
    print("\n=== 智能过滤模式 ===")
    smart_result = smart_filter.filter_entities(test_entities, test_text)
    print(f"过滤后实体: {sum(len(v) for v in smart_result.values())} 个")
    for entity_type, entities in smart_result.items():
        entity_texts = [e['text'] for e in entities]
        print(f"  {entity_type}: {entity_texts}")
    
    smart_stats = smart_filter.get_filter_stats(test_entities, smart_result)
    print(f"过滤统计: {smart_stats}")


if __name__ == "__main__":
    main()