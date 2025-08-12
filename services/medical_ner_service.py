#!/usr/bin/env python3
"""
医学命名实体识别服务
基于 lixin12345/chinese-medical-ner 大模型
用于识别和提取医学文本中的关键实体
"""

import os
import re
from typing import Dict, List, Optional, Any
from loguru import logger
from services.diagnosis_entity_filter import DiagnosisEntityFilter


class MedicalNERService:
    """医学命名实体识别服务"""
    
    def __init__(self, model_name: str = None, use_model: bool = None):
        """
        初始化医学NER服务
        
        Args:
            model_name: NER模型名称，默认使用 lixin12345/chinese-medical-ner
            use_model: 是否使用大模型，从环境变量读取
        """
        # 模型配置
        if model_name is None:
            model_name = os.getenv('MEDICAL_NER_MODEL', 'lixin12345/chinese-medical-ner')
        
        if use_model is None:
            use_model = os.getenv('USE_MEDICAL_NER_MODEL', 'true').lower() == 'true'
        
        self.model_name = model_name
        self.use_model = use_model
        self.ner_pipeline = None
        self.model = None
        self.tokenizer = None
        
        # 初始化诊断实体过滤器
        self.entity_filter = DiagnosisEntityFilter()
        
        # 实体类型映射（从模型标签到标准类型）
        self.entity_type_mapping = {
            'DiseaseNameOrComprehensiveCertificate': 'disease',
            'Symptom': 'symptom', 
            'BodyParts': 'anatomy',
            'OrganOrCellDamage': 'pathology',
            'Drug': 'drug',
            'TreatmentOrPreventionProcedures': 'treatment',
            'TreatmentEquipment': 'equipment',
            'InspectionProcedure': 'inspection',
            'MedicalTestingItems': 'lab_indicator',
            'Department': 'department',
            'Sign': 'sign',
            'InjuryOrPoisoning': 'injury',
            'Microbiology': 'microbiology',
            'MedicalProcedures': 'procedure',
            'InspectEquipment': 'inspect_equipment'
        }
        
        # 初始化模型
        if self.use_model:
            self._init_ner_model()
        else:
            logger.info("未启用大模型NER，将使用规则方法作为回退")
            self._init_fallback_patterns()
    
    def _init_ner_model(self):
        """初始化NER模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            logger.info(f"正在加载医学NER模型: {self.model_name}")
            
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            # 自动检测GPU可用性
            import torch
            device = 0 if torch.cuda.is_available() else -1
            
            # 创建NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=device
            )
            
            logger.info(f"医学NER模型加载成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"医学NER模型加载失败: {e}")
            logger.warning("回退到规则方法")
            self.use_model = False
            self.ner_pipeline = None
            self.model = None
            self.tokenizer = None
            self._init_fallback_patterns()
    
    def _init_fallback_patterns(self):
        """初始化回退的规则模式"""
        self.medical_patterns = {
            # 疾病模式
            'disease': [
                r'(?:急性|慢性|原发性|继发性|复发性|亚急性)?[^，。；\s]{2,12}(?:病|症|炎|癌|瘤|综合征)',
                r'(?:急性|慢性)?[^，。；\s]{2,8}(?:感染|中毒|损伤|破裂|梗死|出血)',
                r'(?:I|II|III|IV|V)+型[^，。；\s]{2,8}(?:病|症)',
                r'[^，。；\s]{2,8}(?:功能不全|功能障碍|衰竭)',
            ],
            
            # 症状模式  
            'symptom': [
                r'(?:反复|持续|间歇性|突发性)?[^，。；\s]{2,6}(?:痛|疼|热|胀|肿|晕|麻|痒)',
                r'(?:大量|少量|血性|脓性)?[^，。；\s]{2,6}(?:出血|分泌|呕吐|腹泻)',
                r'[^，。；\s]{2,6}(?:不适|异常|增大|缩小|肥厚)',
                r'(?:阵发性|持续性)?[^，。；\s]{2,6}(?:咳嗽|气促|心悸|失眠)',
            ],
            
            # 解剖部位模式
            'anatomy': [
                r'(?:左|右|双侧|上|下|前|后)?(?:心|肝|肺|肾|胃|肠|脑|骨|脊柱)[^，。；\s]{0,6}',
                r'(?:左|右|双侧)?(?:乳腺|甲状腺|前列腺|子宫|卵巢)[^，。；\s]{0,4}',
                r'(?:颈|胸|腰|骶|尾)椎[^，。；\s]{0,4}',
                r'(?:主|冠状|肺|肾)动脉[^，。；\s]{0,4}',
            ]
        }
        
        # 停用词列表
        self.stop_words = {
            '待查', '考虑', '疑似', '排除', '？', '?', '诊断为', '患者', '病人',
            '检查', '发现', '显示', '提示', '建议', '需要', '进一步', '复查',
            '治疗', '用药', '服用', '注射', '输液', '手术', '康复'
        }
        
        # 无意义短语
        self.meaningless_phrases = {
            '不详', '不明', '不清', '未明确', '待定', '观察', '随访'
        }
    
    def extract_medical_entities(self, text: str, filter_drugs: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        提取医学实体（支持大模型和规则方法）
        
        Args:
            text: 输入的医学文本
            filter_drugs: 是否过滤非诊断实体（药品、设备、科室等），默认True
            
        Returns:
            字典，键为实体类型，值为实体列表（包含text, start, end, confidence）
        """
        if not text or not text.strip():
            return {}
        
        logger.debug(f"开始提取医学实体: {text}")
        
        # 优先使用大模型
        if self.use_model and self.ner_pipeline:
            try:
                entities = self._extract_entities_with_model(text)
            except Exception as e:
                logger.warning(f"大模型NER失败，回退到规则方法: {e}")
                entities = self._extract_entities_with_rules(text)
        else:
            # 使用规则方法
            entities = self._extract_entities_with_rules(text)
        
        # 如果需要过滤非诊断实体（药品、设备、科室等）
        if filter_drugs:
            logger.debug("开始过滤非诊断实体")
            entities = self.entity_filter.filter_entities(entities, text)
        
        return entities
    
    def _extract_entities_with_model(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """使用大模型提取实体"""
        logger.debug(f"使用大模型提取实体: {text}")
        
        # 使用NER pipeline进行实体识别
        model_entities = self.ner_pipeline(text)
        
        # 转换为标准格式
        entities = {}
        
        for entity in model_entities:
            # 获取实体信息
            entity_text = entity['word'].replace(' ', '').replace('##', '')  # 清理tokenizer artifacts
            entity_label = entity['entity_group'] if 'entity_group' in entity else entity['entity']
            confidence = entity['score']
            start = entity.get('start', 0)
            end = entity.get('end', len(entity_text))
            
            # 映射到标准实体类型
            standard_type = self.entity_type_mapping.get(entity_label, 'other')
            
            # 过滤低质量实体
            if not self._is_valid_model_entity(entity_text, confidence):
                continue
            
            # 添加到结果
            if standard_type not in entities:
                entities[standard_type] = []
            
            entity_info = {
                'text': entity_text,
                'start': start,
                'end': end,
                'confidence': confidence,
                'original_label': entity_label,
                'source': 'model'
            }
            entities[standard_type].append(entity_info)
        
        # 去重和排序
        for entity_type in entities:
            entities[entity_type] = self._deduplicate_entities(entities[entity_type])
        
        total_entities = sum(len(v) for v in entities.values())
        logger.info(f"大模型提取到 {total_entities} 个实体")
        
        # 详细记录每种类型的实体
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_details = [f"{e['text']}({e['confidence']:.3f})" for e in entity_list]
                logger.info(f"  {entity_type}: {entity_details}")
        
        return entities
    
    def _extract_entities_with_rules(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """使用规则方法提取实体"""
        logger.debug(f"使用规则方法提取实体: {text}")
        
        entities = {}
        
        for entity_type, patterns in self.medical_patterns.items():
            entities[entity_type] = []
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text))
                
                for match in matches:
                    entity_text = match.group().strip()
                    
                    # 过滤无效实体
                    if self._is_valid_entity(entity_text):
                        entity_info = {
                            'text': entity_text,
                            'start': match.start(),
                            'end': match.end(),
                            'confidence': self._calculate_entity_confidence(entity_text, entity_type),
                            'pattern': pattern,
                            'source': 'rules'
                        }
                        entities[entity_type].append(entity_info)
        
        # 去重和排序
        for entity_type in entities:
            entities[entity_type] = self._deduplicate_entities(entities[entity_type])
        
        logger.info(f"规则方法提取到 {sum(len(v) for v in entities.values())} 个实体")
        return entities
    
    def _is_valid_model_entity(self, entity_text: str, confidence: float) -> bool:
        """验证大模型提取的实体是否有效"""
        if not entity_text or len(entity_text) < 2:
            return False
        
        # 置信度阈值
        min_confidence = float(os.getenv('MEDICAL_NER_MIN_CONFIDENCE', '0.5'))
        if confidence < min_confidence:
            return False
        
        # 过滤无意义文本（如果有停用词列表）
        if hasattr(self, 'stop_words') and entity_text in self.stop_words:
            return False
        
        return True
    
    def _is_valid_entity(self, entity_text: str) -> bool:
        """判断实体是否有效"""
        if not entity_text or len(entity_text) < 2:
            return False
        
        # 过滤停用词和无意义短语
        if entity_text in self.stop_words or entity_text in self.meaningless_phrases:
            return False
        
        # 过滤纯数字或纯符号
        if re.match(r'^[\d\s\-+.]+$', entity_text):
            return False
        
        return True
    
    def _calculate_entity_confidence(self, entity_text: str, entity_type: str) -> float:
        """计算实体置信度"""
        confidence = 0.5  # 基础置信度
        
        # 长度因子
        if len(entity_text) >= 4:
            confidence += 0.1
        if len(entity_text) >= 6:
            confidence += 0.1
        
        # 特征词加权
        if entity_type == 'disease':
            if any(suffix in entity_text for suffix in ['病', '症', '炎', '癌', '瘤']):
                confidence += 0.2
            if any(prefix in entity_text for prefix in ['急性', '慢性', '原发性']):
                confidence += 0.1
        
        elif entity_type == 'symptom':
            if any(suffix in entity_text for suffix in ['痛', '热', '胀', '肿', '出血']):
                confidence += 0.2
        
        elif entity_type == 'anatomy':
            if any(part in entity_text for part in ['心', '肝', '肺', '肾', '脑']):
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """去重实体列表"""
        if not entities:
            return []
        
        # 按位置排序
        entities.sort(key=lambda x: (x['start'], -x['confidence']))
        
        # 去重逻辑：如果两个实体重叠，保留置信度更高的
        deduplicated = []
        for entity in entities:
            is_duplicate = False
            
            for existing in deduplicated:
                # 检查是否重叠
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start']):
                    # 如果当前实体置信度更高，替换现有实体
                    if entity['confidence'] > existing['confidence']:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(entity)
        
        # 按置信度排序
        return sorted(deduplicated, key=lambda x: x['confidence'], reverse=True)
    
    def identify_diagnosis_keywords(self, text: str) -> List[str]:
        """
        识别诊断关键词（用于向后兼容）
        
        Args:
            text: 输入文本
            
        Returns:
            诊断关键词列表
        """
        entities = self.extract_medical_entities(text)
        keywords = []
        
        # 优先提取疾病实体
        for entity in entities.get('disease', []):
            confidence_threshold = 0.5 if self.use_model else 0.6
            if entity['confidence'] > confidence_threshold:
                keywords.append(entity['text'])
        
        # 如果没有疾病实体，提取症状实体
        if not keywords:
            for entity in entities.get('symptom', []):
                confidence_threshold = 0.6 if self.use_model else 0.7
                if entity['confidence'] > confidence_threshold:
                    keywords.append(entity['text'])
        
        return keywords
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        # 检测GPU可用性
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_device_count = torch.cuda.device_count() if gpu_available else 0
        except ImportError:
            gpu_available = False
            gpu_device_count = 0
        
        return {
            'model_name': self.model_name,
            'use_model': self.use_model,
            'model_loaded': self.ner_pipeline is not None,
            'entity_types': list(self.entity_type_mapping.keys()) if self.use_model else list(self.medical_patterns.keys()),
            'fallback_available': hasattr(self, 'medical_patterns'),
            'gpu_available': gpu_available,
            'gpu_device_count': gpu_device_count,
            'device': 'GPU' if gpu_available and self.use_model else 'CPU'
        }
    
    def get_entity_summary(self, text: str) -> Dict[str, any]:
        """
        获取实体提取摘要
        
        Args:
            text: 输入文本
            
        Returns:
            实体摘要信息
        """
        entities = self.extract_medical_entities(text)
        
        summary = {
            'total_entities': sum(len(entities[key]) for key in entities),
            'entity_types': list(entities.keys()),
            'high_confidence_entities': [],
            'primary_diagnosis_candidates': [],
            'extraction_method': 'model' if self.use_model and self.ner_pipeline else 'rules',
            'model_info': self.get_model_info()
        }
        
        # 动态置信度阈值
        high_confidence_threshold = 0.8 if self.use_model else 0.7
        diagnosis_threshold = 0.5 if self.use_model else 0.6
        
        # 统计高置信度实体
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity['confidence'] > high_confidence_threshold:
                    summary['high_confidence_entities'].append({
                        'type': entity_type,
                        'text': entity['text'],
                        'confidence': entity['confidence'],
                        'source': entity.get('source', 'unknown')
                    })
        
        # 识别主要诊断候选
        disease_entities = entities.get('disease', [])
        if disease_entities:
            summary['primary_diagnosis_candidates'] = [
                entity['text'] for entity in disease_entities[:3]
                if entity['confidence'] > diagnosis_threshold
            ]
        
        return summary
    
    def get_filter_stats(self, text: str) -> Dict[str, Any]:
        """
        获取过滤统计信息
        
        Args:
            text: 输入文本
            
        Returns:
            过滤统计信息
        """
        # 获取未过滤的实体
        original_entities = self.extract_medical_entities(text, filter_drugs=False)
        
        # 获取过滤后的实体（过滤非诊断实体）
        filtered_entities = self.extract_medical_entities(text, filter_drugs=True)
        
        # 生成统计信息
        stats = self.entity_filter.get_filter_stats(original_entities, filtered_entities)
        
        return stats


def main():
    """测试函数"""
    print("=== 医学NER服务测试 ===")
    
    # 测试大模型版本
    print("\n🤖 测试大模型NER服务:")
    model_ner_service = MedicalNERService(use_model=True)
    print(f"模型信息: {model_ner_service.get_model_info()}")
    
    # 测试规则版本
    print("\n📝 测试规则NER服务:")
    rule_ner_service = MedicalNERService(use_model=False)
    print(f"规则信息: {rule_ner_service.get_model_info()}")
    
    test_cases = [
        "急性心肌梗死伴心律失常",
        "慢性肾功能不全 高血压病3级 糖尿病",
        "疑似急性胃肠炎，伴发热腹泻症状",
        "左肺上叶肺癌 胸腔积液 呼吸困难",
        "2型糖尿病伴血糖控制不佳 蛋白尿待查",
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n=== 测试案例 {i}: {test_text} ===")
        
        # 大模型版本
        if model_ner_service.use_model:
            print("🤖 大模型结果:")
            entities = model_ner_service.extract_medical_entities(test_text)
            for entity_type, entity_list in entities.items():
                if entity_list:
                    entity_info = [(e['text'], f"{e['confidence']:.3f}") for e in entity_list[:3]]
                    print(f"  {entity_type}: {entity_info}")
            
            summary = model_ner_service.get_entity_summary(test_text)
            print(f"  摘要: {summary['total_entities']}个实体, 高置信度{len(summary['high_confidence_entities'])}个")
        
        # 规则版本对比
        print("📝 规则方法结果:")
        entities = rule_ner_service.extract_medical_entities(test_text)
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_info = [(e['text'], f"{e['confidence']:.3f}") for e in entity_list[:3]]
                print(f"  {entity_type}: {entity_info}")


if __name__ == "__main__":
    main()