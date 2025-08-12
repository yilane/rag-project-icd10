#!/usr/bin/env python3
"""
不确定性诊断处理服务
用于识别和处理包含不确定性描述的诊断文本
如：待查、疑似、可能等，优先匹配ICD中的"未特指"编码
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger


class UncertaintyDiagnosisService:
    """不确定性诊断处理服务"""
    
    def __init__(self):
        """初始化不确定性诊断服务"""
        
        # 不确定性词汇模式（按优先级排序）
        self.uncertainty_patterns = {
            # 明确的不确定性表达
            'explicit_uncertainty': {
                'patterns': ['待查', '待诊', '待确诊', '待定', '排除', '？', '?'],
                'weight': 1.0,
                'description': '明确不确定性'
            },
            
            # 疑似性表达
            'suspected': {
                'patterns': ['疑似', '疑为', '考虑', '可能', '拟诊', '倾向'],
                'weight': 0.9,
                'description': '疑似性'
            },
            
            # 程度性不确定
            'degree_uncertainty': {
                'patterns': ['不除外', '不能排除', '不明原因', '原因不明', '性质待定'],
                'weight': 0.8,
                'description': '程度不确定性'
            }
        }
        
        # ICD"未特指"匹配模式（按优先级排序）
        self.icd_unspecified_patterns = {
            # 最高优先级：完全匹配的"未特指"
            'exact_unspecified': {
                'patterns': ['未特指的{}', '{}，未特指', '{}未特指'],
                'boost': 0.3,
                'description': '精确未特指匹配'
            },
            
            # 高优先级：包含"未特指"
            'contains_unspecified': {
                'patterns': ['未特指'],
                'boost': 0.25,
                'description': '包含未特指'
            },
            
            # 中优先级：其他不确定性表达
            'other_uncertainty': {
                'patterns': ['其他{}', '{}，其他', '不明{}', '{}不明'],
                'boost': 0.2,
                'description': '其他不确定性'
            },
            
            # 低优先级：编码结构暗示（.9结尾）
            'code_structure': {
                'code_pattern': r'\.9\d*$',  # 以.9结尾的编码
                'boost': 0.15,
                'description': '编码结构暗示'
            }
        }
        
        logger.info("不确定性诊断处理服务初始化完成")
    
    def detect_uncertainty(self, text: str) -> Dict[str, Any]:
        """
        检测文本中的不确定性表达
        
        Args:
            text: 诊断文本
            
        Returns:
            不确定性分析结果
        """
        result = {
            'has_uncertainty': False,
            'uncertainty_type': None,
            'uncertainty_weight': 0.0,
            'matched_patterns': [],
            'clean_text': text,
            'uncertainty_indicators': []
        }
        
        text_lower = text.lower()
        
        # 检测各类不确定性模式
        for uncertainty_type, config in self.uncertainty_patterns.items():
            for pattern in config['patterns']:
                if pattern.lower() in text_lower:
                    result['has_uncertainty'] = True
                    result['uncertainty_type'] = uncertainty_type
                    result['uncertainty_weight'] = max(result['uncertainty_weight'], config['weight'])
                    result['matched_patterns'].append(pattern)
                    result['uncertainty_indicators'].append({
                        'pattern': pattern,
                        'type': uncertainty_type,
                        'weight': config['weight'],
                        'position': text_lower.find(pattern.lower())
                    })
        
        # 生成清理后的文本（移除不确定性词汇）
        if result['has_uncertainty']:
            clean_text = text
            for indicator in result['uncertainty_indicators']:
                pattern = indicator['pattern']
                # 移除不确定性词汇，保留核心诊断内容
                clean_text = re.sub(rf'{re.escape(pattern)}', '', clean_text, flags=re.IGNORECASE)
            
            # 清理多余空格和标点
            result['clean_text'] = re.sub(r'\s+', ' ', clean_text).strip()
            result['clean_text'] = re.sub(r'^[，。、\s]+|[，。、\s]+$', '', result['clean_text'])
        
        logger.debug(f"不确定性检测: '{text}' -> {result}")
        return result
    
    def calculate_unspecified_boost(self, candidate_record: Dict[str, Any], clean_diagnosis: str) -> float:
        """
        计算ICD"未特指"匹配的加权分数
        
        Args:
            candidate_record: 候选ICD记录
            clean_diagnosis: 清理后的诊断文本
            
        Returns:
            加权分数
        """
        boost_score = 0.0
        matched_types = []
        
        candidate_title = candidate_record.get('preferred_zh', '').lower()
        candidate_code = candidate_record.get('code', '')
        clean_diagnosis_lower = clean_diagnosis.lower()
        
        # 1. 检查精确"未特指"匹配
        exact_config = self.icd_unspecified_patterns['exact_unspecified']
        for pattern_template in exact_config['patterns']:
            # 构建具体的匹配模式
            pattern = pattern_template.format(clean_diagnosis_lower)
            if pattern in candidate_title:
                boost_score = max(boost_score, exact_config['boost'])
                matched_types.append('exact_unspecified')
                logger.debug(f"精确未特指匹配: '{pattern}' in '{candidate_title}'")
                break
        
        # 2. 检查包含"未特指"
        if boost_score == 0.0:  # 如果还没有精确匹配
            contains_config = self.icd_unspecified_patterns['contains_unspecified']
            for pattern in contains_config['patterns']:
                if pattern in candidate_title:
                    boost_score = max(boost_score, contains_config['boost'])
                    matched_types.append('contains_unspecified')
                    logger.debug(f"包含未特指匹配: '{pattern}' in '{candidate_title}'")
                    break
        
        # 3. 检查其他不确定性表达
        if boost_score == 0.0:
            other_config = self.icd_unspecified_patterns['other_uncertainty']
            for pattern_template in other_config['patterns']:
                pattern = pattern_template.format(clean_diagnosis_lower)
                if pattern in candidate_title:
                    boost_score = max(boost_score, other_config['boost'])
                    matched_types.append('other_uncertainty')
                    logger.debug(f"其他不确定性匹配: '{pattern}' in '{candidate_title}'")
                    break
        
        # 4. 检查编码结构（.9结尾）
        if boost_score == 0.0:
            code_config = self.icd_unspecified_patterns['code_structure']
            if re.search(code_config['code_pattern'], candidate_code):
                boost_score = code_config['boost']
                matched_types.append('code_structure')
                logger.debug(f"编码结构匹配: '{candidate_code}' matches pattern")
        
        if boost_score > 0:
            logger.info(f"未特指加权: '{clean_diagnosis}' -> '{candidate_title}' (+{boost_score:.3f}, 类型: {matched_types})")
        
        return boost_score
    
    def process_uncertainty_query(self, query_text: str, candidate_records: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        处理包含不确定性的查询
        
        Args:
            query_text: 查询文本
            candidate_records: 候选记录列表
            
        Returns:
            (清理后的查询文本, 加权后的候选记录列表)
        """
        # 1. 检测不确定性
        uncertainty_result = self.detect_uncertainty(query_text)
        
        if not uncertainty_result['has_uncertainty']:
            # 没有不确定性，直接返回
            return query_text, candidate_records
        
        clean_query = uncertainty_result['clean_text']
        uncertainty_weight = uncertainty_result['uncertainty_weight']
        
        logger.info(f"检测到不确定性诊断: '{query_text}' -> '{clean_query}' (权重: {uncertainty_weight})")
        
        # 2. 对候选记录进行"未特指"加权
        enhanced_records = []
        for record in candidate_records:
            enhanced_record = record.copy()
            
            # 计算未特指加权分数
            unspecified_boost = self.calculate_unspecified_boost(record, clean_query)
            
            if unspecified_boost > 0:
                # 应用加权
                original_score = enhanced_record.get('score', 0.0)
                enhanced_score = original_score + (unspecified_boost * uncertainty_weight)
                enhanced_record['score'] = enhanced_score
                enhanced_record['uncertainty_boost'] = unspecified_boost
                enhanced_record['uncertainty_weight'] = uncertainty_weight
                enhanced_record['original_score'] = original_score
                
                logger.debug(f"应用不确定性加权: {record.get('code', '')} "
                           f"{original_score:.3f} -> {enhanced_score:.3f} (+{unspecified_boost * uncertainty_weight:.3f})")
            
            enhanced_records.append(enhanced_record)
        
        # 3. 重新排序
        enhanced_records.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        return clean_query, enhanced_records
    
    def get_uncertainty_explanation(self, query_text: str) -> Dict[str, Any]:
        """
        获取不确定性处理的详细解释
        
        Args:
            query_text: 查询文本
            
        Returns:
            详细解释信息
        """
        uncertainty_result = self.detect_uncertainty(query_text)
        
        explanation = {
            'original_query': query_text,
            'has_uncertainty': uncertainty_result['has_uncertainty'],
            'processed_query': uncertainty_result['clean_text'],
            'uncertainty_analysis': uncertainty_result,
            'processing_strategy': 'none'
        }
        
        if uncertainty_result['has_uncertainty']:
            explanation['processing_strategy'] = 'unspecified_priority'
            explanation['strategy_description'] = (
                f"检测到不确定性表达 {uncertainty_result['matched_patterns']}，"
                f"优先匹配ICD中包含'未特指'、'其他'等不确定性描述的编码"
            )
        
        return explanation


def main():
    """测试函数"""
    print("=== 不确定性诊断处理服务测试 ===")
    
    service = UncertaintyDiagnosisService()
    
    # 测试用例
    test_cases = [
        "颅内损伤待查",
        "急性心肌梗死疑似",
        "肺炎可能",
        "糖尿病排除",
        "高血压病",  # 无不确定性
        "肾功能不全待定",
        "急性胃肠炎？"
    ]
    
    for query in test_cases:
        print(f"\\n测试: '{query}'")
        
        # 不确定性检测
        uncertainty = service.detect_uncertainty(query)
        print(f"  不确定性: {uncertainty['has_uncertainty']}")
        if uncertainty['has_uncertainty']:
            print(f"  清理后: '{uncertainty['clean_text']}'")
            print(f"  匹配模式: {uncertainty['matched_patterns']}")
        
        # 解释
        explanation = service.get_uncertainty_explanation(query)
        print(f"  处理策略: {explanation['processing_strategy']}")


if __name__ == "__main__":
    main()