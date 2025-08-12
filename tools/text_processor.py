#!/usr/bin/env python3
"""
诊断文本处理器
处理多诊断文本的分割和标准化
集成增强的文本处理功能
"""

import re
import os
from typing import List, Dict, Any, Optional
from loguru import logger


class DiagnosisTextProcessor:
    """
    诊断文本处理器
    使用简单的分隔符分割进行诊断提取
    """
    
    def __init__(self, embedding_service=None, use_enhanced_processing=None):
        """
        初始化处理器
        
        Args:
            embedding_service: 嵌入服务实例（用于增强处理）
            use_enhanced_processing: 是否使用增强处理（从环境变量读取）
        """
        # 医学分隔符模式（保持向后兼容）
        self.medical_separators = [
            r'[，,；;]',  # 逗号、分号
            r'[+＋]',     # 加号
            r'\s+',       # 空格
        ]
        
        # 增强处理配置
        if use_enhanced_processing is None:
            use_enhanced_processing = os.getenv('USE_ENHANCED_TEXT_PROCESSING', 'true').lower() == 'true'
        
        self.use_enhanced_processing = use_enhanced_processing
        self.embedding_service = embedding_service
        self._enhanced_processor = None
        
        # 延迟初始化增强处理器
        if self.use_enhanced_processing:
            self._init_enhanced_processor()
    
    def _init_enhanced_processor(self):
        """初始化增强处理器"""
        try:
            from services.enhanced_text_processor import EnhancedTextProcessor
            self._enhanced_processor = EnhancedTextProcessor(self.embedding_service)
            logger.info("增强文本处理器已启用")
        except ImportError as e:
            logger.warning(f"无法导入增强处理器，回退到简单处理: {e}")
            self.use_enhanced_processing = False
        except Exception as e:
            logger.error(f"初始化增强处理器失败: {e}")
            self.use_enhanced_processing = False
    
    def extract_diagnoses(self, text: str) -> List[str]:
        """
        从复杂文本中提取诊断关键词
        支持增强处理和简单处理两种模式
        
        Args:
            text: 输入的医疗文本
            
        Returns:
            提取的诊断关键词列表
        """
        if not text or not text.strip():
            return []
        
        logger.debug(f"开始处理文本: {text}")
        
        # 尝试使用增强处理
        if self.use_enhanced_processing and self._enhanced_processor:
            try:
                enhanced_results = self._enhanced_processor.extract_diagnoses_simple(text)
                logger.info(f"增强处理结果: {enhanced_results}")
                return enhanced_results
            except Exception as e:
                logger.warning(f"增强处理失败，回退到简单处理: {e}")
        
        # 使用简单处理（原始逻辑）
        return self._extract_diagnoses_simple(text)
    
    def _extract_diagnoses_simple(self, text: str) -> List[str]:
        """简单的诊断提取方法（原始逻辑）"""
        # 基于分隔符分割
        segments = self._split_by_separators(text)
        
        # 简单清理
        cleaned_diagnoses = []
        for segment in segments:
            clean_segment = self._clean_diagnosis_text(segment)
            if clean_segment and len(clean_segment) >= 2:  # 最基本的长度过滤
                cleaned_diagnoses.append(clean_segment)
        
        # 去重保序
        unique_diagnoses = []
        seen = set()
        for diagnosis in cleaned_diagnoses:
            if diagnosis not in seen:
                unique_diagnoses.append(diagnosis)
                seen.add(diagnosis)
        
        logger.info(f"简单处理结果: {unique_diagnoses}")
        return unique_diagnoses
    
    def _split_by_separators(self, text: str) -> List[str]:
        """基于医学分隔符分割文本"""
        # 构建分隔符正则表达式
        separator_pattern = '|'.join(self.medical_separators)
        
        # 分割文本
        segments = re.split(separator_pattern, text)
        
        # 清理空白段落
        segments = [seg.strip() for seg in segments if seg.strip()]
        
        return segments
    
    def _clean_diagnosis_text(self, text: str) -> str:
        """
        简单清理诊断文本
        保留有诊断价值的关键词如"待查"、"疑似"等
        """
        if not text:
            return ""
        
        # 去除首尾空白
        text = text.strip()
        
        # 只去除真正无意义的前后缀，保留有诊断价值的词汇
        # "待查"、"疑似"、"考虑"、"排除"等表示诊断不确定性，应保留
        prefixes_to_remove = ['？', '?', '诊断为', '患者']  # 移除了待查、考虑、疑似、排除
        suffixes_to_remove = ['？', '?', '诊断']  # 移除了待查
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        for suffix in suffixes_to_remove:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()
        
        return text
    
    def is_multi_diagnosis(self, text: str) -> bool:
        """判断是否为多诊断文本"""
        # 基于提取的诊断数量判断
        diagnoses = self.extract_diagnoses(text)
        return len(diagnoses) > 1
    
    def extract_diagnoses_enhanced(self, text: str, filter_drugs: bool = True) -> List[Dict[str, Any]]:
        """
        增强的诊断提取方法（返回详细信息）
        
        Args:
            text: 输入文本
            filter_drugs: 是否过滤非诊断实体（药品、设备、科室等），默认True
            
        Returns:
            增强的诊断结果列表，包含元数据
        """
        if not self.use_enhanced_processing or not self._enhanced_processor:
            # 如果没有增强处理器，返回简单格式
            simple_results = self._extract_diagnoses_simple(text)
            return [
                {
                    'text': result,
                    'diagnosis_confidence': 0.5,
                    'metadata': {'is_simple_extraction': True}
                }
                for result in simple_results
            ]
        
        try:
            return self._enhanced_processor.extract_diagnoses_enhanced(text, filter_drugs=filter_drugs)
        except Exception as e:
            logger.error(f"增强提取失败: {e}")
            # 回退到简单格式
            simple_results = self._extract_diagnoses_simple(text)
            return [
                {
                    'text': result,
                    'diagnosis_confidence': 0.5,
                    'metadata': {'is_fallback': True}
                }
                for result in simple_results
            ]
    
    def get_processing_mode(self) -> str:
        """获取当前处理模式"""
        if self.use_enhanced_processing and self._enhanced_processor:
            return "enhanced"
        else:
            return "simple"


def main():
    """测试函数"""
    print("=== 测试诊断文本处理器 ===")
    
    # 测试简单处理器
    simple_processor = DiagnosisTextProcessor(use_enhanced_processing=False)
    print(f"简单处理器模式: {simple_processor.get_processing_mode()}")
    
    # 测试增强处理器（如果可用）
    enhanced_processor = DiagnosisTextProcessor(use_enhanced_processing=True)
    print(f"增强处理器模式: {enhanced_processor.get_processing_mode()}")
    
    test_cases = [
        "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳", 
        "患者诊断为高血压3级，冠心病，心功能不全",
        "肺部阴影 蛋白尿 血肌酐偏高",
        "肾功能不全，建议进一步检查",
        "急性胃肠炎，发热38.5℃，腹泻3天",
        "高血压病 糖尿病 冠状动脉粥样硬化性心脏病",
        "慢性肾小球肾炎 尿毒症 贫血",
    ]
    
    for test_text in test_cases:
        print(f"\n测试文本: {test_text}")
        
        # 简单提取
        simple_diagnoses = simple_processor.extract_diagnoses(test_text)
        print(f"简单提取结果: {simple_diagnoses}")
        
        # 增强提取（如果可用）
        if enhanced_processor.get_processing_mode() == "enhanced":
            enhanced_diagnoses = enhanced_processor.extract_diagnoses(test_text)
            print(f"增强提取结果: {enhanced_diagnoses}")
            
            # 详细增强结果
            enhanced_detailed = enhanced_processor.extract_diagnoses_enhanced(test_text)
            print(f"增强详细结果数量: {len(enhanced_detailed)}")
            for i, result in enumerate(enhanced_detailed):
                print(f"  {i+1}. {result['text']} (置信度: {result.get('diagnosis_confidence', 0):.3f})")
        
        is_multi = simple_processor.is_multi_diagnosis(test_text)
        print(f"多诊断: {is_multi}")


if __name__ == "__main__":
    main() 