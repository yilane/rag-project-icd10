#!/usr/bin/env python3
"""
诊断文本处理器
处理多诊断文本的分割和标准化
"""

import re
from typing import List
from loguru import logger


class DiagnosisTextProcessor:
    """
    诊断文本处理器
    使用简单的分隔符分割进行诊断提取
    """
    
    def __init__(self):
        """初始化处理器"""
        # 医学分隔符模式
        self.medical_separators = [
            r'[，,；;]',  # 逗号、分号
            r'[+＋]',     # 加号
            r'\s+',       # 空格
        ]
    
    def extract_diagnoses(self, text: str) -> List[str]:
        """
        从复杂文本中提取诊断关键词
        
        Args:
            text: 输入的医疗文本
            
        Returns:
            提取的诊断关键词列表
        """
        if not text or not text.strip():
            return []
        
        logger.debug(f"开始处理文本: {text}")
        
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
        
        logger.info(f"分割提取结果: {unique_diagnoses}")
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
        """简单清理诊断文本"""
        if not text:
            return ""
        
        # 去除首尾空白
        text = text.strip()
        
        # 去除常见的无意义前后缀
        prefixes_to_remove = ['待查', '考虑', '疑似', '排除', '？', '?', '诊断为', '患者']
        suffixes_to_remove = ['待查', '？', '?', '诊断']
        
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


def main():
    """测试函数"""
    processor = DiagnosisTextProcessor()
    
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
        
        diagnoses = processor.extract_diagnoses(test_text)
        print(f"提取结果: {diagnoses}")
        
        is_multi = processor.is_multi_diagnosis(test_text)
        print(f"多诊断: {is_multi}")


if __name__ == "__main__":
    main() 