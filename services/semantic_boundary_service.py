#!/usr/bin/env python3
"""
语义边界检测服务
用于智能检测医学文本中的诊断边界
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


class SemanticBoundaryDetector:
    """语义边界检测器"""
    
    def __init__(self, embedding_service=None):
        """
        初始化语义边界检测器
        
        Args:
            embedding_service: 嵌入服务实例
        """
        self.embedding_service = embedding_service
        self.semantic_threshold = 0.75  # 语义相似度阈值
        self.min_segment_length = 2     # 最小段落长度
        
        # 医学分隔符优先级（按分割强度排序）
        self.delimiter_priority = {
            '；': 1, ';': 1,     # 分号 - 强分隔，医学文本中常用于分隔不同诊断
            '。': 2, '.': 2,     # 句号 - 强分隔  
            '，': 3, ',': 3,     # 逗号 - 中等分隔
            '\n': 4,             # 换行 - 很强分隔
            '+': 5, '＋': 5,      # 加号 - 医学连接符
            ' ': 6, '\t': 6,     # 空格制表符 - 弱分隔
        }
        
        # 连接词模式（表示不应该分割的情况）
        self.connection_patterns = [
            r'伴?有?(?:并发|合并)',    # 伴有并发、合并
            r'(?:继发|导致|引起)',      # 继发、导致
            r'(?:急性|慢性)加重',       # 急性加重
            r'(?:病史|既往史)',         # 病史
            r'(?:术后|治疗后)',         # 术后状态
        ]
    
    def detect_diagnosis_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """
        检测诊断边界
        
        Args:
            text: 输入文本
            
        Returns:
            边界列表，每个元素为(start, end, diagnosis_text)
        """
        if not text or not text.strip():
            return []
        
        logger.debug(f"开始检测诊断边界: {text}")
        
        # 1. 初步分割
        initial_segments = self._initial_segmentation(text)
        
        if len(initial_segments) <= 1:
            return [(0, len(text), text.strip())]
        
        # 2. 语义聚合（如果有嵌入服务）
        if self.embedding_service:
            try:
                semantic_groups = self._semantic_clustering(initial_segments)
            except Exception as e:
                logger.warning(f"语义聚合失败，使用初步分割结果: {e}")
                semantic_groups = [[seg] for seg in initial_segments]
        else:
            logger.info("无嵌入服务，跳过语义聚合")
            semantic_groups = [[seg] for seg in initial_segments]
        
        # 3. 边界优化
        optimized_boundaries = self._optimize_boundaries(semantic_groups, text)
        
        logger.info(f"检测到 {len(optimized_boundaries)} 个诊断边界")
        return optimized_boundaries
    
    def _initial_segmentation(self, text: str) -> List[Dict[str, Any]]:
        """初步分割文本"""
        segments = []
        
        # 按分隔符优先级分割
        for delimiter, priority in sorted(self.delimiter_priority.items(), key=lambda x: x[1]):
            if delimiter in text:
                parts = text.split(delimiter)
                if len(parts) > 1:
                    segments = []
                    pos = 0
                    
                    for i, part in enumerate(parts):
                        part = part.strip()
                        if part and len(part) >= self.min_segment_length:
                            start_pos = text.find(part, pos)
                            end_pos = start_pos + len(part)
                            
                            segment = {
                                'text': part,
                                'start': start_pos,
                                'end': end_pos,
                                'delimiter': delimiter,
                                'priority': priority
                            }
                            segments.append(segment)
                            pos = end_pos
                    
                    if segments and len(segments) > 1:  # 确保有多个分段
                        logger.debug(f"使用分隔符 '{delimiter}' 分割得到 {len(segments)} 个段落")
                        break
        
        # 如果没有找到分隔符，返回整个文本
        if not segments:
            segments = [{
                'text': text.strip(),
                'start': 0,
                'end': len(text),
                'delimiter': None,
                'priority': 0
            }]
        
        # 过滤连接词情况
        segments = self._filter_connection_cases(segments, text)
        
        logger.debug(f"初步分割得到 {len(segments)} 个段落")
        # 记录分割详情
        for i, segment in enumerate(segments[:5]):  # 只记录前5个
            logger.debug(f"  段落 {i+1}: [{segment['start']}:{segment['end']}] {segment['text'][:50]}...")
        
        return segments
    
    def _filter_connection_cases(self, segments: List[Dict], text: str) -> List[Dict]:
        """过滤不应该分割的连接词情况"""
        filtered_segments = []
        
        for segment in segments:
            should_keep = True
            
            # 检查是否包含连接词模式
            for pattern in self.connection_patterns:
                if re.search(pattern, segment['text']):
                    # 如果包含连接词，可能需要与前后段落合并
                    should_keep = False
                    break
            
            if should_keep:
                filtered_segments.append(segment)
            else:
                # 尝试与前一个段落合并
                if filtered_segments:
                    prev_segment = filtered_segments[-1]
                    merged_text = prev_segment['text'] + ' ' + segment['text']
                    
                    merged_segment = {
                        'text': merged_text,
                        'start': prev_segment['start'],
                        'end': segment['end'],
                        'delimiter': segment['delimiter'],
                        'priority': min(prev_segment['priority'], segment['priority'])
                    }
                    filtered_segments[-1] = merged_segment
                else:
                    # 如果是第一个段落，保留
                    filtered_segments.append(segment)
        
        return filtered_segments
    
    def _semantic_clustering(self, segments: List[Dict]) -> List[List[str]]:
        """基于语义相似度的聚类"""
        if len(segments) <= 1:
            return [[seg['text']] for seg in segments]
        
        try:
            # 获取段落文本
            segment_texts = [seg['text'] for seg in segments]
            
            # 生成嵌入向量
            embeddings = []
            for text in segment_texts:
                embedding = self.embedding_service.encode_query(text)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(embeddings)
            
            # 转换为距离矩阵（聚类算法需要）
            distance_matrix = 1 - similarity_matrix
            
            # 层次聚类 - 保守策略，减少不合理的合并
            # 对于医学文本，我们希望保持更多的分割，避免将不同诊断合并
            n_clusters = len(segments)  # 保持原有分段数，不进行合并
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # 根据聚类结果分组
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(segment_texts[i])
            
            # 转换为列表格式
            semantic_groups = list(clusters.values())
            
            logger.debug(f"语义聚类得到 {len(semantic_groups)} 个语义组")
            return semantic_groups
            
        except Exception as e:
            logger.error(f"语义聚类失败: {e}")
            # 失败时返回原始分割
            return [[seg['text']] for seg in segments]
    
    def _optimize_boundaries(self, semantic_groups: List[List[str]], original_text: str) -> List[Tuple[int, int, str]]:
        """优化边界位置"""
        boundaries = []
        current_pos = 0
        
        for group in semantic_groups:
            # 合并组内文本
            group_text = ' '.join(group).strip()
            
            if not group_text:
                continue
            
            # 在原文中查找位置
            start_pos = original_text.find(group_text, current_pos)
            if start_pos == -1:
                # 如果直接查找失败，尝试查找组内第一个段落
                first_segment = group[0].strip()
                start_pos = original_text.find(first_segment, current_pos)
                if start_pos == -1:
                    start_pos = current_pos
                
                # 计算结束位置
                end_pos = start_pos + len(group_text)
                if end_pos > len(original_text):
                    end_pos = len(original_text)
            else:
                end_pos = start_pos + len(group_text)
            
            boundaries.append((start_pos, end_pos, group_text))
            current_pos = end_pos + 1
        
        # 如果没有找到边界，返回整个文本
        if not boundaries:
            boundaries = [(0, len(original_text), original_text.strip())]
        
        return boundaries
    
    def get_boundary_confidence(self, boundaries: List[Tuple[int, int, str]]) -> List[float]:
        """计算边界置信度"""
        confidences = []
        
        for i, (start, end, text) in enumerate(boundaries):
            confidence = 0.5  # 基础置信度
            
            # 长度因子
            if len(text) >= 4:
                confidence += 0.1
            if len(text) >= 8:
                confidence += 0.1
            
            # 完整性因子（包含完整的医学术语）
            if re.search(r'[^，。；\s]{2,}(?:病|症|炎|癌|瘤)', text):
                confidence += 0.2
            
            # 独立性因子（不依赖其他段落的语义）
            if not re.search(r'(?:伴有|合并|继发)', text):
                confidence += 0.1
            
            # 边界清晰度因子
            if i < len(boundaries) - 1:  # 不是最后一个边界
                next_text = boundaries[i + 1][2]
                if self.embedding_service:
                    try:
                        # 计算与下一个段落的语义距离
                        emb1 = self.embedding_service.encode_query(text)
                        emb2 = self.embedding_service.encode_query(next_text)
                        similarity = cosine_similarity([emb1], [emb2])[0][0]
                        
                        if similarity < self.semantic_threshold:
                            confidence += 0.1  # 语义距离大，边界清晰
                    except:
                        pass
            
            confidences.append(min(confidence, 1.0))
        
        return confidences
    
    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """分析文本结构"""
        boundaries = self.detect_diagnosis_boundaries(text)
        confidences = self.get_boundary_confidence(boundaries)
        
        analysis = {
            'original_text': text,
            'total_boundaries': len(boundaries),
            'boundaries': [
                {
                    'text': boundary[2],
                    'start': boundary[0],
                    'end': boundary[1],
                    'confidence': conf,
                    'length': len(boundary[2])
                }
                for boundary, conf in zip(boundaries, confidences)
            ],
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'is_multi_diagnosis': len(boundaries) > 1
        }
        
        return analysis


def main():
    """测试函数"""
    # 模拟嵌入服务（实际使用时需要真实的嵌入服务）
    class MockEmbeddingService:
        def encode_query(self, text):
            # 简单的词频向量模拟
            words = text.split()
            vector = np.random.rand(10)  # 简化的10维向量
            return vector
    
    mock_embedding = MockEmbeddingService()
    detector = SemanticBoundaryDetector(mock_embedding)
    
    test_cases = [
        "急性心肌梗死 高血压病 糖尿病",
        "慢性肾功能不全伴蛋白尿，高血压病3级，2型糖尿病血糖控制不佳",
        "疑似急性胃肠炎，患者有发热腹泻症状",
        "左肺上叶肺癌，胸腔积液，呼吸困难",
        "冠状动脉粥样硬化性心脏病伴心功能不全",
    ]
    
    for test_text in test_cases:
        print(f"\n=== 测试文本: {test_text} ===")
        
        # 检测边界
        boundaries = detector.detect_diagnosis_boundaries(test_text)
        print(f"检测到 {len(boundaries)} 个边界:")
        for i, (start, end, text) in enumerate(boundaries):
            print(f"  {i+1}. [{start}:{end}] {text}")
        
        # 分析结构
        analysis = detector.analyze_text_structure(test_text)
        print(f"平均置信度: {analysis['avg_confidence']:.3f}")
        print(f"多诊断: {analysis['is_multi_diagnosis']}")


if __name__ == "__main__":
    main()