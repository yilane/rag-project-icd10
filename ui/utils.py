"""
UI工具函数模块
"""
import pandas as pd
from typing import Dict, List, Any, Tuple
import json
import re


def format_candidates_for_display(candidates: List[Dict]) -> pd.DataFrame:
    """
    将候选结果格式化为DataFrame用于显示
    
    Args:
        candidates: 候选结果列表
        
    Returns:
        格式化的DataFrame
    """
    if not candidates:
        return pd.DataFrame(columns=['ICD编码', '诊断名称', '相似度', '层级', '父节点'])
    
    display_data = []
    for candidate in candidates:
        display_data.append({
            'ICD编码': candidate.get('code', ''),
            '诊断名称': candidate.get('title', ''),
            '相似度': f"{candidate.get('score', 0):.4f}",
            '层级': candidate.get('level', ''),
            '父节点': candidate.get('parent_code', '')
        })
    
    return pd.DataFrame(display_data)


def format_entities_for_display(entities_data: Dict) -> Tuple[pd.DataFrame, str]:
    """
    将实体识别结果格式化为DataFrame和统计信息
    
    Args:
        entities_data: 实体识别结果
        
    Returns:
        (实体DataFrame, 统计信息字符串)
    """
    entities = entities_data.get('entities', {})
    
    if not entities:
        return pd.DataFrame(columns=['实体文本', '实体类型', '置信度']), "未识别到任何医学实体"
    
    # 展开实体数据
    display_data = []
    entity_stats = {}
    
    for entity_type, entity_list in entities.items():
        entity_stats[entity_type] = len(entity_list)
        for entity in entity_list:
            display_data.append({
                '实体文本': entity.get('text', ''),
                '实体类型': entity_type,
                '置信度': f"{entity.get('confidence', 0):.4f}"
            })
    
    df = pd.DataFrame(display_data)
    
    # 生成统计信息
    stats_lines = [f"**实体识别统计:**"]
    for entity_type, count in entity_stats.items():
        stats_lines.append(f"- {entity_type}: {count} 个")
    stats_lines.append(f"- **总计**: {len(display_data)} 个实体")
    
    stats_text = "\n".join(stats_lines)
    
    return df, stats_text


def format_multi_diagnosis_info(query_result: Dict) -> str:
    """
    格式化多诊断识别信息
    
    Args:
        query_result: 查询结果
        
    Returns:
        格式化的多诊断信息
    """
    is_multi = query_result.get('is_multi_diagnosis', False)
    extracted_diagnoses = query_result.get('extracted_diagnoses', [])
    diagnosis_matches = query_result.get('diagnosis_matches', [])
    
    info_lines = []
    
    if is_multi:
        info_lines.append("🔍 **检测到多诊断文本**")
        info_lines.append(f"✅ 识别出 **{len(extracted_diagnoses)}** 个诊断项:")
        
        for i, diagnosis in enumerate(extracted_diagnoses, 1):
            info_lines.append(f"  {i}. {diagnosis}")
        
        # 添加置信度信息
        if diagnosis_matches:
            info_lines.append("\n📊 **各诊断置信度:**")
            for match in diagnosis_matches:
                diagnosis_text = match.get('diagnosis_text', '')
                confidence = match.get('match_confidence', 0)
                confidence_level = match.get('confidence_level', '未知')
                info_lines.append(f"- **{diagnosis_text}**: {confidence:.3f} ({confidence_level})")
    else:
        info_lines.append("📋 **单诊断文本**")
        info_lines.append("系统将进行标准的相似度匹配")
    
    return "\n".join(info_lines)


def format_multi_diagnosis_candidates(query_result: Dict) -> Tuple[str, List[Tuple[str, pd.DataFrame]]]:
    """
    格式化多诊断情况下的候选结果，返回多个分组表格
    
    Args:
        query_result: 查询结果
        
    Returns:
        (多诊断信息字符串, [(诊断标题, 候选表格DataFrame)列表])
    """
    is_multi = query_result.get('is_multi_diagnosis', False)
    
    if not is_multi:
        # 单诊断情况，返回原逻辑
        candidates_df = format_candidates_for_display(query_result.get('candidates', []))
        return format_multi_diagnosis_info(query_result), [("", candidates_df)]
    
    # 多诊断情况
    info_text = format_multi_diagnosis_info(query_result)
    diagnosis_matches = query_result.get('diagnosis_matches', [])
    
    diagnosis_tables = []
    for match in diagnosis_matches:
        diagnosis_text = match.get('diagnosis_text', '')
        confidence = match.get('match_confidence', 0)
        confidence_level = match.get('confidence_level', '未知')
        candidates = match.get('candidates', [])
        
        # 创建诊断标题（包含诊断文本和置信度）
        diagnosis_title = f"{diagnosis_text}  {confidence:.3f} ({confidence_level})"
        
        # 格式化候选结果表格
        candidates_df = format_candidates_for_display(candidates)
        
        # 高亮显示显著的相似度分数
        if not candidates_df.empty:
            candidates_df['相似度'] = candidates_df['相似度'].apply(
                lambda x: highlight_score_significance(float(x))
            )
        
        diagnosis_tables.append((diagnosis_title, candidates_df))
    
    return info_text, diagnosis_tables


def format_standardization_result(result: Dict) -> Tuple[str, str]:
    """
    格式化标准化结果
    
    Args:
        result: 标准化结果
        
    Returns:
        (标准化文本, LLM推理过程)
    """
    standardized_text = result.get('standardized_text', '暂无标准化结果')
    llm_reasoning = result.get('llm_reasoning', '暂无推理过程')
    
    # 格式化推理过程
    reasoning_formatted = f"""
## LLM推理分析

{llm_reasoning}

---
*由 {result.get('llm_provider', 'LLM')} 提供*
    """.strip()
    
    return standardized_text, reasoning_formatted



def format_error_message(result: Dict) -> str:
    """
    格式化错误消息
    
    Args:
        result: API返回结果
        
    Returns:
        格式化的错误消息
    """
    if 'error' not in result:
        return ""
    
    error = result['error']
    
    if 'connected' in result and not result['connected']:
        return f"❌ **连接错误**: {error}\n\n请检查FastAPI服务是否正在运行 (默认地址: http://localhost:8000)"
    
    if 'timeout' in result:
        return f"⏱️ **请求超时**: {error}\n\n请检查网络连接或稍后重试"
    
    if 'status_code' in result:
        return f"🚫 **HTTP错误**: {error}\n\n请检查API服务状态"
    
    return f"⚠️ **处理错误**: {error}"


def create_example_texts():
    """创建示例文本数据"""
    return {
        'entity_examples': [
            "急性心肌梗死伴左心室功能不全，患者服用阿司匹林治疗",
            "慢性肾功能不全，血肌酐升高，建议限制蛋白质摄入",
            "2型糖尿病血糖控制不佳，需要调整胰岛素剂量"
        ],
        'query_examples': [
            "急性胃肠炎",
            "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳",
            "高血压病 糖尿病 冠状动脉粥样硬化性心脏病"
        ],
        'standardize_examples': [
            "疑似埃尔托霍乱爆发，伴有急性胃肠炎症状",
            "患者出现类似感冒的症状，可能是上呼吸道感染",
            "腹痛腹泻，怀疑是食物中毒引起的急性胃肠炎"
        ],
    }


def highlight_score_significance(score: float) -> str:
    """
    为分数添加显著性说明
    
    Args:
        score: 相似度分数
        
    Returns:
        带说明的分数字符串
    """
    if score > 1.0:
        return f"{score:.4f} 🔥 (层级加权)"
    elif score > 0.8:
        return f"{score:.4f} ✅ (高相似度)"
    elif score > 0.6:
        return f"{score:.4f} ⚠️ (中等相似度)"
    else:
        return f"{score:.4f} ❓ (低相似度)"


def format_multi_diagnosis_standardization(result: Dict) -> Tuple[str, List[Tuple[str, str, str, pd.DataFrame]]]:
    """
    格式化多诊断标准化结果，返回分组显示数据
    
    Args:
        result: 标准化结果
        
    Returns:
        (多诊断信息字符串, [(诊断标题, 标准化结果, LLM推理, 候选表格DataFrame)列表])
    """
    if not result or not isinstance(result, list) or len(result) == 0:
        return "处理结果为空", []
    
    main_result = result[0]
    is_multi = main_result.get('is_multi_diagnosis', False)
    
    if not is_multi:
        # 单诊断情况
        standardized_results = main_result.get('standardized_results', [])
        candidates = main_result.get('candidates', [])
        
        if standardized_results:
            std_result = standardized_results[0]
            standardized_text = f"{std_result.get('title', '')} ({std_result.get('code', '')})"
            llm_reasoning = f"标准化置信度: {std_result.get('confidence', 0):.3f}"
        else:
            standardized_text = "暂无标准化结果"
            llm_reasoning = "LLM处理失败"
        
        candidates_df = format_candidates_for_display(candidates)
        
        info_text = "📋 **单诊断标准化**\n系统对整个诊断文本进行了标准化处理。"
        
        return info_text, [("", standardized_text, llm_reasoning, candidates_df)]
    
    # 多诊断情况
    extracted_diagnoses = main_result.get('extracted_diagnoses', [])
    standardization_groups = main_result.get('standardization_groups', [])
    
    info_lines = []
    info_lines.append("🔍 **检测到多诊断文本**")
    info_lines.append(f"✅ 识别出 **{len(extracted_diagnoses)}** 个诊断项:")
    
    for i, diagnosis in enumerate(extracted_diagnoses, 1):
        info_lines.append(f"  {i}. {diagnosis}")
    
    info_lines.append(f"\n📊 **各诊断标准化结果:**")
    
    info_text = "\n".join(info_lines)
    
    # 构建分组数据
    diagnosis_groups = []
    for group in standardization_groups:
        diagnosis_text = group.get('diagnosis_text', '')
        match_confidence = group.get('match_confidence', 0)
        confidence_level = group.get('confidence_level', '未知')
        standardized_results = group.get('standardized_results', [])
        candidates = group.get('candidates', [])
        
        # 创建诊断标题（包含诊断文本和置信度）
        diagnosis_title = f"{diagnosis_text}  {match_confidence:.3f} ({confidence_level})"
        
        # 处理标准化结果
        if standardized_results and len(standardized_results) > 0:
            std_result = standardized_results[0]
            standardized_text = f"{std_result.get('title', '')} ({std_result.get('code', '')})"
            
            # 构建LLM推理信息
            reasoning_lines = [
                f"**标准化结果**: {std_result.get('title', '')}",
                f"**ICD编码**: {std_result.get('code', '')}",
                f"**LLM置信度**: {std_result.get('confidence', 0):.3f}",
                f"**原始诊断**: {std_result.get('diagnosis', diagnosis_text)}"
            ]
            
            if len(standardized_results) > 1:
                reasoning_lines.append(f"\n**其他可能结果**:")
                for i, alt_result in enumerate(standardized_results[1:], 2):
                    reasoning_lines.append(f"  {i}. {alt_result.get('title', '')} ({alt_result.get('code', '')})")
            
            llm_reasoning = "\n".join(reasoning_lines)
        else:
            standardized_text = "暂无标准化结果"
            llm_reasoning = "LLM处理失败或未返回结果"
        
        # 格式化候选结果表格
        candidates_df = format_candidates_for_display(candidates)
        
        # 高亮显示显著的相似度分数
        if not candidates_df.empty:
            candidates_df['相似度'] = candidates_df['相似度'].apply(
                lambda x: highlight_score_significance(float(x))
            )
        
        diagnosis_groups.append((diagnosis_title, standardized_text, llm_reasoning, candidates_df))
    
    return info_text, diagnosis_groups


def generate_standardization_html(diagnosis_groups: list) -> str:
    """
    生成标准化结果的HTML显示内容
    
    Args:
        diagnosis_groups: [(诊断标题, 标准化结果, LLM推理, 候选表格DataFrame)列表]
        
    Returns:
        HTML字符串
    """
    if not diagnosis_groups:
        return "<p>暂无标准化结果</p>"
    
    html_parts = []
    
    for diagnosis_title, standardized_text, llm_reasoning, candidates_df in diagnosis_groups:
        if diagnosis_title.strip():  # 多诊断情况
            html_parts.append(f"""
            <div style="margin: 25px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #f8f9fa;">
                <h3 style="margin: 0 0 15px 0; color: #333; font-size: 18px; border-bottom: 2px solid #28a745; padding-bottom: 10px;">
                    {diagnosis_title}
                </h3>
                
                <div style="margin-bottom: 15px; padding: 15px; background-color: #e8f5e8; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h4 style="margin: 0 0 10px 0; color: #155724; font-size: 16px;">📋 标准化结果</h4>
                    <p style="margin: 0; font-size: 16px; font-weight: bold; color: #155724;">{standardized_text}</p>
                </div>
                
                <div style="margin-bottom: 15px; padding: 15px; background-color: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196f3;">
                    <h4 style="margin: 0 0 10px 0; color: #0d47a1; font-size: 16px;">🧠 LLM推理过程</h4>
                    <div style="font-size: 14px; line-height: 1.6; color: #0d47a1; white-space: pre-line;">{llm_reasoning}</div>
                </div>
                
                <div style="margin-bottom: 0;">
                    <h4 style="margin: 0 0 10px 0; color: #333; font-size: 16px;">📊 推荐的ICD编码</h4>
                    {candidates_df.to_html(index=False, classes='table table-striped', escape=False, table_id=None)}
                </div>
            </div>
            """)
        else:  # 单诊断情况
            html_parts.append(f"""
            <div style="margin: 25px 0;">
                <div style="margin-bottom: 15px; padding: 15px; background-color: #e8f5e8; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h4 style="margin: 0 0 10px 0; color: #155724; font-size: 16px;">📋 标准化结果</h4>
                    <p style="margin: 0; font-size: 16px; font-weight: bold; color: #155724;">{standardized_text}</p>
                </div>
                
                <div style="margin-bottom: 15px; padding: 15px; background-color: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196f3;">
                    <h4 style="margin: 0 0 10px 0; color: #0d47a1; font-size: 16px;">🧠 LLM推理过程</h4>
                    <div style="font-size: 14px; line-height: 1.6; color: #0d47a1; white-space: pre-line;">{llm_reasoning}</div>
                </div>
                
                <div style="margin-bottom: 0;">
                    <h4 style="margin: 0 0 10px 0; color: #333; font-size: 16px;">📊 推荐的ICD编码</h4>
                    {candidates_df.to_html(index=False, classes='table table-striped', escape=False, table_id=None)}
                </div>
            </div>
            """)
    
    # 添加CSS样式
    style = """
    <style>
    .table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin: 10px 0;
    }
    .table th {
        background-color: #f8f9fa;
        font-weight: bold;
        padding: 12px 8px;
        text-align: left;
        border: 1px solid #dee2e6;
    }
    .table td {
        padding: 10px 8px;
        border: 1px solid #dee2e6;
        vertical-align: top;
    }
    .table-striped tbody tr:nth-of-type(odd) {
        background-color: rgba(0, 0, 0, 0.05);
    }
    </style>
    """
    
    return style + "".join(html_parts)