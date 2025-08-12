"""
智能诊断查询Tab页面
"""
import gradio as gr
import pandas as pd
from typing import Tuple
from .api_client import api_client
from .utils import (
    format_candidates_for_display, 
    format_multi_diagnosis_info,
    format_multi_diagnosis_candidates,
    format_error_message, 
    create_example_texts,
    highlight_score_significance
)


def generate_candidates_html(diagnosis_tables: list) -> str:
    """
    生成候选结果的HTML显示内容
    
    Args:
        diagnosis_tables: [(诊断标题, 候选表格DataFrame)列表]
        
    Returns:
        HTML字符串
    """
    if not diagnosis_tables:
        return "<p>暂无匹配结果</p>"
    
    html_parts = []
    
    for diagnosis_title, candidates_df in diagnosis_tables:
        if diagnosis_title.strip():  # 多诊断情况
            html_parts.append(f"""
            <div style="margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f8f9fa;">
                <h4 style="margin: 0 0 15px 0; color: #333; font-size: 16px; border-bottom: 2px solid #007bff; padding-bottom: 8px;">
                    {diagnosis_title}
                </h4>
                {candidates_df.to_html(index=False, classes='table table-striped', escape=False, table_id=None)}
            </div>
            """)
        else:  # 单诊断情况
            html_parts.append(f"""
            <div style="margin: 20px 0;">
                {candidates_df.to_html(index=False, classes='table table-striped', escape=False, table_id=None)}
            </div>
            """)
    
    # 添加CSS样式
    style = """
    <style>
    .table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin: 0;
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


def create_query_tab():
    """创建智能诊断查询Tab"""
    
    examples = create_example_texts()
    
    with gr.TabItem("🔍 智能诊断查询", id="query"):
        gr.Markdown("""
        ## 智能诊断查询
        
        基于语义相似度的ICD编码查询，支持单诊断和多诊断自动识别。
        系统会自动过滤非诊断实体，并提供层级权重优化的相似度评分。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                gr.Markdown("### 📝 输入诊断文本")
                
                input_text = gr.Textbox(
                    label="诊断文本",
                    placeholder="请输入诊断文本，支持单个或多个诊断（如：急性胃肠炎，或：高血压病 糖尿病 冠心病）",
                    lines=3,
                    max_lines=6
                )
                
                # 参数设置
                with gr.Row():
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="返回结果数量 (Top-K)",
                        info="设置返回的候选结果数量"
                    )
                
                enhanced_processing = gr.Checkbox(
                    label="启用增强处理",
                    value=True,
                    info="启用医学NER和语义边界检测等增强功能"
                )
                
                # 控制按钮
                with gr.Row():
                    query_btn = gr.Button("🔍 查询诊断", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 清空", scale=1)
                
                # 示例文本
                gr.Markdown("### 📋 示例文本")
                with gr.Row():
                    for i, example in enumerate(examples['query_examples']):
                        gr.Button(
                            f"示例 {i+1}",
                            size="sm"
                        ).click(
                            fn=lambda x=example: x,
                            outputs=[input_text]
                        )
            
            with gr.Column(scale=2):
                # 结果展示区域
                gr.Markdown("### 📊 查询结果")
                
                # 错误信息显示
                error_msg = gr.Markdown(visible=False)
                
                # 多诊断识别信息
                multi_diagnosis_info = gr.Markdown("等待查询...", visible=True)
                
                # 候选结果显示区域（支持多诊断分组显示）
                candidates_display = gr.HTML(label="匹配的ICD候选结果", visible=True)
                
                # 详细JSON结果（可折叠）
                with gr.Accordion("🔍 详细查询结果", open=False):
                    json_output = gr.JSON(label="完整API响应")
        
        # 事件处理函数
        def query_diagnosis_handler(text: str, top_k_value: int, enhanced: bool) -> Tuple[str, str, str, dict]:
            """处理诊断查询请求"""
            if not text.strip():
                return (
                    "", 
                    "⚠️ 请输入要查询的诊断文本",
                    "",
                    {}
                )
            
            # 调用API
            result = api_client.query_diagnosis(text.strip(), int(top_k_value), enhanced)
            
            # 检查错误
            if 'error' in result:
                error_message = format_error_message(result)
                return (
                    "",
                    error_message,
                    "",
                    result
                )
            
            # 使用新的多诊断格式化逻辑
            multi_info, diagnosis_tables = format_multi_diagnosis_candidates(result)
            
            # 生成HTML显示内容
            candidates_html = generate_candidates_html(diagnosis_tables)
            
            return (
                multi_info,
                "",  # 清空错误信息
                candidates_html,
                result
            )
        
        def clear_all() -> Tuple[str, int, bool, str, str, dict]:
            """清空所有输入和输出"""
            return (
                "",  # input_text
                5,   # top_k (重置为默认值)
                True,  # enhanced_processing (重置为默认值)
                "等待查询...",  # multi_diagnosis_info
                "",  # candidates_display
                {}  # json_output
            )
        
        # 绑定事件
        query_btn.click(
            fn=query_diagnosis_handler,
            inputs=[input_text, top_k, enhanced_processing],
            outputs=[multi_diagnosis_info, error_msg, candidates_display, json_output]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[input_text, top_k, enhanced_processing, multi_diagnosis_info, candidates_display, json_output]
        )
    
    return {
        'input_text': input_text,
        'top_k': top_k,
        'enhanced_processing': enhanced_processing,
        'query_btn': query_btn,
        'clear_btn': clear_btn,
        'multi_diagnosis_info': multi_diagnosis_info,
        'candidates_display': candidates_display,
        'json_output': json_output,
        'error_msg': error_msg
    }