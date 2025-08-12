"""
医学命名实体识别Tab页面
"""
import gradio as gr
import pandas as pd
from typing import Tuple
from .api_client import api_client
from .utils import format_entities_for_display, format_error_message, create_example_texts


def create_entities_tab():
    """创建医学命名实体识别Tab"""
    
    examples = create_example_texts()
    
    with gr.TabItem("🏷️ 医学命名实体识别", id="entities"):
        gr.Markdown("""
        ## 医学命名实体识别
        
        使用专业的中文医学NER模型识别文本中的医学实体，包括疾病、症状、药品、设备等多种类型。
        支持智能过滤功能，可选择过滤药品、设备、科室等非诊断实体，专注于诊断相关信息。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                gr.Markdown("### 📝 输入医学文本")
                
                input_text = gr.Textbox(
                    label="医学文本",
                    placeholder="请输入包含医学术语的文本，如：急性心肌梗死伴左心室功能不全，患者服用阿司匹林治疗",
                    lines=4,
                    max_lines=8
                )
                
                filter_drugs = gr.Checkbox(
                    label="过滤非诊断实体",
                    value=True,
                    info="勾选后将过滤药品、设备、科室等非诊断实体，专注于疾病、症状、实验室指标等诊断相关信息"
                )
                
                # 控制按钮
                with gr.Row():
                    extract_btn = gr.Button("🔍 提取实体", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 清空", scale=1)
                
                # 示例文本
                gr.Markdown("### 📋 示例文本")
                with gr.Row():
                    for i, example in enumerate(examples['entity_examples']):
                        gr.Button(
                            f"示例 {i+1}",
                            size="sm"
                        ).click(
                            fn=lambda x=example: x,
                            outputs=[input_text]
                        )
            
            with gr.Column(scale=2):
                # 结果展示区域
                gr.Markdown("### 📊 识别结果")
                
                # 错误信息显示
                error_msg = gr.Markdown(visible=False)
                
                # 实体统计信息
                stats_info = gr.Markdown("等待处理...", visible=True)
                
                # 实体列表表格
                entities_table = gr.DataFrame(
                    headers=['实体文本', '实体类型', '置信度'],
                    datatype=['str', 'str', 'str'],
                    label="识别出的实体列表",
                    interactive=False,
                    wrap=True
                )
                
                # 详细JSON结果（可折叠）
                with gr.Accordion("🔍 详细识别结果", open=False):
                    json_output = gr.JSON(label="完整API响应")
        
        # 事件处理函数
        def extract_entities_handler(text: str, filter_drugs_flag: bool) -> Tuple[str, str, pd.DataFrame, dict]:
            """处理实体提取请求"""
            if not text.strip():
                return (
                    "", 
                    "⚠️ 请输入要分析的医学文本",
                    pd.DataFrame(columns=['实体文本', '实体类型', '置信度']),
                    {}
                )
            
            # 调用API
            result = api_client.extract_entities(text.strip(), filter_drugs_flag)
            
            # 检查错误
            if 'error' in result:
                error_message = format_error_message(result)
                return (
                    "",
                    error_message,
                    pd.DataFrame(columns=['实体文本', '实体类型', '置信度']),
                    result
                )
            
            # 格式化结果
            entities_df, stats_text = format_entities_for_display(result)
            
            return (
                stats_text,
                "",  # 清空错误信息
                entities_df,
                result
            )
        
        def clear_all() -> Tuple[str, bool, str, pd.DataFrame, dict]:
            """清空所有输入和输出"""
            return (
                "",  # input_text
                True,  # filter_drugs (重置为默认值)
                "等待处理...",  # stats_info
                pd.DataFrame(columns=['实体文本', '实体类型', '置信度']),  # entities_table
                {}  # json_output
            )
        
        # 绑定事件
        extract_btn.click(
            fn=extract_entities_handler,
            inputs=[input_text, filter_drugs],
            outputs=[stats_info, error_msg, entities_table, json_output]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[input_text, filter_drugs, stats_info, entities_table, json_output]
        )
    
    return {
        'input_text': input_text,
        'filter_drugs': filter_drugs,
        'extract_btn': extract_btn,
        'clear_btn': clear_btn,
        'stats_info': stats_info,
        'entities_table': entities_table,
        'json_output': json_output,
        'error_msg': error_msg
    }