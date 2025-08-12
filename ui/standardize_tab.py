"""
诊断标准化Tab页面
"""
import gradio as gr
import pandas as pd
from typing import Tuple
from .api_client import api_client
from .utils import (
    format_candidates_for_display,
    format_standardization_result,
    format_multi_diagnosis_standardization,
    generate_standardization_html,
    format_error_message, 
    create_example_texts
)


def create_standardize_tab():
    """创建诊断标准化Tab"""
    
    examples = create_example_texts()
    
    with gr.TabItem("🤖 诊断标准化", id="standardize"):
        gr.Markdown("""
        ## 诊断标准化
        
        使用大语言模型（LLM）进行智能诊断标准化，将非标准的诊断描述转化为规范的医学术语。
        系统会先进行向量检索获取相关ICD编码，然后由LLM进行智能分析和标准化处理。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                gr.Markdown("### 📝 输入非标准诊断")
                
                input_text = gr.Textbox(
                    label="诊断描述",
                    placeholder="请输入需要标准化的诊断描述，如：疑似埃尔托霍乱爆发，伴有急性胃肠炎症状",
                    lines=4,
                    max_lines=8
                )
                
                # 参数设置
                with gr.Row():
                    llm_provider = gr.Dropdown(
                        choices=["deepseek", "openai", "local"],
                        value="deepseek",
                        label="LLM提供商",
                        info="选择用于标准化的大语言模型"
                    )
                
                with gr.Row():
                    top_k = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        label="检索结果数量",
                        info="为LLM提供的参考ICD编码数量"
                    )
                
                # 控制按钮
                with gr.Row():
                    standardize_btn = gr.Button("🤖 开始标准化", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 清空", scale=1)
                
                # 示例文本
                gr.Markdown("### 📋 示例文本")
                with gr.Row():
                    for i, example in enumerate(examples['standardize_examples']):
                        gr.Button(
                            f"示例 {i+1}",
                            size="sm"
                        ).click(
                            fn=lambda x=example: x,
                            outputs=[input_text]
                        )
            
            with gr.Column(scale=2):
                # 结果展示区域
                gr.Markdown("### 📊 标准化结果")
                
                # 错误信息显示
                error_msg = gr.Markdown(visible=False)
                
                # 多诊断识别信息
                multi_diagnosis_info = gr.Markdown("等待处理...", visible=True)
                
                # 分组标准化结果显示区域（支持多诊断分组显示）
                standardization_display = gr.HTML(label="分组标准化结果", visible=True)
                
                # 详细JSON结果（可折叠）
                with gr.Accordion("🔍 详细标准化结果", open=False):
                    json_output = gr.JSON(label="完整API响应")
        
        # 事件处理函数
        def standardize_diagnosis_handler(text: str, provider: str, top_k_value: int) -> Tuple[str, str, str, dict]:
            """处理诊断标准化请求"""
            if not text.strip():
                return (
                    "", 
                    "⚠️ 请输入要标准化的诊断文本",
                    "",
                    {}
                )
            
            # 调用API (增加超时时间，因为LLM处理可能较慢)
            result = api_client.standardize_diagnosis(text.strip(), provider, int(top_k_value))
            
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
            multi_info, diagnosis_groups = format_multi_diagnosis_standardization(result.get('results', []))
            
            # 生成HTML显示内容
            standardization_html = generate_standardization_html(diagnosis_groups)
            
            return (
                multi_info,
                "",  # 清空错误信息
                standardization_html,
                result
            )
        
        def clear_all() -> Tuple[str, str, int, str, str, dict]:
            """清空所有输入和输出"""
            return (
                "",  # input_text
                "deepseek",  # llm_provider (重置为默认值)
                10,  # top_k (重置为默认值)
                "等待处理...",  # multi_diagnosis_info
                "",  # standardization_display
                {}  # json_output
            )
        
        # 绑定事件
        standardize_btn.click(
            fn=standardize_diagnosis_handler,
            inputs=[input_text, llm_provider, top_k],
            outputs=[multi_diagnosis_info, error_msg, standardization_display, json_output]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[input_text, llm_provider, top_k, multi_diagnosis_info, standardization_display, json_output]
        )
    
    return {
        'input_text': input_text,
        'llm_provider': llm_provider,
        'top_k': top_k,
        'standardize_btn': standardize_btn,
        'clear_btn': clear_btn,
        'multi_diagnosis_info': multi_diagnosis_info,
        'standardization_display': standardization_display,
        'json_output': json_output,
        'error_msg': error_msg
    }