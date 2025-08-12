"""
ICD-10诊断标准化RAG系统 - Gradio Web UI
"""
import gradio as gr
import os
import signal
import sys
import socket
from dotenv import load_dotenv
from loguru import logger

# 加载环境变量
load_dotenv()

# 先设置API配置再导入UI模块
api_port = int(os.getenv('API_PORT', 8005))
api_base_url = f"http://localhost:{api_port}"
os.environ['API_BASE_URL'] = api_base_url

# 导入UI模块
from ui.api_client import api_client
from ui.entities_tab import create_entities_tab
from ui.query_tab import create_query_tab
from ui.standardize_tab import create_standardize_tab

# 确保API客户端使用正确的端口
api_client.base_url = api_base_url

# 注意：API服务需要手动启动
# 使用命令: uvicorn main:app --host 0.0.0.0 --port 8004 --reload


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """寻找可用端口"""
    for i in range(max_attempts):
        port = start_port + i
        try:
            # 尝试绑定端口
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    # 如果都不可用，返回原端口（让Gradio自己处理）
    return start_port


def check_api_status():
    """检查API服务状态"""
    result = api_client.test_connection()
    if result.get('connected', False):
        return "✅ **API服务运行正常** - 所有功能可用"
    else:
        error_msg = result.get('error', '未知错误')
        api_port = int(os.getenv('API_PORT', 8005))
        return f"""❌ **API服务未运行** - {error_msg}

请先启动API服务：
```bash
# 激活conda环境
conda activate rag-project-icd10

# 启动API服务
uvicorn main:app --host 0.0.0.0 --port {api_port} --reload
```

或者使用简化命令：
```bash
python main.py
```"""


def create_app():
    """创建Gradio应用"""
    
    # 自定义CSS样式
    css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
    }
    
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .error-message {
        color: #dc3545 !important;
        background-color: #f8d7da !important;
        border: 1px solid #f5c6cb !important;
        padding: 10px !important;
        border-radius: 5px !important;
        margin: 10px 0 !important;
    }
    
    .success-message {
        color: #155724 !important;
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        padding: 10px !important;
        border-radius: 5px !important;
        margin: 10px 0 !important;
    }
    
    .dataframe table {
        font-size: 14px !important;
    }
    
    .dataframe th {
        background-color: #f8f9fa !important;
        font-weight: bold !important;
    }
    
    /* 进度条样式 */
    .progress {
        height: 20px !important;
        border-radius: 10px !important;
    }
    
    /* 按钮样式优化 */
    .btn-primary {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
    }
    """
    
    # 获取API端口配置
    api_host = os.getenv('API_HOST', '0.0.0.0')
    
    # 应用标题和描述
    title = "🏥 ICD-10 诊断标准化系统"
    description = f"""
    基于检索增强生成(RAG)技术的ICD-10医疗诊断智能标准化工具，支持：
    
    - 🏷️ **医学命名实体识别**: 专业NER模型识别医学实体
    - 🔍 **智能诊断查询**: 多诊断自动识别与向量相似度匹配  
    - 🤖 **诊断标准化**: LLM智能标准化非标准诊断描述
    
    **⚠️ 使用前提**: 请先手动启动API服务 (端口 {api_port})，然后刷新下方的连接状态
    """
    
    with gr.Blocks(
        title=title,
        css=css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="gray"
        )
    ) as app:
        
        # 标题和描述
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        # API连接状态检查
        with gr.Row():
            with gr.Column():
                connection_status = gr.Markdown("🔄 检查API连接状态...")
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新连接状态", size="sm")
                    auto_refresh = gr.Checkbox(label="自动刷新状态", value=True)
        
        # 主要功能Tab页面
        with gr.Tabs() as tabs:
            
            # Tab 1: 医学命名实体识别
            entities_components = create_entities_tab()
            
            # Tab 2: 智能诊断查询  
            query_components = create_query_tab()
            
            # Tab 3: 诊断标准化
            standardize_components = create_standardize_tab()
            
        
        # 页脚信息
        gr.Markdown(f"""
        ---
        **系统信息**
        - 🔗 API服务: FastAPI后端服务 (端口: {api_port}) - **需手动启动**
        - 🧠 向量模型: intfloat/multilingual-e5-large-instruct (1024维)
        - 💾 向量数据库: Milvus Lite (本地部署)
        - 🤖 LLM支持: DeepSeek/OpenAI/Local
        - 📊 数据集: ICD-10 v601 (37k+ 编码)
        
        **启动说明**:
        1. 先启动API服务: `uvicorn main:app --host 0.0.0.0 --port {api_port} --reload`
        2. 然后启动此Gradio界面: `python gradio_app.py`
        3. 确认上方API连接状态为正常后开始使用
        """)
        
        # 使用全局的检查函数
        def check_api_connection():
            """检查API连接状态"""
            return check_api_status()
        
        # 绑定连接状态检查事件
        refresh_btn.click(
            fn=check_api_connection,
            outputs=[connection_status]
        )
        
        # 应用启动时检查连接
        app.load(
            fn=check_api_connection,
            outputs=[connection_status]
        )
        
        # 添加定时器定期刷新状态（每10秒）
        timer = gr.Timer(value=10, active=True)
        timer.tick(
            fn=lambda: check_api_status() if True else None,  # 简化版本，总是检查
            outputs=[connection_status]
        )
    
    return app


def main():
    """主函数"""
    logger.info("🚀 启动ICD-10诊断标准化系统 Gradio UI")
    
    # 从环境变量获取配置
    host = os.getenv('GRADIO_HOST', '0.0.0.0')
    requested_port = int(os.getenv('GRADIO_PORT', 7860))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # 寻找可用端口
    port = find_available_port(requested_port)
    if port != requested_port:
        logger.warning(f"⚠️  端口 {requested_port} 被占用，使用端口 {port}")
    
    logger.info(f"📡 API服务地址: {api_base_url} (需手动启动)")
    logger.info(f"🌐 Gradio UI地址: http://{host}:{port}")
    
    # 注册信号处理器
    def signal_handler(sig, frame):
        """处理中断信号"""
        _ = sig, frame  # 忽略未使用的参数
        logger.info("👋 接收到中断信号，正在关闭Gradio服务...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 创建并启动Gradio应用
        app = create_app()
        
        app.launch(
            server_name=host,
            server_port=port,
            debug=debug,
            share=False,  # 设置为True可生成公共链接
            show_error=True,
            quiet=not debug,
            favicon_path=None,
            # 启动配置
            max_threads=10,
            auth=None,  # 可添加认证: auth=("username", "password")
            # SSL配置 (如需HTTPS)
            # ssl_keyfile=None,
            # ssl_certfile=None,
            # 端口冲突时自动寻找可用端口
            inbrowser=False,
            prevent_thread_lock=False
        )
        
    except KeyboardInterrupt:
        logger.info("👋 用户中断，正在关闭应用...")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        raise
    finally:
        logger.info("🛑 Gradio服务已关闭")


if __name__ == "__main__":
    main()