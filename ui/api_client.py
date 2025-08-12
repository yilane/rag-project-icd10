"""
API客户端模块 - 封装与FastAPI后端的所有通信
"""
import requests
import json
from typing import Dict
from loguru import logger
import os


class APIClient:
    """API客户端，封装所有FastAPI调用"""
    
    def __init__(self, base_url: str = None):
        """
        初始化API客户端
        
        Args:
            base_url: API基础URL，默认从环境变量读取或使用localhost:8005
        """
        self.base_url = base_url or os.getenv('API_BASE_URL', 'http://localhost:8005')
        if not self.base_url.startswith('http'):
            self.base_url = f'http://{self.base_url}'
        
        logger.info(f"API客户端初始化，服务地址: {self.base_url}")
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, timeout: int = 30) -> Dict:
        """
        发起HTTP请求的通用方法
        
        Args:
            method: HTTP方法 (GET, POST)
            endpoint: API端点
            data: 请求数据
            timeout: 请求超时时间
            
        Returns:
            响应数据字典
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=timeout)
            else:
                response = requests.post(
                    url, 
                    json=data, 
                    headers={'Content-Type': 'application/json'},
                    timeout=timeout
                )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            error_msg = f"无法连接到API服务 ({url})，请确保FastAPI服务正在运行"
            logger.error(error_msg)
            return {"error": error_msg, "connected": False}
        except requests.exceptions.Timeout:
            error_msg = f"请求超时 ({timeout}秒)"
            logger.error(error_msg)
            return {"error": error_msg, "timeout": True}
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP错误: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            return {"error": error_msg, "status_code": e.response.status_code}
        except Exception as e:
            error_msg = f"请求异常: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def test_connection(self) -> Dict:
        """测试与API服务的连接"""
        result = self._make_request('GET', '/health')
        if 'error' not in result:
            logger.info("API服务连接正常")
            return {"connected": True, "status": "healthy"}
        else:
            logger.warning(f"API服务连接失败: {result.get('error')}")
            return result
    
    def extract_entities(self, text: str, filter_drugs: bool = True) -> Dict:
        """
        调用医学命名实体识别接口
        
        Args:
            text: 医学文本
            filter_drugs: 是否过滤非诊断实体（药品、设备、科室等）
            
        Returns:
            实体识别结果
        """
        data = {
            "text": text,
            "filter_drugs": filter_drugs  # API参数名保持兼容性
        }
        
        logger.info(f"调用实体识别接口: {text[:50]}...")
        result = self._make_request('POST', '/entities', data)
        
        if 'error' not in result:
            logger.info(f"实体识别成功，识别到 {len(result.get('entities', {}))} 种实体类型")
        
        return result
    
    def query_diagnosis(self, text: str, top_k: int = 5, enhanced_processing: bool = True) -> Dict:
        """
        调用智能诊断查询接口
        
        Args:
            text: 诊断文本
            top_k: 返回结果数量
            enhanced_processing: 是否启用增强处理
            
        Returns:
            诊断查询结果
        """
        data = {
            "text": text,
            "top_k": top_k,
            "enhanced_processing": enhanced_processing
        }
        
        logger.info(f"调用诊断查询接口: {text[:50]}..., top_k={top_k}")
        result = self._make_request('POST', '/query', data)
        
        if 'error' not in result:
            candidates_count = len(result.get('candidates', []))
            is_multi = result.get('is_multi_diagnosis', False)
            logger.info(f"诊断查询成功，返回 {candidates_count} 个候选结果，多诊断: {is_multi}")
        
        return result
    
    def standardize_diagnosis(self, text: str, llm_provider: str = "deepseek", top_k: int = 10) -> Dict:
        """
        调用诊断标准化接口
        
        Args:
            text: 非标准诊断文本
            llm_provider: LLM提供商 (deepseek/openai/local)
            top_k: 检索结果数量
            
        Returns:
            诊断标准化结果
        """
        data = {
            "text": text,
            "llm_provider": llm_provider,
            "top_k": top_k
        }
        
        logger.info(f"调用诊断标准化接口: {text[:50]}..., LLM: {llm_provider}")
        result = self._make_request('POST', '/standardize', data, timeout=60)  # 标准化可能需要更长时间
        
        if 'error' not in result:
            logger.info("诊断标准化成功")
        
        return result
    


# 全局API客户端实例
api_client = APIClient()