import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class LLMService:
    """LLM服务，支持多种模型切换"""
    
    def __init__(self):
        self.config = self._load_config()
        self.provider = self.config.get("llm", {}).get("provider", "deepseek")
        self.client = self._create_client()
    
    def _load_config(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        return {
            "llm": {
                "provider": os.getenv("LLM_PROVIDER", "deepseek"),
                "deepseek": {
                    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                    "max_tokens": 2048,  # 硬编码默认值
                    "temperature": 0.1   # 硬编码默认值
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    "max_tokens": 2048,  # 硬编码默认值
                    "temperature": 0.1   # 硬编码默认值
                },
                "local": {
                    "api_key": os.getenv("LOCAL_API_KEY", "not-required"),
                    "base_url": os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1"),
                    "model": os.getenv("LOCAL_MODEL", "local-medical-model"),
                    "max_tokens": 2048,  # 硬编码默认值
                    "temperature": 0.1   # 硬编码默认值
                }
            }
        }
    
    def _create_client(self) -> OpenAI:
        """创建LLM客户端"""
        llm_config = self.config.get("llm", {})
        provider_config = llm_config.get(self.provider, {})
        
        try:
            client = OpenAI(
                api_key=provider_config.get("api_key", ""),
                base_url=provider_config.get("base_url", ""),
                timeout=120.0  # 设置2分钟超时
            )
            logger.info(f"成功创建 {self.provider} 客户端")
            return client
        except Exception as e:
            logger.error(f"创建 {self.provider} 客户端失败: {e}")
            raise
    
    def switch_provider(self, provider: str) -> bool:
        """切换LLM提供商"""
        if provider not in ["deepseek", "openai", "local"]:
            logger.error(f"不支持的提供商: {provider}")
            return False
        
        try:
            self.provider = provider
            self.client = self._create_client()
            logger.info(f"成功切换到 {provider}")
            return True
        except Exception as e:
            logger.error(f"切换到 {provider} 失败: {e}")
            return False
    
    def _get_standardize_prompt(self, input_text: str, candidates: List[Dict[str, Any]]) -> str:
        """构建标准化Prompt"""
        prompt = f"""您是一名 ICD-10 医学标准化助理，根据输入的诊断内容，识别其中可能包含的多个诊断，并为每个诊断匹配最适合的 ICD-10 code。

用户输入："{input_text}"

候选码值：
"""
        
        for candidate in candidates[:10]:  # 最多显示前10个候选
            score = candidate.get("score", 0)
            code = candidate.get("code", "")
            title = candidate.get("title", "")
            prompt += f"({score:.2f}) {code}: {title}\n"
        
        prompt += """
请按以下格式返回结果：
```json
{
  "results": [
    {
      "diagnosis": "识别出的诊断名称",
      "code": "匹配的ICD-10编码",
      "title": "标准诊断名称",
      "confidence": 0.85
    }
  ]
}
```

注意事项：
1. 只返回JSON格式，不要包含其他文本
2. confidence取值范围0.0-1.0
3. 如果输入包含多个诊断，请分别识别和匹配
4. 优先选择相似度分数高的候选编码
"""
        
        return prompt
    
    def standardize_diagnosis(self, input_text: str, candidates: List[Dict[str, Any]], 
                            provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """标准化诊断"""
        # 如果指定了提供商，临时切换
        original_provider = self.provider
        if provider and provider != self.provider:
            if not self.switch_provider(provider):
                logger.warning(f"切换到 {provider} 失败，继续使用 {self.provider}")
        
        try:
            # 构建Prompt
            prompt = self._get_standardize_prompt(input_text, candidates)
            
            # 获取当前提供商配置
            llm_config = self.config.get("llm", {})
            provider_config = llm_config.get(self.provider, {})
            
            # 调用LLM
            response = self.client.chat.completions.create(
                model=provider_config.get("model", "deepseek-chat"),
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的ICD-10医学编码专家，能够准确识别和标准化医学诊断。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=provider_config.get("max_tokens", 2048),
                temperature=provider_config.get("temperature", 0.1)
            )
            
            # 解析响应
            content = response.choices[0].message.content.strip()
            
            # 提取JSON部分
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content
            
            # 解析JSON
            try:
                result = json.loads(json_content)
                return result.get("results", [])
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}, 原文本: {json_content}")
                # 返回简化结果
                return self._create_fallback_result(input_text, candidates)
        
        except Exception as e:
            logger.error(f"LLM标准化失败: {e}")
            return self._create_fallback_result(input_text, candidates)
        
        finally:
            # 恢复原始提供商
            if provider and provider != original_provider:
                self.switch_provider(original_provider)
    
    def _create_fallback_result(self, input_text: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """创建备用结果（当LLM调用失败时）"""
        if not candidates:
            return []
        
        # 返回最高分的候选作为结果
        best_candidate = candidates[0]
        return [{
            "diagnosis": input_text,
            "code": best_candidate.get("code", ""),
            "title": best_candidate.get("title", ""),
            "confidence": min(best_candidate.get("score", 0), 0.95)  # 降低置信度
        }]
    
    def generate_response(self, prompt: str, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        生成LLM响应
        
        Args:
            prompt: 输入提示词
            provider: 指定提供商（可选）
            
        Returns:
            响应结果字典
        """
        # 如果指定了提供商，临时切换
        original_provider = self.provider
        if provider and provider != self.provider:
            if not self.switch_provider(provider):
                logger.warning(f"切换到 {provider} 失败，继续使用 {self.provider}")
        
        try:
            # 获取当前提供商配置
            llm_config = self.config.get("llm", {})
            provider_config = llm_config.get(self.provider, {})
            
            # 调用LLM
            response = self.client.chat.completions.create(
                model=provider_config.get("model", "deepseek-chat"),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=provider_config.get("max_tokens", 2048),
                temperature=provider_config.get("temperature", 0.1)
            )
            
            # 返回格式化结果
            return {
                "content": response.choices[0].message.content.strip(),
                "provider": self.provider,
                "model": provider_config.get("model"),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"LLM响应生成失败: {e}")
            return {
                "error": str(e),
                "provider": self.provider
            }
        
        finally:
            # 恢复原始提供商
            if provider and provider != original_provider:
                self.switch_provider(original_provider)
    
    def test_connection(self) -> Dict[str, Any]:
        """测试LLM连接"""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"正在测试 {self.provider} 连接...")
            
            # 发送简单测试请求
            llm_config = self.config.get("llm", {})
            provider_config = llm_config.get(self.provider, {})
            
            response = self.client.chat.completions.create(
                model=provider_config.get("model", "deepseek-chat"),
                messages=[
                    {"role": "user", "content": "你好"}
                ],
                max_tokens=10,
                timeout=90  # 90秒超时
            )
            
            duration = time.time() - start_time
            logger.info(f"{self.provider} 连接测试成功 (耗时: {duration:.1f}秒)")
            
            return {
                "connected": True,
                "provider": self.provider,
                "model": provider_config.get("model", ""),
                "response": response.choices[0].message.content,
                "duration": duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            logger.warning(f"{self.provider} 连接测试失败 (耗时: {duration:.1f}秒): {error_msg}")
            
            # 分析错误类型
            if "timeout" in error_msg.lower():
                error_type = "timeout"
            elif "authentication" in error_msg.lower() or "401" in error_msg:
                error_type = "auth"
            elif "404" in error_msg:
                error_type = "endpoint"
            else:
                error_type = "unknown"
            
            return {
                "connected": False,
                "provider": self.provider,
                "error": error_msg,
                "error_type": error_type,
                "duration": duration
            }
    
    def get_provider_info(self) -> Dict[str, Any]:
        """获取当前提供商信息"""
        llm_config = self.config.get("llm", {})
        provider_config = llm_config.get(self.provider, {})
        
        return {
            "current_provider": self.provider,
            "model": provider_config.get("model", ""),
            "base_url": provider_config.get("base_url", ""),
            "max_tokens": provider_config.get("max_tokens", 2048),
            "temperature": provider_config.get("temperature", 0.1),
            "available_providers": ["deepseek", "openai", "local"]
        }


def main():
    """测试LLM服务"""
    service = LLMService()
    
    # 测试连接
    connection_test = service.test_connection()
    print(f"连接测试: {connection_test}")
    
    # 获取提供商信息
    info = service.get_provider_info()
    print(f"提供商信息: {info}")
    
    # 测试诊断标准化
    test_candidates = [
        {"code": "A00.905", "title": "霍乱暴发型", "score": 0.88},
        {"code": "K59.1", "title": "急性胃肠炎", "score": 0.75}
    ]
    
    results = service.standardize_diagnosis("急性胃肠炎", test_candidates)
    print(f"标准化结果: {results}")


if __name__ == "__main__":
    main() 