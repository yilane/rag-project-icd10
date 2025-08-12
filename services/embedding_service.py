import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class EmbeddingService:
    """向量化服务"""
    
    def __init__(self):
        self.config = self._load_config()
        self.model = None
        self.device = self._get_device()
        self._load_model()
    
    def _load_config(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        return {
            "embedding": {
                "model_name": os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large-instruct"),
                "max_length": 512,  # 硬编码默认值
                "batch_size": 32,   # 硬编码默认值
                "device": os.getenv("EMBEDDING_DEVICE", "auto")
            }
        }
    
    def _get_device(self) -> str:
        """获取计算设备"""
        device_config = self.config.get("embedding", {}).get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return device_config
    
    def _load_model(self):
        """加载嵌入模型"""
        model_name = self.config.get("embedding", {}).get("model_name", 
                                                         "intfloat/multilingual-e5-large-instruct")
        
        try:
            logger.info(f"正在加载嵌入模型: {model_name}")
            logger.info(f"模型名称类型: {type(model_name)}, 值: {repr(model_name)}")
            logger.info(f"目标设备: {self.device}")
            
            # 确保模型名称不为None
            if not model_name:
                raise ValueError("模型名称不能为空")
            
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"模型加载成功，使用设备: {self.device}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error(f"配置信息: {self.config}")
            raise
    
    def _prepare_text_for_embedding(self, text: str) -> str:
        """为E5模型准备文本"""
        # E5模型推荐的格式：query: 或者 passage:
        if not text.startswith(("query:", "passage:")):
            text = f"passage: {text}"
        return text
    
    def encode_single(self, text: str) -> np.ndarray:
        """对单个文本进行向量化"""
        if not self.model:
            raise RuntimeError("嵌入模型未加载")
        
        prepared_text = self._prepare_text_for_embedding(text)
        embedding = self.model.encode(prepared_text, normalize_embeddings=True)
        return embedding
    
    def encode_batch(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """批量向量化"""
        if not self.model:
            raise RuntimeError("嵌入模型未加载")
        
        if not texts:
            return []
        
        # 准备文本
        prepared_texts = [self._prepare_text_for_embedding(text) for text in texts]
        
        # 批量编码
        batch_size = self.config.get("embedding", {}).get("batch_size", 32)
        embeddings = self.model.encode(
            prepared_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        
        return embeddings.tolist()
    
    def encode_icd_record(self, icd_record: Dict[str, Any]) -> np.ndarray:
        """为ICD记录生成向量"""
        # 只对preferred_zh字段进行向量化
        preferred_zh = icd_record.get("preferred_zh", "")
        
        if not preferred_zh.strip():
            # 如果preferred_zh为空，使用默认文本
            preferred_zh = f"ICD代码 {icd_record.get('code', 'unknown')}"
        
        return self.encode_single(preferred_zh)
    
    def encode_query(self, query: str) -> np.ndarray:
        """为查询文本生成向量"""
        query_text = f"query: {query}"
        return self.model.encode(query_text, normalize_embeddings=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.model:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.config.get("embedding", {}).get("model_name"),
            "device": self.device,
            "max_seq_length": getattr(self.model, 'max_seq_length', None),
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }
    
    def test_embedding(self, test_text: str = "测试文本") -> Dict[str, Any]:
        """测试向量化功能"""
        try:
            embedding = self.encode_single(test_text)
            return {
                "success": True,
                "embedding_shape": embedding.shape,
                "embedding_type": str(type(embedding)),
                "sample_values": embedding[:5].tolist()  # 前5个值作为示例
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """测试向量化服务"""
    service = EmbeddingService()
    
    # 测试单个文本
    test_text = "急性胃肠炎"
    result = service.test_embedding(test_text)
    print(f"测试结果: {result}")
    
    # 测试批量
    test_texts = ["霍乱", "伤寒", "急性胃肠炎"]
    embeddings = service.encode_batch(test_texts)
    print(f"批量向量化完成，生成 {len(embeddings)} 个向量")
    
    # 模型信息
    info = service.get_model_info()
    print(f"模型信息: {info}")


if __name__ == "__main__":
    main() 