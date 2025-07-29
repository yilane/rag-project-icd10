from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_serializer
import numpy as np
import dataclasses


def numpy_serializer(obj):
    """Custom serializer for numpy types"""
    if isinstance(obj, (np.integer, np.floating, np.ndarray)):
        return obj.item() if hasattr(obj, 'item') else float(obj)
    return obj


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif dataclasses.is_dataclass(obj):
        # Convert dataclass to dict and process recursively
        return convert_numpy_types(dataclasses.asdict(obj))
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__ (like Pydantic models)
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = convert_numpy_types(value)
        return result
    else:
        return obj


class ICDCode(BaseModel):
    """ICD-10 编码数据模型"""
    code: str = Field(..., description="ICD-10编码")
    preferred_zh: str = Field(..., description="主推中文名")
    preferred_en: Optional[str] = Field(None, description="主推英文名")
    synonyms: List[str] = Field(default_factory=list, description="同义词列表")
    parents: List[str] = Field(default_factory=list, description="父节点编码列表")
    notes: str = Field(default="", description="说明或附注")
    clinical_tags: Dict[str, Any] = Field(default_factory=dict, description="临床标签")
    
    # 组合编码相关字段
    main_code: Optional[str] = Field(None, description="主编码(组合编码时)")
    secondary_code: Optional[str] = Field(None, description="次编码(组合编码时)")
    has_complication: bool = Field(False, description="是否为组合编码")


class Candidate(BaseModel):
    """候选结果模型"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
            # Handle dataclasses (like SimilarityFactors)
            object: lambda x: dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x
        }
    )
    
    code: str = Field(..., description="ICD-10编码")
    title: str = Field(..., description="诊断名称")
    score: float = Field(..., description="相似度分数", ge=0.0)
    
    # 层级信息字段 (Enhanced features)
    level: Optional[int] = Field(default=1, description="ICD层级级别")
    parent_code: Optional[str] = Field(default="", description="父级编码")
    
    # 增强评分字段
    enhanced_score: Optional[float] = Field(default=None, description="增强后的分数")
    original_score: Optional[float] = Field(default=None, description="原始相似度分数")
    similarity_factors: Optional[Any] = Field(default=None, description="相似度计算因子")
    
    @field_serializer('similarity_factors')
    def serialize_similarity_factors(self, value):
        """Custom serializer for similarity_factors to handle numpy types"""
        if value is None:
            return None
        return convert_numpy_types(value)


class DiagnosisMatch(BaseModel):
    """单个诊断匹配结果"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
            # Handle dataclasses (like SimilarityFactors)
            object: lambda x: dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x
        }
    )
    
    diagnosis_text: str = Field(..., description="提取的诊断文本")
    candidates: List[Candidate] = Field(..., description="匹配的候选结果")
    match_confidence: float = Field(..., description="整体匹配置信度", ge=0.0, le=1.0)
    
    # 增强置信度字段 (Enhanced features)
    confidence_metrics: Optional[Any] = Field(default=None, description="置信度指标详情")
    confidence_factors: Optional[Any] = Field(default=None, description="置信度因子")
    confidence_level: Optional[str] = Field(default=None, description="置信度等级")
    
    @field_serializer('confidence_metrics')
    def serialize_confidence_metrics(self, value):
        """Custom serializer for confidence_metrics to handle numpy types"""
        if value is None:
            return None
        return convert_numpy_types(value)
    
    @field_serializer('confidence_factors')
    def serialize_confidence_factors(self, value):
        """Custom serializer for confidence_factors to handle numpy types"""
        if value is None:
            return None
        return convert_numpy_types(value)


class DiagnosisResult(BaseModel):
    """诊断结果模型"""
    diagnosis: str = Field(..., description="识别出的诊断")
    code: str = Field(..., description="ICD-10编码")
    title: str = Field(..., description="标准诊断名称")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    """查询请求模型"""
    text: str = Field(..., description="输入的诊断文本", min_length=1)
    top_k: int = Field(default=5, description="返回候选数量", ge=1, le=50)


class QueryResponse(BaseModel):
    """查询响应模型"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
            # Handle dataclasses (like SimilarityFactors)
            object: lambda x: dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x
        }
    )
    
    candidates: List[Candidate] = Field(..., description="候选结果列表")
    # 多诊断支持字段
    is_multi_diagnosis: bool = Field(default=False, description="是否为多诊断查询")
    extracted_diagnoses: List[str] = Field(default_factory=list, description="提取的诊断词列表（多诊断时）")
    diagnosis_matches: List[DiagnosisMatch] = Field(default_factory=list, description="每个诊断的匹配结果（多诊断时）")


class StandardizeRequest(BaseModel):
    """标准化请求模型"""
    text: str = Field(..., description="输入的诊断文本", min_length=1)
    top_k: int = Field(default=10, description="检索候选数量", ge=1, le=50)
    llm_provider: Optional[str] = Field(default="deepseek", description="LLM提供商", pattern="^(deepseek|openai|local)$")


class StandardizeResponse(BaseModel):
    """标准化响应模型"""
    results: List[DiagnosisResult] = Field(..., description="标准化结果列表")


class EmbeddingRequest(BaseModel):
    """向量化请求模型"""
    texts: List[str] = Field(..., description="要向量化的文本列表")


class EmbeddingResponse(BaseModel):
    """向量化响应模型"""
    embeddings: List[List[float]] = Field(..., description="向量列表")
    model: str = Field(..., description="使用的模型名称")


class MultiDiagnosisRequest(BaseModel):
    """多诊断匹配请求模型"""
    text: str = Field(..., description="包含多个诊断的文本", min_length=1)
    top_k: int = Field(default=5, description="每个诊断返回候选数量", ge=1, le=20)
    separator: Optional[str] = Field(default=None, description="分隔符，如果为空则自动识别")


class MultiDiagnosisResponse(BaseModel):
    """多诊断匹配响应模型"""
    original_text: str = Field(..., description="原始输入文本")
    extracted_diagnoses: List[str] = Field(..., description="提取的诊断词列表")
    matches: List[DiagnosisMatch] = Field(..., description="每个诊断的匹配结果")
    total_matches: int = Field(..., description="总匹配数量")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    milvus_connected: bool = Field(..., description="Milvus连接状态")
    embedding_model_loaded: bool = Field(..., description="嵌入模型加载状态")
    total_records: int = Field(..., description="数据库记录总数") 