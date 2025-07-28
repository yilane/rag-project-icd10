from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


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
    code: str = Field(..., description="ICD-10编码")
    title: str = Field(..., description="诊断名称")
    score: float = Field(..., description="相似度分数", ge=0.0)


class DiagnosisMatch(BaseModel):
    """单个诊断匹配结果"""
    diagnosis_text: str = Field(..., description="提取的诊断文本")
    candidates: List[Candidate] = Field(..., description="匹配的候选结果")
    match_confidence: float = Field(..., description="整体匹配置信度", ge=0.0, le=1.0)


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