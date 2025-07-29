# ICD-10 诊断标准化RAG系统

基于检索增强生成(RAG)技术的ICD-10医疗诊断内容标准化工具，支持中文医学术语的智能匹配和标准化。

## 📋 项目概述

本系统通过向量化技术和大语言模型，为医生输入的纯文本诊断内容提供准确的ICD-10标准编码匹配，支持：

- **多诊断识别**：智能识别复合诊断文本中的多个诊断项，自动过滤药品实体
- **智能检索**：基于多语言E5模型的语义向量检索，集成层级权重优化
- **标准化推理**：支持DeepSeek/OpenAI等多种LLM模型，默认deepseek
- **本地部署**：使用Milvus Lite本地向量数据库，无需外部依赖
- **增强处理**：集成医学NER、语义边界检测、多维度置信度评估
- **层级匹配**：支持ICD-10三级层级结构的智能权重计算

## 🏗️ 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI服务   │    │   向量化模型     │    │   Milvus数据库   │
│   (API接口)     │────│  (E5-Large)     │────│  (向量存储)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         │              ┌─────────────────┐             │
         └──────────────│   LLM服务       │─────────────┘
                        │ (DeepSeek/GPT)  │
                        └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 多诊断处理服务   │    │  医学NER服务    │    │ 层级相似度服务   │
│   (智能分割)    │────│ (实体识别)     │────│ (权重计算)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         └──────────────│ 多维度置信度服务 │─────────────┘
                        │ (综合评估)      │
                        └─────────────────┘
```

### 核心组件

- **数据层**：37k+ ICD-10编码，支持中文医学术语和层级结构解析
- **向量化**：`intfloat/multilingual-e5-large-instruct` 模型（1024维）
- **向量库**：Milvus Lite本地部署，HNSW索引，支持层级字段
- **多诊断服务**：集成医学NER的智能诊断识别和分割
- **医学NER**：基于`lixin12345/chinese-medical-ner`模型的实体识别
- **层级相似度**：ICD-10三级层级的智能权重计算（1.2x/1.0x/0.8x）
- **置信度评估**：12维度综合置信度评分系统
- **药品过滤**：自动过滤非诊断相关的药品和设备实体
- **推理层**：DeepSeek/OpenAI LLM，支持多模型切换
- **API层**：FastAPI，完整的资源生命周期管理

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建conda虚拟环境
conda create -n rag-project-icd10 python=3.10
conda activate rag-project-icd10

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

```bash
# 复制配置文件
cp env.example .env

# 编辑环境变量（配置API密钥等）
vim .env
```

### 3. 构建数据库

```bash
# 完整构建（数据加载 + 向量化 + 建索引）
python tools/build_database.py

# 重建数据库（清空现有数据）
python tools/build_database.py --rebuild

# 验证数据库
python tools/build_database.py --verify-only
```

### 4. 启动API服务

```bash
# 启动FastAPI服务
python main.py

# 或使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 测试API

访问 http://localhost:8000/docs 查看API文档

```bash
# 健康检查
curl http://localhost:8000/health

# 基础查询（支持多诊断自动识别，自动过滤药品）
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "急性胃肠炎 发热", "top_k": 5}'

# 多诊断查询示例（含药品过滤）
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳 服用二甲双胍", "top_k": 5}'

# 智能标准化（集成多诊断识别和层级权重）
curl -X POST "http://localhost:8000/standardize" \
  -H "Content-Type: application/json" \
  -d '{"text": "疑似埃尔托霍乱爆发，伴有急性胃肠炎症状", "top_k": 10, "llm_provider": "deepseek"}'
```

## 📚 API文档

### 核心接口

#### 1. 向量搜索 `POST /query`

基于语义相似度的ICD编码查询，**自动支持多诊断识别、药品过滤、层级权重**

**请求示例：**
```json
{
  "text": "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳 服用二甲双胍",
  "top_k": 5
}
```

**响应示例：**
```json
{
  "candidates": [
    {
      "code": "N18.9",
      "title": "慢性肾功能不全",
      "score": 1.1234,
      "level": 2,
      "parent_code": "N18"
    },
    {
      "code": "E11.9", 
      "title": "2型糖尿病",
      "score": 0.9967,
      "level": 2,
      "parent_code": "E11"
    }
  ],
  "is_multi_diagnosis": true,
  "extracted_diagnoses": ["蛋白尿", "肾功能不全", "2型糖尿病伴血糖控制不佳"],
  "diagnosis_matches": [
    {
      "diagnosis_text": "蛋白尿",
      "candidates": [...],
      "match_confidence": 0.85,
      "confidence_level": "高置信度",
      "confidence_metrics": {...}
    }
  ]
}
```

**新增功能：**
- 🚫 **自动药品过滤**：自动识别并过滤"二甲双胍"等药品实体
- 📊 **层级权重**：分数>1.0表示层级加权（主类别1.2x，亚类别1.0x，详细0.8x）
- 🎯 **置信度评估**：多维度置信度评分和等级判断
- 🏷️ **层级信息**：返回ICD层级和父节点信息

#### 2. 诊断标准化 `POST /standardize`

使用LLM进行智能诊断标准化，**集成多诊断查询逻辑**

**请求示例：**
```json
{
  "text": "疑似埃尔托霍乱爆发，伴有急性胃肠炎症状",
  "top_k": 10,
  "llm_provider": "deepseek"
}
```

**工作流程：**
1. 多诊断识别和向量检索（复用query接口逻辑）
2. LLM标准化处理
3. 返回结构化结果

#### 3. 文本向量化 `POST /embed`

获取文本的向量表示

```json
{
  "texts": ["霍乱", "急性胃肠炎", "伤寒"]
}
```

#### 4. 系统状态 `GET /health`

检查系统健康状态，包括各服务组件状态

### 管理接口

- `GET /stats` - 系统统计信息（Milvus、向量化、LLM状态）
- `POST /llm/switch` - 切换LLM提供商
- `GET /llm/test` - 测试LLM连接
- `GET /resource/status` - 获取详细的资源状态
- `POST /resource/release` - 释放系统资源
- `POST /resource/reload` - 重新加载Milvus集合

## 🛠️ 开发指南

### 项目结构

```
rag-project-icd10/
├── main.py                               # FastAPI主应用（含完整生命周期管理）
├── env.example                           # 环境变量配置示例  
├── requirements.txt                      # Python依赖包
├── README.md                             # 项目说明文档
├── CLAUDE.md                             # Claude Code操作指南
├── data/                                 # 数据目录
│   └── ICD_10v601.csv                   # 原始ICD数据（37k+记录）
├── db/                                   # Milvus数据存储
├── logs/                                 # 日志文件
├── models/                               # 数据模型
│   ├── __init__.py
│   └── icd_models.py                    # Pydantic模型（支持层级和置信度）
├── services/                             # 业务服务
│   ├── embedding_service.py             # 向量化服务（E5-Large）
│   ├── milvus_service.py                # 向量数据库服务（HNSW索引）
│   ├── llm_service.py                   # LLM服务（多提供商支持）
│   ├── multi_diagnosis_service.py       # 多诊断处理服务（核心）
│   ├── medical_ner_service.py           # 医学NER服务（实体识别）
│   ├── hierarchical_similarity_service.py  # 层级相似度服务
│   ├── multidimensional_confidence_service.py  # 多维度置信度评估
│   ├── enhanced_text_processor.py       # 增强文本处理器
│   ├── semantic_boundary_service.py     # 语义边界检测服务
│   └── diagnosis_entity_filter.py       # 诊断实体过滤器
├── tools/                                # 工具脚本
│   ├── text_processor.py                # 文本处理器（规则化分割）
│   └── build_database.py                # 数据库构建（支持层级解析）
└── tests/                                # 测试文件
    ├── test_chinese_medical_ner.py      # NER服务测试
    ├── test_hierarchical_similarity.py  # 层级相似度测试
    ├── test_multidimensional_confidence.py  # 置信度评估测试
    └── test_query_api_with_multi_diagnosis.py  # API集成测试
```

### 多诊断处理特性

#### 智能分割策略
- **分隔符识别**：逗号(，,)、分号(；;)、空格、加号(+＋)
- **医学NER增强**：集成`lixin12345/chinese-medical-ner`模型
- **语义边界检测**：智能判断诊断边界
- **实体过滤**：自动过滤药品、设备等非诊断实体
- **长度过滤**：保留≥2字符的有效片段

#### 增强识别流程
1. **多模式提取**：结合规则分割和NER实体识别
2. **语义边界优化**：使用语义相似度优化分割点
3. **实体分类过滤**：区分诊断、药品、设备、治疗等实体类型
4. **层级权重检索**：为每个诊断项进行增强向量检索
5. **置信度评估**：12维度综合置信度评分
6. **结果整合**：按加权分数排序，支持>1.0的层级加权分数

### 配置说明

主要配置在 `.env` 文件中：

```bash
# LLM配置（默认deepseek）
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# OpenAI配置（可选）
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# 向量化配置
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large-instruct
EMBEDDING_DEVICE=auto

# Milvus配置
MILVUS_MODE=local
MILVUS_DB_PATH=./db/milvus_icd10.db
MILVUS_COLLECTION_NAME=icd10_e5

# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=info

# 医学NER模型配置（增强功能）
USE_MEDICAL_NER_MODEL=true
MEDICAL_NER_MODEL=lixin12345/chinese-medical-ner
MEDICAL_NER_MIN_CONFIDENCE=0.5
MEDICAL_NER_USE_GPU=false

# 增强文本处理配置
USE_ENHANCED_TEXT_PROCESSING=true
SEMANTIC_BOUNDARY_THRESHOLD=0.75
ENHANCED_CONFIDENCE_WEIGHT=0.4

# 文本处理优化
MIN_DIAGNOSIS_LENGTH=2
MAX_DIAGNOSIS_LENGTH=50
ENTITY_DENSITY_THRESHOLD=0.1
BOUNDARY_CONFIDENCE_THRESHOLD=0.8
```

**注意**：大部分技术参数（如max_tokens、temperature、dimension等）已硬编码在代码中，只有核心的连接配置需要在.env文件中设置。

## 🔧 高级功能

### 多诊断识别示例

```python
# 输入：复合诊断文本（包含药品）
text = "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳 服用二甲双胍"

# 系统自动识别和过滤：
# 提取的诊断：["蛋白尿", "肾功能不全", "2型糖尿病伴血糖控制不佳"]
# 过滤的药品：["二甲双胍"]

# 每个诊断分别进行增强向量检索，支持层级权重和置信度评估
```

### 置信度评估系统

系统采用12维度综合置信度评分：

```python
# 置信度维度示例
confidence_metrics = {
    "overall_confidence": 0.856,
    "reliability_score": 0.782,
    "confidence_interval": [0.801, 0.911],
    "factor_contributions": {
        "vector_similarity": 0.234,
        "hierarchy_boost": 0.156,
        "entity_match_score": 0.198,
        "terminology_accuracy": 0.145,
        "professional_specificity": 0.123
    }
}
```

### LLM模型切换

系统支持多种LLM提供商动态切换：

```bash
# 切换到DeepSeek (默认推荐)
curl -X POST "http://localhost:8000/llm/switch" \
  -H "Content-Type: application/json" \
  -d '"deepseek"'

# 切换到OpenAI
curl -X POST "http://localhost:8000/llm/switch" \
  -H "Content-Type: application/json" \
  -d '"openai"'

# 测试当前LLM连接
curl -X GET "http://localhost:8000/llm/test"
```

### 增强文本处理

```python
# 增强诊断提取（集成NER和语义边界检测）
from services.enhanced_text_processor import EnhancedTextProcessor
from services.embedding_service import EmbeddingService

embedding_service = EmbeddingService()
enhanced_processor = EnhancedTextProcessor(embedding_service)

# 复杂医学文本处理
result = enhanced_processor.extract_diagnoses_enhanced(
    "慢性肾功能不全伴蛋白尿，高血压病3级，2型糖尿病血糖控制不佳"
)

# 返回详细的诊断信息，包括置信度、实体类型、边界信息
for diagnosis in result:
    print(f"诊断: {diagnosis['text']}")
    print(f"置信度: {diagnosis['diagnosis_confidence']:.3f}")
    print(f"实体密度: {diagnosis['entity_density']:.3f}")
```

### 医学实体识别

```bash
# 测试医学NER服务
python -c "
from services.medical_ner_service import MedicalNERService
ner_service = MedicalNERService()

text = '急性心肌梗死伴左心室功能不全，服用阿司匹林'
entities = ner_service.extract_medical_entities(text, filter_drugs=True)

print('提取的医学实体:')
for entity_type, entity_list in entities.items():
    print(f'{entity_type}: {[e[\"text\"] for e in entity_list]}')
"
```

### 批量处理

```bash
# 批量向量化
python -c "
from services.embedding_service import EmbeddingService
service = EmbeddingService()
texts = ['霍乱', '伤寒', '急性胃肠炎']
embeddings = service.encode_batch(texts)
print(f'生成了 {len(embeddings)} 个向量')
"

# 测试增强多诊断处理
python -c "
from tools.text_processor import DiagnosisTextProcessor
from services.embedding_service import EmbeddingService

embedding_service = EmbeddingService()
processor = DiagnosisTextProcessor(embedding_service, use_enhanced_processing=True)
text = '高血压病 糖尿病 冠状动脉粥样硬化性心脏病 服用降压药'
diagnoses = processor.extract_diagnoses_enhanced(text)
print(f'提取诊断数: {len(diagnoses)}')
for d in diagnoses:
    print(f'  - {d[\"text\"]} (置信度: {d[\"diagnosis_confidence\"]:.3f})')
"
```

### 性能优化

- **向量维度**：1024维，平衡精度和性能
- **多诊断并发**：支持并行检索多个诊断项
- **索引优化**：HNSW索引，快速近似搜索
- **层级权重**：ICD-10三级层级智能加权（1.2x/1.0x/0.8x）
- **药品过滤**：自动识别并过滤非诊断相关的药品实体
- **内存管理**：完整的资源生命周期管理
- **语义边界检测**：智能识别诊断边界，提高分割准确性
- **置信度优化**：12维度综合评分，提供可靠性量化指标

## 📊 功能特性

### 多诊断支持

| 特性 | 说明 | 示例 |
|------|------|------|
| 自动识别 | 无需手动指定分隔符 | "高血压，糖尿病，冠心病" |
| 灵活分割 | 支持多种分隔符 | 逗号、分号、空格、加号 |
| 独立检索 | 每个诊断单独向量检索 | 提高匹配精度 |
| 结果整合 | 按分数排序合并 | 统一候选池 |

### 标准化集成

| 特性 | 说明 | 优势 |
|------|------|------|
| 查询集成 | 复用query接口逻辑 | 一致的多诊断处理 |
| LLM增强 | 基于检索结果标准化 | 提高准确性 |
| 提供商切换 | 支持多种LLM | 灵活配置 |
| 详细日志 | 完整处理过程记录 | 便于调试分析 |

## 🐛 问题排查

### 常见问题

1. **向量化模型加载失败**
   ```bash
   # 检查网络连接，手动下载模型
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large-instruct')"
   
   # 或设置镜像源
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **Milvus连接错误**
   ```bash
   # 确保使用本地文件模式
   export MILVUS_MODE=local
   export MILVUS_DB_PATH="./db/milvus_icd10.db"
   
   # 检查数据库文件权限
   ls -la ./db/milvus_icd10.db
   ```

3. **医学NER模型加载失败**
   ```bash
   # 测试NER模型加载
   python -c "
   from services.medical_ner_service import MedicalNERService
   ner = MedicalNERService(use_model=True)
   print('NER模型加载成功')
   "
   
   # 如果失败，可以禁用NER功能
   export USE_MEDICAL_NER_MODEL=false
   ```

4. **多诊断分割效果不佳**
   ```python
   # 检查增强文本处理器
   from tools.text_processor import DiagnosisTextProcessor
   from services.embedding_service import EmbeddingService
   
   embedding_service = EmbeddingService()
   processor = DiagnosisTextProcessor(embedding_service, use_enhanced_processing=True)
   
   # 测试诊断提取
   result = processor.extract_diagnoses_enhanced("你的测试文本")
   print(f"提取到 {len(result)} 个诊断")
   for d in result:
       print(f"- {d['text']} (置信度: {d['diagnosis_confidence']:.3f})")
   ```

5. **置信度评分异常**
   ```python
   # 检查置信度服务
   from services.multidimensional_confidence_service import MultiDimensionalConfidenceService
   from services.embedding_service import EmbeddingService
   
   embedding_service = EmbeddingService()
   confidence_service = MultiDimensionalConfidenceService(embedding_service)
   
   # 测试置信度计算
   test_diagnosis = "急性心肌梗死"
   test_candidates = [{"code": "I21.9", "title": "急性心肌梗死", "score": 0.95}]
   metrics, factors = confidence_service.calculate_comprehensive_confidence(
       test_diagnosis, test_candidates
   )
   print(f"综合置信度: {metrics.overall_confidence:.3f}")
   ```

6. **LLM连接失败**
   ```bash
   # 测试LLM连接
   curl -X GET "http://localhost:8000/llm/test"
   
   # 切换LLM提供商
   curl -X POST "http://localhost:8000/llm/switch" \
     -H "Content-Type: application/json" \
     -d '"deepseek"'
   
   # 检查API密钥配置
   grep -E "DEEPSEEK_API_KEY|OPENAI_API_KEY" .env
   ```

7. **层级权重计算错误**
   ```python
   # 测试层级相似度服务
   from services.hierarchical_similarity_service import HierarchicalSimilarityService
   
   hierarchical_service = HierarchicalSimilarityService(embedding_service)
   
   # 检查层级解析
   test_code = "I21.9"
   level_info = hierarchical_service._parse_icd_level(test_code)
   print(f"编码 {test_code} 层级信息: {level_info}")
   ```

### 日志查看

```bash
# API日志
tail -f logs/api.log

# 查看多诊断处理日志
grep "多诊断" logs/api.log

# 查看增强处理日志
grep "增强" logs/api.log

# 查看置信度评估日志
grep "置信度" logs/api.log

# 查看药品过滤日志
grep "药品过滤" logs/api.log

# 查看层级权重计算日志
grep "层级" logs/api.log

# 查看标准化流程日志
grep "标准化" logs/api.log
```

### 系统监控

```bash
# 检查系统健康状态（包含所有服务组件）
curl http://localhost:8000/health

# 获取详细资源状态（包括增强服务状态）
curl http://localhost:8000/resource/status

# 查看系统统计信息
curl http://localhost:8000/stats

# 测试医学实体提取功能
curl -X POST "http://localhost:8000/entities" \
  -H "Content-Type: application/json" \
  -d '{"text": "急性心肌梗死伴心律失常，服用阿司匹林"}'

# 获取详细置信度报告（新功能）
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "慢性肾功能不全", "top_k": 3}' | jq '.diagnosis_matches[0].confidence_metrics'
```

### 性能监控

```bash
# 监控内存使用情况
curl http://localhost:8000/resource/status | jq '.milvus.memory_usage'

# 手动释放系统资源
curl -X POST http://localhost:8000/resource/release

# 重新加载Milvus集合到内存
curl -X POST http://localhost:8000/resource/reload
```

## 🎯 核心增强特性

### 1. 智能药品过滤系统
- **自动识别**：基于医学NER模型识别药品、设备等非诊断实体
- **智能过滤**：自动从查询结果中过滤药品实体，专注于诊断匹配
- **配置灵活**：支持通过环境变量启用/禁用药品过滤功能

### 2. 多维度置信度评估
- **12维度评分**：包括向量相似度、层级权重、实体匹配、术语准确性等
- **综合置信度**：集成多个因子的加权评分，提供可靠性量化
- **置信度区间**：提供置信度的统计区间和可靠性评估
- **改进建议**：基于置信度分析提供诊断文本优化建议

### 3. ICD-10层级智能权重
- **三级层级**：主类别、亚类别、详细编码的智能识别
- **动态权重**：主类别1.2x、亚类别1.0x、详细编码0.8x的差异化加权
- **层级增强**：基于ICD-10层级结构的语义增强匹配

### 4. 增强文本处理器
- **语义边界检测**：智能识别医学文本中的诊断边界
- **NER集成**：融合医学命名实体识别提高分割准确性
- **多模式提取**：结合规则分割和语义分析的混合方法

### 5. 医学实体识别服务
- **专业模型**：基于`lixin12345/chinese-medical-ner`的中文医学NER
- **实体分类**：区分疾病、症状、药品、设备、治疗方式等实体类型
- **置信度评估**：为每个识别的实体提供置信度分数

### 6. 系统性能优化
- **内存管理**：完整的资源生命周期管理和自动清理
- **批处理优化**：根据数据规模动态调整批处理大小
- **GPU支持**：智能GPU缓存管理和自动清理
- **并发处理**：多诊断并行检索和处理

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 🙏 致谢

- **HuggingFace** - 提供优秀的多语言E5模型
- **Milvus** - 高性能向量数据库
- **FastAPI** - 现代化的API框架
- **DeepSeek** - 优秀的中文大语言模型

## 📞 联系方式

如有问题或建议，请提交 [Issue](https://github.com/yilane/rag-project-icd10/issues) 或联系项目维护者。

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！** 