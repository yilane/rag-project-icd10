# ICD-10 诊断标准化RAG系统

基于检索增强生成(RAG)技术的ICD-10医疗诊断内容标准化工具，支持中文医学术语的智能匹配和标准化。

## 📋 项目概述

本系统通过向量化技术和大语言模型，为医生输入的纯文本诊断内容提供准确的ICD-10标准编码匹配，支持：

- **多诊断识别**：智能识别复合诊断文本中的多个诊断项
- **智能检索**：基于多语言E5模型的语义向量检索
- **标准化推理**：支持DeepSeek/OpenAI等多种LLM模型，默认deepseek
- **本地部署**：使用Milvus Lite本地向量数据库
- **规则化处理**：基于分隔符的灵活文本分割（无复杂依赖）

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
                        ┌─────────────────┐
                        │ 多诊断处理服务   │
                        │(规则化文本分割) │
                        └─────────────────┘
```

### 核心组件

- **数据层**：40k+ ICD-10编码，支持中文医学术语
- **向量化**：`intfloat/multilingual-e5-large-instruct` 模型
- **向量库**：Milvus本地部署，HNSW索引
- **多诊断服务**：智能识别和分割复合诊断文本
- **文本处理器**：基于分隔符的简洁文本分割（逗号、分号、空格、加号）
- **推理层**：DeepSeek/OpenAI LLM，支持多模型切换，默认deepseek
- **API层**：FastAPI，提供RESTful接口

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

# 基础查询（支持多诊断自动识别）
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "急性胃肠炎 发热", "top_k": 5}'

# 多诊断查询示例
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳", "top_k": 5}'

# 智能标准化（集成多诊断识别）
curl -X POST "http://localhost:8000/standardize" \
  -H "Content-Type: application/json" \
  -d '{"text": "疑似埃尔托霍乱爆发，伴有急性胃肠炎症状", "top_k": 10, "llm_provider": "deepseek"}'
```

## 📚 API文档

### 核心接口

#### 1. 向量搜索 `POST /query`

基于语义相似度的ICD编码查询，**自动支持多诊断识别**

**请求示例：**
```json
{
  "text": "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳",
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
      "score": 0.9234
    },
    {
      "code": "E11.9", 
      "title": "2型糖尿病",
      "score": 0.8967
    }
  ],
  "is_multi_diagnosis": true,
  "extracted_diagnoses": ["蛋白尿", "肾功能不全", "2型糖尿病伴血糖控制不佳"],
  "diagnosis_matches": [
    {
      "diagnosis_text": "蛋白尿",
      "candidates": [...],
      "match_confidence": 0.85
    }
  ]
}
```

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
├── main.py                      # FastAPI主应用（含完整生命周期管理）
├── env.example                  # 环境变量配置示例  
├── requirements.txt             # Python依赖包
├── README.md                    # 项目说明文档
├── README_MILVUS_CONFIG.md      # Milvus配置详细说明
├── icd_prd.md                   # 产品需求文档
├── data/                        # 数据目录
│   └── ICD_10v601.csv          # 原始ICD数据
├── db/                          # Milvus数据存储
├── logs/                        # 日志文件
├── models/                      # 数据模型
│   ├── __init__.py
│   └── icd_models.py           # Pydantic模型（含多诊断支持）
├── services/                    # 业务服务
│   ├── embedding_service.py    # 向量化服务
│   ├── milvus_service.py       # 向量数据库服务
│   ├── llm_service.py          # LLM服务（多提供商支持）
│   └── multi_diagnosis_service.py  # 多诊断处理服务
├── tools/                       # 工具脚本
│   ├── text_processor.py       # 文本处理器（规则化分割）
│   └── build_database.py       # 数据库构建
└── tests/                       # 测试文件（可选）
```

### 多诊断处理特性

#### 文本分割策略
- **基于分隔符**：逗号(，,)、分号(；;)、空格、加号(+＋)
- **简洁高效**：无复杂规则，无外部词典依赖
- **长度过滤**：保留≥2字符的有效片段
- **去重保序**：维护原始诊断顺序

#### 智能识别流程
1. **分割提取**：使用`DiagnosisTextProcessor`分割文本
2. **多诊断判断**：检测是否包含多个诊断项
3. **分别检索**：为每个诊断项进行独立向量检索
4. **结果整合**：合并候选结果，按分数排序

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
MILVUS_DB_PATH=./db/milvus
MILVUS_COLLECTION_NAME=icd10

# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=info
```

**注意**：大部分技术参数（如max_tokens、temperature、dimension等）已硬编码在代码中，只有核心的连接配置需要在.env文件中设置。

## 🔧 高级功能

### 多诊断识别示例

```python
# 输入：复合诊断文本
text = "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳"

# 系统自动识别为3个诊断：
# 1. "蛋白尿"
# 2. "肾功能不全" 
# 3. "2型糖尿病伴血糖控制不佳"

# 分别进行向量检索并合并结果
```

### LLM模型切换

系统支持多种LLM提供商：

```python
# DeepSeek (默认推荐)
{
  "llm_provider": "deepseek"
}

# OpenAI
{
  "llm_provider": "openai"  
}

# 本地模型
{
  "llm_provider": "local"
}
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

# 测试多诊断处理
python -c "
from tools.text_processor import DiagnosisTextProcessor
processor = DiagnosisTextProcessor()
text = '高血压病 糖尿病 冠状动脉粥样硬化性心脏病'
diagnoses = processor.extract_diagnoses(text)
print(f'提取诊断: {diagnoses}')
"
```

### 性能优化

- **向量维度**：1024维，平衡精度和性能
- **多诊断并发**：支持并行检索多个诊断项
- **索引优化**：HNSW索引，快速近似搜索
- **内存管理**：完整的资源生命周期管理
- **规则化分割**：避免复杂NLP依赖，提升处理速度

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

1. **模型加载失败**
   ```bash
   # 检查网络连接，手动下载模型
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large-instruct')"
   ```

2. **Milvus连接错误**
   ```bash
   # 使用本地文件模式
   export MILVUS_MODE=local
   export MILVUS_DB_PATH="./db/milvus"
   ```

3. **多诊断分割效果不佳**
   ```python
   # 检查文本处理器
   from tools.text_processor import DiagnosisTextProcessor
   processor = DiagnosisTextProcessor()
   processor.extract_diagnoses("你的测试文本")
   ```

4. **LLM连接失败**
   ```bash
   # 测试LLM连接
   curl -X GET "http://localhost:8000/llm/test"
   
   # 切换LLM提供商
   curl -X POST "http://localhost:8000/llm/switch" \
     -H "Content-Type: application/json" \
     -d '"deepseek"'
   ```

### 日志查看

```bash
# API日志
tail -f logs/api.log

# 查看多诊断处理日志
grep "多诊断" logs/api.log

# 查看标准化流程日志
grep "标准化" logs/api.log
```

### 系统监控

```bash
# 检查系统状态
curl http://localhost:8000/health

# 获取详细资源状态
curl http://localhost:8000/resource/status

# 查看系统统计
curl http://localhost:8000/stats
```

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- **HuggingFace** - 提供优秀的多语言E5模型
- **Milvus** - 高性能向量数据库
- **FastAPI** - 现代化的API框架
- **DeepSeek** - 优秀的中文大语言模型

## 📞 联系方式

如有问题或建议，请提交 [Issue](https://github.com/your-repo/issues) 或联系项目维护者。

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！** 