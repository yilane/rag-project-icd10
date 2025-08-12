# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ICD-10 medical diagnosis standardization system built with RAG (Retrieval Augmented Generation) technology. The system provides intelligent matching and standardization of Chinese medical terminology to standard ICD-10 codes.

### Core Technologies
- **Vector Search**: Uses `shibing624/text2vec-base-chinese` embedding model (768 dimensions) optimized for Chinese medical terminology
- **Vector Database**: Milvus Lite for local vector storage with HNSW indexing
- **LLM**: DeepSeek (default) with OpenAI/Local fallback support
- **Web Framework**: FastAPI with comprehensive lifecycle management and async context managers
- **UI Framework**: Gradio web interface with 3 main tabs (Entity Recognition, Query, Standardization)
- **Multi-diagnosis Processing**: Rule-based text splitting with hierarchical ICD-10 support
- **Enhanced NER**: Medical Named Entity Recognition with non-diagnostic entity filtering
- **Hierarchical Search**: Layer-weighted scoring system (main categories 1.2x, subcategories 1.0x, detailed 0.8x)

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n rag-project-icd10 python=3.10
conda activate rag-project-icd10

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp env.example .env
# Edit .env with your API keys
```

**Important**: Always use the conda environment `rag-project-icd10` for this project. Activate it before running any commands:
```bash
conda activate rag-project-icd10
```

### Database Management
```bash
# Full database build with hierarchy support (data loading + vectorization + indexing)
python tools/build_database.py

# Rebuild database (clear existing data) - includes layer parsing and semantic enhancement
python tools/build_database.py --rebuild

# Verify database with hierarchy features
python tools/build_database.py --verify-only

# Build with specific input file
python tools/build_database.py --input data/ICD_10v601.csv --rebuild
```

### Running the Application

#### Method 1: Manual Separate Launch (推荐)
```bash
# 终端1 - 启动API服务
uvicorn main:app --host 0.0.0.0 --port 8005 --reload

# 终端2 - 启动Gradio界面
python gradio_app.py
```

#### Method 2: FastAPI Only
```bash
# Start only FastAPI server
python main.py
# Or
uvicorn main:app --host 0.0.0.0 --port 8005 --reload
```

**重要**: Gradio界面需要API服务先启动才能正常使用所有功能。界面会显示API连接状态。默认API端口为8005，Gradio端口为7860。

### Testing
```bash
# Health check
curl http://localhost:8005/health

# Test query endpoint with multi-diagnosis and hierarchy weights
curl -X POST "http://localhost:8005/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "蛋白尿待查 肾功能不全 2型糖尿病伴血糖控制不佳", "top_k": 5}'

# Test standardization with LLM integration
curl -X POST "http://localhost:8005/standardize" \
  -H "Content-Type: application/json" \
  -d '{"text": "疑似埃尔托霍乱爆发，伴有急性胃肠炎症状", "top_k": 10, "llm_provider": "deepseek"}'

# Test batch embedding
curl -X POST "http://localhost:8005/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["急性胃肠炎", "蛋白尿", "肾功能不全"]}'

# Test API functionality directly
python -c "from services.medical_ner_service import MedicalNERService; ner = MedicalNERService(); print('NER service initialized successfully')"
python -c "from services.enhanced_text_processor import EnhancedTextProcessor; print('Enhanced text processor available')"
```

## Architecture Overview

### Core Services
- **`main.py`**: FastAPI application with complete lifecycle management and service orchestration
- **`gradio_app.py`**: Gradio web interface with 3 main tabs (Entity Recognition, Query, Standardization)
- **`services/embedding_service.py`**: Text vectorization using Chinese text2vec model
- **`services/milvus_service.py`**: Vector database operations with HNSW indexing
- **`services/llm_service.py`**: Multi-provider LLM service (DeepSeek/OpenAI/Local)
- **`services/multi_diagnosis_service.py`**: Multi-diagnosis detection and processing
- **`services/diagnosis_entity_filter.py`**: Non-diagnostic entity filtering (drugs, equipment, departments)
- **`tools/text_processor.py`**: Rule-based text splitting using delimiters (comma, semicolon, space, plus)

### Enhanced Services
- **`services/enhanced_text_processor.py`**: Advanced text processing integrating medical NER and semantic boundary detection
- **`services/medical_ner_service.py`**: Medical Named Entity Recognition using `lixin12345/chinese-medical-ner` model
- **`services/hierarchical_similarity_service.py`**: Multi-dimensional similarity calculation with ICD-10 hierarchy-aware scoring
- **`services/semantic_boundary_service.py`**: Semantic boundary detection for improved diagnosis segmentation
- **`services/multidimensional_confidence_service.py`**: Advanced confidence scoring with multiple evaluation dimensions
- **`services/uncertainty_diagnosis_service.py`**: Handles uncertainty patterns in medical diagnoses (e.g., "待查", "疑似")

### Data Flow
1. **Input Processing**: Text → Multi-diagnosis detection → Individual diagnosis extraction
2. **Vector Search**: Each diagnosis → Embedding → Milvus similarity search
3. **Result Aggregation**: Individual results merged and ranked by score
4. **LLM Standardization**: Vector search results → LLM → Standardized ICD codes

### Multi-diagnosis Processing
The system automatically detects and processes compound diagnoses like "高血压病 糖尿病 冠状动脉粥样硬化性心脏病":
- Uses delimiter-based splitting (，,；;空格+＋) 
- Filters segments ≥2 characters
- Performs independent vector search for each diagnosis
- Merges results with confidence scoring

## Configuration

### Environment Variables (`.env`)
```bash
# LLM Configuration (defaults to deepseek)
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Local LLM Configuration
LOCAL_BASE_URL=http://localhost:8000/v1
LOCAL_MODEL=local-medical-model
LOCAL_API_KEY=not-required

# Embedding Configuration
EMBEDDING_MODEL_NAME=shibing624/text2vec-base-chinese
EMBEDDING_DEVICE=auto

# Milvus Configuration
MILVUS_MODE=local
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_DB_PATH=./db/milvus_icd10.db
MILVUS_COLLECTION_NAME=icd10_collection

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8005
API_WORKERS=1
API_LOG_LEVEL=info

# Gradio Configuration
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860

# Debug Configuration
DEBUG=false
LOG_LEVEL=INFO
```

## Key Implementation Details

### Service Initialization
All services are initialized during FastAPI lifespan with proper error handling and cleanup. The application uses an `@asynccontextmanager` for complete resource lifecycle management.

### Vector Database Setup
- **Dimension**: 768 (from shibing624/text2vec-base-chinese)
- **Index**: HNSW for fast approximate search
- **Metric**: Inner Product (IP) for similarity scoring
- **Storage**: Local SQLite-based Milvus Lite
- **Hierarchy Fields**: `level`, `parent_code`, `category_path`, `semantic_text`
- **Dynamic Batching**: 32-256 batch size based on dataset size
- **Layer Weighting**: Main categories (1.2x), Subcategories (1.0x), Detailed (0.8x)

### Multi-diagnosis Architecture
- **Detection**: `DiagnosisTextProcessor.extract_diagnoses()` splits text using multiple delimiters
- **Processing**: `MultiDiagnosisService.match_multiple_diagnoses()` handles parallel vector searches
- **Integration**: Both `/query` and `/standardize` endpoints support automatic multi-diagnosis detection
- **Standardization**: Multi-diagnosis standardization processes each diagnosis separately through LLM
- **UI Display**: Grouped display for both query results and standardization with confidence levels
- **Entity Filtering**: Automatic filtering of non-diagnostic entities (drugs, equipment, departments)

### Resource Management
The application includes comprehensive resource management:
- Automatic service cleanup on shutdown
- Memory release endpoints (`/resource/release`, `/resource/reload`)
- GPU cache clearing when available
- Collection loading/unloading for memory optimization

## API Endpoints

### Core Endpoints
- `POST /query`: Vector similarity search with automatic multi-diagnosis support
- `POST /standardize`: LLM-based diagnosis standardization with integrated vector search
- `POST /embed`: Batch text vectorization
- `POST /entities`: Medical entity extraction with non-diagnostic filtering
- `GET /health`: System health check with service status

### Management Endpoints  
- `GET /stats`: System statistics (Milvus, embedding, LLM status)
- `POST /llm/switch`: Switch LLM provider dynamically
- `GET /llm/test`: Test LLM connection
- `GET /resource/status`: Detailed resource status
- `POST /resource/release`: Release system resources
- `POST /resource/reload`: Reload Milvus collection

## Gradio Web Interface

The system includes a comprehensive web interface built with Gradio, consisting of 3 main tabs:

### Tab Structure
1. **🏷️ 医学命名实体识别** (`ui/entities_tab.py`)
   - Medical Named Entity Recognition interface
   - Toggle for non-diagnostic entity filtering (drugs, equipment, departments)
   - Real-time entity extraction and classification

2. **🔍 智能诊断查询** (`ui/query_tab.py`) 
   - Multi-diagnosis text input and processing
   - Vector similarity search with grouped results display
   - Hierarchical scoring visualization with confidence levels

3. **🤖 诊断标准化** (`ui/standardize_tab.py`)
   - LLM-based diagnosis standardization 
   - Multi-diagnosis grouped standardization display
   - Interactive LLM provider selection and reasoning display

### UI Components
- **`ui/utils.py`**: Utility functions for data formatting and HTML generation
- **`ui/api_client.py`**: API client for communication with FastAPI backend (default port: 8005)
- **Auto-refresh**: Connection status monitoring and health check display

## Development Guidelines

### Service Dependencies
When modifying services, note the dependency chain:
- `EmbeddingService` → `MilvusService` → `MultiDiagnosisService`
- Enhanced services depend on core services: `EnhancedTextProcessor` → `MedicalNERService` + `SemanticBoundaryDetector`
- `HierarchicalSimilarityService` integrates with vector search for improved scoring
- All services are injected into the main FastAPI application
- Services use async context managers for proper resource management

### ICD-10 Hierarchy Implementation
The system parses ICD-10 codes into three levels:
- **Level 1 (Main)**: A00, B15, etc. (no dots)
- **Level 2 (Sub)**: A00.0, B15.1, etc. (single digit after dot)
- **Level 3 (Detailed)**: A00.001, B15.101, etc. (multiple digits after dot)

Hierarchy parsing logic in `tools/build_database.py:_parse_hierarchy()`

### Text Processing Critical Fix
**Important**: The `tools/text_processor.py` file contains a critical fix for diagnosis text cleaning. The `_clean_diagnosis_text()` method preserves medically significant terms like "待查" (pending investigation), "疑似" (suspected), "考虑" (considering), and "排除" (rule out) which are essential for proper medical diagnosis interpretation.

### Multi-diagnosis Testing
Use `tools/text_processor.py` to test diagnosis extraction:
```python
from tools.text_processor import DiagnosisTextProcessor
processor = DiagnosisTextProcessor()
diagnoses = processor.extract_diagnoses("高血压病 糖尿病 冠心病")
```

### Enhanced Text Processing
Test the enhanced text processing capabilities:
```python
from services.enhanced_text_processor import EnhancedTextProcessor
from services.embedding_service import EmbeddingService

embedding_service = EmbeddingService()
enhanced_processor = EnhancedTextProcessor(embedding_service)
result = enhanced_processor.extract_diagnoses_enhanced("复杂的医学诊断文本")
```

### Medical NER Testing
Test medical entity recognition:
```python
from services.medical_ner_service import MedicalNERService
ner_service = MedicalNERService()
entities = ner_service.extract_entities("急性心肌梗死伴左心室功能不全")
```

### Entity Filtering Configuration
The system includes sophisticated entity filtering to focus on diagnostic content:
```python
from services.diagnosis_entity_filter import DiagnosisEntityFilter

# Filter configuration - keeps lab_indicators, filters drugs/equipment
filter_service = DiagnosisEntityFilter()
filtered_entities = filter_service.filter_non_diagnostic_entities(entities, keep_lab_indicators=True)
```

**Important**: The system uses special confidence thresholds for lab_indicator entities (0.5) to prevent them from being incorrectly filtered while maintaining the specified confidence threshold for other entity types.

### Database Rebuild Process
When rebuilding the database with hierarchy support:
1. **Data Loading**: CSV parsing with hierarchy detection (`_parse_hierarchy()`)
2. **Semantic Enhancement**: Creates enriched text with parent information (`_build_semantic_text()`)
3. **Dynamic Batching**: Calculates optimal batch size based on record count (`_calculate_optimal_batch_size()`)
4. **Layer Statistics**: Logs distribution of main/sub/detailed categories (`_log_hierarchy_stats()`)

### Logging
All logs are written to `logs/api.log` with rotation. Use structured logging for multi-diagnosis processing to track the complete flow.

## Troubleshooting

### Common Issues
1. **Model Loading**: If embedding model fails to load, check network connectivity and HuggingFace cache
2. **Milvus Connection**: Ensure `MILVUS_MODE=local` and database path is writable
3. **LLM API**: Test connection with `GET /llm/test` and verify API keys in `.env`
4. **Multi-diagnosis**: Check text processor with various delimiter combinations
5. **Medical NER Model**: If NER model fails to load, verify network access for model download
6. **Gradio UI Issues**: If standardization results don't display, check API port configuration (should be 8005)
7. **Entity Filtering**: If lab_indicator entities are incorrectly filtered, verify confidence threshold settings
8. **Empty Search Results**: If no candidates returned, check database health with `/health` endpoint and verify collection has data
9. **Vector Dimension Mismatch**: Ensure database was built with correct embedding model (768 dimensions for text2vec-base-chinese)

### Performance Optimization
- Vector dimension is optimized at 768 for Chinese medical terminology
- Multi-diagnosis processing uses concurrent vector searches
- HNSW indexing provides fast approximate similarity search
- Complete resource lifecycle management prevents memory leaks
- Dynamic batch sizing: 32 (<1K records), 64 (<10K), 128 (<50K), 256 (50K+)
- Hierarchy-aware semantic text vectorization improves search precision
- Layer weighting system prioritizes main categories over detailed subcategories

### Score Validation
The `Candidate` model in `models/icd_models.py` supports scores >1.0 to accommodate hierarchy weighting. Main category matches can have scores up to 1.2x the original similarity score.

### Data Models Architecture
- **StandardizeResponse**: Updated to support flexible result formats including multi-diagnosis grouping
- **DiagnosisMatch**: Contains confidence metrics, factors, and levels for enhanced matching
- **Candidate**: Supports hierarchy information (level, parent_code) and enhanced scoring
- **Multi-diagnosis Support**: All models accommodate both single and multi-diagnosis scenarios

### UI Data Flow
1. **API Response**: Backend returns grouped standardization results for multi-diagnosis
2. **Format Processing**: `format_multi_diagnosis_standardization()` processes API results  
3. **HTML Generation**: `generate_standardization_html()` creates grouped display
4. **Gradio Display**: HTML components show organized results with confidence levels