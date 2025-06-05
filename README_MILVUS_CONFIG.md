# Milvus 服务配置说明

## 概述
Milvus 服务现在支持两种模式：本地模式（Milvus Lite）和远程模式（远程 Milvus 服务器）。

## 配置参数

通过环境变量来配置 Milvus 服务：

### 基础配置
- `MILVUS_MODE`: 运行模式，可选值：
  - `local`: 使用 Milvus Lite 本地数据库（默认）
  - `remote`: 连接远程 Milvus 服务器

### 本地模式配置（MILVUS_MODE=local）
- `MILVUS_DB_PATH`: 本地数据库文件路径（默认：`./db/milvus_icd10.db`）
- `MILVUS_COLLECTION_NAME`: 集合名称（默认：`icd10`）

### 远程模式配置（MILVUS_MODE=remote）
- `MILVUS_HOST`: Milvus 服务器地址（默认：`localhost`）
- `MILVUS_PORT`: Milvus 服务器端口（默认：`19530`）
- `MILVUS_USERNAME`: 用户名（可选）
- `MILVUS_PASSWORD`: 密码（可选）
- `MILVUS_DB_NAME`: 数据库名称（默认：`default`）
- `MILVUS_SECURE`: 是否使用 HTTPS 连接（默认：`false`）
- `MILVUS_COLLECTION_NAME`: 集合名称（默认：`icd10`）

### 通用配置
- `MILVUS_COLLECTION_NAME`: 集合名称（默认：`icd10`）

## 使用示例

### 本地模式（默认）
```bash
# 使用默认本地模式
export MILVUS_MODE=local
export MILVUS_DB_PATH=./db/milvus_icd10.db

# 或者不设置，使用默认值
python your_script.py
```

### 远程模式 - 无认证
```bash
export MILVUS_MODE=remote
export MILVUS_HOST=your-milvus-server.com
export MILVUS_PORT=19530
export MILVUS_DB_NAME=icd10_db
python your_script.py
```

### 远程模式 - 有认证
```bash
export MILVUS_MODE=remote
export MILVUS_HOST=your-milvus-server.com
export MILVUS_PORT=19530
export MILVUS_USERNAME=your_username
export MILVUS_PASSWORD=your_password
export MILVUS_DB_NAME=icd10_db
python your_script.py
```

### 远程模式 - HTTPS 连接
```bash
export MILVUS_MODE=remote
export MILVUS_HOST=your-secure-milvus-server.com
export MILVUS_PORT=443
export MILVUS_SECURE=true
export MILVUS_USERNAME=your_username
export MILVUS_PASSWORD=your_password
export MILVUS_DB_NAME=icd10_db
python your_script.py
```

## 连接测试

可以使用 `test_connection()` 方法测试连接：

```python
from services.milvus_service import MilvusService

milvus_service = MilvusService()
result = milvus_service.test_connection()
print(result)
```

返回结果示例：

### 本地模式
```json
{
    "connected": true,
    "mode": "local",
    "collection_stats": {...},
    "client_type": "MilvusClient",
    "local_info": {
        "db_path": "./db/milvus_icd10.db"
    }
}
```

### 远程模式
```json
{
    "connected": true,
    "mode": "remote",
    "collection_stats": {...},
    "client_type": "MilvusClient",
    "remote_info": {
        "host": "your-milvus-server.com",
        "port": 19530,
        "db_name": "icd10_db",
        "secure": false
    }
}
```

## 注意事项

1. **本地模式**：
   - 自动创建数据库目录
   - 数据存储在本地文件系统
   - 适合开发和测试环境

2. **远程模式**：
   - 需要确保远程 Milvus 服务器可访问
   - 支持用户名/密码认证
   - 支持 HTTPS 安全连接
   - 适合生产环境

3. **切换模式**：
   - 切换模式后数据不会自动迁移
   - 建议在切换前备份重要数据

4. **性能考虑**：
   - 本地模式：读写速度快，但受本地硬件限制
   - 远程模式：可能有网络延迟，但可以利用更强大的服务器资源 