import os
import json
import numpy as np
from typing import Dict, Any, List
from loguru import logger
from pymilvus import MilvusClient, DataType


class MilvusService:
    """Milvus向量数据库服务"""
    
    def __init__(self, embedding_service=None):
        self.config = self._load_config()
        self.collection_name = self.config.get("milvus", {}).get("collection_name", "icd10")
        self.embedding_service = embedding_service
        self.dimension = self._get_vector_dimension()
        self.client = None
        self._connect()
        self._setup_collection()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        return {
            "milvus": {
                "mode": os.getenv("MILVUS_MODE", "local"),  # local 或 remote
                "host": os.getenv("MILVUS_HOST", "localhost"),
                "port": int(os.getenv("MILVUS_PORT", "19530")),
                "username": os.getenv("MILVUS_USERNAME", ""),
                "password": os.getenv("MILVUS_PASSWORD", ""),
                "db_name": os.getenv("MILVUS_DB_NAME", "default"),
                "db_path": os.getenv("MILVUS_DB_PATH", "./db/milvus_icd10.db"),
                "collection_name": os.getenv("MILVUS_COLLECTION_NAME", "icd10"),
                "index_type": "FLAT",
                "metric_type": "IP",
                "secure": os.getenv("MILVUS_SECURE", "false").lower() == "true"
            }
        }
    
    def _get_vector_dimension(self) -> int:
        """动态获取向量维度"""
        if self.embedding_service:
            try:
                # 使用测试文本获取向量维度
                test_text = "测试文本"
                test_vector = self.embedding_service.encode_query(test_text)
                dimension = len(test_vector)
                logger.info(f"从嵌入模型获取向量维度: {dimension}")
                return dimension
            except Exception as e:
                logger.warning(f"无法从嵌入服务获取维度: {e}")
        
        # 如果无法获取，使用默认值
        default_dimension = 1024
        logger.warning(f"使用默认向量维度: {default_dimension}")
        return default_dimension
    
    def _connect(self):
        """连接Milvus数据库"""
        milvus_config = self.config.get("milvus", {})
        mode = milvus_config.get("mode", "local")
        
        try:
            # 如果连接失败，删除旧的连接重新开始
            if hasattr(self, 'client') and self.client:
                try:
                    self.client.close()
                except:
                    pass
            
            if mode == "local":
                # 使用Milvus Lite本地模式
                db_path = milvus_config.get("db_path", "./db/milvus_icd10.db")
                
                # 确保db目录存在
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
                    logger.info(f"创建数据库目录: {db_dir}")
                
                # 使用MilvusClient连接Milvus Lite
                self.client = MilvusClient(uri=db_path)
                logger.info(f"成功连接到Milvus Lite: {db_path}")
                
            elif mode == "remote":
                # 使用远程Milvus服务
                host = milvus_config.get("host", "localhost")
                port = milvus_config.get("port", 19530)
                username = milvus_config.get("username", "")
                password = milvus_config.get("password", "")
                db_name = milvus_config.get("db_name", "default")
                secure = milvus_config.get("secure", False)
                
                # 构建连接URI
                if secure:
                    uri = f"https://{host}:{port}"
                else:
                    uri = f"http://{host}:{port}"
                
                # 准备连接参数
                connection_params = {"uri": uri, "token":"root:Milvus"}
                
                if username and password:
                    connection_params["user"] = username
                    connection_params["password"] = password
                
                if db_name and db_name != "default":
                    connection_params["db_name"] = db_name
                
                # 连接远程Milvus
                self.client = MilvusClient(**connection_params)
                logger.info(f"成功连接到远程Milvus: {host}:{port} (数据库: {db_name})")
                
            else:
                raise ValueError(f"不支持的Milvus模式: {mode}，请使用 'local' 或 'remote'")
            
        except Exception as e:
            logger.error(f"Milvus连接失败 (模式: {mode}): {e}")
            raise
    
    def _setup_collection(self):
        """设置集合并加载到内存"""
        try:
            # 检查集合是否存在
            if self.client.has_collection(collection_name=self.collection_name):
                logger.info(f"集合 {self.collection_name} 已存在")
            else:
                logger.info(f"集合 {self.collection_name} 不存在，创建新集合")
                self._create_collection()
            
            # 加载集合到内存（搜索前必需）
            self._load_collection_to_memory()
            
        except Exception as e:
            logger.error(f"设置集合失败: {e}")
            raise
    
    def _load_collection_to_memory(self):
        """将集合加载到内存中"""
        try:
            # 检查集合加载状态
            load_state = self.client.get_load_state(collection_name=self.collection_name)
            
            if load_state == "Loaded":
                logger.info(f"✅ 集合 {self.collection_name} 已经在内存中")
                return
            
            logger.info(f"📤 正在加载集合 {self.collection_name} 到内存...")
            
            # 加载集合
            self.client.load_collection(collection_name=self.collection_name)
            
            # 验证加载状态
            load_state = self.client.get_load_state(collection_name=self.collection_name)
            if load_state == "Loaded":
                logger.info(f"✅ 集合 {self.collection_name} 已成功加载到内存")
            else:
                logger.warning(f"⚠️  集合 {self.collection_name} 加载状态: {load_state}")
            
        except Exception as e:
            logger.error(f"❌ 加载集合到内存失败: {e}")
            raise
    
    def _create_collection(self):
        """创建新集合"""
        logger.info(f"创建新集合: {self.collection_name}")
        
        try:
            # 定义schema
            schema = self.client.create_schema(
                enable_dynamic_field=True
            )
            
            # 添加字段（优化版：支持层级关系）
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
            schema.add_field(field_name="code", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="preferred_zh", datatype=DataType.VARCHAR, max_length=500)
            schema.add_field(field_name="has_complication", datatype=DataType.BOOL)
            schema.add_field(field_name="main_code", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="secondary_code", datatype=DataType.VARCHAR, max_length=50)
            
            # 层级关系字段
            schema.add_field(field_name="level", datatype=DataType.INT32)  # 层级深度: 1=主类,2=亚类,3=细分类
            schema.add_field(field_name="parent_code", datatype=DataType.VARCHAR, max_length=50)  # 父级编码
            schema.add_field(field_name="category_path", datatype=DataType.VARCHAR, max_length=200)  # 完整分类路径
            schema.add_field(field_name="semantic_text", datatype=DataType.VARCHAR, max_length=1000)  # 增强的语义文本
            
            # 定义索引参数
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type=self.config.get("milvus", {}).get("index_type", "FLAT"),
                metric_type=self.config.get("milvus", {}).get("metric_type", "IP")
            )
            
            # 创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info("集合创建完成")
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise
    
    def insert_records(self, records: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> bool:
        """插入记录到数据库"""
        if len(records) != len(embeddings):
            raise ValueError("记录数量与向量数量不匹配")
        
        logger.info(f"准备插入 {len(records)} 条记录到集合 {self.collection_name}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 准备数据
                data = []
                for i, record in enumerate(records):
                    # 处理null值，确保所有字段都有有效值
                    secondary_code = record.get("secondary_code")
                    if secondary_code is None:
                        secondary_code = ""
                    
                    main_code = record.get("main_code")
                    if main_code is None:
                        main_code = ""
                    
                    data.append({
                        "vector": embeddings[i].tolist(),
                        "code": record["code"],
                        "preferred_zh": record.get("preferred_zh", ""),
                        "has_complication": record.get("has_complication", False),
                        "main_code": main_code,
                        "secondary_code": secondary_code,
                        # 层级字段
                        "level": record.get("level", 1),
                        "parent_code": record.get("parent_code", ""),
                        "category_path": record.get("category_path", ""),
                        "semantic_text": record.get("semantic_text", "")
                    })
               
                # 验证数据完整性
                logger.info(f"验证批次数据：共 {len(data)} 条记录")
                
                # 检查每个字段的数据完整性
                field_counts = {}
                for record in data:
                    for field_name, field_value in record.items():
                        if field_name not in field_counts:
                            field_counts[field_name] = 0
                        if field_value is not None and field_value != "":
                            field_counts[field_name] += 1
                
                logger.info(f"字段数据统计: {field_counts}")
                
                # 插入数据
                self.client.insert(
                    collection_name=self.collection_name,
                    data=data
                )
                
                logger.info(f"成功插入 {len(records)} 条记录")
                return True
            except Exception as e:
                logger.error(f"插入记录失败: {e}")
                return False
        return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """向量搜索"""
        try:
            # 确保集合已加载（搜索前必需）
            if not self.client.has_collection(collection_name=self.collection_name):
                logger.error(f"集合 {self.collection_name} 不存在")
                return []
            
            # 执行搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector.tolist()],
                limit=top_k,
                output_fields=["code", "preferred_zh", "has_complication", "main_code", "secondary_code", "level", "parent_code", "category_path", "semantic_text"]
            )
            
            # 处理结果
            candidates = []
            if results and len(results) > 0:
                for hit in results[0]:
                    # 层级权重调整
                    base_score = float(hit.get("distance", 0))
                    level = hit.get("level", 1)
                    level_weight = self._calculate_level_weight(level)
                    adjusted_score = float(base_score * level_weight)
                    
                    candidates.append({
                        "code": hit.get("code"),
                        "title": hit.get("preferred_zh"),
                        "score": float(adjusted_score),
                        "original_score": float(base_score),
                        "metadata": {
                            "has_complication": hit.get("has_complication", False),
                            "main_code": hit.get("main_code", ""),
                            "secondary_code": hit.get("secondary_code", ""),
                            "level": level,
                            "parent_code": hit.get("parent_code", ""),
                            "category_path": hit.get("category_path", ""),
                            "semantic_text": hit.get("semantic_text", "")
                        }
                    })
                
                # 按调整后的分数重新排序
                candidates.sort(key=lambda x: x["score"], reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "exists": self.client.has_collection(collection_name=self.collection_name),
                "dimension": self.dimension
            }
            
            if stats["exists"]:
                collection_stats = self.client.get_collection_stats(collection_name=self.collection_name)
                stats["num_entities"] = collection_stats.get("row_count", 0)
            else:
                stats["num_entities"] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
    
    def load_collection(self) -> bool:
        """加载集合到内存中"""
        try:
            if not self.client.has_collection(collection_name=self.collection_name):
                logger.error(f"集合 {self.collection_name} 不存在")
                return False
            
            # 加载集合
            self.client.load_collection(collection_name=self.collection_name)
            logger.info(f"集合 {self.collection_name} 已加载到内存")
            return True
            
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            if self.client.has_collection(collection_name=self.collection_name):
                self.client.drop_collection(collection_name=self.collection_name)
                logger.info(f"集合 {self.collection_name} 已删除")
            
            self._setup_collection()
            return True
            
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """测试数据库连接"""
        try:
            # 测试集合操作
            stats = self.get_collection_stats()
            mode = self.config.get("milvus", {}).get("mode", "local")
            
            connection_info = {
                "connected": True,
                "mode": mode,
                "collection_stats": stats,
                "client_type": "MilvusClient"
            }
            
            # 添加连接详细信息
            if mode == "remote":
                milvus_config = self.config.get("milvus", {})
                connection_info["remote_info"] = {
                    "host": milvus_config.get("host"),
                    "port": milvus_config.get("port"),
                    "db_name": milvus_config.get("db_name"),
                    "secure": milvus_config.get("secure")
                }
            else:
                connection_info["local_info"] = {
                    "db_path": self.config.get("milvus", {}).get("db_path")
                }
            
            return connection_info
            
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return {
                "connected": False,
                "error": str(e),
                "mode": self.config.get("milvus", {}).get("mode", "local")
            }
    
    def release_collection(self) -> Dict[str, Any]:
        """释放集合内存资源"""
        try:
            if not self.client:
                return {"success": False, "message": "客户端未连接"}
            
            if not self.client.has_collection(collection_name=self.collection_name):
                return {"success": False, "message": f"集合 {self.collection_name} 不存在"}
            
            # 释放集合内存
            self.client.release_collection(collection_name=self.collection_name)
            logger.info(f"✅ 集合 {self.collection_name} 内存已释放")
            
            return {
                "success": True,
                "message": f"集合 {self.collection_name} 内存已释放",
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            error_msg = f"释放集合内存失败: {e}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def get_collection_load_state(self) -> Dict[str, Any]:
        """获取集合加载状态"""
        try:
            if not self.client:
                return {"loaded": False, "message": "客户端未连接"}
            
            if not self.client.has_collection(collection_name=self.collection_name):
                return {"loaded": False, "message": f"集合 {self.collection_name} 不存在"}
            
            # 获取加载状态
            load_state = self.client.get_load_state(collection_name=self.collection_name)
            
            return {
                "loaded": load_state == "Loaded",
                "state": load_state,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            error_msg = f"获取集合加载状态失败: {e}"
            logger.warning(error_msg)
            return {"loaded": False, "message": error_msg}
    
    def disconnect(self) -> Dict[str, Any]:
        """断开连接并清理资源"""
        try:
            if not self.client:
                return {"success": True, "message": "客户端已经断开"}
            
            # 首先释放集合内存
            release_result = self.release_collection()
            
            # 关闭客户端连接
            try:
                if hasattr(self.client, 'close'):
                    self.client.close()
                elif hasattr(self.client, 'disconnect'):
                    self.client.disconnect()
                    
                self.client = None
                logger.info("🔌 Milvus客户端连接已断开")
                
                return {
                    "success": True,
                    "message": "Milvus连接已断开，资源已清理",
                    "release_result": release_result
                }
                
            except Exception as close_err:
                logger.warning(f"关闭Milvus客户端时出错: {close_err}")
                self.client = None
                return {
                    "success": True,
                    "message": "连接已断开（可能有警告）",
                    "warning": str(close_err)
                }
                
        except Exception as e:
            error_msg = f"断开Milvus连接失败: {e}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取集合内存使用情况"""
        try:
            if not self.client:
                return {"memory_usage": 0, "message": "客户端未连接"}
            
            if not self.client.has_collection(collection_name=self.collection_name):
                return {"memory_usage": 0, "message": f"集合 {self.collection_name} 不存在"}
            
            # 获取集合统计信息
            stats = self.get_collection_stats()
            load_state = self.get_collection_load_state()
            
            return {
                "collection_name": self.collection_name,
                "loaded": load_state.get("loaded", False),
                "load_state": load_state.get("state", "Unknown"),
                "num_entities": stats.get("num_entities", 0),
                "estimated_memory_mb": stats.get("num_entities", 0) * self.dimension * 4 / (1024 * 1024),  # 4 bytes per float
                "message": "内存使用为估算值（基于向量维度和实体数量）"
            }
            
        except Exception as e:
            error_msg = f"获取内存使用情况失败: {e}"
            logger.warning(error_msg)
            return {"memory_usage": 0, "message": error_msg}
    
    def health_check(self) -> Dict[str, Any]:
        """完整的健康检查"""
        try:
            connection_test = self.test_connection()
            load_state = self.get_collection_load_state()
            memory_usage = self.get_memory_usage()
            
            is_healthy = (
                connection_test.get("connected", False) and 
                load_state.get("loaded", False)
            )
            
            return {
                "healthy": is_healthy,
                "connection": connection_test,
                "load_state": load_state,
                "memory_usage": memory_usage,
                "timestamp": __import__("datetime").datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": __import__("datetime").datetime.now().isoformat()
            } 
    def _calculate_level_weight(self, level: int) -> float:
        """计算层级权重"""
        # 主类别权重更高，细分类权重较低
        level_weights = {
            1: 1.2,  # 主类别
            2: 1.0,  # 亚类别  
            3: 0.8   # 细分类
        }
        return level_weights.get(level, 1.0)
