import json
import os
import sys
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.milvus_service import MilvusService


class DatabaseBuilder:
    """ICD数据库构建器"""
    
    def __init__(self):
        self.embedding_service = None
        self.milvus_service = None
        
        # 配置日志
        logger.add("logs/database_build.log", rotation="50 MB", level="INFO")
    
    def initialize_services(self):
        """初始化服务"""
        logger.info("初始化服务...")
        
        try:
            # 初始化向量化服务
            logger.info("加载向量化模型...")
            self.embedding_service = EmbeddingService()
            
            # 测试向量化服务
            test_result = self.embedding_service.test_embedding("测试")
            if not test_result.get("success"):
                raise Exception(f"向量化服务测试失败: {test_result.get('error')}")
            
            logger.info(f"向量化模型加载成功: {self.embedding_service.get_model_info()}")
            
            # 初始化Milvus服务（传入嵌入服务以获取向量维度）
            logger.info("连接Milvus数据库...")
            self.milvus_service = MilvusService(embedding_service=self.embedding_service)
            
            # 测试Milvus连接
            connection_test = self.milvus_service.test_connection()
            if not connection_test.get("connected"):
                raise Exception(f"Milvus连接失败: {connection_test.get('error')}")
            
            logger.info(f"Milvus连接成功: {connection_test}")
            logger.info(f"向量维度: {self.milvus_service.dimension}")
            
        except Exception as e:
            logger.error(f"服务初始化失败: {e}")
            raise
    
    def load_csv_data(self, input_file: str) -> List[Dict]:
        """直接从CSV文件加载数据"""
        logger.info(f"开始加载数据: {input_file}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(input_file, encoding='utf-8')
            logger.info(f"成功读取 {len(df)} 条记录")
            
            # 转换为标准格式的记录列表
            records = []
            for idx, row in df.iterrows():
                # 简单数据转换，不进行复杂的清洗
                code = str(row.get('code', '')).strip()
                disease = str(row.get('disease', '')).strip()
                
                # 基本验证：跳过空记录
                if not code or not disease or code == 'nan' or disease == 'nan':
                    continue
                
                # 解析组合编码（保留基本解析）
                main_code = code
                secondary_code = ""
                has_complication = False
                
                if '+' in code and '*' in code:
                    parts = code.split('+')
                    if len(parts) == 2:
                        main_code = parts[0].strip()
                        secondary_code = parts[1].replace('*', '').strip()
                        has_complication = True
                
                # 构建记录（简化版）
                record = {
                    "code": code,
                    "preferred_zh": disease,
                    "main_code": main_code,
                    "secondary_code": secondary_code,
                    "has_complication": has_complication
                }
                
                records.append(record)
            
            logger.info(f"转换完成，获得 {len(records)} 条有效记录")
            return records
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def vectorize_and_index(self, records: List[Dict]) -> bool:
        """向量化数据并建立索引"""
        logger.info(f"开始向量化 {len(records)} 条记录")
        
        try:
            # 批量向量化  
            batch_size = 50  # 批处理大小
            total_batches = (len(records) + batch_size - 1) // batch_size
            
            logger.info(f"开始批量向量化，每批 {batch_size} 条，共 {total_batches} 批")
            
            for batch_idx in tqdm(range(total_batches), desc="向量化进度", unit="批"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(records))
                batch_records = records[start_idx:end_idx]
                current_batch_size = len(batch_records)
                
                logger.info(f"正在处理第 {batch_idx + 1}/{total_batches} 批，记录 {start_idx + 1}-{end_idx} 共 {current_batch_size} 条")
                
                # 为每条记录生成向量
                batch_embeddings = []
                failed_count = 0
                
                for i, record in enumerate(batch_records):
                    try:
                        embedding = self.embedding_service.encode_icd_record(record)
                        batch_embeddings.append(embedding)
                        
                        # 每处理100条显示一次进度
                        if (i + 1) % 100 == 0:
                            logger.info(f"  - 已向量化 {i + 1}/{current_batch_size} 条记录")
                            
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"记录 {record.get('code')} 向量化失败: {e}")
                        # 使用零向量作为备用
                        zero_embedding = [0.0] * self.milvus_service.dimension
                        batch_embeddings.append(zero_embedding)
                
                if failed_count > 0:
                    logger.warning(f"批次 {batch_idx + 1} 中有 {failed_count} 条记录向量化失败")
                
                # 插入到Milvus
                logger.info(f"正在插入第 {batch_idx + 1} 批数据到数据库...")
                success = self.milvus_service.insert_records(batch_records, batch_embeddings)
                if not success:
                    logger.error(f"批次 {batch_idx + 1} 插入失败...")
                    return False
                
                processed_count = (batch_idx + 1) * batch_size if batch_idx < total_batches - 1 else len(records)
                logger.info(f"✅ 批次 {batch_idx + 1}/{total_batches} 完成，已处理 {processed_count}/{len(records)} 条记录")
                
            logger.info("向量化和索引建立完成")
            
            # 加载集合到内存中以支持搜索
            logger.info("加载集合到内存中...")
            load_success = self.milvus_service.load_collection()
            if not load_success:
                logger.warning("集合加载失败，但数据插入成功")
            
            return True
            
        except Exception as e:
            logger.error(f"向量化和索引失败: {e}")
            return False
    
    def verify_database(self) -> Dict[str, Any]:
        """验证数据库状态"""
        logger.info("验证数据库状态...")
        
        try:
            # 获取集合统计信息
            stats = self.milvus_service.get_collection_stats()
            
            # 确保集合已加载到内存中（搜索前必须的步骤）
            logger.info("加载集合到内存中...")
            load_success = self.milvus_service.load_collection()
            if not load_success:
                logger.warning("集合加载失败，可能影响搜索结果")
            
            # 测试搜索功能
            logger.info("执行搜索测试...")
            test_vector = self.embedding_service.encode_query("急性胃肠炎")
            search_results = self.milvus_service.search(test_vector, top_k=5)
            
            verification_result = {
                "database_stats": stats,
                "search_test": {
                    "query": "急性胃肠炎",
                    "results_count": len(search_results),
                    "top_results": search_results[:3] if search_results else []
                }
            }
            
            logger.info(f"数据库验证完成: {verification_result}")
            return verification_result
            
        except Exception as e:
            logger.error(f"数据库验证失败: {e}")
            return {"error": str(e)}
    
    def build_full_database(self, input_file: str = "data/ICD_10v601.csv", 
                           rebuild: bool = False) -> bool:
        """完整构建数据库"""
        logger.info("开始完整构建ICD诊断数据库")
        
        try:
            # 检查是否需要重建
            if rebuild:
                logger.info("重建模式：清空现有数据")
                self.initialize_services()
                self.milvus_service.clear_collection()
            else:
                logger.info("增量模式：基于现有数据")
                self.initialize_services()
            
            # 直接加载CSV数据（跳过数据清洗步骤）
            logger.info("直接从CSV文件加载数据...")
            records = self.load_csv_data(input_file)
            
            # 向量化并建立索引
            logger.info("开始向量化和索引...")
            vectorize_success = self.vectorize_and_index(records)
            if not vectorize_success:
                logger.error("向量化失败")
                return False
            
            # 验证数据库
            logger.info("验证数据库状态...")
            verification = self.verify_database()
            if "error" in verification:
                logger.error(f"数据库验证失败: {verification['error']}")
                return False
            
            logger.info("数据库构建完成!")
            logger.info(f"最终统计: {verification['database_stats']}")
            
            return True
            
        except Exception as e:
            logger.error(f"数据库构建失败: {e}")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ICD数据库构建工具（简化版）")
    parser.add_argument("--input", default="data/ICD_10v601.csv", help="输入CSV文件路径")
    parser.add_argument("--rebuild", action="store_true", help="重建数据库（清空现有数据）")
    parser.add_argument("--verify-only", action="store_true", help="仅验证现有数据库")
    
    args = parser.parse_args()
    
    builder = DatabaseBuilder()
    
    try:
        if args.verify_only:
            # 只验证现有数据库
            logger.info("验证现有数据库...")
            builder.initialize_services()
            verification = builder.verify_database()
            if "error" not in verification:
                logger.info("数据库验证成功")
                print("数据库状态正常")
                return True
            else:
                logger.error(f"数据库验证失败: {verification['error']}")
                return False
        else:
            # 构建数据库
            success = builder.build_full_database(args.input, rebuild=args.rebuild)
            if success:
                logger.info("数据库构建成功")
                print("数据库构建完成")
                return True
            else:
                logger.error("数据库构建失败")
                return False
            
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        print("操作已中断")
        return False
    except Exception as e:
        logger.error(f"运行出错: {e}")
        print(f"错误: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 