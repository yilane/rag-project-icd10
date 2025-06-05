import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from models.icd_models import (
    QueryRequest, QueryResponse, StandardizeRequest, StandardizeResponse,
    EmbeddingRequest, EmbeddingResponse, HealthCheckResponse,
    DiagnosisMatch, Candidate
)
from services.embedding_service import EmbeddingService
from services.milvus_service import MilvusService
from services.llm_service import LLMService
from services.multi_diagnosis_service import MultiDiagnosisService


# 全局服务实例
embedding_service = None
milvus_service = None
llm_service = None
multi_diagnosis_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 启动和关闭时的资源管理"""
    # 启动时初始化服务
    global embedding_service, milvus_service, llm_service, multi_diagnosis_service
    
    startup_success = False
    
    try:
        logger.info("🚀 开始初始化应用服务...")
        
        # 初始化向量化服务
        logger.info("📊 初始化向量化服务...")
        embedding_service = EmbeddingService()
        logger.info("✅ 向量化服务初始化完成")
        
        # 初始化Milvus服务
        logger.info("💾 初始化Milvus服务...")
        milvus_service = MilvusService(embedding_service)
        
        # 验证Milvus连接和集合状态
        connection_test = milvus_service.test_connection()
        if not connection_test.get("connected", False):
            logger.warning("⚠️  Milvus连接测试失败，但继续启动")
        else:
            # 检查集合加载状态
            load_state = milvus_service.get_collection_load_state()
            if load_state.get("loaded", False):
                logger.info(f"✅ 集合 {milvus_service.collection_name} 已加载到内存")
            else:
                logger.warning(f"⚠️  集合 {milvus_service.collection_name} 未完全加载: {load_state.get('state', 'Unknown')}")
        
        logger.info("✅ Milvus服务初始化完成")
        
        # 初始化LLM服务
        logger.info("🤖 初始化LLM服务...")
        llm_service = LLMService()
        
        # 测试LLM连接（非关键，失败也继续）
        try:
            llm_test = llm_service.test_connection()
            if not llm_test.get("connected", False):
                logger.warning("⚠️  LLM连接测试失败，标准化功能可能不可用")
        except Exception as llm_err:
            logger.warning(f"⚠️  LLM测试失败: {llm_err}")
        
        logger.info("✅ LLM服务初始化完成")
        
        # 初始化多诊断服务（使用规则方案）
        logger.info("🔍 初始化多诊断服务...")
        multi_diagnosis_service = MultiDiagnosisService(embedding_service, milvus_service)
        logger.info("✅ 多诊断服务初始化完成")
        
        startup_success = True
        logger.info("🎉 所有服务初始化完成，应用启动成功！")
        
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        # 如果启动失败，尝试清理已初始化的资源
        await _cleanup_services()
        raise
    
    try:
        # 应用运行期间
        yield
        
    finally:
        # 关闭时清理资源
        logger.info("🧹 开始清理应用资源...")
        await _cleanup_services()
        logger.info("✅ 应用资源清理完成")


async def _cleanup_services():
    """清理所有服务资源"""
    global embedding_service, milvus_service, llm_service, multi_diagnosis_service
    
    cleanup_tasks = []
    
    try:
        # 清理多诊断服务
        if multi_diagnosis_service:
            logger.info("🔍 清理多诊断服务...")
            multi_diagnosis_service = None
            logger.info("✅ 多诊断服务清理完成")
        
        # 清理LLM服务
        if llm_service:
            logger.info("🤖 清理LLM服务...")
            # LLM服务主要是API客户端，关闭连接
            if hasattr(llm_service, 'client') and llm_service.client:
                try:
                    if hasattr(llm_service.client, 'close'):
                        llm_service.client.close()
                except Exception as e:
                    logger.warning(f"LLM客户端关闭失败: {e}")
            llm_service = None
            logger.info("✅ LLM服务清理完成")
        
        # 清理Milvus服务
        if milvus_service:
            logger.info("💾 清理Milvus服务...")
            try:
                # 释放载入内存的集合
                await _release_milvus_collections()
                
                # 断开Milvus连接
                if hasattr(milvus_service, 'disconnect'):
                    milvus_service.disconnect()
                elif hasattr(milvus_service, 'close'):
                    milvus_service.close()
                    
            except Exception as e:
                logger.warning(f"Milvus服务清理失败: {e}")
            finally:
                milvus_service = None
                logger.info("✅ Milvus服务清理完成")
        
        # 清理向量化服务
        if embedding_service:
            logger.info("📊 清理向量化服务...")
            try:
                # 释放模型内存
                if hasattr(embedding_service, 'model') and embedding_service.model:
                    del embedding_service.model
                
                # 清理GPU缓存（如果使用GPU）
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("🔥 GPU缓存已清理")
                except ImportError:
                    pass  # 没有安装PyTorch
                    
            except Exception as e:
                logger.warning(f"向量化服务清理失败: {e}")
            finally:
                embedding_service = None
                logger.info("✅ 向量化服务清理完成")
                
    except Exception as e:
        logger.error(f"❌ 资源清理过程中发生错误: {e}")


async def _release_milvus_collections():
    """释放Milvus集合的内存资源"""
    try:
        if milvus_service and hasattr(milvus_service, 'collection_name'):
            from pymilvus import Collection, connections
            
            # 获取当前加载的集合
            collection_name = milvus_service.collection_name
            
            # 检查集合是否存在并已加载
            try:
                collection = Collection(collection_name)
                
                # 检查集合加载状态
                load_progress = collection.loading_progress
                if load_progress.get('loading_progress') == "100%":
                    logger.info(f"📤 释放集合内存: {collection_name}")
                    collection.release()
                    logger.info(f"✅ 集合 {collection_name} 内存已释放")
                else:
                    logger.info(f"ℹ️  集合 {collection_name} 未完全加载，无需释放")
                    
            except Exception as collection_err:
                logger.warning(f"处理集合 {collection_name} 时出错: {collection_err}")
            
            # 断开所有Milvus连接
            try:
                connections.disconnect("default")
                logger.info("🔌 Milvus连接已断开")
            except Exception as conn_err:
                logger.warning(f"断开Milvus连接失败: {conn_err}")
                
    except Exception as e:
        logger.warning(f"释放Milvus集合内存失败: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="ICD-10 诊断标准化API",
    description="基于RAG的ICD-10诊断内容标准化系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志
logger.add("logs/api.log", rotation="50 MB", level="INFO")


@app.get("/", tags=["根路径"])
async def root():
    """根路径"""
    return {
        "message": "ICD-10 诊断标准化API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    try:
        # 检查向量化服务
        embedding_loaded = embedding_service and embedding_service.get_model_info().get("loaded", False)
        
        # 检查Milvus连接
        milvus_connected = False
        total_records = 0
        if milvus_service:
            connection_test = milvus_service.test_connection()
            milvus_connected = connection_test.get("connected", False)
            if milvus_connected:
                stats = milvus_service.get_collection_stats()
                total_records = stats.get("num_entities", 0)
        
        status = "healthy" if (embedding_loaded and milvus_connected) else "unhealthy"
        
        return HealthCheckResponse(
            status=status,
            milvus_connected=milvus_connected,
            embedding_model_loaded=embedding_loaded,
            total_records=total_records
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["向量搜索"])
async def query_similar(request: QueryRequest):
    """基于向量相似度的ICD编码查询（支持自动多诊断识别）"""
    try:
        logger.info(f"收到查询请求: {request.text}")
        
        if not embedding_service or not milvus_service or not multi_diagnosis_service:
            raise HTTPException(status_code=503, detail="服务未就绪")
        
        # 首先检查是否可能包含多个诊断
        potential_diagnoses = multi_diagnosis_service.text_processor.extract_diagnoses(request.text)
        
        # 判断是否为多诊断查询
        if len(potential_diagnoses) > 1:
            logger.info(f"检测到多诊断查询，提取的诊断: {potential_diagnoses}")
            
            # 执行多诊断匹配
            multi_result = multi_diagnosis_service.match_multiple_diagnoses(
                text=request.text,
                top_k=request.top_k
            )
            
            # 合并所有候选结果
            all_candidates = []
            diagnosis_matches = []
            
            for match_data in multi_result["matches"]:
                # 添加到总候选列表
                all_candidates.extend(match_data.candidates)
                # 保存诊断匹配详情
                diagnosis_matches.append(match_data)
            
            # 按分数排序并限制数量
            all_candidates.sort(key=lambda x: x.score, reverse=True)
            all_candidates = all_candidates[:request.top_k]
            
            logger.info(f"多诊断查询完成，共找到 {len(all_candidates)} 个候选结果")
            
            return QueryResponse(
                candidates=all_candidates,
                is_multi_diagnosis=True,
                extracted_diagnoses=multi_result["extracted_diagnoses"],
                diagnosis_matches=diagnosis_matches
            )
        
        else:
            logger.info("检测到单一诊断查询，使用传统向量搜索")
            
            # 单一诊断的传统处理方式
            query_vector = embedding_service.encode_query(request.text)
            candidates = milvus_service.search(query_vector, top_k=request.top_k)
            
            logger.info(f"单一诊断查询完成，返回 {len(candidates)} 个候选")
            
            return QueryResponse(
                candidates=candidates,
                is_multi_diagnosis=False
            )
        
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/standardize", response_model=StandardizeResponse, tags=["诊断标准化"])
async def standardize_diagnosis(request: StandardizeRequest):
    """基于LLM的诊断标准化（集成多诊断查询逻辑）"""
    try:
        logger.info(f"收到标准化请求: {request.text}")
        
        if not embedding_service or not milvus_service or not llm_service or not multi_diagnosis_service:
            raise HTTPException(status_code=503, detail="服务未就绪")
        
        # 第一步：使用query接口逻辑进行多诊断识别和向量检索
        logger.info("开始多诊断识别和向量检索...")
        
        # 首先检查是否可能包含多个诊断
        potential_diagnoses = multi_diagnosis_service.text_processor.extract_diagnoses(request.text)
        
        all_candidates = []
        extracted_diagnoses = []
        diagnosis_matches = []
        
        # 判断是否为多诊断查询
        if len(potential_diagnoses) > 1:
            logger.info(f"检测到多诊断查询，提取的诊断: {potential_diagnoses}")
            
            # 执行多诊断匹配
            multi_result = multi_diagnosis_service.match_multiple_diagnoses(
                text=request.text,
                top_k=request.top_k
            )
            
            # 合并所有候选结果
            for match_data in multi_result["matches"]:
                all_candidates.extend(match_data.candidates)
                diagnosis_matches.append(match_data)
            
            extracted_diagnoses = multi_result["extracted_diagnoses"]
            
            # 按分数排序并限制数量
            all_candidates.sort(key=lambda x: x.score, reverse=True)
            all_candidates = all_candidates[:request.top_k]
            
            logger.info(f"多诊断查询完成，共找到 {len(all_candidates)} 个候选结果")
            
        else:
            logger.info("检测到单一诊断查询，使用传统向量搜索")
            
            # 单一诊断的传统处理方式
            query_vector = embedding_service.encode_query(request.text)
            candidates_dict = milvus_service.search(query_vector, top_k=request.top_k)
            
            # 转换为Candidate对象格式（与多诊断保持一致）
            from models.icd_models import Candidate
            for candidate_dict in candidates_dict:
                candidate = Candidate(
                    code=candidate_dict.get("code", ""),
                    title=candidate_dict.get("title", ""),
                    score=candidate_dict.get("score", 0.0)
                )
                all_candidates.append(candidate)
            
            extracted_diagnoses = [request.text]
            
            logger.info(f"单一诊断查询完成，返回 {len(all_candidates)} 个候选")
        
        if not all_candidates:
            logger.warning(f"未找到相关候选: {request.text}")
            return StandardizeResponse(results=[])
        
        # 记录向量检索的候选结果
        candidate_info = []
        for i, candidate in enumerate(all_candidates, 1):
            candidate_info.append(f"{i}. {candidate.title} (编码: {candidate.code}, 分数: {candidate.score:.4f})")
        
        logger.info(f"向量检索获取到 {len(all_candidates)} 个候选词:")
        for info in candidate_info:
            logger.info(f"  {info}")
        
        # 第二步：使用LLM进行标准化
        logger.info(f"开始LLM标准化，使用提供商: {request.llm_provider}")
        
        # 转换为字典格式（LLM服务需要的格式）
        candidates_for_llm = []
        for candidate in all_candidates:
            candidates_for_llm.append({
                "code": candidate.code,
                "title": candidate.title,
                "score": candidate.score
            })
        
        results = llm_service.standardize_diagnosis(
            request.text,
            candidates_for_llm,
            request.llm_provider
        )
        
        logger.info(f"标准化完成，返回 {len(results)} 个结果")
        
        # 如果是多诊断，记录详细信息
        if len(potential_diagnoses) > 1:
            logger.info(f"多诊断标准化详情:")
            logger.info(f"  原始文本: {request.text}")
            logger.info(f"  提取诊断: {extracted_diagnoses}")
            logger.info(f"  诊断匹配数: {len(diagnosis_matches)}")
        
        return StandardizeResponse(results=results)
        
    except Exception as e:
        logger.error(f"标准化失败: {e}")
        raise HTTPException(status_code=500, detail=f"标准化失败: {str(e)}")


@app.post("/embed", response_model=EmbeddingResponse, tags=["向量化"])
async def embed_texts(request: EmbeddingRequest):
    """文本向量化"""
    try:
        logger.info(f"收到向量化请求: {len(request.texts)} 个文本")
        
        if not embedding_service:
            raise HTTPException(status_code=503, detail="向量化服务未就绪")
        
        # 批量向量化
        embeddings = embedding_service.encode_batch(request.texts, show_progress=False)
        
        # 获取模型信息
        model_info = embedding_service.get_model_info()
        model_name = model_info.get("model_name", "unknown")
        
        logger.info(f"向量化完成，生成 {len(embeddings)} 个向量")
        
        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_name
        )
        
    except Exception as e:
        logger.error(f"向量化失败: {e}")
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")



@app.get("/stats", tags=["统计信息"])
async def get_stats():
    """获取系统统计信息"""
    try:
        stats = {}
        
        # Milvus统计
        if milvus_service:
            milvus_stats = milvus_service.get_collection_stats()
            stats["milvus"] = milvus_stats
        
        # 向量化模型信息
        if embedding_service:
            embedding_info = embedding_service.get_model_info()
            stats["embedding"] = embedding_info
        
        # LLM提供商信息
        if llm_service:
            llm_info = llm_service.get_provider_info()
            stats["llm"] = llm_info
        
        return stats
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.post("/llm/switch", tags=["LLM管理"])
async def switch_llm_provider(provider: str):
    """切换LLM提供商"""
    try:
        if not llm_service:
            raise HTTPException(status_code=503, detail="LLM服务未就绪")
        
        success = llm_service.switch_provider(provider)
        if success:
            return {"message": f"成功切换到 {provider}", "current_provider": provider}
        else:
            raise HTTPException(status_code=400, detail=f"切换到 {provider} 失败")
            
    except Exception as e:
        logger.error(f"切换LLM提供商失败: {e}")
        raise HTTPException(status_code=500, detail=f"切换失败: {str(e)}")


@app.get("/llm/test", tags=["LLM管理"])
async def test_llm_connection():
    """测试LLM连接"""
    try:
        if not llm_service:
            raise HTTPException(status_code=503, detail="LLM服务未就绪")
        
        result = llm_service.test_connection()
        return result
        
    except Exception as e:
        logger.error(f"测试LLM连接失败: {e}")
        raise HTTPException(status_code=500, detail=f"测试失败: {str(e)}")


@app.get("/resource/status", tags=["资源管理"])
async def get_resource_status():
    """获取系统资源状态"""
    try:
        resource_status = {}
        
        # Milvus资源状态
        if milvus_service:
            milvus_health = milvus_service.health_check()
            resource_status["milvus"] = milvus_health
        else:
            resource_status["milvus"] = {"healthy": False, "message": "服务未初始化"}
        
        # 向量化服务状态
        if embedding_service:
            embedding_info = embedding_service.get_model_info()
            resource_status["embedding"] = {
                "loaded": embedding_info.get("loaded", False),
                "model_name": embedding_info.get("model_name", "unknown"),
                "model_size_mb": embedding_info.get("model_size_mb", 0)
            }
        else:
            resource_status["embedding"] = {"loaded": False, "message": "服务未初始化"}
        
        # LLM服务状态
        if llm_service:
            llm_info = llm_service.get_provider_info()
            resource_status["llm"] = llm_info
        else:
            resource_status["llm"] = {"connected": False, "message": "服务未初始化"}
        
        # 多诊断服务状态
        if multi_diagnosis_service:
            resource_status["multi_diagnosis"] = {
                "initialized": True,
                "text_processor": "DiagnosisTextProcessor"
            }
        else:
            resource_status["multi_diagnosis"] = {"initialized": False}
        
        return resource_status
        
    except Exception as e:
        logger.error(f"获取资源状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取资源状态失败: {str(e)}")


@app.post("/resource/release", tags=["资源管理"])
async def release_resources():
    """释放系统资源（不关闭应用）"""
    try:
        results = {}
        
        # 释放Milvus集合内存
        if milvus_service:
            release_result = milvus_service.release_collection()
            results["milvus_collection"] = release_result
        
        # 清理GPU缓存（如果有）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                results["gpu_cache"] = {"success": True, "message": "GPU缓存已清理"}
        except ImportError:
            results["gpu_cache"] = {"success": False, "message": "未安装PyTorch"}
        
        # 显式垃圾回收
        import gc
        collected = gc.collect()
        results["garbage_collection"] = {"collected_objects": collected}
        
        logger.info("手动资源释放完成")
        return {"status": "success", "results": results}
        
    except Exception as e:
        logger.error(f"释放资源失败: {e}")
        raise HTTPException(status_code=500, detail=f"释放资源失败: {str(e)}")


@app.post("/resource/reload", tags=["资源管理"])
async def reload_collection():
    """重新加载Milvus集合到内存"""
    try:
        if not milvus_service:
            raise HTTPException(status_code=503, detail="Milvus服务未就绪")
        
        # 先释放现有集合
        release_result = milvus_service.release_collection()
        
        # 重新加载集合
        load_success = milvus_service.load_collection()
        
        if load_success:
            # 获取加载状态
            load_state = milvus_service.get_collection_load_state()
            memory_usage = milvus_service.get_memory_usage()
            
            return {
                "status": "success",
                "message": "集合重新加载成功",
                "release_result": release_result,
                "load_state": load_state,
                "memory_usage": memory_usage
            }
        else:
            raise HTTPException(status_code=500, detail="集合重新加载失败")
        
    except Exception as e:
        logger.error(f"重新加载集合失败: {e}")
        raise HTTPException(status_code=500, detail=f"重新加载集合失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level=os.getenv("API_LOG_LEVEL", "info")
    ) 