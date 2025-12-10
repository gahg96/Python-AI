"""
统一模型网关 - 支持多种模型接入
"""
import httpx
import time
import asyncio
from typing import Dict, Optional, List, Any
from enum import Enum
import json


class ModelType(Enum):
    """模型类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL_LLM = "local_llm"
    RAG = "rag"
    AGENT = "agent"


class ModelGateway:
    """统一模型网关"""
    
    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def register_model(self, model_id: str, model_type: ModelType, config: Dict):
        """注册模型
        
        Args:
            model_id: 模型唯一标识
            model_type: 模型类型
            config: 配置信息，包含API密钥、端点等
        """
        self.models[model_id] = {
            'id': model_id,
            'type': model_type,
            'config': config,
            'status': 'active'
        }
        print(f"✅ 模型 {model_id} ({model_type.value}) 已注册")
    
    async def call_model(self, model_id: str, prompt: str, context: Optional[List] = None) -> Dict:
        """统一调用接口
        
        Args:
            model_id: 模型ID
            prompt: 用户提示
            context: 对话上下文
            
        Returns:
            {
                'response': str,  # 模型响应
                'latency': float,  # 响应延时（秒）
                'tokens': int,     # 使用的token数（如果有）
                'error': str       # 错误信息（如果有）
            }
        """
        if model_id not in self.models:
            return {
                'response': '',
                'latency': 0,
                'tokens': 0,
                'error': f'模型 {model_id} 未注册'
            }
        
        model_info = self.models[model_id]
        model_type = model_info['type']
        config = model_info['config']
        
        start_time = time.time()
        
        try:
            if model_type == ModelType.OPENAI:
                result = await self._call_openai(model_id, prompt, context, config)
            elif model_type == ModelType.ANTHROPIC:
                result = await self._call_anthropic(model_id, prompt, context, config)
            elif model_type == ModelType.LOCAL_LLM:
                result = await self._call_local_llm(model_id, prompt, context, config)
            else:
                # 模拟调用（用于演示）
                result = await self._call_mock(model_id, prompt, context, config)
            
            latency = time.time() - start_time
            result['latency'] = latency
            
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            return {
                'response': '',
                'latency': latency,
                'tokens': 0,
                'error': str(e)
            }
    
    async def _call_openai(self, model_id: str, prompt: str, context: Optional[List], config: Dict) -> Dict:
        """调用OpenAI API"""
        api_key = config.get('api_key', '')
        endpoint = config.get('endpoint', 'https://api.openai.com/v1/chat/completions')
        model_name = config.get('model_name', 'gpt-3.5-turbo')
        
        messages = []
        if context:
            messages.extend(context)
        messages.append({'role': 'user', 'content': prompt})
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model_name,
            'messages': messages,
            'temperature': config.get('temperature', 0.7)
        }
        
        try:
            response = await self.client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return {
                'response': data['choices'][0]['message']['content'],
                'tokens': data.get('usage', {}).get('total_tokens', 0),
                'error': None
            }
        except Exception as e:
            return {
                'response': '',
                'tokens': 0,
                'error': f'OpenAI API错误: {str(e)}'
            }
    
    async def _call_anthropic(self, model_id: str, prompt: str, context: Optional[List], config: Dict) -> Dict:
        """调用Anthropic API"""
        # 类似OpenAI的实现
        return await self._call_mock(model_id, prompt, context, config)
    
    async def _call_local_llm(self, model_id: str, prompt: str, context: Optional[List], config: Dict) -> Dict:
        """调用本地LLM"""
        endpoint = config.get('endpoint', 'http://localhost:8000/v1/chat/completions')
        
        messages = []
        if context:
            messages.extend(context)
        messages.append({'role': 'user', 'content': prompt})
        
        payload = {
            'messages': messages,
            'temperature': config.get('temperature', 0.7)
        }
        
        try:
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return {
                'response': data.get('choices', [{}])[0].get('message', {}).get('content', ''),
                'tokens': 0,
                'error': None
            }
        except Exception as e:
            return {
                'response': '',
                'tokens': 0,
                'error': f'本地LLM错误: {str(e)}'
            }
    
    async def _call_mock(self, model_id: str, prompt: str, context: Optional[List], config: Dict) -> Dict:
        """模拟调用（用于演示，不实际调用API）"""
        # 模拟不同模型的响应风格
        model_name = config.get('model_name', model_id)
        
        # 模拟响应延时
        await asyncio.sleep(0.5 + hash(model_id) % 3 * 0.3)
        
        # 根据模型ID生成不同的响应
        responses = {
            'gpt-4': f"[GPT-4模拟] 针对您的问题：{prompt[:50]}...，我的回答是：这是一个需要综合考虑的问题。根据金融行业最佳实践，我建议...",
            'claude-3': f"[Claude-3模拟] 关于您提到的：{prompt[:50]}...，从合规和风险控制的角度，我认为...",
            'local-model': f"[本地模型模拟] 对于这个问题，我的理解是：{prompt[:50]}...，建议采取以下措施...",
            'rag-model': f"[RAG模型模拟] 根据知识库检索，关于：{prompt[:50]}...，相关文档显示...",
        }
        
        response_text = responses.get(model_id, f"[{model_name}模拟] 这是对问题的回答：{prompt[:100]}...")
        
        return {
            'response': response_text,
            'tokens': len(response_text) // 4,  # 粗略估算
            'error': None
        }
    
    def list_models(self) -> List[Dict]:
        """获取所有已注册的模型"""
        result = []
        for model_id, model_info in self.models.items():
            result.append({
                'id': model_info.get('id', model_id),
                'type': model_info.get('type'),
                'status': model_info.get('status', 'active'),
                'config': model_info.get('config', {})
            })
        return result
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """获取模型信息"""
        return self.models.get(model_id)
    
    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()


# 全局网关实例
gateway = ModelGateway()

# 导入asyncio用于异步调用
import asyncio

