"""
Gamium 日志工具
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class GamiumLogger:
    """
    Gamium 训练日志记录器
    
    功能：
    - 控制台彩色输出
    - 文件日志记录
    - 训练指标追踪
    - JSON 格式导出
    """
    
    def __init__(
        self,
        name: str = "Gamium",
        log_dir: str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '\033[36m[%(name)s]\033[0m %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # 指标追踪
        self.metrics: Dict[str, list] = {}
        self.run_info = {
            'start_time': timestamp,
            'name': name,
        }
    
    def info(self, msg: str):
        """信息日志"""
        self.logger.info(msg)
    
    def debug(self, msg: str):
        """调试日志"""
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        """警告日志"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """错误日志"""
        self.logger.error(msg)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """记录训练指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        entry = {'value': value}
        if step is not None:
            entry['step'] = step
        entry['time'] = datetime.now().isoformat()
        
        self.metrics[name].append(entry)
    
    def log_dict(self, data: Dict[str, Any], prefix: str = ""):
        """记录字典数据"""
        for key, value in data.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)):
                self.log_metric(metric_name, value)
    
    def save_metrics(self, filename: str = None):
        """保存指标到 JSON 文件"""
        if filename is None:
            filename = f"metrics_{self.run_info['start_time']}.json"
        
        filepath = self.log_dir / filename
        
        data = {
            'run_info': self.run_info,
            'metrics': self.metrics,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.info(f"指标已保存到: {filepath}")
    
    def print_summary(self):
        """打印训练摘要"""
        print("\n" + "=" * 60)
        print(f"📊 训练摘要 - {self.name}")
        print("=" * 60)
        
        for name, values in self.metrics.items():
            if values:
                recent = values[-1]['value']
                avg = sum(v['value'] for v in values) / len(values)
                print(f"  {name}: 最新={recent:.4f}, 平均={avg:.4f}, 样本数={len(values)}")
        
        print("=" * 60)


if __name__ == "__main__":
    # 测试
    logger = GamiumLogger("test")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.log_metric("reward", 10.5, step=1)
    logger.log_metric("reward", 12.3, step=2)
    logger.log_metric("loss", 0.05, step=1)
    logger.print_summary()

