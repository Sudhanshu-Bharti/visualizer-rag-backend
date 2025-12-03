#!/usr/bin/env python3
"""
Memory optimization utility for RAG backend
"""
import gc
import os
import psutil
import torch
from typing import Dict, Any


class MemoryManager:
    """Memory management utility for the RAG application"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get current memory usage information"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        info = {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "total_mb": psutil.virtual_memory().total / 1024 / 1024,
        }
        
        if torch.cuda.is_available():
            info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return info
    
    @staticmethod
    def cleanup_memory(force_gc: bool = True) -> None:
        """Perform memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        if force_gc:
            gc.collect()
    
    @staticmethod
    def check_memory_threshold(threshold_mb: int = 1000) -> bool:
        """Check if memory usage is below threshold"""
        memory_info = MemoryManager.get_memory_info()
        return memory_info["rss_mb"] < threshold_mb
    
    @staticmethod
    def log_memory_usage(logger, context: str = ""):
        """Log current memory usage"""
        info = MemoryManager.get_memory_info()
        logger.info(f"Memory usage {context}: RSS={info['rss_mb']:.1f}MB, "
                   f"Available={info['available_mb']:.1f}MB, "
                   f"Percent={info['percent']:.1f}%")