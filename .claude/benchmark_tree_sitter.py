#!/usr/bin/env python3
"""
Performance benchmark: AST vs Tree-sitter parsing
Compare parsing performance and chunk quality between methods
"""

import time
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "semantic"))
from tree_sitter_chunker import TreeSitterSemanticChunker

def benchmark_parsing_performance():
    """Benchmark AST vs Tree-sitter parsing performance"""
    
    # Test with a real trading module
    test_file = Path(__file__).parent / "phase2b/advanced_risk_management.py"
    
    if not test_file.exists():
        print("⚠️  Test file not found, using simple test code")
        test_content = """
def calculate_kelly_fraction(win_rate, avg_win, avg_loss):
    edge = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    odds = avg_win / avg_loss
    kelly_fraction = edge / odds
    return kelly_fraction

class RiskManager:
    def __init__(self):
        self.stop_loss = 0.02
        self.position_limit = 0.5
        
    def check_position_limits(self, position_size):
        return position_size <= self.position_limit

def execute_trade(signal, position_size):
    if signal == 'BUY':
        return f"Buying {position_size} units"
    elif signal == 'SELL':
        return f"Selling {position_size} units"
    return "No action"

import numpy as np
import pandas as pd

def ensemble_predict(models, features):
    predictions = []
    for model in models:
        pred = model.predict(features)
        predictions.append(pred)
    return np.mean(predictions)
"""
    else:
        with open(test_file, 'r') as f:
            test_content = f.read()
    
    print("🔬 Tree-sitter vs AST Performance Benchmark")
    print("=" * 60)
    print(f"📄 Test content length: {len(test_content)} characters")
    print(f"📏 Test content lines: {len(test_content.splitlines())} lines")
    print()
    
    # Initialize chunker
    chunker = TreeSitterSemanticChunker()
    
    # Benchmark tree-sitter parsing
    print("🌳 Testing Tree-sitter parsing...")
    tree_sitter_times = []
    tree_sitter_chunks = 0
    
    for i in range(5):
        start_time = time.time()
        
        # Force tree-sitter parsing
        chunks = chunker._tree_sitter_chunk(test_content, "test_module.py")
        
        end_time = time.time()
        parse_time = (end_time - start_time) * 1000
        tree_sitter_times.append(parse_time)
        tree_sitter_chunks = len(chunks)
        
        print(f"  Run {i+1}: {parse_time:.2f}ms, {len(chunks)} chunks")
    
    tree_sitter_avg = sum(tree_sitter_times) / len(tree_sitter_times)
    
    print()
    print("🔧 Testing AST parsing...")
    ast_times = []
    ast_chunks = 0
    
    for i in range(5):
        start_time = time.time()
        
        # Force AST parsing
        chunks = chunker._enhanced_ast_chunk(test_content, "test_module.py")
        
        end_time = time.time()
        parse_time = (end_time - start_time) * 1000
        ast_times.append(parse_time)
        ast_chunks = len(chunks)
        
        print(f"  Run {i+1}: {parse_time:.2f}ms, {len(chunks)} chunks")
    
    ast_avg = sum(ast_times) / len(ast_times)
    
    print()
    print("📊 Performance Comparison:")
    print(f"├── Tree-sitter avg: {tree_sitter_avg:.2f}ms ({tree_sitter_chunks} chunks)")
    print(f"├── AST avg: {ast_avg:.2f}ms ({ast_chunks} chunks)")
    print(f"├── Speed ratio: {ast_avg/tree_sitter_avg:.2f}x")
    print(f"└── Chunk difference: {tree_sitter_chunks - ast_chunks} chunks")
    
    # Performance target analysis
    target_time = 2.0  # < 2ms target
    print()
    print("🎯 Performance Target Analysis:")
    print(f"├── Target: < {target_time}ms")
    print(f"├── Tree-sitter: {'✅ PASS' if tree_sitter_avg < target_time else '❌ FAIL'}")
    print(f"└── AST: {'✅ PASS' if ast_avg < target_time else '❌ FAIL'}")
    
    # Quality analysis
    print()
    print("🔍 Quality Analysis:")
    
    # Test tree-sitter chunks
    ts_chunks = chunker._tree_sitter_chunk(test_content, "test_module.py")
    ast_chunks = chunker._enhanced_ast_chunk(test_content, "test_module.py")
    
    ts_trading_relevant = sum(1 for c in ts_chunks if c.trading_relevance > 0.5)
    ast_trading_relevant = sum(1 for c in ast_chunks if c.trading_relevance > 0.5)
    
    print(f"├── Tree-sitter trading chunks: {ts_trading_relevant}/{len(ts_chunks)}")
    print(f"├── AST trading chunks: {ast_trading_relevant}/{len(ast_chunks)}")
    
    # Check semantic tags
    ts_tags = sum(len(c.semantic_tags) for c in ts_chunks)
    ast_tags = sum(len(c.semantic_tags) for c in ast_chunks)
    
    print(f"├── Tree-sitter semantic tags: {ts_tags}")
    print(f"└── AST semantic tags: {ast_tags}")
    
    return {
        'tree_sitter_avg': tree_sitter_avg,
        'ast_avg': ast_avg,
        'tree_sitter_chunks': tree_sitter_chunks,
        'ast_chunks': ast_chunks,
        'speed_ratio': ast_avg/tree_sitter_avg,
        'target_met': tree_sitter_avg < target_time
    }

if __name__ == "__main__":
    benchmark_parsing_performance()