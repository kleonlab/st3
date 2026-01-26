#!/usr/bin/env python3
"""Quick test to verify AMP implementation works correctly."""

import torch
import torch.nn as nn

# Simple test to verify AMP setup
def test_amp_setup():
    """Test that AMP context managers work correctly."""
    
    print("Testing AMP Implementation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, AMP will not provide speedup")
        return
    
    device = torch.device('cuda')
    
    # Test BF16
    print("\n1. Testing BF16 autocast...")
    try:
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.matmul(x, y)
            print(f"   ✓ BF16 autocast works - output dtype: {z.dtype}")
    except Exception as e:
        print(f"   ✗ BF16 failed: {e}")
    
    # Test FP16
    print("\n2. Testing FP16 autocast...")
    try:
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.matmul(x, y)
            print(f"   ✓ FP16 autocast works - output dtype: {z.dtype}")
    except Exception as e:
        print(f"   ✗ FP16 failed: {e}")
    
    # Test GradScaler
    print("\n3. Testing GradScaler...")
    try:
        scaler = torch.cuda.amp.GradScaler()
        model = nn.Linear(10, 10).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        
        x = torch.randn(5, 10, device=device)
        
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            output = model(x)
            loss = output.sum()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"   ✓ GradScaler works - scale: {scaler.get_scale()}")
    except Exception as e:
        print(f"   ✗ GradScaler failed: {e}")
    
    # Test BF16 without scaler
    print("\n4. Testing BF16 without GradScaler...")
    try:
        model = nn.Linear(10, 10).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        
        x = torch.randn(5, 10, device=device)
        
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model(x)
            loss = output.sum()
        
        loss.backward()
        optimizer.step()
        
        print(f"   ✓ BF16 without GradScaler works")
    except Exception as e:
        print(f"   ✗ BF16 without GradScaler failed: {e}")
    
    print("\n✅ AMP implementation test complete!")
    print("\nRecommendations:")
    print("  - Use BF16 (bfloat16) for best stability on modern GPUs")
    print("  - Use FP16 (float16) on older GPUs if BF16 is not supported")
    print("  - Set use_amp: true in your config file to enable")

if __name__ == "__main__":
    test_amp_setup()

