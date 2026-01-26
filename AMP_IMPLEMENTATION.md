# Mixed Precision (BF16/FP16) Training Implementation

## Overview
Added automatic mixed precision (AMP) training support to the SEDD perturbation prediction training pipeline. This enables training in bfloat16 or float16 precision for faster training and reduced memory usage.

## Changes Made

### 1. Configuration File (`configs/perturbseq_small.yaml`)
Added AMP settings to the training configuration:
```yaml
training:
  # ... existing settings ...
  # Mixed precision training
  use_amp: true  # Enable automatic mixed precision
  amp_dtype: bfloat16  # Options: bfloat16, float16
```

### 2. Training Script (`scripts/train_perturbseq.py`)
- Added command-line arguments for AMP:
  - `--use_amp`: Enable/disable mixed precision training
  - `--amp_dtype`: Choose between `bfloat16` and `float16`
- Modified trainer initialization to pass AMP settings
- Added dtype mapping from string to torch dtype

### 3. Trainer Classes (`sedd/trainer.py`)
Modified both `SEDDTrainer` and `PerturbationTrainer` classes:

#### New Parameters:
- `use_amp` (bool): Enable automatic mixed precision
- `amp_dtype` (torch.dtype): Precision dtype (torch.bfloat16 or torch.float16)

#### Implementation Details:
- **Forward Pass**: Wrapped model forward passes in `torch.cuda.amp.autocast()` context
- **Backward Pass**: 
  - For `float16`: Uses `GradScaler` for gradient scaling to prevent underflow
  - For `bfloat16`: No gradient scaling needed (better numerical stability)
- **Gradient Clipping**: Properly integrated with gradient scaling
- **Validation**: Also uses autocast for consistent precision

## Usage

### Basic Usage (BF16 - Recommended)
```bash
python scripts/train_perturbseq.py --config configs/perturbseq_small.yaml
```
With the config file set to `use_amp: true` and `amp_dtype: bfloat16`, training will automatically use bfloat16 precision.

### Command Line Override
```bash
# Enable BF16
python scripts/train_perturbseq.py --use_amp --amp_dtype bfloat16

# Enable FP16
python scripts/train_perturbseq.py --use_amp --amp_dtype float16

# Disable AMP (use FP32)
python scripts/train_perturbseq.py
```

## Benefits

### BFloat16 (Recommended)
- ✅ Better numerical stability than FP16
- ✅ No gradient scaling needed
- ✅ ~2x faster training on modern GPUs (A100, H100)
- ✅ ~50% memory reduction
- ✅ Same dynamic range as FP32

### Float16
- ✅ ~2x faster training on older GPUs
- ✅ ~50% memory reduction
- ⚠️ Requires gradient scaling
- ⚠️ Less stable than BF16

## Technical Details

### Forward Pass
```python
with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    loss = self.model.get_loss(...)
```

### Backward Pass (FP16 with scaling)
```python
if self.scaler is not None:
    self.scaler.scale(loss).backward()
    if self.gradient_clip > 0:
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
    self.scaler.step(self.optimizer)
    self.scaler.update()
```

### Backward Pass (BF16 - no scaling)
```python
else:
    loss.backward()
    if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
    self.optimizer.step()
```

## Checkpoint Compatibility
- Checkpoints saved with AMP training are fully compatible with FP32 training
- Model weights are always saved in FP32 precision
- Optimizer states include AMP scaler state when using FP16

## Recommendations
1. **Use BF16 on modern GPUs** (A100, H100, RTX 30xx+) for best stability
2. **Use FP16 on older GPUs** (V100, RTX 20xx) if BF16 is not supported
3. **Monitor training loss** - if you see NaN values with FP16, switch to BF16
4. **Batch size**: You may be able to increase batch size by ~2x due to memory savings

## Testing
To verify AMP is working:
1. Check the training log for: "Using automatic mixed precision training with dtype: bfloat16"
2. Monitor GPU memory usage (should be ~50% lower than FP32)
3. Check training speed (should be ~2x faster on supported hardware)

