# BFloat16 Conversion Error - Fixed ✅

## Issue

Users encountered the following error when running the notebook script:

```
Error in embedding generation: Got unsupported ScalarType BFloat16
```

## Root Cause

When using bfloat16 or float16 dtype for GPU optimization, PyTorch tensors cannot be directly converted to numpy arrays because numpy doesn't support these data types natively. The error occurred in the embedding generation function when trying to convert embeddings to numpy:

```python
embeddings = embeddings.cpu().numpy()  # ❌ Fails with bfloat16/float16
```

## Solution

Added automatic dtype conversion to float32 before numpy conversion in the `qwen3_embedding` function:

```python
# Convert to float32 if using bfloat16 or float16 (for consistency)
if embeddings.dtype in (torch.bfloat16, torch.float16):
    embeddings = embeddings.to(torch.float32)

# Move to CPU and convert to numpy
embeddings = embeddings.cpu().numpy()  # ✅ Now works correctly
```

## Fixed in Commit

- **Commit**: 4402b44
- **Date**: 2026-01-04
- **File**: `notebook_hirag_qwen3.py`

## Additional Notes

### Model Name Clarification

Users should use the correct model name:
- ✅ **Correct**: `Qwen/Qwen3-8B-Base` (non-instruct version)
- ❌ **Incorrect**: `Qwen/Qwen3-8B` (different model)

The non-instruct version is recommended for the HiRAG system as specified in the original requirements.

## Testing

The fix has been validated for:
- ✅ BFloat16 dtype (H100/A100 GPUs)
- ✅ Float16 dtype (older GPUs)
- ✅ Float32 dtype (CPU or explicit float32)

## Performance Impact

The conversion from bfloat16/float16 to float32 has minimal performance impact:
- Conversion happens only once per batch
- Memory usage slightly increases during conversion (temporary)
- No impact on model inference speed
- Final numpy arrays are in float32 (standard precision)

## Documentation Updates

Updated the following files with troubleshooting information:
- `NOTEBOOK_USAGE.md` (English)
- `NOTEBOOK_USAGE_CN.md` (Chinese)

Both documentation files now include:
1. BFloat16 conversion error explanation
2. Model name clarification
3. Code snippet for the fix

## Related Issues

This fix resolves the issue reported in:
- Comment ID: 3707999719
- User: @alpassss

## References

- PyTorch BFloat16: https://pytorch.org/docs/stable/tensors.html#torch.bfloat16
- NumPy dtypes: https://numpy.org/doc/stable/reference/arrays.scalars.html
- Original implementation: commit e7da990
