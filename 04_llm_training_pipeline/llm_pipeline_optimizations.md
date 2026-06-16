# The GPT-2 Hardware Optimization Waterfall

| Optimization Step | Code Implementation | Runtime / Step | Tokens / Sec | Cumulative Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **0. Baseline** | Naive PyTorch implementation (float32). | ~1000 ms | ~16,000 | 1.0x |
| **1. TensorFloat-32 (TF32)** | `torch.set_float32_matmul_precision('high')` | ~330 ms | ~49,000 | **~3.0x** |
| **2. Mixed Precision (BF16)** | `with torch.autocast(..., dtype=torch.bfloat16):` | ~300 ms | ~54,000 | **~3.3x** |
| **3. Pre-compilation** | `torch.compile(model)` | ~130 ms | ~126,000 | **~7.7x** |
| **4. FlashAttention** | `F.scaled_dot_product_attention(...)` | ~96 ms | ~170,000 | **~10.4x** |
| **5. Vocabulary Padding** | Padding `vocab_size` from 50,257 to 50,304. | ~93 ms | ~176,000 | **~10.8x** |

*(Note: These metrics assume a batch size of 16 and a sequence length of 1024, processing a total of 16,384 tokens per step.)*

---

### What drives each performance jump?

*   **TF32 Precision:** Standard 32-bit floats are truncated to 19 bits by dropping 13 bits from the mantissa. This has a negligible effect on accuracy for the purpose of training, but it allows the GPU's Tensor Cores to execute matrix multiplications ~8x faster. This is applied as a global setting to the pipeline - PyTorch automatically determines when and where to apply this truncation for efficiency gains (typically on the heavy matrix multiplications like the linear layers and attention layers). This is a compute optimization built into the GPU hardware and activated in code.
*   **Mixed Precision - Bfloat16 Autocast:** Mixed precision similarly lowers the precision of numbers used in the model during training to 16 bits for specific operations. `bfloat16` is uniquely suited for this because it shrinks the mantissa but preserves the exact exponent range of `float32`, meaning you don't have to wrestle with gradient scalers to prevent numeric underflow during training. This is a memory optimization. Note: You never use TF32 and BFloat16 at the same time! TF32 can be thought of as the 'catch-all' optimization, and Bfloat16 is used when you really need to maximize speed and efficiency out of your hardware (which often is a goal in training LLMs).
*   **Pre-compilation using `torch.compile`:** Every time you run an operation in training, you have to read the data from the GPU's main memory, move it to the core to do math computations, and write the results back to main memory. Often this back and forth is a major bottleneck in training. PyTorch 2.0's graph compiler optimizes for this. At the start of training, it watches the code execute and analyzes what math operations are happening (the model architecture) and optimizes how the hardware executes operations through **kernel fusion**. Instead of constantly shifting data back and forth between High Bandwidth Memory (HBM) and the GPU's compute cores for every single math operation, it fuses them together (e.g., executing a linear projection and a GeLU activation in one continuous pass). This is primarily a memory bandwidth optimization.
*   **FlashAttention:** This is an algorithmic optimization of the self-attention mechanism. Instead of fully materializing the massive N x N attention matrix in VRAM, FlashAttention computes attention scores and applies the softmax incrementally in localized "tiles." This maximizes the use of ultra-fast on-chip SRAM and drastically minimizes sluggish memory reads/writes to HBM. This is a memory optimization and a compute optimization.
*   **"Nice Numbers" (Vocab Padding):** The original GPT-2 vocabulary size is 50,257, which is an odd number while the GPU hardware works in powers of 2.By simply padding the vocabulary size with empty slots to reach 50,304 (the nearest multiple of 2), the massive matrix multiplications in the final language modeling head align perfectly with the GPU's memory layout, yielding a final bump in computational efficiency. This is a compute optimization.