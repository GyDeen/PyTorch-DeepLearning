
# Practical guide: implementing a paper architecture

**Prerequisite:** understand Module composition, convolution output shapes, flatten/permute, trainable parameters, and at least one complete training run. This is an optional advanced exercise. You can return to CS189 or your AI project without implementing a Vision Transformer first.

## Image patches are a shape transformation (Tasks 1–29)
The model's input is $(N,C,H,W)$. A Conv2d with kernel and stride equal to patch size maps non-overlapping image patches to embedding vectors. Its out_channels is the embedding width, not the number of classes.

A small worked shape example:

```python
projection = nn.Conv2d(3, 8, kernel_size=4, stride=4)
projected = projection(torch.randn(2, 3, 8, 8))   # (2, 8, 2, 2)
tokens = projected.flatten(start_dim=2).permute(0, 2, 1)
print(tokens.shape)  # (2, 4, 8): batch, patch count, embedding width
```

`Flatten(start_dim=2)` combines spatial axes, then permute reorders them. It is not interchangeable with an arbitrary reshape of the same sizes.

## Class token and position parameter (Tasks 30–36)
`nn.Parameter` can hold a learned vector/tensor, not just a scalar. A single stored class-token parameter of shape $(1,1,D)$ can be `.expand(batch_size,-1,-1)` to appear in every image's token sequence. Here -1 means keep the existing size; expansion shares storage.

`torch.cat(tensors,dim=...)` joins along an **existing** axis, unlike stack which adds a new axis. Prepending the class token increases sequence length by one. Adding positional parameters requires compatible shapes and keeps sequence length unchanged.

## New layer APIs (Tasks 37–44)
| API | Main arguments | Shape/return contract |
| --- | --- | --- |
| `nn.LayerNorm(normalized_shape)` | Embedding width, usually an integer | Normalizes final feature axis; preserves shape |
| `nn.MultiheadAttention(embed_dim,num_heads,dropout=...,batch_first=True)` | Feature width divisible by head count | Call with query,key,value; returns output and attention weights |
| `nn.GELU()` | No required arguments | Element-wise nonlinearity, same shape |
| `nn.Dropout(p)` | Probability of setting an entry to zero during training | Preserves shape; disabled in eval mode |
| `nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,...)` | Embedding width, head count, hidden MLP width | With batch_first=True, consumes/returns $(N,S,D)$ |

Attention usage on an unrelated small example:

```python
attention = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
example_tokens = torch.randn(2, 4, 8)
attended, attention_weights = attention(example_tokens, example_tokens, example_tokens)
```

For **self** attention, query/key/value come from the same sequence. `output, _ = ...` discards an unused second return. A residual connection adds a block's output to its input, so their shapes must match.

## Compose and train (Tasks 45–50)
Store blocks in __init__, not forward. Sequential registers and automatically executes its child modules in order; ModuleList registers its children but requires an explicit forward loop. An ordinary Python list would not register those layers for parameter discovery/device movement.

The class-token output is selected with `sequence[:,0]`, preserving batch and embedding axes. A final Linear maps that embedding to class logits. The existing cross-entropy training loop then works unchanged.

Large parameter counts and scarce data can make scratch training slow and poor. The checkpoint is understanding the implementation and observing its limitations, not matching a published result.

## Pretrained comparison (Tasks 51–63)
Use torchvision's pretrained weights API, freeze backbone parameters, and replace the class head. This is the same transfer-learning pattern as Chapter 06; inspect the specific head attribute rather than assuming it is called classifier in every model.

Reference: [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
