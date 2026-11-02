# Practical guide: the course's reusable PyTorch helpers

This section assumes you already know Python modules, imports, classes, and functions. Its purpose is to expose the **PyTorch interfaces** that must survive moving code out of a notebook.

| Helper | Required inputs | Returns / side effects |
| --- | --- | --- |
| `data_setup.create_dataloaders` | train/test image paths, transform, batch_size, num_workers | train/test DataLoaders and ordered class names |
| `engine.train_step` | model, loader, loss_fn, optimizer, device | Numeric train loss/accuracy; updates model parameters |
| `engine.test_step` | model, loader, loss_fn, device | Numeric evaluation loss/accuracy; no parameter updates |
| `engine.train` | both loaders, model, optimizer, loss, epochs, device | Four per-epoch history lists |
| `utils.save_model` | model, target directory, checkpoint filename | Writes the model state_dict |

These are course helpers, not built-in torch functions. Their implementations are included under going_modular/. Inspect the signature before calling one, since your own helpers may use different argument names or metric units.

The model should remain an nn.Module with registered child layers/parameters. The optimizer must reference the exact model instance being trained. Keep preprocessing, target dtype, and device choices explicit when passing objects between helpers. Do not accidentally construct a new optimizer/model for every batch.

When moving existing working code into these helpers, preserve the sequence forward, loss, zero_grad, backward, step. Evaluation stays under eval/inference mode. Store detached numeric metrics rather than graph-tracking loss tensors. Reuse a complete working run to understand each helper's role before reorganizing the whole workflow.
