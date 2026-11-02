
# Practical guide: recording experiments

**Prerequisite:** finish one training/evaluation run and understand its metric history dictionary. Logging is a way to record those same numbers; it is not another optimizer or training algorithm.

## PyTorch seeds and course-helper contracts (Tasks 1–11)
`torch.manual_seed(seed)` seeds PyTorch's device generators. The course helper set_seeds wraps this operation; it does not seed sklearn or NumPy. The helper download_data(source,destination,remove_source=True) returns the extracted dataset directory as a Path. Use that returned directory for your loaders. Seeds alone do not guarantee identical results across hardware/software configurations.

The model/data preparation is the Chapter 06 workflow: weight-specific transforms, frozen backbone, fresh classifier, matching loss and optimizer.

## SummaryWriter and TensorBoard (Tasks 12–19)
`SummaryWriter(log_dir=...)` writes numeric events into files. TensorBoard is a viewer that reads those files. Your Python loop is still responsible for producing every number.

```python
from torch.utils.tensorboard import SummaryWriter
example_writer = SummaryWriter(log_dir="runs/example")
example_writer.add_scalar("loss/train", 0.75, global_step=0)
example_writer.add_scalar("loss/train", 0.50, global_step=1)
example_writer.close()
```

`add_scalar(tag,value,global_step)` logs one number; tag groups related points, value is a Python number, global_step is usually epoch/update index. `add_scalars(main_tag,tag_scalar_dict,global_step)` groups several series under one chart. Call `close()` to flush and finish the writer. Do not close a writer before the loop that needs it.

Each training run should receive its own writer. Logging into a shared writer would mix runs even when the models are different.

`%load_ext tensorboard` and `%tensorboard --logdir runs` are notebook commands; the `%` syntax belongs to IPython. A shell equivalent is `python -m tensorboard.main --logdir runs` in your activated environment.

## Compare runs fairly (Tasks 20–30)
A model factory function creates a fresh model **each time it is called**. A name string such as 'effnetb0' is not a model object; use it to decide which factory to call. Create a new optimizer after constructing the model, and a distinct writer for each run.

A nested loop enumerates dataset choice, epoch count, and architecture. Give each run a name containing all three. Use a common held-out dataset. An evaluation metric used to choose among runs serves as a validation metric; reserve a separate untouched test set before reporting a final result on your own project.

## Reload and inspect (Tasks 31–34)
Select the checkpoint using your actual results, not the course's assumed winner. Recreate its architecture, load the state dictionary, set eval mode, and apply its matching transforms. Record size and latency separately from predictive quality.

**Independent transfer:** try two learning rates on your Chapter 01 or 02 model and log the histories. That is sufficient first practice; eight image-model experiments are optional.
