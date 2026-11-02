
# Practical guide: images, batches, and CNNs

**Prerequisite:** train/evaluate the Chapter 02 classifier independently. This is an optional vision application of that same workflow; it is not required before returning to your AI project.

## Dataset and DataLoader (Tasks 1–12)
A **Dataset** supplies individual examples. `dataset[i]` normally returns `(input, target)` and `len(dataset)` is sample count. A **DataLoader** groups examples into batches and optionally shuffles their order. Its length is **batch count**, not sample count.

Try the interface with tiny tensors before downloading images:

```python
from torch.utils.data import TensorDataset, DataLoader
small_data = TensorDataset(torch.randn(6, 3), torch.tensor([0, 1, 0, 1, 0, 1]))
small_loader = DataLoader(small_data, batch_size=2, shuffle=False, num_workers=0)
for inputs, targets in small_loader:
    print(inputs.shape, targets.shape)  # (2, 3), (2,)
```

| Constructor argument | Meaning |
| --- | --- |
| `dataset` | Dataset object whose examples should be batched |
| `batch_size` | Maximum examples per batch; last batch can be smaller |
| `shuffle` | Randomize sample order; normally True for training and False for testing |
| `num_workers` | Additional data-loading processes; use zero in these Mac notebooks |

`next(iter(loader))` creates an iterator and requests its first batch. It is useful for inspecting the interface; a training epoch instead loops through the entire loader.

`datasets.FashionMNIST(root, train, transform, download)` creates the image dataset: root is the storage folder, train selects the official split, transform preprocesses each retrieved image, and download permits fetching missing data. `ToTensor()` converts PIL images into channel-first float tensors. A grayscale image is $(1,28,28)$; a full batch here is $(32,1,28,28)$. Labels collate into int64 shape $(32,)$.

## A model consumes features; images have extra axes (Tasks 13–24)
`nn.Flatten(start_dim=1)` preserves the batch axis and collapses all remaining dimensions into features. The default start_dim is one. Thus $(N,1,28,28)$ becomes $(N,784)$. Flattening an **unbatched** image with the same default does not mean the same thing: inspect shapes before guessing.

Use Linear and Sequential exactly as in Chapter 02, with 784 input features and ten output logits. CrossEntropyLoss still takes int64 class labels, not one-hot vectors.

## Minibatch training helpers (Tasks 19–29)
An epoch now contains several optimizer updates, one per training batch:

1. Enter train mode, loop over `(X_batch,y_batch)`, move both to device.
2. Forward, loss, zero gradients, backward, optimizer step for each batch.
3. Accumulate detached numeric loss and accuracy; average using the method stated in the task.
4. Evaluate test batches in eval/inference mode without updates.

A `train_step` helper performs one whole training epoch in this course, despite its name. Give it `model`, loader, loss, optimizer, and device explicitly. A `test_step` helper has no optimizer. Return numbers so the outer loop can record them. Keep accuracy units consistent; these FashionMNIST tasks use percentages.

`timer()` from timeit returns a time reading; subtract the before reading from the after reading for elapsed seconds. GPU operations can be asynchronous, so simple host timing is only an approximate comparison unless you synchronize. The core task here is learning batching, not establishing a hardware benchmark.

## Convolution and pooling (Tasks 30–42)
`nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0)` creates a learnable convolution layer. The first two arguments describe **channels**, not image height/width. A kernel slides over spatial dimensions and produces out_channels feature maps. `padding` adds border positions; `stride` is movement between applications.

```python
example_conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
example_pool = nn.MaxPool2d(kernel_size=2)
feature_maps = example_conv(torch.randn(2, 1, 8, 8))
print(feature_maps.shape)            # (2, 4, 8, 8)
print(example_pool(feature_maps).shape)  # (2, 4, 4, 4)
```

Pool layers select a maximum in each spatial window and have no learned weights here. ReLU changes values but not shape. Before defining a final Linear layer, work out or print the flattened feature count after your convolution/pooling sequence.

## Interpret results, then save (Tasks 43–61)
`torch.softmax(logits,dim=1)` produces class probabilities; argmax selects class index. Pair predictions with original labels and class_names to identify errors. A confusion matrix counts true/predicted label pairs; the axes must be labelled before interpreting off-diagonal cells.

`pd.DataFrame([result_dict_a,result_dict_b])` creates one comparison row per model. A named Series supports plotting with `.plot(kind='barh')`. You can reuse matplotlib line plots for histories.

Save/reload with the Chapter 01 state_dict workflow. The reconstructed CNN must have exactly the same layer/parameter shapes as the checkpoint.

References: [Dataset/DataLoader tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html), [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
