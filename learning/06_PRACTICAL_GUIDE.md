
# Practical guide: transfer learning

**Prerequisite:** train a classifier and understand parameters, submodules, DataLoader, train/eval modes, and checkpoints. This is an optional shortcut for image problems, not a prerequisite for training a feature-based AI model.

## Weights and transforms (Tasks 1–10)
`torchvision.models` contains model constructors. A weights enum such as `EfficientNet_B0_Weights.DEFAULT` identifies a pretrained parameter set and its metadata. It is not the model. Pass it as `weights=...` to the model constructor. `weights.transforms()` gives its matching callable image preprocessing pipeline.

```python
from torchvision.models import resnet18, ResNet18_Weights
example_weights = ResNet18_Weights.DEFAULT
example_preprocess = example_weights.transforms()
example_model = resnet18(weights=example_weights)
```

This ResNet example illustrates the API; your task uses EfficientNet. The first pretrained construction can download weights. Use the paired transforms rather than assuming every architecture uses exactly the same resize/crop/normalization.

`data_setup.create_dataloaders(train_dir,test_dir,transform,batch_size,num_workers=...)` is a **course helper**, not a built-in torch function. It creates ImageFolder datasets and DataLoaders, returning `(train_loader,test_loader,class_names)`; supply zero workers in these notebooks.

## Freeze the backbone and replace the head (Tasks 11–16)
Backbone/features extract representations; the head maps them to task-specific logits. Set existing backbone parameters' `requires_grad=False` so backward does not compute their gradients. Replace the classifier with newly constructed layers of the appropriate input width. New nn.Linear parameters require gradients by default.

Use `for parameter in model.features.parameters():` to access only that submodule. `model.parameters()` covers the whole model. Freezing gradients and evaluation mode are separate: train mode may still update running statistics of normalization layers. The course workflow trains the whole model in train mode while freezing backbone parameter gradients; it is not a promise that every backbone buffer is unchanged.

Inspect the existing head before choosing its replacement width. Move a newly assigned head to device too. An optimizer constructed after replacement discovers the new parameters.

## Training helpers (Tasks 17–18)
`engine.train` accepts model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, and device. It returns a dictionary of four metric lists. You already implemented an equivalent loop earlier; open the supplied helper if you want to see how the arguments are used.

Start with a single forward batch to understand shape, then perform the specified training run. Predict with eval/inference mode, using exactly the paired pretrained transforms. The objective still consumes raw logits and int64 labels.

## Custom-image prediction (Tasks 19–21)
A prediction helper is an ordinary function. It accepts a model and a file path; it must open RGB input, transform, batch, place on device, call the model, and interpret output. `transform=None` means the caller can omit that optional argument, not that the model accepts raw images. Set a meaningful default pipeline or explicitly pass the selected weight transforms.

Reference: [torchvision pretrained models](https://docs.pytorch.org/vision/stable/models.html).
