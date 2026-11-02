
# Practical guide: your own dataset and reusable functions

**Prerequisite:** understand Dataset/DataLoader, image axes, a CNN forward pass, and a minibatch loop from Chapter 03. This chapter teaches interfaces and preprocessing; the same Dataset pattern can later serve non-image features in your AI project.

## Files, PIL, and transforms (Tasks 1–19)
A `Path` is a filename/directory object. `path.glob('pattern')` iterates over matching paths; `list(...)` collects them. `path.name` gives a filename; `path.parent.name` can identify a class folder. Use `Image.open(path).convert('RGB')` from PIL to read a consistent three-channel image.

`transforms.Compose([transform_a,transform_b,...])` calls transforms in sequence. Constructors build callable objects; applying the pipeline to an image executes it.

```python
from torchvision import transforms
resize_and_tensor = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
# transformed = resize_and_tensor(a_pil_image)
```

`Resize((height,width))` takes a pair for exact spatial size. `RandomHorizontalFlip(p=...)` probabilistically flips a training image. `ToTensor()` converts a PIL image into float32 channel-first values, usually scaling uint8 pixels into the unit interval. Do not divide by 255 again if ToTensor already performed that conversion.

`datasets.ImageFolder(root,transform=...)` expects one subfolder per class. It discovers sorted classes, assigns integer labels, and returns a transformed image/label pair. `class_to_idx` is a mapping, not a model prediction.

## Implement the Dataset interface (Tasks 20–36)
Subclassing Dataset is Python inheritance, like nn.Module, but a Dataset is not itself a neural network. Store file paths and transforms in `__init__`. Implement these methods at the same indentation level:

```python
from torch.utils.data import Dataset
class FeaturePairs(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.targets[index]
```

In your **image** implementation, store image paths rather than all decoded images. `__getitem__(index)` opens that file, finds its parent-folder class, applies `self.transform` when provided, and returns `(image_tensor,label_int)`. DataLoader calls these methods for you when making batches.

A regular function such as `find_classes(directory)` receives its directory as an argument and returns a list/dictionary. Class methods instead receive `self` automatically. Do not access an unrelated global directory from a supposedly reusable scanner.

## Train functions and result dictionaries (Tasks 37–58)
The model is still a TinyVGG-style classifier; the input now has three channels and a different size. Recalculate the final flatten size. `torchinfo.summary(model,input_size=...,device=...)` performs a forward-based summary using a generated input. Its `input_size` includes batch size; it is not the model constructor's `input_shape` channel argument.

Write each helper separately: `train_step` returns one epoch's train metrics; `test_step` returns test metrics; `train` repeats both and stores lists. A result dictionary uses string keys so plotting code can retrieve the correct lists. A new experiment needs a fresh model and optimizer.

Augmentation is an experimental change to the training distribution. Keep testing deterministic so changes in evaluation reflect the model rather than random image augmentation.

## Inference input contract (Tasks 59–73)
An image-read function often returns uint8 channel-first data. Your model needs float32, the same scaling and spatial size as training, a batch dimension, and a matching device. Check these **before** the forward call.

```python
# For a raw uint8 tensor only:
float_pixels = raw_pixels.float() / 255
# Add exactly one batch axis after applying the intended resizing transform:
# batch = transformed_image.unsqueeze(0).to(device)
```

A tensor image for the model is channel-first; `image.permute(1,2,0)` is only for display. Classify with raw logits, apply softmax across classes, then use `.argmax(dim=1).item()` to obtain one Python class index. Use the training class_names ordering.

**Independent transfer:** implement FeaturePairs for an existing small table of features and labels. The training loop can stay nearly unchanged, even though the model might become an MLP instead of a CNN.
