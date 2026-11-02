# Practical learning path: library APIs before implementation

**Assumed background:** you know Python, including classes/functions, and have studied ML theory. The gap this material addresses is practical use of PyTorch, matplotlib, sklearn/pandas, torchvision, Jupyter, and related tools. It does not reteach Python syntax or derive ML theory.

## How the notebooks are organized

Before asking you to use a new library interface, the notebook introduces its purpose, important arguments, input/output types and shapes, and a minimal use example where useful. The concrete task then asks you to apply it. Previously introduced tools can be reused without repeating their entire explanation; API notes remain immediately above their first relevant task in each lesson.

For an nn.Module exercise, the explanation focuses on parameter/submodule registration, forward dispatch, device/mode management, and checkpoint interfaces—not what a Python class or self means. Likewise a plotting task introduces matplotlib's calls, not how to define a function.

## Core route

1. Chapter 00: tensor operations, shapes/dtypes/devices, indexing, and storage sharing.
2. Chapter 01: model/loss/optimizer APIs, matplotlib, a real training/evaluation loop, and checkpoint saving/loading.
3. Chapter 02: data-tool APIs, binary/multiclass logits, classification losses, and metrics.
4. Apply the same workflow to a small problem from CS189 or your AI project. Define the sample/target contract and evaluation split before choosing the model.

Vision datasets/CNNs, transfer learning, logging, transformers, and web deployment are optional extensions. Use the ones relevant to your next project rather than treating all later chapters as prerequisites.

## Tool conventions worth knowing

- Creating an nn.Linear object initializes its parameters. Calling it on inputs computes predictions; neither action trains it.
- Module parameters are found through registration, not through arbitrary tensor variables in scope.
- backward computes gradients; optimizer.step updates weights; zero_grad clears accumulated gradients.
- train/eval modes affect certain layers; inference_mode controls gradient recording. These are separate interfaces.
- DataLoader provides batches; it does not split datasets or decide your prediction target.
- Helpers imported from going_modular or helper_functions are course code; their signatures are explained separately from PyTorch APIs.

## Jupyter workflow

The kernel holds imports/objects/variables; running a cell changes that live state. Saving (Cmd+S) writes notebook source and outputs to disk, not Python objects. A kernel restart requires rerunning prerequisite cells. After an external notebook update, use File → Reload Notebook from Disk rather than overwriting it from an older open view. Save your work before requesting edits.

Use Shift+Tab inside a call or help(api_name) for docstrings. Notes in the notebook supply the subset of arguments needed for each exercise. If the error concerns shape/dtype/device, inspect those properties before changing the model formula.

The tiny examples are API demonstrations, not a request to fill every cell with copied solutions. Write the concrete exercise in your own task cell. Your previous code and outputs are preserved.
