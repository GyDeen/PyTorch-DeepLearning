# Practical learning route

Begin with [Start here](learning/START_HERE.md), written for someone who knows Python and ML theory but is new to these libraries. Learn first sections introduce package behavior, arguments, shape/dtype contracts, and small examples before practice. They assume Python knowledge. Complete Chapters 00–02 before deciding which optional later chapters serve your own project.

# Working through the PyTorch lessons

Open this repository folder in VS Code or launch JupyterLab from this folder. Select the project `.venv` / **Python (PyTorch Learning)** kernel.

## What to write

Read the practice outline at the start of each lesson. Each **Your task** above a code cell states what to implement; **Inspect when finished** describes the expected shape, output, or observation. Keep the named variables/functions so later tasks can reuse them. API hints are there to help you locate a tool, not to supply the implementation.

The lesson notebooks teach a complete workflow in small steps. The notebooks under `extras/exercises/` are separate end-of-chapter assignments and already contain their own questions. Finish the lesson first, then attempt those exercises independently.

## Work in this order

| Lesson | Main outcome |
| --- | --- |
| 00 | Manipulate tensors, shapes, indexing, and devices |
| 01 | Fit a line, evaluate predictions, and save/reload weights |
| 02 | Train binary and multiclass classifiers |
| 03 | Train and compare image classifiers on FashionMNIST |
| 04 | Load your own image folders and implement dataset/training helpers |
| 05 exercise | Move reusable code into modules; the upstream chapter is linked below |
| 06 | Adapt a pretrained model to three food classes |
| 07 | Log and compare controlled experiments |
| 08 | Build a small ViT implementation and compare pretrained transfer |
| 09 | Measure models and package a local image-classification app |

Chapter 05 is [Going Modular](https://www.learnpytorch.io/05_pytorch_going_modular/); the original course teaches it through scripts rather than a root lesson notebook. Its exercise template is in `extras/exercises/`. Supplied `going_modular/` modules let later notebooks import the shared helpers without cloning another copy of the course.

## Your Mac

Use float32 for model inputs unless the task says otherwise. Select CUDA when available, otherwise MPS when available, otherwise CPU. A Mac normally uses MPS or CPU. Keep a model and its input tensors on the same device. Convert detached CPU copies for NumPy/matplotlib. DataLoader `num_workers=0` avoids notebook multiprocessing complications; increase workers only as a deliberate later experiment.

The tasks that download datasets name the source files. They do not ask you to train every model immediately: write one section at a time and inspect its output. Heavy ViT/Food101 extensions are identified explicitly.

## Save and resume

Save the notebook in your editor before switching between VS Code and browser JupyterLab. Both must open this same repository. An editor may need the notebook reopened to display changes saved by another editor. Avoid editing the same notebook simultaneously in both. Notebook variables live in the kernel, so after restarting it rerun prerequisite cells in order.

Commit your saved notebooks in VS Code Source Control, then choose **Sync Changes** / **Push** to update GitHub. Saving a local notebook alone does not upload it.

## Support code and attribution

Lesson text, notebooks, and shared helper modules are adapted from [Daniel Bourke's Learn PyTorch course](https://github.com/mrdbourke/pytorch-deep-learning), under the included LICENSE. The guided tasks/outlines are added for independent practice. Shared helpers are provided as utilities; solution code has not been inserted into your notebook cells. The boundary plotting helper additionally restores the model's device after making its CPU visualization.
