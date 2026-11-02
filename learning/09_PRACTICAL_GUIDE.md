
# Practical guide: from a trained model to a local app

**Prerequisite:** train and reload a model, preprocess one input correctly, and interpret its output. This entire chapter is an optional application/deployment extension. It is not required for using a trained model inside your AI project.

## Pretrained-model and preprocessing contracts (Tasks 1–29)
The model constructors in this lesson return a model and its weight-specific preprocessing. Keep those paired: input scaling, resizing, and normalization are part of the inference contract.

Measure three different things: metric quality on held-out examples, parameter count (`sum(p.numel() for p in model.parameters())`), and serialized file size (`Path(...).stat().st_size`). None is a substitute for the others. Use the same metric units across models.

## Per-image prediction records (Tasks 30–46)
A list of dictionaries is a convenient format for detailed results. Each row should identify input, true label, predicted label, confidence, latency, and correctness. `pd.DataFrame(records)` turns these into a table. Select a column with `frame['column']`, aggregate with `.mean()` or `.value_counts()`, and keep units visible.

A timing interval covers only the calls between your start/end readings. Put the same stages inside the interval for both models and use the same device. CPU inference simplifies this exercise. Confidence is the selected model probability, not a guarantee that a prediction is correct.

## Gradio input/output components (Tasks 47–52)
Gradio binds the existing predict function to input/output components. It does not train or automatically preprocess for the model.

```python
import gradio as gr
# predict must already accept a PIL image and return (probability_dict, seconds).
# demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"),
#                     outputs=[gr.Label(), gr.Number()])
# demo.launch(share=False)
```

`gr.Image(type='pil')` specifies the type passed to predict. Label displays a dictionary mapping class names to probabilities; Number displays a scalar. The order and count of returned values must match outputs. Pass the callable as fn, and use share=False for a local app.

Inside predict, use CPU model evaluation/inference mode, the matching transforms, and one batch dimension. Do not recreate/randomly initialize the model for each image.

## Package files without notebook state (Tasks 53–62)
A standalone app needs Python source defining its model architecture, a saved checkpoint, preprocessing code, and the class-name mapping. It cannot use a class merely because that class exists in a separate notebook kernel.

`model.py` defines a reusable factory. `app.py` imports it, loads weights, defines the callback, and launches an interface. Keep launching separate from model construction so importing the model does not launch the app.

`%%writefile path/to/file.py` at the first line of a notebook cell writes the rest of the cell into a file **instead of executing that body**. Repeating it overwrites that file. Alternatively create files in your editor. `shutil.copy2(source,destination)` copies a file; it does not remove the source.

Requirements describe packages needed to run the app. Use versions that actually work together in your environment, not the historical course's old pins. A ZIP is a distribution artifact, not a deployment by itself. No publishing is required here.

## Optional Food101 extension (Tasks 63–86)
The larger dataset uses 101 classes, so architecture head, class-name list, checkpoint, and displayed output labels must all agree on 101. `random_split(dataset,lengths,generator=...)` creates index subsets; transforms still belong to the underlying dataset. A seeded `torch.Generator().manual_seed(seed)` makes the subset selection reproducible without resetting unrelated global randomness.

Reuse the three-class pipeline with a different dataset/head only after you understand the smaller app. Downloading and training the large extension is optional.

Reference: [Gradio quickstart](https://www.gradio.app/guides/quickstart).
