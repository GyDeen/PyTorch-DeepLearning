
# Practical guide: from a trained model to a local app

**Prerequisite:** train and reload a model, preprocess one input correctly, and interpret its output. This entire chapter is an optional application/deployment extension. It is not required for using a trained model inside your AI project.

## Reusable model factories (Tasks 1–29)
A function returning `(model, transforms)` is an ordinary Python function returning a tuple. Unpack both with `model, preprocess = factory(...)`. Returning preprocessing alongside the model reduces accidental mismatch.

Measure three different things: metric quality on held-out examples, parameter count (`sum(p.numel() for p in model.parameters())`), and serialized file size (`Path(...).stat().st_size`). None is a substitute for the others. Use the same metric units across models.

## Per-image prediction records (Tasks 30–46)
A list of dictionaries is a convenient format for detailed results. Each row should identify input, true label, predicted label, confidence, latency, and correctness. `pd.DataFrame(records)` turns these into a table. Select a column with `frame['column']`, aggregate with `.mean()` or `.value_counts()`, and keep units visible.

A timing interval covers only the calls between your start/end readings. Put the same stages inside the interval for both models and use the same device. CPU inference simplifies this exercise. Confidence is the selected model probability, not a guarantee that a prediction is correct.

## A callback is just a function (Tasks 47–52)
A web interface does not train your model. It receives an image, calls a Python prediction function, and displays its returned values.

Here is an unrelated miniature Gradio example:

```python
import gradio as gr

def describe_text(text):
    return f"You entered {len(text)} characters"

example_app = gr.Interface(fn=describe_text, inputs="text", outputs="text")
# example_app.launch(share=False)
```

`fn` is a **function object**. Pass `predict`, not `predict(...)`: the app will call it later when a user submits input. For your image task, `gr.Image(type='pil')` provides a PIL image; `gr.Label()` can display a dictionary from class names to probabilities; `gr.Number()` displays elapsed time. A function returning two values needs two output components in the same order. Launch locally with share=False.

Inside predict, use the trained CPU model in eval/inference mode, the matching transforms, and one batch dimension. Do not recreate/randomly initialize the model on every image submission.

## Package files without notebook state (Tasks 53–62)
A standalone app needs Python source defining its model architecture, a saved checkpoint, preprocessing code, and the class-name mapping. It cannot use a class merely because that class exists in a separate notebook kernel.

`model.py` defines a reusable factory. `app.py` imports it, loads weights, defines the callback, and launches an interface. The guard `if __name__ == '__main__':` runs launch code only when the file is executed as the main script, not when imported by another module.

`%%writefile path/to/file.py` at the first line of a notebook cell writes the rest of the cell into a file **instead of executing that body**. Repeating it overwrites that file. Alternatively create files in your editor. `shutil.copy2(source,destination)` copies a file; it does not remove the source.

Requirements describe packages needed to run the app. Use versions that actually work together in your environment, not the historical course's old pins. A ZIP is a distribution artifact, not a deployment by itself. No publishing is required here.

## Optional Food101 extension (Tasks 63–86)
The larger dataset uses 101 classes, so architecture head, class-name list, checkpoint, and displayed output labels must all agree on 101. `random_split(dataset,lengths,generator=...)` creates index subsets; transforms still belong to the underlying dataset. A seeded `torch.Generator().manual_seed(seed)` makes the subset selection reproducible without resetting unrelated global randomness.

Reuse the three-class pipeline with a different dataset/head only after you understand the smaller app. Downloading and training the large extension is optional.

Reference: [Gradio quickstart](https://www.gradio.app/guides/quickstart).
