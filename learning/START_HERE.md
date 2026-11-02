
# Start here: from ML theory to a working training workflow

This material assumes you have seen ML theory (for example CS189), but have not used PyTorch, matplotlib, Jupyter, or Python model classes. The objective is practical independence: prepare tensors, implement a model, train/evaluate it, and save/load it. The course is a starting point before your own coursework or AI project, not a requirement to learn every image architecture.

## A manageable path

1. **Chapter 00:** get comfortable with shapes, indexing, dtype, device, and shared storage. You have already practiced much of this.
2. **Chapter 01:** the main hands-on introduction. Read each **Learn first** block, optionally run its tiny example, then implement the following task. Complete plotting, object structure, one training update, a loop, evaluation, and checkpoints.
3. **Chapter 02:** apply the same loop to classification. Learn new APIs for data splitting, logits, classification losses, and metrics.
4. **Independent checkpoint:** train a small model from your own data and explain every line of the training loop. At this point you can return to CS189 or your AI project; the later vision, experiment-tracking, transformer, and web-app chapters are optional extensions.

If a step introduces an unfamiliar API, its practical guide or Learn first block should state what object it creates, which arguments you supply, what types/shapes it expects, and what it returns. Use the worked example as a pattern, then write the task implementation yourself. You do not need to invent a library API from its name.

## Three kinds of things you will encounter

| Kind | Example | What you do |
| --- | --- | --- |
| Function | `torch.randn(3)` | Call it with arguments; receive a result |
| Class / constructor | `nn.Linear(2, 1)` | Construct a persistent layer object with stored parameters |
| Method on an object | `model.parameters()` | Ask that particular object to perform an operation or expose its state |

Some objects are callable: `model(inputs)` and `loss_fn(predictions,targets)` call objects created earlier. Construction and use are different actions. You will see this distinction explained before the first model exercise.

## Jupyter mechanics

A notebook combines Markdown explanations, code cells, and saved outputs. Its **kernel** is a live Python process. Run a cell with Shift+Enter; names it defines remain available to later cells in that kernel. Notebook position is not execution order: a cell run later can overwrite a name used earlier.

- **Edit:** changes the in-memory document.
- **Run:** executes code and changes kernel state; displayed output is not proof the current source has been rerun.
- **Save (Cmd+S):** writes notebook text/code/outputs to its `.ipynb` file. It does not save live Python objects.
- **Restart kernel:** clears variables/imports; rerun prerequisite cells from the top.
- **Reload from disk:** use when an external edit changed the file. Do not save an older open copy over the new version. Before requesting notebook changes, save your current work.

An error is feedback: inspect the failing line, then the input types, shapes, dtype, and device. A SyntaxError means Python could not parse the source; it is not an ML problem. An AttributeError often indicates a misspelled name, a method nested at the wrong indentation, or an object of the wrong type.

## Practical study pattern

Read one short concept, run its tiny example if useful, then fill one task cell. For a larger task, get one stage working before adding the next. First understand a forward pass; then one loss; then one update; then a loop. Do not attempt an entire training pipeline from an unexplained outline.

When you run provided examples, use a temporary scratch cell and keep their `demo_`/`example_` variable names separate from the task's model/optimizer. Worked examples teach the mechanics; your task cells remain your implementation.

## Bridge to your next project

Before training on project data, write down what one sample represents, its feature shape/dtype, the target, how labels are obtained, and how train/validation/test data are separated. Build a simple baseline first. Reuse Chapter 01/02's model, loss, optimizer, and evaluation structure; change only the problem-specific parts. If your data is a feature vector, you do not need the vision chapters to begin.

Tools learned here are transferable. An `nn.Module` encodes a prediction rule; `.backward()` computes gradients; an optimizer updates parameters; a DataLoader provides batches. None of those decides your project's target or evaluation design for you.
