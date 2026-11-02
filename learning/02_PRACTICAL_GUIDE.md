
# Practical guide: classification APIs

**Prerequisite:** complete Chapter 01 through saving/reloading a trained model. You should be able to explain Module parameter registration, forward, loss, backward, optimizer step, and inference mode. If any is unfamiliar, revisit the corresponding Learn first cell there. This chapter changes the data and objective; it keeps that training workflow.

## Data tools: sklearn and pandas (Tasks 1–10)
`sklearn` provides dataset generation/splitting functions. They return data; they do not train a PyTorch model for you.

| API | Arguments to understand | Result |
| --- | --- | --- |
| `make_circles(n_samples, noise, random_state)` | Number of samples, noise level, reproducible dataset seed | NumPy features and binary labels |
| `train_test_split(X, y, test_size, random_state)` | Aligned arrays/tensors, fraction held out, split seed | X_train, X_test, y_train, y_test in that order |
| `pd.DataFrame(dict_of_columns)` | Dictionary with equally long columns | A labelled table; does not convert your model data automatically |
| `table.head(n)` | Number of rows to display | A small table preview |
| `table['label'].value_counts()` | One selected column | Counts per distinct label |

Miniature example you can run separately:

```python
from sklearn.model_selection import train_test_split
features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
labels = torch.tensor([0, 0, 0, 1, 1, 1])
small_train_x, small_test_x, small_train_y, small_test_y = train_test_split(
    features, labels, test_size=0.33, random_state=7)
```

Inspect one row and its label. Split both together to preserve pairing. Convert NumPy inputs with `torch.from_numpy(array).float()`; class indices for multiclass loss use `.long()` instead. `random_state` controls sklearn's own randomness; it is not set by torch.manual_seed.

## Build a classifier: containers and activations (Tasks 11–40)
A classifier still maps inputs to predictions with an nn.Module. With two coordinate inputs and one binary score, its shape contract is $(N,2)\to(N,1)$.

`nn.Sequential(layer1, layer2, ...)` stores layers and calls them in order. Adjacent feature counts must match. Example:

```python
small_network = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
print(small_network(torch.randn(5, 3)).shape)  # (5, 2)
```

`nn.ReLU()` creates a callable activation layer; `torch.relu(tensor)` applies an activation directly. An activation is not an optimizer or trainable parameter. Without nonlinear activations, multiple affine layers still describe one affine mapping.

For your custom class, create sublayers in `__init__`; in `forward`, call them in order. Do not recreate layers in forward: that would initialize new parameters on each prediction.

## Logits, probabilities, labels, and the loss
A **logit** is a raw score. A **probability** is a transformed score; a **label** is the final discrete class decision. Use these separately:

| Problem | Model output / target | Loss input | Display/predict |
| --- | --- | --- | --- |
| Binary | float logits $(N,)$; float targets $(N,)$ containing 0/1 | `nn.BCEWithLogitsLoss()(logits, targets)` | `torch.sigmoid(logits)` then round or threshold |
| Multiclass | float logits $(N,C)$; int64 targets $(N,)$ | `nn.CrossEntropyLoss()(logits, targets)` | `torch.softmax(logits, dim=1)` then `.argmax(dim=1)` |

Create the loss object once before the loop. These losses apply the needed probability transformations internally; do not feed sigmoid/softmax probabilities into the corresponding logits-based loss.

A binary model often outputs $(N,1)$. Use `.squeeze(dim=1)` to remove **only** its final size-one axis; targets then match $(N,)$. Avoid broadcasting a target of the wrong shape into a different objective.

## Accuracy and training (Tasks 15, 20, 25, 38, 56)
Compare predicted **labels** with targets, count matches, divide by sample count. `.sum().item()` converts a Boolean match tensor into a Python count. A fraction and a percentage are different units: use one consistently.

Reuse Chapter 01's five steps. Only the loss and conversion to labels change. Backpropagate the scalar loss, not accuracy: argmax/round are decision operations, not the training objective. Use raw logits for loss; probabilities/labels are for interpretation and metrics. For one batch of five examples in two classes, inspect logits shape before training and explicitly verify its alignment with five targets.

## Plots and generalization
`plt.scatter(X[:,0], X[:,1], c=y)` uses coordinates as positions and label values as colors. Here `c` takes an array of per-point values, whereas `color='blue'` chooses one fixed color. Move plotting copies to CPU.

A decision-boundary helper samples a coordinate grid and asks the model for a label at each position. It visualizes the learned mapping; it does not perform training. Training and test plots help expose whether the mapping generalizes.

**Independent transfer:** once you can train the circle classifier, replace its generated data with a small feature/label dataset from your own problem. First define exactly what one input row represents and what target you want it to predict. Keep the split and baseline comparison explicit.

Reference: [PyTorch classification losses](https://docs.pytorch.org/docs/stable/nn.html#loss-functions), [sklearn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
