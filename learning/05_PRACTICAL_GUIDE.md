
# Practical guide: moving code out of a notebook

**Prerequisite:** write and call an ordinary Python function and train a model in Chapter 01 or 02. Modularization is organizing the same working code into files; it is not a new ML algorithm.

## A module is a Python file
If `example_tools.py` contains:

```python
def describe_shape(tensor):
    return str(tuple(tensor.shape))
```

then a notebook running in that directory can use:

```python
from example_tools import describe_shape
print(describe_shape(torch.randn(2, 3)))
```

Importing executes a file's top-level statements once for that Python process. Avoid putting expensive training loops at top level in a module you want to import. A function name is defined by executing its definition; its body runs only when called.

## Pass dependencies explicitly
A notebook function can accidentally depend on globals defined by old cells. A reusable module should take model, data, device, and other dependencies as arguments. It cannot see a notebook's `X_train` merely because the notebook imported it.

Copy one already-working function at a time into a file, import it, then use it directly in your normal workflow. Do not rewrite the entire training system before you have a working run.

## Packages and imports
A directory containing modules can be a package. An `__init__.py` file makes its package role explicit. `from package_name import module_name` imports a module, then `module_name.function_name(...)` calls a function it defines. The supplied course helpers live under `going_modular/going_modular/` to match the upstream imports; they are shared utilities, not another cloned project.

`%%writefile relative/path.py` writes a notebook cell into a source file instead of running its body. It replaces the target file's contents. You may create/edit the same file in VS Code instead.

After changing an imported module, the kernel may still use the imported old definition. Restart and rerun prerequisite cells, or deliberately reload the module with importlib. Saving a source file and importing it into an already-running kernel are separate steps.
