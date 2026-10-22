# NumPy-Analyzer 🧮

> A beginner-friendly Python and NumPy project for exploring numerical arrays through a menu-driven command-line analyzer.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-Rami--Manan%2FNumPy--Analyzer-181717?logo=github)](https://github.com/Rami-Manan/NumPy-Analyzer)

> **Development status:** This repository is currently an early-stage scaffold. The checked-in project contains the README, license, and Git ignore configuration, but it does **not** yet contain the Python application, dependency manifest, or automated tests. Features described as *planned* below come from the project brief and are not presented as shipped functionality.

## 📖 About the Project

**NumPy-Analyzer** is intended to be a small, interactive learning project that makes common NumPy operations accessible from a terminal menu. The planned application will combine NumPy's array-processing capabilities with a class-based Python design so learners can practice both numerical computing and object-oriented programming in one project.

The target experience is a guided console workflow for creating arrays, inspecting values, transforming data, running arithmetic and statistical operations, and learning how array shape affects the result. The project is deliberately focused on standard Python, NumPy, and a normal terminal rather than a web or graphical interface.

The implementation is not present in the current repository revision yet. This README therefore documents the verified repository state and separates the proposed application scope from functionality that can currently be executed.

## ✨ Planned Features

The following capabilities are part of the project brief and are proposed for the first implementation:

- [ ] **Class-based analyzer** — provide a `DataAnalytics` class that owns the current array and related operations.
- [ ] **1D, 2D, and 3D array creation** — collect dimensions and values from the user and build NumPy arrays.
- [ ] **Indexing and slicing** — access elements and select ranges using NumPy/Python indexing conventions.
- [ ] **Array combination and splitting** — support vertical stacking, horizontal stacking, concatenation, and splitting where shapes are compatible.
- [ ] **Element-wise arithmetic** — add, subtract, multiply, and divide compatible arrays without mutating the original.
- [ ] **Linear-algebra operations** — provide dot products and matrix multiplication with dimension validation.
- [ ] **Search, sort, and filter** — locate matching values, sort multidimensional data along a documented axis, and filter using comparison conditions.
- [ ] **Aggregations** — calculate sum, mean, median, standard deviation, and variance over an array or valid axis.
- [ ] **Statistics** — calculate minimum, maximum, percentiles, and correlation coefficients for suitable inputs.
- [ ] **Input validation** — handle malformed numbers, dimensions, indexes, shapes, split counts, percentiles, and division by zero with readable messages.
- [ ] **Menu-driven navigation** — let users return from submenus to the main menu and exit cleanly.

> The checklist above is a roadmap, not an implementation status report. No Python source file is currently tracked in this repository.

## 🧰 Technologies

The intended technology stack is small and portable:

| Technology | Role | Current repository status |
| --- | --- | --- |
| [Python](https://www.python.org/) | Application language and command-line interface | Planned; no Python source is committed yet |
| [NumPy](https://numpy.org/) | Multidimensional arrays, vectorized arithmetic, and statistics | Planned; no dependency file is committed yet |
| Python `unittest` | Automated regression tests | Proposed; no test suite is committed yet |
| Terminal / PowerShell / Bash | Running the interactive application | Intended interface |

The intended dependency list contains only NumPy; the current checkout does not declare any dependencies.

## 🗂️ Project Structure

### Current repository

This is the directory structure that is actually present in the repository at the time of writing:

```text
NumPy-Analyzer/
├── .gitignore   # Standard Python and tooling exclusions
├── LICENSE      # MIT License
└── README.md    # Project documentation
```

### Planned implementation layout

The project brief proposes the following files for the application phase. They are shown here as planned locations and do not exist in the current revision:

```text
NumPy-Analyzer/
├── numpy_analyzer.py       # Planned executable CLI and DataAnalytics class
├── test_numpy_analyzer.py  # Planned unittest-based automated tests
├── requirements.txt        # Planned dependency declaration for NumPy
├── LICENSE
└── README.md
```

## 🚀 Installation and Setup

### Clone the repository

**PowerShell**

```powershell
git clone https://github.com/Rami-Manan/NumPy-Analyzer.git
Set-Location NumPy-Analyzer
```

**Linux or macOS**

```bash
git clone https://github.com/Rami-Manan/NumPy-Analyzer.git
cd NumPy-Analyzer
```

### Create a virtual environment

**PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux or macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install the planned dependency

The repository does not currently contain a `requirements.txt` file. When the Python implementation is added, NumPy can be installed with:

```bash
python -m pip install numpy
```

For a future dependency-managed version, the expected command will be:

```bash
python -m pip install -r requirements.txt
```

## ▶️ Running the Application

There is no executable entry point in the current checkout, so the following command cannot run successfully until `numpy_analyzer.py` is added:

```bash
python numpy_analyzer.py
```

Once the planned CLI exists, it is expected to present a main menu similar to:

```text
Welcome to the NumPy Analyzer!
========================================

Choose an option:
1. Create a NumPy Array
2. Perform Mathematical Operations
3. Combine or Split Arrays
4. Search, Sort, or Filter Arrays
5. Compute Aggregates and Statistics
6. Exit
```

The exact prompts and output should be treated as implementation targets until the source code and tests are committed.

### Planned automated tests

The project brief calls for a `unittest` suite. After `test_numpy_analyzer.py` is added, the intended command is:

```bash
python -m unittest test_numpy_analyzer.py
```

This command is not runnable in the current checkout because the test file does not exist yet.

## 🧭 Intended Usage Guide

The planned workflow is organized around the six options in the main menu:

1. **Create a NumPy Array** — choose a 1D, 2D, or 3D array, enter its dimensions, and provide the required values.
2. **Perform Mathematical Operations** — choose element-wise addition, subtraction, multiplication, division, dot product, or matrix multiplication and provide a compatible second array when needed.
3. **Combine or Split Arrays** — stack or concatenate compatible arrays, or split an existing array into smaller parts.
4. **Search, Sort, or Filter Arrays** — find values and indices, sort in ascending or descending order, or select values using a comparison condition.
5. **Compute Aggregates and Statistics** — calculate reductions such as the mean or median, then inspect minimum, maximum, percentile, or correlation results.
6. **Exit** — leave the program without terminating unexpectedly.

Each item above describes the intended interface from the project brief. It is not currently executable because the corresponding implementation files have not yet been added to the repository.

## 🧪 NumPy Examples

The following standalone example demonstrates the NumPy concepts that the planned analyzer is intended to expose. It is valid Python/NumPy, but it is **not** an import from this repository because no application module has been committed yet.

```python
import numpy as np

data = np.array([[10, 20, 30], [40, 50, 60]])
increment = np.full(data.shape, 5)

print("Selected columns:")
print(data[:, 1:3])

print("Element-wise addition:")
print(data + increment)
```

Expected NumPy display:

```text
Selected columns:
[[20 30]
 [50 60]]
Element-wise addition:
[[15 25 35]
 [45 55 65]]
```

## 🧠 NumPy Concepts Demonstrated by the Planned Design

The project brief is structured around several useful NumPy fundamentals:

- **Array creation and shape** — construct arrays from user values and inspect dimensions, rows, columns, and compatible shapes.
- **Indexing and slicing** — use zero-based indexes and Python's end-exclusive slice notation such as `0:2`.
- **Vectorized arithmetic** — apply element-wise operations to arrays without writing a manual loop for every value.
- **Stacking and splitting** — combine arrays along an axis or divide one array into smaller arrays.
- **Boolean filtering** — create a condition mask and select values that satisfy it.
- **Reductions and statistics** — calculate values such as sum, mean, median, standard deviation, variance, minimum, maximum, and percentiles.
- **Shape-aware linear algebra** — distinguish element-wise multiplication from dot products and matrix multiplication.

These are learning goals inferred from the supplied project brief. They should not be read as evidence that the current repository already implements them.

## 🖥️ Reference Output

The project brief provides the following target behavior for a 2D array containing `10 20 30 40 50 60`:

```text
Array created successfully:
[[10 20 30]
 [40 50 60]]

Result of Addition:
[[15 25 35]
 [45 55 65]]

Median of Array: 35.0
```

This is reference output supplied for the planned application, not output verified by running the current checkout. The repository currently has no executable program with which to reproduce it.

## 📋 Requirements

### Verified for the current repository

- Git, to clone the repository.
- A Markdown viewer or GitHub account, to read the current documentation.

### Intended for the application phase

- Python 3.x. The repository does not currently declare a minimum Python version.
- NumPy, installed with `python -m pip install numpy` until a `requirements.txt` file is added.
- A standard terminal such as PowerShell, Command Prompt, Bash, or a VS Code terminal.

No database, web framework, GUI toolkit, or other third-party package is part of the verified repository contents.

## 🛣️ Roadmap

The following improvements are proposed for future commits:

- [ ] Add `numpy_analyzer.py` with the `DataAnalytics` class and a focused CLI controller.
- [ ] Add `requirements.txt` with a pinned or bounded NumPy dependency.
- [ ] Add `test_numpy_analyzer.py` covering valid operations, shape errors, invalid input, and the reference examples.
- [ ] Add clear error handling for malformed numeric input, invalid dimensions, out-of-range indexes, incompatible shapes, empty arrays, and division by zero.
- [ ] Document supported axes and dimensionality for every operation.
- [ ] Add continuous integration to run the test suite on supported Python versions.
- [ ] Add more educational examples and terminal transcripts after the implementation is available.

## 🤝 Contributing

Contributions are welcome as the implementation takes shape. A simple workflow is:

1. Fork the repository.
2. Create a focused feature branch:

   ```bash
   git checkout -b feature/your-change
   ```

3. Make the change and add or update tests where applicable.
4. Run the documented checks locally.
5. Open a pull request describing the behavior changed and how it was verified.

Please keep changes focused, avoid committing virtual environments or generated files, and update the README when user-facing behavior changes.

## 📄 License

This project is licensed under the [MIT License](LICENSE). The repository identifies **Rami Manan** as the copyright holder for 2026.

## 👤 Author

Created and maintained by [Rami Manan](https://github.com/Rami-Manan).

- Repository: [github.com/Rami-Manan/NumPy-Analyzer](https://github.com/Rami-Manan/NumPy-Analyzer)
- Issues and discussions: [GitHub Issues](https://github.com/Rami-Manan/NumPy-Analyzer/issues)

## 🙏 Acknowledgements and Support

This learning project is designed around the official [Python documentation](https://docs.python.org/3/) and [NumPy documentation](https://numpy.org/doc/). If you find a documentation problem or want to help implement one of the roadmap items, please open an issue or submit a pull request in the repository.

⭐ If this project helps you practice Python and numerical computing, consider starring the repository and sharing constructive feedback.


[![Architecture diagram of rami-manan/numpy-analyzer](https://gitdiagram.com/rami-manan/numpy-analyzer/diagram.png)](https://gitdiagram.com/rami-manan/numpy-analyzer?utm_source=readme&utm_medium=picture)
