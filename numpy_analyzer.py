"""Interactive NumPy array analyzer.

The :class:`DataAnalytics` class contains the testable array operations.  The
menu-driven command line interface at the bottom of this module is a thin
input/output layer around that class, so the operations can also be imported
and used from another Python program.
"""

from __future__ import annotations

import operator as operator_module
from typing import Any, Callable, Iterable, Sequence

import numpy as np


class DataAnalytics:
    """Store one NumPy array and provide common numerical operations.

    Every operation that transforms or combines data returns a new array (or
    scalar) and leaves ``self.array`` unchanged.  Use :meth:`set_array` to
    replace the working array intentionally.
    """

    _COMPARATORS: dict[str, Callable[[Any, Any], Any]] = {
        ">": operator_module.gt,
        "<": operator_module.lt,
        "==": operator_module.eq,
        ">=": operator_module.ge,
        "<=": operator_module.le,
        "!=": operator_module.ne,
    }

    def __init__(self, array: Any | None = None) -> None:
        self.array: np.ndarray | None = None
        if array is not None:
            self.set_array(array)

    @classmethod
    def from_array(cls, array: Any) -> "DataAnalytics":
        """Create an analyzer initialized with a defensive copy of *array*."""

        return cls(array)

    @classmethod
    def from_1d(cls, values: Any) -> "DataAnalytics":
        """Create an analyzer from a one-dimensional sequence."""

        analyzer = cls()
        analyzer.create_1d_array(values)
        return analyzer

    @classmethod
    def from_2d(cls, values: Any, rows: int | None = None, columns: int | None = None) -> "DataAnalytics":
        """Create an analyzer from a two-dimensional sequence.

        When *rows* and *columns* are supplied, ``values`` may be a flat
        sequence and is reshaped after its size has been validated.
        """

        analyzer = cls()
        analyzer.create_2d_array(values, rows, columns)
        return analyzer

    @classmethod
    def from_3d(
        cls,
        values: Any,
        depth: int | None = None,
        rows: int | None = None,
        columns: int | None = None,
    ) -> "DataAnalytics":
        """Create an analyzer from a three-dimensional sequence."""

        analyzer = cls()
        analyzer.create_3d_array(values, depth, rows, columns)
        return analyzer

    @staticmethod
    def is_numeric_array(value: Any) -> bool:
        """Return whether *value* can be represented by a numeric NumPy array."""

        try:
            candidate = np.asarray(value)
            return np.issubdtype(candidate.dtype, np.number)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _copy_result(result: Any) -> Any:
        """Copy ndarray results while keeping scalar results convenient to use."""

        return result.copy() if isinstance(result, np.ndarray) else result

    @staticmethod
    def _as_numeric_array(value: Any, name: str = "array") -> np.ndarray:
        try:
            candidate = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain numeric values.") from exc
        if not np.issubdtype(candidate.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values.")
        return candidate

    def _require_array(self) -> np.ndarray:
        if self.array is None:
            raise ValueError("No array has been set. Create or set an array first.")
        return self.array

    @staticmethod
    def _validate_axis(array: np.ndarray, axis: int | None) -> None:
        if axis is not None and (
            isinstance(axis, (bool, np.bool_))
            or not isinstance(axis, (int, np.integer))
        ):
            raise TypeError("Axis must be an integer or None.")
        if axis is not None and (axis >= array.ndim or axis < -array.ndim):
            raise ValueError(f"Axis {axis} is invalid for an {array.ndim}D array.")

    validate_axis = staticmethod(_validate_axis)

    @staticmethod
    def _validate_shape(shape: Sequence[int]) -> tuple[int, ...]:
        if not isinstance(shape, (tuple, list)) or not shape:
            raise ValueError("Shape must be a non-empty sequence of positive integers.")
        validated: list[int] = []
        for dimension in shape:
            if isinstance(dimension, bool) or not isinstance(
                dimension, (int, np.integer)
            ):
                raise ValueError("Every dimension must be an integer.")
            if dimension <= 0:
                raise ValueError("Every dimension must be greater than zero.")
            validated.append(int(dimension))
        return tuple(validated)

    # Public validation helpers are useful to callers building their own UI
    # around the class, while the underscored versions keep internal calls
    # concise.
    validate_shape = staticmethod(_validate_shape)

    def set_array(self, array: Any) -> np.ndarray:
        """Set and return a defensive copy of a numeric array."""

        candidate = self._as_numeric_array(array)
        if candidate.ndim == 0:
            raise ValueError("An array must have at least one dimension.")
        self.array = np.array(candidate, copy=True)
        return self.array.copy()

    def create_array(
        self, values: Any, shape: Sequence[int] | None = None
    ) -> np.ndarray:
        """Create the working array from values, optionally reshaping it."""

        candidate = self._as_numeric_array(values, "values")
        if shape is not None:
            validated_shape = self._validate_shape(shape)
            expected_size = int(np.prod(validated_shape))
            if candidate.size != expected_size:
                raise ValueError(
                    f"Shape {validated_shape} needs {expected_size} values, "
                    f"but received {candidate.size}."
                )
            candidate = candidate.reshape(validated_shape)
        return self.set_array(candidate)

    def _create_with_dimensions(
        self, values: Any, shape: Sequence[int] | None, expected_dimensions: int
    ) -> np.ndarray:
        if shape is not None:
            validated = self._validate_shape(shape)
            if len(validated) != expected_dimensions:
                raise ValueError(
                    f"This constructor requires exactly {expected_dimensions} dimensions."
                )
            return self.create_array(values, validated)

        candidate = self._as_numeric_array(values, "values")
        if candidate.ndim != expected_dimensions:
            raise ValueError(
                f"Expected a {expected_dimensions}D array, got {candidate.ndim}D data."
            )
        return self.set_array(candidate)

    def create_1d_array(self, values: Any, length: int | None = None) -> np.ndarray:
        """Create a one-dimensional working array."""

        shape = (length,) if length is not None else None
        return self._create_with_dimensions(values, shape, 1)

    def create_2d_array(
        self,
        values: Any,
        rows: int | None = None,
        columns: int | None = None,
    ) -> np.ndarray:
        """Create a two-dimensional working array."""

        if (rows is None) != (columns is None):
            raise ValueError("Both rows and columns must be supplied together.")
        shape = (rows, columns) if rows is not None else None
        return self._create_with_dimensions(values, shape, 2)

    def create_3d_array(
        self,
        values: Any,
        depth: int | None = None,
        rows: int | None = None,
        columns: int | None = None,
    ) -> np.ndarray:
        """Create a three-dimensional working array."""

        supplied = (depth, rows, columns)
        if any(item is not None for item in supplied) and not all(
            item is not None for item in supplied
        ):
            raise ValueError("Depth, rows, and columns must be supplied together.")
        shape = supplied if all(item is not None for item in supplied) else None
        return self._create_with_dimensions(values, shape, 3)

    create_1d = create_1d_array
    create_2d = create_2d_array
    create_3d = create_3d_array

    def index(self, index: Any) -> Any:
        """Return one indexed item or sub-array without changing the array."""

        array = self._require_array()
        try:
            return self._copy_result(array[index])
        except (IndexError, TypeError) as exc:
            raise IndexError(f"Invalid index {index!r} for array with shape {array.shape}.") from exc

    get_element = index
    indexing = index

    def slice_array(
        self,
        start: int | slice | tuple[Any, ...] | None = None,
        stop: int | None = None,
        step: int | None = None,
        axis: int = 0,
    ) -> Any:
        """Return a slice along *axis* without changing the source array.

        A ready-made ``slice`` or tuple of indexes can be passed as *start*;
        otherwise ``start:stop:step`` is applied to the selected axis.
        """

        array = self._require_array()
        if isinstance(start, tuple) and stop is None and step is None:
            selection: Any = start
        elif isinstance(start, slice) and stop is None and step is None:
            selection = start
        else:
            self._validate_axis(array, axis)
            selection = slice(start, stop, step)
            indexes = [slice(None)] * array.ndim
            indexes[axis] = selection
            selection = tuple(indexes)
        try:
            return self._copy_result(array[selection])
        except (IndexError, TypeError) as exc:
            raise IndexError("The requested slice is not valid for this array.") from exc

    slice = slice_array
    slicing = slice_array

    def _binary_operation(self, other: Any, operation: Callable[..., Any], name: str) -> Any:
        array = self._require_array()
        other_array = self._as_numeric_array(other, "other array")
        try:
            return self._copy_result(operation(array, other_array))
        except ValueError as exc:
            raise ValueError(
                f"Cannot perform {name}: operands have incompatible shapes "
                f"{array.shape} and {other_array.shape}."
            ) from exc

    def addition(self, other: Any) -> Any:
        """Return element-wise addition with *other*."""

        return self._binary_operation(other, np.add, "addition")

    def subtraction(self, other: Any) -> Any:
        """Return element-wise subtraction from the current array."""

        return self._binary_operation(other, np.subtract, "subtraction")

    def multiplication(self, other: Any) -> Any:
        """Return element-wise multiplication with *other*."""

        return self._binary_operation(other, np.multiply, "multiplication")

    def division(self, other: Any) -> Any:
        """Return element-wise division, rejecting every zero divisor."""

        array = self._require_array()
        other_array = self._as_numeric_array(other, "other array")
        if np.any(other_array == 0):
            raise ValueError("Division by zero is not allowed.")
        try:
            return self._copy_result(np.divide(array, other_array))
        except ValueError as exc:
            raise ValueError(
                f"Cannot perform division: operands have incompatible shapes "
                f"{array.shape} and {other_array.shape}."
            ) from exc

    # Short mathematical names are convenient in code and keep the longer
    # names above explicit for the CLI and beginner-facing documentation.
    add = addition
    subtract = subtraction
    multiply = multiplication
    divide = division
    add_arrays = addition
    subtract_arrays = subtraction
    multiply_arrays = multiplication
    divide_arrays = division

    def dot_product(self, other: Any) -> Any:
        """Return the NumPy dot product of the current and another array."""

        array = self._require_array()
        other_array = self._as_numeric_array(other, "other array")
        try:
            return self._copy_result(np.dot(array, other_array))
        except ValueError as exc:
            raise ValueError(
                f"Dot product requires compatible shapes, got {array.shape} and "
                f"{other_array.shape}."
            ) from exc

    dot = dot_product
    dot_arrays = dot_product

    def matrix_multiplication(self, other: Any) -> np.ndarray:
        """Return a matrix product for two-dimensional arrays."""

        array = self._require_array()
        other_array = self._as_numeric_array(other, "other matrix")
        if array.ndim != 2 or other_array.ndim != 2:
            raise ValueError("Matrix multiplication requires two-dimensional arrays.")
        if array.shape[1] != other_array.shape[0]:
            raise ValueError(
                f"Matrix shapes {array.shape} and {other_array.shape} cannot be multiplied."
            )
        return np.matmul(array, other_array).copy()

    matmul = matrix_multiplication
    matrix_multiply = matrix_multiplication
    matmul_arrays = matrix_multiplication

    def _combine_arrays(self, arrays: Iterable[Any], operation: Callable[..., Any], name: str) -> np.ndarray:
        current = self._require_array()
        converted = [current] + [self._as_numeric_array(item, "combined array") for item in arrays]
        try:
            return self._copy_result(operation(converted))
        except ValueError as exc:
            raise ValueError(f"Arrays cannot be combined using {name} due to their shapes.") from exc

    def combine_arrays(
        self, other: Any, method: str = "vertical", axis: int = 0
    ) -> np.ndarray:
        """Combine the current array with *other* using a named strategy.

        ``method`` accepts ``vertical``/``vstack``, ``horizontal``/``hstack``,
        and ``concatenate``.  This convenience method keeps the common
        two-array use case discoverable while the dedicated methods remain
        available for more than one additional array.
        """

        normalized = method.strip().lower() if isinstance(method, str) else ""
        if normalized in {"vertical", "vstack", "vertical_stack"}:
            return self.vertical_stack(other)
        if normalized in {"horizontal", "hstack", "horizontal_stack"}:
            return self.horizontal_stack(other)
        if normalized in {"concatenate", "concat"}:
            return self.concatenate(other, axis=axis)
        raise ValueError("Combine method must be vertical, horizontal, or concatenate.")

    combine = combine_arrays

    def vertical_stack(self, *arrays: Any) -> np.ndarray:
        """Stack the current array and *arrays* vertically."""

        return self._combine_arrays(arrays, np.vstack, "vertical stacking")

    def horizontal_stack(self, *arrays: Any) -> np.ndarray:
        """Stack the current array and *arrays* horizontally."""

        return self._combine_arrays(arrays, np.hstack, "horizontal stacking")

    def concatenate(self, *arrays: Any, axis: int = 0) -> np.ndarray:
        """Concatenate the current array and *arrays* along *axis*."""

        current = self._require_array()
        self._validate_axis(current, axis)
        converted = [current] + [self._as_numeric_array(item, "combined array") for item in arrays]
        try:
            return np.concatenate(converted, axis=axis).copy()
        except ValueError as exc:
            raise ValueError("Arrays cannot be concatenated because their shapes do not match.") from exc

    vstack = vertical_stack
    hstack = horizontal_stack
    combine_vertical = vertical_stack
    combine_horizontal = horizontal_stack
    concatenate_arrays = concatenate

    def split(self, sections: int, axis: int = 0) -> list[np.ndarray]:
        """Split the current array into *sections* along *axis*."""

        array = self._require_array()
        if isinstance(sections, bool) or not isinstance(sections, (int, np.integer)):
            raise ValueError("The number of split sections must be a positive integer.")
        if sections <= 0:
            raise ValueError("The number of split sections must be a positive integer.")
        self._validate_axis(array, axis)
        try:
            return [part.copy() for part in np.array_split(array, int(sections), axis=axis)]
        except ValueError as exc:
            raise ValueError("The array cannot be split along the requested axis.") from exc

    split_array = split
    split_arrays = split

    def search(self, value: Any) -> tuple[np.ndarray, ...]:
        """Return the index arrays where *value* occurs."""

        array = self._require_array()
        try:
            return tuple(indexes.copy() for indexes in np.where(array == value))
        except (TypeError, ValueError) as exc:
            raise TypeError("The search value is not compatible with this array.") from exc

    find = search

    def sort(self, order: str = "ascending", axis: int = -1) -> np.ndarray:
        """Return a sorted copy in ascending or descending order."""

        array = self._require_array()
        if not isinstance(order, str):
            raise ValueError("Sort order must be 'ascending' or 'descending'.")
        normalized = order.strip().lower()
        if normalized in {"asc", "ascending", "up"}:
            ascending = True
        elif normalized in {"desc", "descending", "down"}:
            ascending = False
        else:
            raise ValueError("Sort order must be 'ascending' or 'descending'.")
        self._validate_axis(array, axis)
        result = np.sort(array, axis=axis)
        return result if ascending else np.flip(result, axis=axis)

    def sort_array(self, ascending: bool | str = True, axis: int = -1) -> np.ndarray:
        """Return a sorted copy using a boolean ascending flag."""

        if isinstance(ascending, str):
            return self.sort(ascending, axis=axis)
        if not isinstance(ascending, (bool, np.bool_)):
            raise ValueError("Ascending must be a boolean.")
        return self.sort("ascending" if ascending else "descending", axis=axis)

    def filter(self, condition: str, value: Any) -> np.ndarray:
        """Return flattened values satisfying a comparison condition."""

        array = self._require_array()
        if not isinstance(condition, str) or condition.strip() not in self._COMPARATORS:
            raise ValueError("Condition must be one of >, <, ==, >=, <=, !=.")
        comparator = self._COMPARATORS[condition.strip()]
        try:
            mask = comparator(array, value)
            return array[mask].copy()
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("The filter value is not compatible with this array.") from exc

    filter_array = filter

    def search_value(self, value: Any) -> tuple[np.ndarray, ...]:
        """Alias with an explicit name for menu and teaching code."""

        return self.search(value)

    def aggregate(self, operation: str, axis: int | None = None) -> Any:
        """Calculate sum, mean, median, standard deviation, or variance."""

        array = self._require_array()
        self._validate_axis(array, axis)
        operations: dict[str, Callable[..., Any]] = {
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "standard deviation": np.std,
            "standard_deviation": np.std,
            "var": np.var,
            "variance": np.var,
        }
        key = operation.strip().lower() if isinstance(operation, str) else ""
        if key not in operations:
            raise ValueError("Aggregation must be sum, mean, median, std, or var.")
        if array.size == 0:
            raise ValueError("Cannot aggregate an empty array.")
        return self._copy_result(operations[key](array, axis=axis))

    def sum(self, axis: int | None = None) -> Any:
        return self.aggregate("sum", axis)

    def mean(self, axis: int | None = None) -> Any:
        return self.aggregate("mean", axis)

    def median(self, axis: int | None = None) -> Any:
        return self.aggregate("median", axis)

    def std(self, axis: int | None = None) -> Any:
        return self.aggregate("std", axis)

    def var(self, axis: int | None = None) -> Any:
        return self.aggregate("var", axis)

    standard_deviation = std
    variance = var

    def minimum(self, axis: int | None = None) -> Any:
        """Return the minimum value or minimum along an axis."""

        array = self._require_array()
        self._validate_axis(array, axis)
        if array.size == 0:
            raise ValueError("Cannot calculate the minimum of an empty array.")
        return self._copy_result(np.min(array, axis=axis))

    def maximum(self, axis: int | None = None) -> Any:
        """Return the maximum value or maximum along an axis."""

        array = self._require_array()
        self._validate_axis(array, axis)
        if array.size == 0:
            raise ValueError("Cannot calculate the maximum of an empty array.")
        return self._copy_result(np.max(array, axis=axis))

    min = minimum
    max = maximum

    def percentile(self, value: float, axis: int | None = None) -> Any:
        """Return the requested percentile in the inclusive range 0 to 100."""

        array = self._require_array()
        self._validate_axis(array, axis)
        if array.size == 0:
            raise ValueError("Cannot calculate a percentile for an empty array.")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError("Percentile must be a number from 0 to 100.")
        if not 0 <= float(value) <= 100:
            raise ValueError("Percentile must be between 0 and 100.")
        return self._copy_result(np.percentile(array, float(value), axis=axis))

    percentiles = percentile
    calculate_percentile = percentile

    def correlation(self, other: Any | None = None) -> Any:
        """Return a correlation coefficient or correlation matrix.

        With *other*, both inputs are treated as observations and must have
        equal size.  Without it, a two-dimensional current array produces its
        NumPy correlation matrix.
        """

        array = self._require_array()
        if other is None:
            if array.ndim != 2:
                raise ValueError("A second array is required unless the current array is 2D.")
            if array.shape[0] < 2:
                raise ValueError("At least two observations are required for correlation.")
            data = array
            rowvar = False
            constant_axis = 0
        else:
            other_array = self._as_numeric_array(other, "other array")
            left = array.ravel()
            right = other_array.ravel()
            if left.size != right.size:
                raise ValueError("Correlation arrays must contain the same number of values.")
            if left.size < 2:
                raise ValueError("At least two observations are required for correlation.")
            data = np.vstack((left, right))
            rowvar = True
            constant_axis = 1
        if np.any(np.std(data, axis=constant_axis) == 0):
            raise ValueError("Correlation is undefined for constant-valued data.")
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.corrcoef(data, rowvar=rowvar)
        if other is None:
            return result.copy()
        return float(result[0, 1])

    correlation_coefficient = correlation

    def statistics(self, axis: int | None = None) -> dict[str, Any]:
        """Return the core aggregate and range statistics in one mapping."""

        return {
            "sum": self.sum(axis),
            "mean": self.mean(axis),
            "median": self.median(axis),
            "std": self.std(axis),
            "variance": self.var(axis),
            "minimum": self.minimum(axis),
            "maximum": self.maximum(axis),
        }

    calculate_statistics = statistics


class AnalyzerCLI:
    """Input/output controller for the :class:`DataAnalytics` menu."""

    def __init__(self, analyzer: DataAnalytics | None = None) -> None:
        self.analyzer = analyzer or DataAnalytics()

    @staticmethod
    def _prompt_int(prompt: str, minimum: int | None = None) -> int:
        while True:
            try:
                value = int(input(prompt).strip())
                if minimum is not None and value < minimum:
                    raise ValueError
                return value
            except EOFError as exc:
                raise KeyboardInterrupt from exc
            except ValueError:
                print("Please enter a valid whole number.")

    @staticmethod
    def _prompt_float(prompt: str) -> float:
        while True:
            try:
                return float(input(prompt).strip())
            except EOFError as exc:
                raise KeyboardInterrupt from exc
            except ValueError:
                print("Please enter a valid number.")

    @staticmethod
    def _prompt_shape() -> tuple[int, ...]:
        while True:
            raw = input("Enter dimensions (for example, 2 3): ").strip()
            try:
                pieces = raw.replace("x", " ").replace(",", " ").split()
                shape = tuple(int(piece) for piece in pieces)
                return DataAnalytics._validate_shape(shape)
            except EOFError as exc:
                raise KeyboardInterrupt from exc
            except (ValueError, TypeError):
                print("Enter one or more positive integer dimensions.")

    @staticmethod
    def _prompt_values(count: int) -> list[int | float]:
        while True:
            raw = input(f"Enter {count} numeric values separated by spaces or commas: ").strip()
            try:
                pieces = raw.replace(",", " ").split()
                values: list[int | float] = []
                for piece in pieces:
                    number = float(piece)
                    values.append(int(number) if number.is_integer() else number)
                if len(values) != count:
                    raise ValueError
                return values
            except EOFError as exc:
                raise KeyboardInterrupt from exc
            except ValueError:
                print(f"Please enter exactly {count} numeric values.")

    def _require_current(self) -> bool:
        if self.analyzer.array is None:
            print("Create an array first.")
            return False
        return True

    def _create_menu(self) -> None:
        print("1. Create a 1D array")
        print("2. Create a 2D array")
        print("3. Create a 3D array")
        choice = input("Choose the array dimension (or press Enter to return): ").strip()
        if not choice:
            return
        if choice not in {"1", "2", "3"}:
            print("Please choose 1, 2, or 3.")
            return

        if choice == "1":
            shape = (self._prompt_int("Length: ", minimum=1),)
        elif choice == "2":
            shape = (
                self._prompt_int("Rows: ", minimum=1),
                self._prompt_int("Columns: ", minimum=1),
            )
        else:
            shape = (
                self._prompt_int("Depth: ", minimum=1),
                self._prompt_int("Rows: ", minimum=1),
                self._prompt_int("Columns: ", minimum=1),
            )
        count = int(np.prod(shape))
        values = self._prompt_values(count)
        try:
            created = self.analyzer.create_array(values, shape)
            print("Array created successfully:")
            print(created)
        except (TypeError, ValueError) as exc:
            print(f"Error: {exc}")

    def _index_menu(self) -> None:
        if not self._require_current():
            return
        try:
            mode = input("1. Index  2. Slice (or press Enter to return): ").strip()
            if not mode:
                return
            if mode == "1":
                print("Enter a zero-based index. Use comma-separated indexes for multiple dimensions.")
                raw = input("Index: ").strip()
                pieces = [int(part.strip()) for part in raw.split(",")]
                index: Any = pieces[0] if len(pieces) == 1 else tuple(pieces)
                result = self.analyzer.index(index)
            elif mode == "2":
                axis_raw = input("Axis (blank for 0): ").strip()
                axis = int(axis_raw) if axis_raw else 0
                start = input("Start (blank for beginning): ").strip()
                stop = input("Stop (blank for end): ").strip()
                step = input("Step (blank for 1): ").strip()
                result = self.analyzer.slice_array(
                    int(start) if start else None,
                    int(stop) if stop else None,
                    int(step) if step else None,
                    axis=axis,
                )
            else:
                print("Please choose 1 or 2.")
                return
            print("Result:")
            print(result)
        except (ValueError, IndexError, EOFError) as exc:
            print(f"Error: invalid index or slice ({exc}).")

    def _arithmetic_menu(self) -> None:
        if not self._require_current():
            return
        print("1. Addition  2. Subtraction  3. Multiplication  4. Division")
        print("5. Dot product  6. Matrix multiplication  7. Index or slice")
        choice = input("Choose an operation (or press Enter to return): ").strip()
        if not choice:
            return
        if choice == "7":
            self._index_menu()
            return
        operations = {
            "1": self.analyzer.addition,
            "2": self.analyzer.subtraction,
            "3": self.analyzer.multiplication,
            "4": self.analyzer.division,
            "5": self.analyzer.dot_product,
            "6": self.analyzer.matrix_multiplication,
        }
        operation = operations.get(choice)
        if operation is None:
            print("Please choose an option from 1 to 6.")
            return
        try:
            if choice in {"5", "6"}:
                shape = self._prompt_shape()
                values = self._prompt_values(int(np.prod(shape)))
                other: Any = np.asarray(values).reshape(shape)
            else:
                operand_mode = input(
                    "Operand: 1. scalar  2. array (press Enter for scalar): "
                ).strip()
                if operand_mode in {"", "1"}:
                    other = self._prompt_float("Enter a scalar operand: ")
                elif operand_mode == "2":
                    if self.analyzer.array is None:  # defensive; _require_current ran above
                        return
                    shape = self.analyzer.array.shape
                    print(f"Enter an array with shape {shape}.")
                    values = self._prompt_values(int(np.prod(shape)))
                    other = np.asarray(values).reshape(shape)
                else:
                    print("Please choose scalar or array operand.")
                    return
            print("Result:")
            print(operation(other))
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            print(f"Error: {exc}")

    def _combine_menu(self) -> None:
        if not self._require_current():
            return
        print("1. Vertical stack  2. Horizontal stack  3. Concatenate  4. Split")
        choice = input("Choose an operation (or press Enter to return): ").strip()
        if not choice:
            return
        try:
            if choice == "4":
                sections = self._prompt_int("Number of sections: ", minimum=1)
                print(self.analyzer.split(sections))
                return
            shape = self._prompt_shape()
            values = self._prompt_values(int(np.prod(shape)))
            other = np.asarray(values).reshape(shape)
            if choice == "1":
                result = self.analyzer.vertical_stack(other)
            elif choice == "2":
                result = self.analyzer.horizontal_stack(other)
            elif choice == "3":
                axis = self._prompt_int("Axis: ")
                result = self.analyzer.concatenate(other, axis=axis)
            else:
                print("Please choose an option from 1 to 4.")
                return
            print(result)
        except (TypeError, ValueError, EOFError) as exc:
            print(f"Error: {exc}")

    def _search_menu(self) -> None:
        if not self._require_current():
            return
        print("1. Search  2. Sort  3. Filter")
        choice = input("Choose an operation (or press Enter to return): ").strip()
        if not choice:
            return
        try:
            if choice == "1":
                value = self._prompt_float("Value to search for: ")
                matches = self.analyzer.search(value)
                if matches and matches[0].size:
                    print("Matching indexes:", matches)
                else:
                    print("No matching value was found.")
            elif choice == "2":
                order = input("Order (ascending/descending): ").strip()
                print(self.analyzer.sort(order))
            elif choice == "3":
                condition = input("Condition (>, <, ==, >=, <=, !=): ").strip()
                value = self._prompt_float("Comparison value: ")
                print(self.analyzer.filter(condition, value))
            else:
                print("Please choose an option from 1 to 3.")
        except (TypeError, ValueError, EOFError) as exc:
            print(f"Error: {exc}")

    def _statistics_menu(self) -> None:
        if not self._require_current():
            return
        print("1. Aggregate  2. Minimum/maximum  3. Percentile  4. Correlation")
        choice = input("Choose an operation (or press Enter to return): ").strip()
        if not choice:
            return
        try:
            if choice == "1":
                operation = input("Aggregation (sum/mean/median/std/var): ").strip()
                print(self.analyzer.aggregate(operation))
            elif choice == "2":
                print("Minimum:", self.analyzer.minimum())
                print("Maximum:", self.analyzer.maximum())
            elif choice == "3":
                value = self._prompt_float("Percentile (0-100): ")
                print(self.analyzer.percentile(value))
            elif choice == "4":
                shape = self._prompt_shape()
                values = self._prompt_values(int(np.prod(shape)))
                other = np.asarray(values).reshape(shape)
                print(self.analyzer.correlation(other))
            else:
                print("Please choose an option from 1 to 4.")
        except (TypeError, ValueError, EOFError) as exc:
            print(f"Error: {exc}")

    def run(self) -> None:
        """Run the main menu until the user chooses Exit or sends EOF."""

        print("Welcome to the NumPy Analyzer!")
        while True:
            print("\n" + "=" * 40)
            print("1. Create a NumPy Array")
            print("2. Perform Mathematical Operations")
            print("3. Combine or Split Arrays")
            print("4. Search, Sort, or Filter Arrays")
            print("5. Compute Aggregates and Statistics")
            print("6. Exit")
            try:
                choice = input("Choose an option: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return
            try:
                if choice == "1":
                    self._create_menu()
                elif choice == "2":
                    self._arithmetic_menu()
                elif choice == "3":
                    self._combine_menu()
                elif choice == "4":
                    self._search_menu()
                elif choice == "5":
                    self._statistics_menu()
                elif choice == "6":
                    print("Goodbye!")
                    return
                else:
                    print("Please choose an option from 1 to 6.")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                return


def main() -> None:
    """Application entry point."""

    AnalyzerCLI().run()


if __name__ == "__main__":
    main()
