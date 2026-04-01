## 2024-03-27 - Vectorizing Pandas iterrows()
**Learning:** Iterating over Pandas DataFrames using `iterrows()` is a massive performance bottleneck due to the overhead of creating a Series object for each row. When iterating sequentially, converting the relevant columns to NumPy arrays (`df[cols].to_numpy()`) and iterating over indices (`for i in range(len(df)):`) yields a ~17x speedup.
**Action:** Always prefer vectorized operations or iterating over pre-extracted NumPy arrays instead of `iterrows()` or `itertuples()` when sequential processing is unavoidable.

## 2024-05-20 - Pandas DataFrame Concatenation Overhead
**Learning:** Concatenating DataFrames (`pd.concat`) and doing substring searches on MultiIndex columns (`str.startswith`) to split them apart again is extremely expensive and scales poorly. In our `ReturnsBuilder` feature construction, aligning indexes directly (`lagged.index.intersection(forward.index)`) and filtering invalid rows (`notna().all(axis=1)`) avoids creating massive intermediate DataFrames in memory and bypasses expensive string checks.
**Action:** When merging, filtering, and returning subsets of multiple aligned DataFrames, avoid `.concat()` unless the final combined shape is strictly required. Use `.intersection()` for alignment and bitwise `&` on `.notna().all(axis=1)` masks for efficient multi-frame `dropna()`.

## 2024-05-24 - Pandas DataFrame apply(axis=1) Overhead
**Learning:** Using `apply(..., axis=1)` to process rows in a Pandas DataFrame is surprisingly slow because it instantiates a Pandas Series for every single row before passing it to the function. When vectorization is not an option (e.g. for row-wise string concatenation), replacing `df.apply(lambda row: func(row), axis=1)` with a list comprehension iterating over `df.itertuples(index=False, name=None)` yields a ~3x speedup.
**Action:** When performing row-by-row string operations or non-vectorizable logic, avoid `.apply(..., axis=1)`. Use `itertuples(index=False, name=None)` coupled with a list comprehension instead.

## 2025-04-01 - Vectorized Pandas Operations

**Learning:** `DataFrame.apply(np.log1p)` incurs heavy Python overhead when applied over thousands of rows or numerous columns, because it dispatches row-by-row or column-by-column. Instead, `np.log1p(DataFrame)` passes the entire C-contiguous memory block to NumPy's compiled ufunc, maintaining pandas indexing and vastly outperforming `.apply()`. Additionally, iterating over columns to apply string transformations (`.astype(str).str.rstrip('%')` and `pd.to_numeric()`) is significantly slower than using `.replace('%', '', regex=True)` on the whole DataFrame followed by a single `.apply(pd.to_numeric)`.

**Action:** Whenever applying standard mathematical functions to pandas DataFrames, check if a NumPy universal function (ufunc) can be applied directly to the DataFrame. Avoid `.apply()` unless custom Python logic is required. Replace loops over DataFrame columns with vectorized operations across the entire DataFrame structure where applicable.
