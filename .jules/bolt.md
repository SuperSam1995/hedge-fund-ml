## 2024-03-27 - Vectorizing Pandas iterrows()
**Learning:** Iterating over Pandas DataFrames using `iterrows()` is a massive performance bottleneck due to the overhead of creating a Series object for each row. When iterating sequentially, converting the relevant columns to NumPy arrays (`df[cols].to_numpy()`) and iterating over indices (`for i in range(len(df)):`) yields a ~17x speedup.
**Action:** Always prefer vectorized operations or iterating over pre-extracted NumPy arrays instead of `iterrows()` or `itertuples()` when sequential processing is unavoidable.

## 2024-05-20 - Pandas DataFrame Concatenation Overhead
**Learning:** Concatenating DataFrames (`pd.concat`) and doing substring searches on MultiIndex columns (`str.startswith`) to split them apart again is extremely expensive and scales poorly. In our `ReturnsBuilder` feature construction, aligning indexes directly (`lagged.index.intersection(forward.index)`) and filtering invalid rows (`notna().all(axis=1)`) avoids creating massive intermediate DataFrames in memory and bypasses expensive string checks.
**Action:** When merging, filtering, and returning subsets of multiple aligned DataFrames, avoid `.concat()` unless the final combined shape is strictly required. Use `.intersection()` for alignment and bitwise `&` on `.notna().all(axis=1)` masks for efficient multi-frame `dropna()`.

## 2024-05-24 - Pandas DataFrame apply(axis=1) Overhead
**Learning:** Using `apply(..., axis=1)` to process rows in a Pandas DataFrame is surprisingly slow because it instantiates a Pandas Series for every single row before passing it to the function. When vectorization is not an option (e.g. for row-wise string concatenation), replacing `df.apply(lambda row: func(row), axis=1)` with a list comprehension iterating over `df.itertuples(index=False, name=None)` yields a ~3x speedup.
**Action:** When performing row-by-row string operations or non-vectorizable logic, avoid `.apply(..., axis=1)`. Use `itertuples(index=False, name=None)` coupled with a list comprehension instead.

## 2024-05-24 - Precomputing redundant array multiplications and extracting diagonal optimally
**Learning:** `np.diag(2D_array)` allocates a new 1D array by copying the diagonal. For a large covariance matrix, this wastes time and memory compared to `2D_array.diagonal()`, which simply returns a read-only view. Also, in complex math formulas like `phi * new_arr * diag * delta - old_arr * diag * delta - 0.5 * np.square(delta) * diag`, `diag * delta` is computed multiple times which introduces unnecessary overhead.
**Action:** When working with large matrices, extract the diagonal via `arr.diagonal()` to return a view without copying. Look out for common element-wise operations and factor them out (`common = diag * delta`).
