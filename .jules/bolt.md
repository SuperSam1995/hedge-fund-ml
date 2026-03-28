## 2024-03-27 - Vectorizing Pandas iterrows()
**Learning:** Iterating over Pandas DataFrames using `iterrows()` is a massive performance bottleneck due to the overhead of creating a Series object for each row. When iterating sequentially, converting the relevant columns to NumPy arrays (`df[cols].to_numpy()`) and iterating over indices (`for i in range(len(df)):`) yields a ~17x speedup.
**Action:** Always prefer vectorized operations or iterating over pre-extracted NumPy arrays instead of `iterrows()` or `itertuples()` when sequential processing is unavoidable.

## 2024-05-20 - Pandas DataFrame Concatenation Overhead
**Learning:** Concatenating DataFrames (`pd.concat`) and doing substring searches on MultiIndex columns (`str.startswith`) to split them apart again is extremely expensive and scales poorly. In our `ReturnsBuilder` feature construction, aligning indexes directly (`lagged.index.intersection(forward.index)`) and filtering invalid rows (`notna().all(axis=1)`) avoids creating massive intermediate DataFrames in memory and bypasses expensive string checks.
**Action:** When merging, filtering, and returning subsets of multiple aligned DataFrames, avoid `.concat()` unless the final combined shape is strictly required. Use `.intersection()` for alignment and bitwise `&` on `.notna().all(axis=1)` masks for efficient multi-frame `dropna()`.
