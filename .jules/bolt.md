## 2024-03-27 - Vectorizing Pandas iterrows()
**Learning:** Iterating over Pandas DataFrames using `iterrows()` is a massive performance bottleneck due to the overhead of creating a Series object for each row. When iterating sequentially, converting the relevant columns to NumPy arrays (`df[cols].to_numpy()`) and iterating over indices (`for i in range(len(df)):`) yields a ~17x speedup.
**Action:** Always prefer vectorized operations or iterating over pre-extracted NumPy arrays instead of `iterrows()` or `itertuples()` when sequential processing is unavoidable.
