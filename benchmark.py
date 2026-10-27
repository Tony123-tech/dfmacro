import time

import dfmacro_rs
import numpy as np
import pandas as pd

# 建立 5000 萬行嘅龐大數據 (佔用大約 400MB 記憶體)
N = 50_000_000
print(f"Creating DataFrame with {N:,} rows...")
arr = np.random.rand(N)
df = pd.DataFrame({"price": arr})

# 1. Pandas (單核)
t0 = time.perf_counter()
sum_pd = df["price"].sum()
t_pd = time.perf_counter() - t0
print(f"🐼 Pandas Sum: {t_pd:.4f}s | Result: {sum_pd:.2f}")

# 2. Rust (多核 Rayon)
t0 = time.perf_counter()
sum_rs = dfmacro_rs.sum_masked(arr, None)
t_rs = time.perf_counter() - t0
print(f"🦀 Rust Sum:   {t_rs:.4f}s | Result: {sum_rs:.2f}")

print(f"\n🏁 Speedup: {t_pd / t_rs:.2f}x faster")

# 3. 測試 Mean (平均值)
t0 = time.perf_counter()
mean_pd = df["price"].mean()
t_pd = time.perf_counter() - t0
print(f"\n🐼 Pandas Mean: {t_pd:.4f}s | Result: {mean_pd:.4f}")

t0 = time.perf_counter()
mean_rs = dfmacro_rs.mean_masked(arr, None)
t_rs = time.perf_counter() - t0
print(f"🦀 Rust Mean:   {t_rs:.4f}s | Result: {mean_rs:.4f}")

print(f"\n🏁 Speedup: {t_pd / t_rs:.2f}x faster")
