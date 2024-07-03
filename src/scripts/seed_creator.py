import pandas as pd
import numpy as np

num_seeds = 500000
seeds = np.random.randint(0, 2**32 - 1, size=num_seeds, dtype=np.uint32)
seeds_df = pd.DataFrame(seeds, columns=['seed'])
seeds_df.to_csv('seeds.csv', index=False)