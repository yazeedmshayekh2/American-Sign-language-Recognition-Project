def add_padding(number, target_divisor):
    padding = (target_divisor - (number % target_divisor)) % target_divisor
    return number + padding

print(add_padding(74664, 543) // 3)

import pandas as pd
import numpy as np

# Sample DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Number of padding rows to add
num_padding_rows = 3

# Create a DataFrame with NaN values for padding rows
padding_rows = pd.DataFrame(np.nan, index=range(num_padding_rows), columns=df.columns)

# Append the padding rows to the original DataFrame
df_padded = pd.concat([df, padding_rows], ignore_index=True)

print(df_padded)
