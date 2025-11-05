import pandas as pd
import json
import numpy as np
from lsms_library.local_tools import to_parquet
from lsms_library.local_tools import get_dataframe

X = []
for t in ['2014', '2018-19', '2021-22']:
    try:
        X.append(get_dataframe('../%s/_/household_roster.parquet' % t))
    except FileNotFoundError:
        print(f"Warning: No household_roster.parquet for {t}")

if X:
    x = pd.concat(X, axis=0)
    to_parquet(x, '../var/household_roster.parquet')

