import numpy as np
import pandas as pd
import sys
sys.path.append('../../../_/')
from lsms_library.local_tools import df_data_grabber, to_parquet

def extract_string(x):
    """Extract string value, converting to standard format"""
    try:
        return str(x).strip()
    except AttributeError:
        return ''

def sex_mapper(x):
    """Convert sex: 1=male, 2=female (lowercase m/f)"""
    try:
        if pd.isna(x):
            return ''
        # Handle numeric values (1/2 or 1.0/2.0)
        try:
            if float(x) == 2 or float(x) == 2.0:
                return 'f'
            if float(x) == 1 or float(x) == 1.0:
                return 'm'
        except (ValueError, TypeError):
            pass
        # Handle string values (categorical labels)
        x_str = str(x).strip()
        if x_str.lower() in ['male', 'm', '1']:
            return 'm'
        if x_str.lower() in ['female', 'f', '2']:
            return 'f'
        return ''
    except (ValueError, TypeError):
        return ''

def make_i(x):
    """Create household id from household_id"""
    try:
        if pd.isna(x):
            return ''
        return str(x).strip()
    except (ValueError, TypeError):
        return ''

def relation_mapper_ethiopia(x):
    """Map relation codes to English for Ethiopia waves"""
    # Standard Ethiopia LSMS relationship codes
    mapping = {
        1: 'Head',
        2: 'Spouse',
        3: 'Child',
        4: 'Son-in-law/Daughter-in-law',
        5: 'Grandchild',
        6: 'Parent',
        7: 'Parent-in-law',
        8: 'Sibling',
        9: 'Niece/Nephew',
        10: 'Other relative',
        11: 'Adopted/Foster/Stepchild',
        12: 'Non-relative',
        13: 'Other relative',  # Extended codes
        14: 'Other relative',
        15: 'Other relative',
        98: '',  # Don't know
        99: '',  # Missing
    }
    try:
        if pd.isna(x):
            return ''
        # Try numeric first
        try:
            code = int(float(x))
            return mapping.get(code, '')
        except (ValueError, TypeError):
            pass
        # Try string lookup
        x_str = str(x).strip()
        return mapping.get(int(float(x_str)), '') if x_str.replace('.', '').isdigit() else ''
    except (ValueError, TypeError):
        return ''

# Load data
idxvars = dict(i=('household_id', make_i))

myvars = dict(sex=('hh_s1q03', sex_mapper),
              age='hh_s1q04_a',
              relation=('hh_s1q02', relation_mapper_ethiopia))

df = df_data_grabber('../Data/sect1_hh_w1.dta', idxvars, convert_categoricals=False, **myvars)

# Create pid from individual_id
# First we need to load the individual_id separately since it's not in idxvars
from lsms_library.local_tools import get_dataframe
df_temp = get_dataframe('../Data/sect1_hh_w1.dta', convert_categoricals=False)
df_temp = df_temp.set_index('household_id')

# Reset index to add individual_id
df = df.reset_index()
df = df.merge(df_temp[['individual_id']].reset_index(), left_on='i', right_on='household_id', how='left')
df['pid'] = df['individual_id'].astype(str)

# Drop helper columns
df = df.drop(columns=['household_id', 'individual_id'], errors='ignore')

# Set index
df = df.set_index(['i', 'pid'])

# Add time dimension (wave year)
df.index = pd.MultiIndex.from_tuples(
    [('2011-12',) + idx for idx in df.index],
    names=['t'] + df.index.names
)

# Clean up any completely empty rows (all columns NaN)
df = df.replace('', np.nan)
df = df[~df.isna().all(axis=1)]  # Only drop rows where ALL columns are NaN
df = df.sort_index()

# Ensure we have data
if len(df) == 0:
    raise ValueError("household_roster is empty! Check data files and column mappings.")

to_parquet(df, 'household_roster.parquet')

