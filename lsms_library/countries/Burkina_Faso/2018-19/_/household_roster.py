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
    """Convert sex: 1=male, 2=female (lowercase m/f) or handle string values"""
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
        if x_str.lower() in ['masculin', 'masculine', 'm', 'male']:
            return 'm'
        if x_str.lower() in ['féminin', 'feminin', 'féminine', 'feminine', 'f', 'female']:
            return 'f'
        return ''
    except (ValueError, TypeError):
        return ''

def make_i(x):
    """Create household id from grappe and menage"""
    try:
        grappe_val = int(x[0]) if not pd.isna(x[0]) else 0
        menage_val = int(x[1]) if not pd.isna(x[1]) else 0
        return str(grappe_val) + str(menage_val).rjust(3, '0')
    except (ValueError, TypeError, IndexError) as e:
        return ''


def make_age(x):
    """Calculate age from s01q04a or birth year s01q03c"""
    if not pd.isna(x[0]):
        return x[0]
    # Calculate from birth year (2019 for wave 1, 2020 for wave 2)
    if not pd.isna(x[1]):
        return 2019 - x[1]  # Using 2019 as base year for 2018-19 survey
    return np.nan

def relation_mapper_2018(x):
    """Map relation codes to English for 2018-19 wave"""
    # Mapping for 2018-19: codes 1-10
    mapping = {
        1: 'Head',
        2: 'Spouse',
        3: 'Child',
        4: 'Parent',
        5: 'Grandchild',
        6: 'Grandparent',
        7: 'Sibling',
        8: 'Other relative',
        9: 'Non-relative',
        10: 'Non-relative',
        # Handle string values if categoricals were converted
        'Chef de ménage': 'Head',
        'Conjoint ( e )': 'Spouse',
        'Conjoint(e)': 'Spouse',
        'Fils, Fille': 'Child',
        'Père, Mère': 'Parent',
        'Petit fils, petite fille': 'Grandchild',
        'Grand-parents': 'Grandparent',
        'Frère, sœur': 'Sibling',
        'Autres Parents du CM/Conjoint': 'Other relative',
        'Personne non apparentée au CM/Conjoint': 'Non-relative',
        'Domestique ou parent du domestique': 'Non-relative',
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
        return mapping.get(x_str, mapping.get(x_str.lower(), ''))
    except (ValueError, TypeError):
        return ''

# Load all data first
idxvars = dict(i=(['grappe', 'menage'], make_i),
               vague='vague')

myvars = dict(sex=('s01q01', sex_mapper),
              age=(['s01q04a', 's01q03c'], make_age),
              relation=('s01q02', relation_mapper_2018))

df_all = df_data_grabber('../Data/s01_me_bfa2018.dta', idxvars, convert_categoricals=False, **myvars)

# Check if we have data
if len(df_all) == 0:
    raise ValueError("No data loaded from s01_me_bfa2018.dta! Check file and column mappings.")

# Reset index to work with columns
df_all = df_all.reset_index()

# Wave 1 (vague = 1 or 1.0): 2018
w1 = df_all[df_all['vague'].isin([1, 1.0])].copy()
w1 = w1.drop(columns=['vague'])
w1['pid'] = w1.groupby('i').cumcount() + 1
w1['pid'] = w1['i'] + w1['pid'].astype(str).str.rjust(2, '0')
w1 = w1.set_index(['i', 'pid'])
w1.index = pd.MultiIndex.from_tuples(
    [('2018-19',) + idx for idx in w1.index],
    names=['t'] + w1.index.names
)

# Wave 2 (vague = 2 or 2.0): 2019
w2 = df_all[df_all['vague'].isin([2, 2.0])].copy()
w2 = w2.drop(columns=['vague'])
w2['pid'] = w2.groupby('i').cumcount() + 1
w2['pid'] = w2['i'] + w2['pid'].astype(str).str.rjust(2, '0')
w2 = w2.set_index(['i', 'pid'])
w2.index = pd.MultiIndex.from_tuples(
    [('2018-19',) + idx for idx in w2.index],
    names=['t'] + w2.index.names
)

# Concatenate both waves
df = pd.concat([w1, w2])

# Clean up any completely empty rows (all columns NaN)
# But keep rows even if some columns are NaN
df = df.replace('', np.nan)
df = df[~df.isna().all(axis=1)]  # Only drop rows where ALL columns are NaN
df = df.sort_index()

# Ensure we have data
if len(df) == 0:
    raise ValueError("household_roster is empty! Check data files and column mappings.")

to_parquet(df, 'household_roster.parquet')

