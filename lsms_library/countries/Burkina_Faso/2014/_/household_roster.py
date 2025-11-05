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
    """Convert sex to m/f format (lowercase)"""
    x_str = extract_string(x)
    if x_str == 'Masculin':
        return 'm'
    if x_str in ['Feminin', 'Féminin']:
        return 'f'
    return x_str.lower() if x_str else ''

def make_i(x):
    """Create household id from zd and menage"""
    try:
        zd_val = x[0] if not pd.isna(x[0]) else ''
        menage_val = int(x[1]) if not pd.isna(x[1]) else 0
        return str(zd_val) + str(menage_val).rjust(3, '0')
    except (ValueError, TypeError, IndexError) as e:
        # Return a placeholder if conversion fails
        return ''

def relation_mapper_2014(x):
    """Map relation codes to English for 2014 wave"""
    # Mapping for 2014: B5 codes 1-9
    mapping = {
        1: 'Head',
        2: 'Spouse',
        3: 'Child',
        4: 'Grandchild',
        5: 'Parent',
        6: 'Sibling',
        7: 'Other relative',
        8: 'Non-relative',
        9: 'Non-relative',
        # Handle string values if categoricals were converted
        'Chef de ménage': 'Head',
        'Conjoint(e)': 'Spouse',
        'Fils ou fille': 'Child',
        'Petit fils/fille': 'Grandchild',
        'Père / mère': 'Parent',
        'Frère/sœur': 'Sibling',
        'Autre parent': 'Other relative',
        'Domestique/personnel de maison': 'Non-relative',
        'Sans lien de parenté': 'Non-relative',
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

# File 1 (p1): 2013_Q4 -> use wave year 2014
idxvars = dict(t=('zd', lambda x: "2014"),
               i=(['zd', 'menage'], make_i))

myvars = dict(sex=('B2', sex_mapper),
              age='B4',
              relation=('B5', relation_mapper_2014))

p1 = df_data_grabber('../Data/emc2014_p1_individu_27022015.dta', idxvars, **myvars)
# Create pid from row number within each household
p1 = p1.reset_index()
p1['pid'] = p1.groupby(['t', 'i']).cumcount() + 1
p1['pid'] = p1['i'] + p1['pid'].astype(str).str.rjust(2, '0')
p1 = p1.set_index(['t', 'i', 'pid'])

# File 2 (p2): 2014_Q1 -> use wave year 2014
idxvars = dict(t=('zd', lambda x: "2014"),
               i=(['zd', 'menage'], make_i))

myvars = dict(sex=('B2', sex_mapper),
              age='B4',
              relation=('B5', relation_mapper_2014))

p2 = df_data_grabber('../Data/emc2014_p2_individu_27022015.dta', idxvars, **myvars)
# Create pid from row number within each household
p2 = p2.reset_index()
p2['pid'] = p2.groupby(['t', 'i']).cumcount() + 1
p2['pid'] = p2['i'] + p2['pid'].astype(str).str.rjust(2, '0')
p2 = p2.set_index(['t', 'i', 'pid'])

# File 3 (p3): 2014_Q2 -> use wave year 2014
# Note: p3 may not have relation column, so we'll check and handle gracefully
idxvars = dict(t=('zd', lambda x: "2014"),
               i=(['zd', 'menage'], make_i))

# First check what columns exist
from lsms_library.local_tools import get_dataframe
df_temp = get_dataframe('../Data/emc2014_p3_individu_27022015.dta', convert_categoricals=False)

myvars = dict(sex=('sexe3', sex_mapper),
              age='age3')

# Try to add relation if column exists
if 'B5' in df_temp.columns:
    myvars['relation'] = ('B5', relation_mapper_2014)
elif 'relation3' in df_temp.columns:
    myvars['relation'] = ('relation3', relation_mapper_2014)
# If neither exists, relation will be NaN (we'll add it after loading)

p3 = df_data_grabber('../Data/emc2014_p3_individu_27022015.dta', idxvars, **myvars)

# If relation wasn't loaded, add it as NaN
if 'relation' not in p3.columns:
    p3['relation'] = np.nan

# Create pid from row number within each household
p3 = p3.reset_index()
p3['pid'] = p3.groupby(['t', 'i']).cumcount() + 1
p3['pid'] = p3['i'] + p3['pid'].astype(str).str.rjust(2, '0')
p3 = p3.set_index(['t', 'i', 'pid'])

# File 4 (p4): 2014_Q3 -> use wave year 2014
idxvars = dict(t=('zd', lambda x: "2014"),
               i=(['zd', 'menage'], make_i))

myvars = dict(sex=('B2', sex_mapper),
              age='B4B',
              relation=('B5', relation_mapper_2014))

p4 = df_data_grabber('../Data/emc2014_p4_individu_27022015.dta', idxvars, **myvars)
# Create pid from row number within each household
p4 = p4.reset_index()
p4['pid'] = p4.groupby(['t', 'i']).cumcount() + 1
p4['pid'] = p4['i'] + p4['pid'].astype(str).str.rjust(2, '0')
p4 = p4.set_index(['t', 'i', 'pid'])

# Concatenate all periods
df = pd.concat([p1, p2, p3, p4])

# Clean up any completely empty rows (all columns NaN)
# But keep rows even if some columns are NaN
df = df.replace('', np.nan)
df = df[~df.isna().all(axis=1)]  # Only drop rows where ALL columns are NaN
df = df.sort_index()

# Ensure we have data
if len(df) == 0:
    raise ValueError("household_roster is empty! Check data files and column mappings.")

to_parquet(df, 'household_roster.parquet')

