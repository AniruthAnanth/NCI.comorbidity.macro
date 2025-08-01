# NCI_comorbidity_macro.py

import pandas as pd
import numpy as np
from collections import defaultdict

def comorb(

    INFILE,                # Path to SAS dataset
    id_col,                # Patient ID column name (str)
    startdate_col,         # Start date of window (str)
    enddate_col,           # End date of window (str)
    claimstart_col,        # Claim start date column (str)
    claimend_col,          # Claim end date column (str)
    claimtype_col,         # Claim type column (str)
    dxvarlist,             # List of diagnosis code column names (list of str)
    ruleout,               # Ruleout flag (str or bool)
    outfile=None           # Output file path (optional)
):
    """
    Pythonic implementation of the NCI comorbidity SAS macro.
    Reads a SAS dataset, flags comorbidities, applies the ruleout algorithm, and computes indices.
    Returns a DataFrame (and optionally writes to CSV).
    """

    # Define comorbidity conditions and ICD code logic
    # ICD-9 and ICD-10 code sets for each condition (truncated for brevity; expand as needed)
    icd_map = {
        'acute_mi': {
            9: ['410'],
            10: ['I21', 'I22']
        },
        'history_mi': {
            9: ['412'],
            10: ['I252']
        },
        'chf': {
            9: ['428'],
            10: ['I50']
        },
        'pvd': {
            9: ['4439', '441', '7854', 'V434'],
            10: ['I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959']
        },
        'cvd': {
            9: ['430', '431', '432', '433', '434', '435', '436', '437', '438'],
            10: ['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69']
        },
        'copd': {
            9: ['490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505', '5064'],
            10: ['J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J66', 'J67']
        },
        'dementia': {
            9: ['290', '2941', '3312'],
            10: ['F00', 'F01', 'F02', 'F03', 'F051', 'G30', 'G311']
        },
        'paralysis': {
            9: ['342', '343', '344'],
            10: ['G81', 'G82', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839']
        },
        'diabetes': {
            9: ['250'],
            10: ['E10', 'E11', 'E13']  # E12/E14 excluded per macro update
        },
        'diabetes_comp': {
            9: ['2504', '2505', '2506', '2507', '2508', '2509'],
            10: ['E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E132', 'E133', 'E134', 'E135', 'E136', 'E137', 'E138', 'E139']
        },
        'renal_disease': {
            9: ['582', '5830', '5831', '5832', '5834', '5836', '585', '586', '5880'],
            10: ['N18', 'N19', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992']
        },
        'mild_liver_disease': {
            9: ['570', '5712', '5714', '5715', '5716', '5718', '5719', '5733', '5734', '5738', '5739', 'V427'],
            10: ['B18', 'K700', 'K701', 'K702', 'K703', 'K709', 'K713', 'K714', 'K715', 'K717', 'K73', 'K74', 'K760', 'K762', 'K763', 'K764', 'K768', 'K769', 'Z944']
        },
        'liver_disease': {
            9: ['5722', '5723', '5724', '5728'],
            10: ['I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765', 'K766', 'K767']
        },
        'ulcers': {
            9: ['531', '532', '533', '534'],
            10: ['K25', 'K26', 'K27', 'K28']
        },
        'rheum_disease': {
            9: ['4465', '7100', '7101', '7104', '7140', '7141', '7142', '7148', '725'],
            10: ['M05', 'M06', 'M315', 'M32', 'M33', 'M34', 'M351', 'M353', 'M360']
        },
        'aids': {
            9: ['042', '043', '044'],
            10: ['B20', 'B21', 'B22', 'B24']
        }
    }

    # Read in the SAS dataset
    claims = pd.read_sas(INFILE, encoding='utf-8')

    # Keep only relevant columns
    columns_to_keep = [id_col, startdate_col, enddate_col, claimstart_col, claimend_col, claimtype_col] + dxvarlist
    claims = claims[columns_to_keep].copy()

    # Convert date columns to datetime
    for col in [startdate_col, enddate_col, claimstart_col, claimend_col]:
        claims[col] = pd.to_datetime(claims[col], errors='coerce')

    # Select claim records in appropriate window
    if str(ruleout).upper() in ("Y", "1", "R"):
        mask = (
            (claims[startdate_col] - pd.Timedelta(days=30) <= claims[claimstart_col]) &
            (claims[claimstart_col] <= claims[enddate_col] + pd.Timedelta(days=30))
        )
        claims = claims[mask].copy()
        claims['inwindow'] = (
            (claims[startdate_col] <= claims[claimstart_col]) &
            (claims[claimstart_col] <= claims[enddate_col])
        )
    else:
        mask = (
            (claims[startdate_col] <= claims[claimstart_col]) &
            (claims[claimstart_col] <= claims[enddate_col])
        )
        claims = claims[mask].copy()
        claims['inwindow'] = True

    # Determine ICD version by claim end date
    claims['ICDVRSN'] = np.where(claims[claimend_col] < pd.Timestamp('2015-10-01'), 9, 10)

    # Melt diagnosis columns to long format for easier processing
    dx_long = claims.melt(
        id_vars=[id_col, startdate_col, enddate_col, claimstart_col, claimend_col, claimtype_col, 'ICDVRSN', 'inwindow'],
        value_vars=dxvarlist,
        var_name='DXVAR',
        value_name='DXCODE'
    ).dropna(subset=['DXCODE'])
    dx_long['DXCODE'] = dx_long['DXCODE'].astype(str).str.upper().str.strip()

    # --- Comorbidity flagging logic ---
    for cond, icd_versions in icd_map.items():
        dx_long[cond + '_flag'] = 0
        for icdver, code_list in icd_versions.items():
            if icdver == 9:
                # ICD-9: match by prefix (3-5 digits)
                for code in code_list:
                    dx_long.loc[(dx_long['ICDVRSN'] == 9) & dx_long['DXCODE'].str.startswith(code), cond + '_flag'] = 1
            else:
                # ICD-10: match by prefix (can be 3-5 chars)
                for code in code_list:
                    dx_long.loc[(dx_long['ICDVRSN'] == 10) & dx_long['DXCODE'].str.startswith(code), cond + '_flag'] = 1

    # --- Ruleout Algorithm ---
    if str(ruleout).upper() in ("Y", "1", "R"):
        # For each patient and condition, apply ruleout logic
        # 1. MedPAR claims (claimtype M) always count
        # 2. Outpatient/Carrier (O/N): need 2+ claims >30 days apart, or confirmation in MedPAR
        # We'll build a dict: {patient: {condition: [dates]}}
        medpar_types = ['M']
        outpatient_types = ['O', 'N']
        flagged = defaultdict(lambda: defaultdict(list))
        for cond in icd_map:
            # MedPAR claims
            medpar = dx_long[(dx_long[cond + '_flag'] == 1) & (dx_long[claimtype_col].isin(medpar_types))]
            for pid, group in medpar.groupby(id_col):
                flagged[pid][cond].extend(group[claimstart_col].tolist())
            # Outpatient/Carrier claims
            outpt = dx_long[(dx_long[cond + '_flag'] == 1) & (dx_long[claimtype_col].isin(outpatient_types))]
            for pid, group in outpt.groupby(id_col):
                dates = sorted(group[claimstart_col].dropna())
                if len(dates) >= 2:
                    # Check if any two dates are >30 days apart
                    found = False
                    for i in range(len(dates)):
                        for j in range(i+1, len(dates)):
                            if abs((dates[j] - dates[i]).days) > 30:
                                flagged[pid][cond].extend([dates[i], dates[j]])
                                found = True
                                break
                        if found:
                            break

        # Build summary DataFrame
        summary = []
        for pid in claims[id_col].unique():
            row = {id_col: pid}
            for cond in icd_map:
                cond_dates = flagged[pid][cond]
                if cond_dates:
                    row[cond] = 1
                    row[cond + '_first_date'] = min(cond_dates)
                else:
                    row[cond] = 0
                    row[cond + '_first_date'] = pd.NaT
            summary.append(row)
        out = pd.DataFrame(summary)
    else:
        # No ruleout: any claim in window counts
        summary = []
        for pid, group in dx_long.groupby(id_col):
            row = {id_col: pid}
            for cond in icd_map:
                cond_claims = group[group[cond + '_flag'] == 1]
                if not cond_claims.empty:
                    row[cond] = 1
                    row[cond + '_first_date'] = cond_claims[claimstart_col].min()
                else:
                    row[cond] = 0
                    row[cond + '_first_date'] = pd.NaT
            summary.append(row)
        out = pd.DataFrame(summary)

    # Merge with patient IDs to ensure all patients are included
    out = claims[[id_col]].drop_duplicates().merge(out, on=id_col, how='left').fillna({k: 0 for k in icd_map.keys()})

    # Calculate Charlson and NCI indices
    out['Charlson'] = (
        1 * ((out['acute_mi'] == 1) | (out['history_mi'] == 1)).astype(int) +
        1 * (out['chf'] == 1).astype(int) +
        1 * (out['pvd'] == 1).astype(int) +
        1 * (out['cvd'] == 1).astype(int) +
        1 * (out['copd'] == 1).astype(int) +
        1 * (out['dementia'] == 1).astype(int) +
        2 * (out['paralysis'] == 1).astype(int) +
        1 * ((out['diabetes'] == 1) & (out['diabetes_comp'] == 0)).astype(int) +
        2 * (out['diabetes_comp'] == 1).astype(int) +
        2 * (out['renal_disease'] == 1).astype(int) +
        1 * ((out['mild_liver_disease'] == 1) & (out['liver_disease'] == 0)).astype(int) +
        3 * (out['liver_disease'] == 1).astype(int) +
        1 * (out['ulcers'] == 1).astype(int) +
        1 * (out['rheum_disease'] == 1).astype(int) +
        6 * (out['aids'] == 1).astype(int)
    )

    out['NCI_index'] = (
        0.12624 * (out['acute_mi'] == 1).astype(int) +
        0.07999 * (out['history_mi'] == 1).astype(int) +
        0.64441 * (out['chf'] == 1).astype(int) +
        0.26232 * (out['pvd'] == 1).astype(int) +
        0.27868 * (out['cvd'] == 1).astype(int) +
        0.52487 * (out['copd'] == 1).astype(int) +
        0.72219 * (out['dementia'] == 1).astype(int) +
        0.39882 * (out['paralysis'] == 1).astype(int) +
        0.29408 * ((out['diabetes'] == 1) | (out['diabetes_comp'] == 1)).astype(int) +
        0.47010 * (out['renal_disease'] == 1).astype(int) +
        0.73803 * ((out['mild_liver_disease'] == 1) | (out['liver_disease'] == 1)).astype(int) +
        0.07506 * (out['ulcers'] == 1).astype(int) +
        0.21905 * (out['rheum_disease'] == 1).astype(int) +
        0.58362 * (out['aids'] == 1).astype(int)
    )

    # Sort by patient ID
    out = out.sort_values(by=id_col)

    # Output
    if outfile:
        out.to_csv(outfile, index=False)
    return out