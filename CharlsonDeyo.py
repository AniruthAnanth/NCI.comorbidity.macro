from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd

ICD10_CUTOFF = pd.Timestamp("2015-10-01")


@dataclass
class Rule:
    icd9_prefix: List[str] = field(default_factory=list)
    icd9_ranges: List[Tuple[str, str]] = field(default_factory=list)
    icd10_prefix: List[str] = field(default_factory=list)
    icd10_ranges: List[Tuple[str, str]] = field(default_factory=list)


CHARLSON_DEYO_RULES: Dict[str, Rule] = {
    "acute_mi": Rule(icd9_prefix=["410"], icd10_prefix=["I21", "I22"]),
    "history_mi": Rule(icd9_prefix=["412"], icd10_prefix=["I252"]),
    "chf": Rule(
        icd9_prefix=["39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493", "428"],
        icd9_ranges=[("4254", "4259")],
        icd10_prefix=["I099", "I110", "I130", "I132", "I255", "I420", "I43", "I50", "P290"],
        icd10_ranges=[("I425", "I429")],
    ),
    "pvd": Rule(
        icd9_prefix=["0930", "440", "441", "4471", "5571", "5579", "V434"],
        icd9_ranges=[("4431", "4439")],
        icd10_prefix=["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"],
    ),
    "cvd": Rule(icd9_prefix=["36234"], icd9_ranges=[("430", "438")], icd10_prefix=["G45", "G46", "H340", "I6"]),
    "copd": Rule(
        icd9_prefix=["4168", "4169", "5064", "5081", "5088"],
        icd9_ranges=[("490", "505")],
        icd10_prefix=["I278", "I279", "J684", "J701", "J703"],
        icd10_ranges=[("J40", "J47"), ("J60", "J67")],
    ),
    "dementia": Rule(icd9_prefix=["290", "2941", "3312"], icd10_prefix=["F051", "G30", "G311"], icd10_ranges=[("F00", "F03")]),
    "paralysis": Rule(
        icd9_prefix=["3341", "342", "343", "3449"],
        icd9_ranges=[("3440", "3446")],
        icd10_prefix=["G041", "G114", "G801", "G802", "G81", "G82", "G839"],
        icd10_ranges=[("G830", "G834")],
    ),
    "diabetes": Rule(
        icd9_prefix=["2500", "2501", "2502", "2503", "2508", "2509"],
        icd10_prefix=["E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E130", "E131", "E136", "E138", "E139"],
    ),
    "diabetes_comp": Rule(
        icd9_prefix=["2504", "2505", "2506", "2507"],
        icd10_prefix=["E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115", "E117", "E132", "E133", "E134", "E135", "E137"],
    ),
    "renal_disease": Rule(
        icd9_prefix=["40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493", "582", "585", "586", "5880", "V420", "V451", "V56"],
        icd9_ranges=[("5830", "5837")],
        icd10_prefix=["I120", "I131", "N18", "N19", "N250", "Z940", "Z992"],
        icd10_ranges=[("N032", "N037"), ("N052", "N057"), ("Z490", "Z492")],
    ),
    "mild_liver_disease": Rule(
        icd9_prefix=["07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709", "570", "571", "5733", "5734", "5738", "5739", "V427"],
        icd10_prefix=["B18", "K709", "K717", "K73", "K74", "K760", "K768", "K769", "Z944"],
        icd10_ranges=[("K700", "K703"), ("K713", "K715"), ("K762", "K764")],
    ),
    "liver_disease": Rule(
        icd9_ranges=[("4560", "4562"), ("5722", "5728")],
        icd10_prefix=["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"],
    ),
    "ulcers": Rule(icd9_ranges=[("531", "534")], icd10_ranges=[("K25", "K28")]),
    "rheum_disease": Rule(
        icd9_prefix=["4465", "7148", "725"],
        icd9_ranges=[("7100", "7104"), ("7140", "7142")],
        icd10_prefix=["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
    ),
    "aids": Rule(icd9_ranges=[("042", "044")], icd10_prefix=["B20", "B21", "B22", "B24"]),
}


def normalize_icd(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .fillna("")
    )


def prefix_match(codes: pd.Series, prefixes: List[str]) -> pd.Series:
    if not prefixes:
        return pd.Series(False, index=codes.index)
    out = pd.Series(False, index=codes.index)
    for p in prefixes:
        out = out | codes.str.startswith(p)
    return out


def range_match(codes: pd.Series, ranges: List[Tuple[str, str]]) -> pd.Series:
    if not ranges:
        return pd.Series(False, index=codes.index)
    out = pd.Series(False, index=codes.index)
    for start, end in ranges:
        n = len(start)
        slice_ = codes.str.slice(0, n)
        out = out | ((slice_ >= start) & (slice_ <= end))
    return out


def match_rule(codes: pd.Series, icd_version: pd.Series, rule: Rule) -> pd.Series:
    codes = normalize_icd(codes)
    is_icd9 = icd_version.eq("icd9")
    out = pd.Series(False, index=codes.index)

    if is_icd9.any():
        c9 = codes[is_icd9]
        out.loc[is_icd9] = prefix_match(c9, rule.icd9_prefix) | range_match(c9, rule.icd9_ranges)
    if (~is_icd9).any():
        c10 = codes[~is_icd9]
        out.loc[~is_icd9] = prefix_match(c10, rule.icd10_prefix) | range_match(c10, rule.icd10_ranges)

    return out


def to_dx_long(dx_df: pd.DataFrame) -> pd.DataFrame:
    if {"PATIENT_ID", "CLM_THRU_DT", "DX"}.issubset(dx_df.columns):
        out = dx_df.copy()
        out["DX"] = normalize_icd(out["DX"])
        out = out[out["DX"].ne("")]
        return out

    dx_cols = [
        c
        for c in dx_df.columns
        if c.startswith("ICD_DGNS") or c.startswith("PRNCPAL_DGNS") or c.startswith("FST_DGNS")
    ]
    if not dx_cols:
        raise ValueError("Input claims must include either DX or ICD_DGNS_*/PRNCPAL_DGNS/FST_DGNS columns.")

    out = dx_df.melt(
        id_vars=[c for c in dx_df.columns if c not in dx_cols],
        value_vars=dx_cols,
        var_name="dx_col",
        value_name="DX",
    )
    out = out.dropna(subset=["DX"]).drop(columns=["dx_col"])
    out["DX"] = normalize_icd(out["DX"])
    out = out[out["DX"].ne("")]
    return out


def calculate_charlson_deyo(
    dx_claims: pd.DataFrame,
    seer_df: pd.DataFrame,
    window_before_days: int = 395,
    window_after_days: int = 30,
    ruleout: bool = False,
    ruleout_window_days: int = 30,
) -> pd.DataFrame:
    required_claim_cols = {"PATIENT_ID", "CLM_THRU_DT", "source"}
    required_seer_cols = {"PATIENT_ID", "dx.date.end"}
    if not required_claim_cols.issubset(dx_claims.columns):
        raise ValueError("dx_claims must include PATIENT_ID, CLM_THRU_DT, and source.")
    if not required_seer_cols.issubset(seer_df.columns):
        raise ValueError("seer_df must include PATIENT_ID and dx.date.end.")

    claims = dx_claims.copy()
    claims["CLM_THRU_DT"] = pd.to_datetime(claims["CLM_THRU_DT"])
    if "CLM_FROM_DT" in claims.columns:
        claims["CLM_FROM_DT"] = pd.to_datetime(claims["CLM_FROM_DT"])
    seer = seer_df.copy()
    seer["dx.date.end"] = pd.to_datetime(seer["dx.date.end"])

    dx_long = to_dx_long(claims)
    if "CLM_FROM_DT" in dx_long.columns:
        dx_long["claim_start_date"] = dx_long["CLM_FROM_DT"].where(
            dx_long["CLM_FROM_DT"].notna(), dx_long["CLM_THRU_DT"]
        )
    else:
        dx_long["claim_start_date"] = dx_long["CLM_THRU_DT"]
    dx_long["claim_end_date"] = dx_long["CLM_THRU_DT"]
    dx_long = dx_long.merge(seer[["PATIENT_ID", "dx.date.end"]], on="PATIENT_ID", how="inner")
    dx_long["start_date"] = dx_long["dx.date.end"] - pd.to_timedelta(window_before_days, unit="D")
    dx_long["end_date"] = dx_long["dx.date.end"] + pd.to_timedelta(window_after_days, unit="D")
    extra_days = int(ruleout_window_days) if ruleout else 0
    dx_long["scan_start"] = dx_long["start_date"] - pd.to_timedelta(extra_days, unit="D")
    dx_long["scan_end"] = dx_long["end_date"] + pd.to_timedelta(extra_days, unit="D")
    dx_long["in_window"] = (dx_long["claim_start_date"] >= dx_long["start_date"]) & (
        dx_long["claim_start_date"] <= dx_long["end_date"]
    )
    dx_long = dx_long[
        (dx_long["claim_start_date"] >= dx_long["scan_start"]) & (dx_long["claim_start_date"] <= dx_long["scan_end"])
    ].copy()
    dx_long["icd9or10"] = dx_long["claim_end_date"].ge(ICD10_CUTOFF).map({True: "icd10", False: "icd9"})
    dx_long["claim_month"] = dx_long["claim_start_date"].dt.to_period("M").astype(str)

    charlson = seer[["PATIENT_ID"]].drop_duplicates().copy()

    for name, rule in CHARLSON_DEYO_RULES.items():
        flagged = dx_long.copy()
        flagged["flag"] = match_rule(flagged["DX"], flagged["icd9or10"], rule)
        flagged = flagged[flagged["flag"]]

        if flagged.empty:
            summary = pd.DataFrame(
                columns=[
                    "PATIENT_ID",
                    f"{name}_pre_count",
                    f"{name}_pre_month_count",
                    f"{name}_medpar",
                    f"{name}_otherfile",
                    f"{name}_binary",
                ]
            )
        else:
            grouped = flagged.groupby("PATIENT_ID")
            summary = grouped.apply(
                lambda g: pd.Series(
                    {
                        f"{name}_pre_count": int(g["in_window"].sum()),
                        f"{name}_pre_month_count": int(g.loc[g["in_window"], "claim_month"].nunique()),
                        f"{name}_medpar": (g["source"] == "medpar").any(),
                        f"{name}_otherfile": g["source"].isin(["carrierbase", "outpat"]).any(),
                        "first_date": g["claim_start_date"].min(),
                        "last_date": g["claim_start_date"].max(),
                        "first_in_window": g.loc[g["in_window"], "claim_start_date"].min()
                        if g["in_window"].any()
                        else pd.NaT,
                    }
                )
            , include_groups=False).reset_index()

            if ruleout:
                ruleout_excluded = (
                    summary["first_in_window"].notna()
                    & ~summary[f"{name}_medpar"].astype(bool)
                    & ((summary["last_date"] - summary["first_date"]).dt.days <= int(ruleout_window_days))
                )
            else:
                ruleout_excluded = pd.Series(False, index=summary.index)

            summary[f"{name}_binary"] = (
                summary["first_in_window"].notna() & ~ruleout_excluded
            ).astype(int)
            summary = summary.drop(columns=["first_date", "last_date", "first_in_window"])

        charlson = charlson.merge(summary, on="PATIENT_ID", how="left")

    count_cols = [c for c in charlson.columns if c.endswith("_count")]
    charlson[count_cols] = charlson[count_cols].fillna(0).astype(int)

    for x in CHARLSON_DEYO_RULES:
        pre_col = f"{x}_pre_count"
        medpar_col = f"{x}_medpar"
        other_col = f"{x}_otherfile"
        status_col = f"{x}_status"
        binary_col = f"{x}_binary"

        charlson[medpar_col] = charlson[medpar_col].astype("boolean").fillna(False).astype(bool)
        charlson[other_col] = charlson[other_col].astype("boolean").fillna(False).astype(bool)
        charlson[binary_col] = charlson[binary_col].fillna(0).astype(int)
        charlson[status_col] = "DNE"
        charlson.loc[charlson[binary_col] == 1, status_col] = "Valid"
        charlson.loc[(charlson[binary_col] == 0) & (charlson[pre_col] > 0), status_col] = "Exist but Invalid"

    charlson["Charlson"] = (
        ((charlson["acute_mi_binary"] == 1) | (charlson["history_mi_binary"] == 1)).astype(int)
        + (charlson["chf_binary"] == 1).astype(int)
        + (charlson["pvd_binary"] == 1).astype(int)
        + (charlson["cvd_binary"] == 1).astype(int)
        + (charlson["copd_binary"] == 1).astype(int)
        + (charlson["dementia_binary"] == 1).astype(int)
        + 2 * (charlson["paralysis_binary"] == 1).astype(int)
        + ((charlson["diabetes_binary"] == 1) & (charlson["diabetes_comp_binary"] == 0)).astype(int)
        + 2 * (charlson["diabetes_comp_binary"] == 1).astype(int)
        + 2 * (charlson["renal_disease_binary"] == 1).astype(int)
        + ((charlson["mild_liver_disease_binary"] == 1) & (charlson["liver_disease_binary"] == 0)).astype(int)
        + 3 * (charlson["liver_disease_binary"] == 1).astype(int)
        + (charlson["ulcers_binary"] == 1).astype(int)
        + (charlson["rheum_disease_binary"] == 1).astype(int)
        + 6 * (charlson["aids_binary"] == 1).astype(int)
    )

    charlson["charlson.bin"] = pd.cut(
        charlson["Charlson"],
        bins=[-float("inf"), 0, 1, 2, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype("Int64")

    charlson["NCI_index"] = (
        0.12624 * (charlson["acute_mi_binary"] == 1).astype(int)
        + 0.07999 * (charlson["history_mi_binary"] == 1).astype(int)
        + 0.64441 * (charlson["chf_binary"] == 1).astype(int)
        + 0.26232 * (charlson["pvd_binary"] == 1).astype(int)
        + 0.27868 * (charlson["cvd_binary"] == 1).astype(int)
        + 0.52487 * (charlson["copd_binary"] == 1).astype(int)
        + 0.72219 * (charlson["dementia_binary"] == 1).astype(int)
        + 0.39882 * (charlson["paralysis_binary"] == 1).astype(int)
        + 0.29408 * ((charlson["diabetes_binary"] == 1) | (charlson["diabetes_comp_binary"] == 1)).astype(int)
        + 0.47010 * (charlson["renal_disease_binary"] == 1).astype(int)
        + 0.73803 * ((charlson["mild_liver_disease_binary"] == 1) | (charlson["liver_disease_binary"] == 1)).astype(int)
        + 0.07506 * (charlson["ulcers_binary"] == 1).astype(int)
        + 0.21905 * (charlson["rheum_disease_binary"] == 1).astype(int)
        + 0.58362 * (charlson["aids_binary"] == 1).astype(int)
    )

    return charlson


# Example:
# scores = calculate_charlson_deyo(dx_claims=dx_total_df, seer_df=seer_total_df)
# scores.to_parquet("charlson_codes_clean.parquet", index=False)
