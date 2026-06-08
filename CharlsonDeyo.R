library(dplyr)
library(tidyr)
library(lubridate)
library(stringr)
library(rlang)

ICD10_CUTOFF <- as.Date("2015-10-01")

CHARLSON_DEYO_RULES <- list(
  acute_mi = list(
    icd9_prefix = c("410"),
    icd10_prefix = c("I21", "I22")
  ),
  history_mi = list(
    icd9_prefix = c("412"),
    icd10_prefix = c("I252")
  ),
  chf = list(
    icd9_prefix = c("39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493", "428"),
    icd9_ranges = list(c("4254", "4259")),
    icd10_prefix = c("I099", "I110", "I130", "I132", "I255", "I420", "I43", "I50", "P290"),
    icd10_ranges = list(c("I425", "I429"))
  ),
  pvd = list(
    icd9_prefix = c("0930", "440", "441", "4471", "5571", "5579", "V434"),
    icd9_ranges = list(c("4431", "4439")),
    icd10_prefix = c("I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959")
  ),
  cvd = list(
    icd9_prefix = c("36234"),
    icd9_ranges = list(c("430", "438")),
    icd10_prefix = c("G45", "G46", "H340", "I6")
  ),
  copd = list(
    icd9_prefix = c("4168", "4169", "5064", "5081", "5088"),
    icd9_ranges = list(c("490", "505")),
    icd10_prefix = c("I278", "I279", "J684", "J701", "J703"),
    icd10_ranges = list(c("J40", "J47"), c("J60", "J67"))
  ),
  dementia = list(
    icd9_prefix = c("290", "2941", "3312"),
    icd10_prefix = c("F051", "G30", "G311"),
    icd10_ranges = list(c("F00", "F03"))
  ),
  paralysis = list(
    icd9_prefix = c("3341", "342", "343", "3449"),
    icd9_ranges = list(c("3440", "3446")),
    icd10_prefix = c("G041", "G114", "G801", "G802", "G81", "G82", "G839"),
    icd10_ranges = list(c("G830", "G834"))
  ),
  diabetes = list(
    icd9_prefix = c("2500", "2501", "2502", "2503", "2508", "2509"),
    icd10_prefix = c("E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E130", "E131", "E136", "E138", "E139")
  ),
  diabetes_comp = list(
    icd9_prefix = c("2504", "2505", "2506", "2507"),
    icd10_prefix = c("E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115", "E117", "E132", "E133", "E134", "E135", "E137")
  ),
  renal_disease = list(
    icd9_prefix = c("40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493", "582", "585", "586", "5880", "V420", "V451", "V56"),
    icd9_ranges = list(c("5830", "5837")),
    icd10_prefix = c("I120", "I131", "N18", "N19", "N250", "Z940", "Z992"),
    icd10_ranges = list(c("N032", "N037"), c("N052", "N057"), c("Z490", "Z492"))
  ),
  mild_liver_disease = list(
    icd9_prefix = c("07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709", "570", "571", "5733", "5734", "5738", "5739", "V427"),
    icd10_prefix = c("B18", "K709", "K717", "K73", "K74", "K760", "K768", "K769", "Z944"),
    icd10_ranges = list(c("K700", "K703"), c("K713", "K715"), c("K762", "K764"))
  ),
  liver_disease = list(
    icd9_ranges = list(c("4560", "4562"), c("5722", "5728")),
    icd10_prefix = c("I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767")
  ),
  ulcers = list(
    icd9_ranges = list(c("531", "534")),
    icd10_ranges = list(c("K25", "K28"))
  ),
  rheum_disease = list(
    icd9_prefix = c("4465", "7148", "725"),
    icd9_ranges = list(c("7100", "7104"), c("7140", "7142")),
    icd10_prefix = c("M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360")
  ),
  aids = list(
    icd9_ranges = list(c("042", "044")),
    icd10_prefix = c("B20", "B21", "B22", "B24")
  )
)

normalize_icd <- function(x) {
  str_to_upper(str_replace_all(as.character(x), "[^A-Za-z0-9]", ""))
}

prefix_match <- function(codes, prefixes) {
  if (length(prefixes) == 0 || length(codes) == 0) {
    return(rep(FALSE, length(codes)))
  }
  Reduce(`|`, lapply(prefixes, function(p) startsWith(codes, p)))
}

range_match <- function(codes, ranges) {
  if (length(ranges) == 0 || length(codes) == 0) {
    return(rep(FALSE, length(codes)))
  }
  out <- rep(FALSE, length(codes))
  for (r in ranges) {
    n <- nchar(r[1])
    code_slice <- substr(codes, 1, n)
    out <- out | (code_slice >= r[1] & code_slice <= r[2])
  }
  out
}

match_rule <- function(codes, icd_version, rule) {
  codes <- normalize_icd(codes)
  out <- rep(FALSE, length(codes))
  is_icd9 <- icd_version == "icd9"

  if (any(is_icd9)) {
    out[is_icd9] <- prefix_match(codes[is_icd9], rule$icd9_prefix %||% character()) |
      range_match(codes[is_icd9], rule$icd9_ranges %||% list())
  }

  if (any(!is_icd9)) {
    out[!is_icd9] <- prefix_match(codes[!is_icd9], rule$icd10_prefix %||% character()) |
      range_match(codes[!is_icd9], rule$icd10_ranges %||% list())
  }

  out
}

to_dx_long <- function(dx_df) {
  if (all(c("PATIENT_ID", "CLM_THRU_DT", "DX") %in% colnames(dx_df))) {
    return(dx_df %>% mutate(DX = normalize_icd(DX)) %>% filter(!is.na(DX) & DX != ""))
  }

  dx_cols <- grep("^(ICD_DGNS|PRNCPAL_DGNS|FST_DGNS)", colnames(dx_df), value = TRUE)
  if (length(dx_cols) == 0) {
    stop("Input claims must include either DX or ICD_DGNS_*/PRNCPAL_DGNS/FST_DGNS columns.")
  }

  dx_df %>%
    pivot_longer(cols = all_of(dx_cols), names_to = "dx_col", values_to = "DX", values_drop_na = TRUE) %>%
    select(-dx_col) %>%
    mutate(DX = normalize_icd(DX)) %>%
    filter(!is.na(DX) & DX != "")
}

calculate_charlson_deyo <- function(dx_claims, seer_df, window_before_days = 395L, window_after_days = 30L,
                                    ruleout = FALSE, ruleout_window_days = 30L) {
  required_claim_cols <- c("PATIENT_ID", "CLM_THRU_DT", "source")
  required_seer_cols <- c("PATIENT_ID", "dx.date.end")
  if (!all(required_claim_cols %in% colnames(dx_claims))) {
    stop("dx_claims must include PATIENT_ID, CLM_THRU_DT, and source.")
  }
  if (!all(required_seer_cols %in% colnames(seer_df))) {
    stop("seer_df must include PATIENT_ID and dx.date.end.")
  }

  dx_claims <- dx_claims %>% mutate(CLM_THRU_DT = as.Date(CLM_THRU_DT))
  if ("CLM_FROM_DT" %in% colnames(dx_claims)) {
    dx_claims <- dx_claims %>% mutate(CLM_FROM_DT = as.Date(CLM_FROM_DT))
  }
  seer_df <- seer_df %>% mutate(dx.date.end = as.Date(dx.date.end))

  dx_long <- to_dx_long(dx_claims) %>%
    mutate(claim_end_date = as.Date(CLM_THRU_DT))

  if ("CLM_FROM_DT" %in% colnames(dx_long)) {
    dx_long <- dx_long %>% mutate(claim_start_date = coalesce(as.Date(CLM_FROM_DT), claim_end_date))
  } else {
    dx_long <- dx_long %>% mutate(claim_start_date = claim_end_date)
  }

  extra_days <- if (isTRUE(ruleout)) as.integer(ruleout_window_days) else 0L

  dx_long <- dx_long %>%
    inner_join(seer_df %>% select(PATIENT_ID, dx.date.end), by = "PATIENT_ID") %>%
    mutate(
      start_date = dx.date.end - as.integer(window_before_days),
      end_date = dx.date.end + as.integer(window_after_days),
      scan_start = start_date - extra_days,
      scan_end = end_date + extra_days,
      in_window = claim_start_date >= start_date & claim_start_date <= end_date
    ) %>%
    filter(claim_start_date >= scan_start & claim_start_date <= scan_end) %>%
    mutate(
      icd9or10 = if_else(claim_end_date >= ICD10_CUTOFF, "icd10", "icd9"),
      claim_month = format(claim_start_date, "%Y-%m")
    )

  patient_base <- seer_df %>% distinct(PATIENT_ID)
  charlson_codes <- patient_base

  for (name in names(CHARLSON_DEYO_RULES)) {
    rule <- CHARLSON_DEYO_RULES[[name]]
    flagged <- dx_long %>%
      mutate(flag = match_rule(DX, icd9or10, rule)) %>%
      filter(flag)

    summary <- flagged %>%
      group_by(PATIENT_ID) %>%
      summarise(
        pre_count = sum(in_window),
        pre_month_count = if (any(in_window)) n_distinct(claim_month[in_window]) else 0L,
        medpar = any(source == "medpar"),
        otherfile = any(source %in% c("carrierbase", "outpat")),
        first_date = min(claim_start_date, na.rm = TRUE),
        last_date = max(claim_start_date, na.rm = TRUE),
        first_in_window = if (any(in_window)) min(claim_start_date[in_window], na.rm = TRUE) else as.Date(NA),
        .groups = "drop"
      ) %>%
      mutate(
        ruleout_excluded = if (isTRUE(ruleout)) {
          !is.na(first_in_window) & !medpar & (as.numeric(last_date - first_date) <= as.integer(ruleout_window_days))
        } else {
          FALSE
        },
        binary = if_else(!is.na(first_in_window) & !ruleout_excluded, 1L, 0L),
        pre_count = as.integer(pre_count),
        pre_month_count = as.integer(pre_month_count)
      ) %>%
      transmute(
        PATIENT_ID,
        !!paste0(name, "_pre_count") := pre_count,
        !!paste0(name, "_pre_month_count") := pre_month_count,
        !!paste0(name, "_medpar") := medpar,
        !!paste0(name, "_otherfile") := otherfile,
        !!paste0(name, "_binary") := binary
      )

    charlson_codes <- charlson_codes %>% left_join(summary, by = "PATIENT_ID")
  }

  charlson_codes <- charlson_codes %>% mutate(across(contains("_count"), ~ replace_na(., 0L)))

  x_list <- names(CHARLSON_DEYO_RULES)
  for (x in x_list) {
    pre_col <- sym(paste0(x, "_pre_count"))
    medpar_col <- sym(paste0(x, "_medpar"))
    otherfile_col <- sym(paste0(x, "_otherfile"))
    status_col <- paste0(x, "_status")
    binary_col <- paste0(x, "_binary")

    charlson_codes <- charlson_codes %>%
      mutate(
        !!medpar_col := coalesce(!!medpar_col, FALSE),
        !!otherfile_col := coalesce(!!otherfile_col, FALSE),
        !!binary_col := as.integer(coalesce(.data[[binary_col]], 0L)),
        !!status_col := case_when(
          .data[[binary_col]] == 1L ~ "Valid",
          !!pre_col > 0 ~ "Exist but Invalid",
          TRUE ~ "DNE"
        )
      )
  }

  charlson_codes %>%
    mutate(
      Charlson =
        1 * ((acute_mi_binary == 1) | (history_mi_binary == 1)) +
        1 * (chf_binary == 1) +
        1 * (pvd_binary == 1) +
        1 * (cvd_binary == 1) +
        1 * (copd_binary == 1) +
        1 * (dementia_binary == 1) +
        2 * (paralysis_binary == 1) +
        1 * ((diabetes_binary == 1) & (diabetes_comp_binary == 0)) +
        2 * (diabetes_comp_binary == 1) +
        2 * (renal_disease_binary == 1) +
        1 * ((mild_liver_disease_binary == 1) & (liver_disease_binary == 0)) +
        3 * (liver_disease_binary == 1) +
        1 * (ulcers_binary == 1) +
        1 * (rheum_disease_binary == 1) +
        6 * (aids_binary == 1),
      charlson.bin = case_when(
        Charlson == 0 ~ 0L,
        Charlson == 1 ~ 1L,
        Charlson == 2 ~ 2L,
        Charlson >= 3 ~ 3L,
        TRUE ~ NA_integer_
      ),
      NCI_index =
        0.12624 * (acute_mi_binary == 1) +
        0.07999 * (history_mi_binary == 1) +
        0.64441 * (chf_binary == 1) +
        0.26232 * (pvd_binary == 1) +
        0.27868 * (cvd_binary == 1) +
        0.52487 * (copd_binary == 1) +
        0.72219 * (dementia_binary == 1) +
        0.39882 * (paralysis_binary == 1) +
        0.29408 * ((diabetes_binary == 1) | (diabetes_comp_binary == 1)) +
        0.47010 * (renal_disease_binary == 1) +
        0.73803 * ((mild_liver_disease_binary == 1) | (liver_disease_binary == 1)) +
        0.07506 * (ulcers_binary == 1) +
        0.21905 * (rheum_disease_binary == 1) +
        0.58362 * (aids_binary == 1)
    )
}

# Example:
# scores <- calculate_charlson_deyo(dx_claims = dx.total, seer_df = seer.total)
# saveRDS(scores, "charlson.codes.clean.rds")
