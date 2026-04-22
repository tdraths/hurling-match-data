# =============================================================================
# HURLING MATCH DATA — CLEANING SCRIPT
# clean_match_data.py
#
# PURPOSE:
#   Cleans raw CSV exports from the match tagger and produces processed files
#   ready for analysis. Handles missing values intentionally rather than
#   dropping rows or letting NaNs silently affect results.
#
# USAGE:
#   Run this script in Colab after uploading your raw CSV exports.
#   Edit the MATCH CONFIG block at the top for each new match.
#   Processed files are saved to data/processed/.
#
# WHAT THIS SCRIPT TEACHES:
#   - The difference between "missing" and "not applicable"
#   - How to use fillna(), map(), and apply() safely
#   - How to flag data completeness without dropping rows
#   - How to derive new columns from existing ones
# =============================================================================

import pandas as pd
import numpy as np


# ── MATCH CONFIG ──────────────────────────────────────────────────────────────
# Edit this block for each match. Everything below it is reusable.

MATCH_LABEL = '2025_AISF_KIK_TIP'   # used in file names
TEAM_A      = 'Kilkenny'
TEAM_B      = 'Tipperary'

# Raw file paths — update to wherever your exports are
PO_RAW_PATH = f'data/raw/puckouts/puckouts_{MATCH_LABEL}.csv'
FR_RAW_PATH = f'data/raw/frees/frees_{MATCH_LABEL}.csv'

# Processed output paths
PO_OUT_PATH = f'data/processed/puckouts/{MATCH_LABEL}.csv'
FR_OUT_PATH = f'data/processed/frees/{MATCH_LABEL}.csv'

# Goalkeeper dictionary — add an entry for every match/team combination
# Format: (match_label, team_name) -> goalkeeper_name
# Build this up as you tag more matches
GOALKEEPER_MAP = {
    ('2025_AIF_COR_TIP',  'Cork'):      'Patrick Collins',
    ('2025_AIF_COR_TIP',  'Tipperary'): 'Rhys Shelly',
    ('2025_AISF_KIK_TIP', 'Kilkenny'):  'Eoin Murphy',
    ('2025_AISF_KIK_TIP', 'Tipperary'): 'Rhys Shelly',
}

# Free-taker dictionary — add designated takers as you identify them
# Format: (match_label, team_name) -> player_name
# Leave out teams where frees are shared or unknown
FREE_TAKER_MAP = {
    ('2025_AISF_KIK_TIP', 'Kilkenny'): 'TJ Reid',
}

# ── END CONFIG ────────────────────────────────────────────────────────────────


# =============================================================================
# LOOKUP TABLES
# These are used to derive new columns from existing ones.
# Defined once here so they're easy to update if the schema changes.
# =============================================================================

# Maps target_zone to puckout type
# DL/DC/DR = short (defensive zones)
# ML/MC/MR = medium (midfield zones)
# AL/AC/AR = long (attacking zones)
ZONE_TO_TYPE = {
    'DL': 'short',  'DC': 'short',  'DR': 'short',
    'ML': 'medium', 'MC': 'medium', 'MR': 'medium',
    'AL': 'long',   'AC': 'long',   'AR': 'long',
}

# Maps poss_outcome to a numeric net score value
# Used as the target variable for the Expected Points model
# Positive = pucking team scored, Negative = opposition scored
POSS_OUTCOME_VALUE = {
    'point':     1,
    'goal':      3,
    'no-score':  0,
    'opp-point': -1,
    'opp-goal':  -3,
}

# Maps score_diff to a game state label
# Used to analyse how puckout strategy changes with scoreline
def get_game_state(diff):
    """
    Converts a numeric score differential into a human-readable game state.

    diff is from the pucking team's perspective:
      positive = pucking team winning
      negative = pucking team losing
      zero     = level

    Using 4+ as a threshold because in hurling a 4-point lead
    is generally considered comfortable — a goal wins it in one.
    """
    if diff <= -4:   return 'Losing 4+'
    elif diff <= -1: return 'Losing 1-3'
    elif diff == 0:  return 'Level'
    elif diff <= 3:  return 'Winning 1-3'
    else:            return 'Winning 4+'


# =============================================================================
# PUCKOUT CLEANING FUNCTION
# =============================================================================

def clean_puckouts(path, match_label, team_a, team_b):
    """
    Loads and cleans a raw puckout CSV export from the tagger.

    Steps:
      1. Load and standardise team names
      2. Fix data types (minute, half)
      3. Standardise missing values — the key step
      4. Derive new columns (type, goalkeeper, game_state etc.)
      5. Handle the touch chain — distinguish 'chain ended here'
         from 'chain not logged at all'
      6. Sort and re-sequence IDs
      7. Return cleaned DataFrame

    Parameters
    ----------
    path        : str  — path to the raw CSV
    match_label : str  — match identifier, e.g. '2025_AISF_KIK_TIP'
    team_a      : str  — Team A name as entered in the tagger
    team_b      : str  — Team B name as entered in the tagger

    Returns
    -------
    pandas.DataFrame — cleaned puckout data
    """

    print(f"\n{'='*55}")
    print(f"  Cleaning puckouts: {match_label}")
    print(f"{'='*55}")

    # ── STEP 1: Load ──────────────────────────────────────────
    # low_memory=False prevents pandas from guessing column types
    # on the first pass, which can cause mixed-type warnings
    po = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(po)} rows from {path}")

    # ── STEP 2: Standardise team names ───────────────────────
    # .str.strip() removes any accidental leading/trailing spaces
    # .str.title() converts to Title Case (Cork, Tipperary, etc.)
    # This ensures joins and groupbys work correctly later
    for col in ['pucking_team', 'team_a', 'team_b']:
        po[col] = po[col].str.strip().str.title()

    # ── STEP 3: Fix data types ────────────────────────────────
    # pd.to_numeric with errors='coerce' converts anything that
    # can't be a number to NaN instead of raising an error.
    # This catches typos like '6263' or '?' in the minute field.
    po['minute'] = pd.to_numeric(po['minute'], errors='coerce')
    po['half']   = pd.to_numeric(po['half'],   errors='coerce')

    # Flag and report any bad minutes (> 100 or missing)
    bad_minutes = po[po['minute'].isna() | (po['minute'] > 100)]
    if len(bad_minutes) > 0:
        print(f"\n  ⚠ {len(bad_minutes)} rows with suspect minute values:")
        print(bad_minutes[['id','minute','pucking_team','target_zone']].to_string())
        print("  → Fix these manually before proceeding.\n")
    else:
        print("  ✓ All minute values look valid")

    # ── STEP 4: Standardise missing values ───────────────────
    #
    # This is the most important step. Pandas uses NaN (Not a Number)
    # to represent missing values, but NaN has tricky behaviour:
    #
    #   - NaN != NaN  (NaN is not equal to itself)
    #   - NaN propagates through calculations (1 + NaN = NaN)
    #   - groupby silently skips NaN values
    #   - value_counts() excludes NaN by default
    #
    # Strategy: replace NaN with explicit string values that have
    # clear meanings, so there's no ambiguity in analysis.
    #
    # We use three distinct fill values:
    #   'unknown' → field was not filled in during tagging (data gap)
    #   'n/a'     → field is not applicable to this row (structural)
    #   ''        → intentionally empty (e.g. possession outcome not
    #               yet logged on second pass)
    #
    # We handle each column group differently based on its semantics.

    # Categorical fields that should have been filled during tagging
    # If they're NaN, it's a data gap — fill with 'unknown'
    categorical_cols = ['type', 'delivery', 'target_zone', 'retained',
                        'next_action']
    for col in categorical_cols:
        if col in po.columns:
            # fillna replaces NaN with the given value
            po[col] = po[col].fillna('unknown')
            # Also catch empty strings that slipped through
            po[col] = po[col].replace('', 'unknown')

    # Notes field — NaN here just means no note was written
    # Replace with empty string so string operations work safely
    po['notes'] = po['notes'].fillna('')

    # Possession outcome — intentionally left blank until second pass
    # Keep as empty string, not 'unknown', to distinguish from
    # rows where tagging was attempted but left blank
    if 'poss_outcome' in po.columns:
        po['poss_outcome'] = po['poss_outcome'].fillna('')
    else:
        # Column may not exist in older exports — add it
        po['poss_outcome'] = ''

    # ── STEP 5: Handle the touch chain ───────────────────────
    #
    # The touch chain requires careful handling because empty cells
    # can mean two very different things:
    #
    #   Case A: Chain was logged, but ended early because a shot
    #           occurred at Touch 1 or 2. Touches after the shot
    #           are empty because there were no more touches.
    #           → Fill post-shot cells with 'n/a'
    #
    #   Case B: Chain was never logged at all for this puckout.
    #           All touch cells are empty.
    #           → Leave as empty, flag with has_chain = False
    #
    # Getting this right means analyses of the chain will only
    # include rows where data was actually collected.

    chain_act_cols  = ['t1_action', 't2_action', 't3_action']
    chain_team_cols = ['t1_team',   't2_team',   't3_team'  ]
    chain_all_cols  = chain_act_cols + chain_team_cols + ['shot_outcome']

    # Ensure all chain columns exist (older exports may not have them)
    for col in chain_all_cols:
        if col not in po.columns:
            po[col] = ''

    # Replace NaN with empty string across all chain columns
    # We'll assign meaningful values in the next steps
    po[chain_all_cols] = po[chain_all_cols].fillna('')

    # has_chain: True if Touch 1 action was logged
    # This distinguishes rows where chain data exists from rows
    # where no chain data was collected at all
    po['has_chain'] = (po['t1_action'] != '') & (po['t1_action'] != 'unknown')

    # For rows WITH chain data: fill cells after a shot with 'n/a'
    # This makes it explicit that those cells are empty because
    # the chain ended, not because data is missing
    def fill_post_shot(row):
        """
        If a shot occurred at touch N, marks touches N+1 onwards
        as 'n/a' to distinguish structural absence from missing data.

        Works through touches 1, 2, 3 in order. As soon as it finds
        a shot, it fills everything after it and stops.
        """
        # Only process rows that have chain data
        if not row['has_chain']:
            return row

        shot_at = None
        for n in [1, 2, 3]:
            if row[f't{n}_action'] == 'shot':
                shot_at = n
                break

        if shot_at is not None:
            # Fill all touch columns after the shot with 'n/a'
            for n in range(shot_at + 1, 4):
                row[f't{n}_action'] = 'n/a'
                row[f't{n}_team']   = 'n/a'

        return row

    # apply() runs fill_post_shot on every row, axis=1 means row-wise
    # This is slower than vectorised operations but necessary here
    # because the logic depends on values across multiple columns
    po = po.apply(fill_post_shot, axis=1)

    # chain_length: how many touches were logged (0, 1, 2, or 3)
    # Useful for filtering analyses to rows with sufficient chain data
    def get_chain_length(row):
        if not row['has_chain']:
            return 0
        # Count how many touch actions are filled with real values
        # (i.e. not empty string and not 'n/a')
        count = 0
        for n in [1, 2, 3]:
            val = row[f't{n}_action']
            if val != '' and val != 'n/a':
                count = n
        return count

    po['chain_length'] = po.apply(get_chain_length, axis=1)

    print(f"  ✓ Touch chain: {po['has_chain'].sum()} of {len(po)} rows have chain data")
    chain_dist = po[po['has_chain']]['chain_length'].value_counts().sort_index()
    for length, count in chain_dist.items():
        print(f"      chain_length={length}: {count} rows")

    # ── STEP 6: Derive new columns ────────────────────────────

    # type: derived from target_zone using the lookup table
    # .map() replaces each value using the dictionary
    # Values not in the dictionary become NaN — then we fill with 'unknown'
    po['type'] = po['target_zone'].map(ZONE_TO_TYPE).fillna('unknown')

    # Check for zone/type mismatches in older data where type was entered manually
    if 'type' in po.columns:
        mismatches = po[
            (po['target_zone'] != 'unknown') &
            (po['type'] != po['target_zone'].map(ZONE_TO_TYPE).fillna('unknown'))
        ]
        if len(mismatches) > 0:
            print(f"\n  ⚠ {len(mismatches)} type/zone mismatches found (type will be overridden):")
            print(mismatches[['id','minute','target_zone','type']].to_string())

    # match_label: stored on every row for joining with other matches
    po['match_label'] = match_label

    # opposition: the team that is not the pucking team
    # np.where is a vectorised if/else:
    #   where pucking_team == team_a, use team_b, else use team_a
    po['opposition'] = np.where(
        po['pucking_team'] == team_a, team_b, team_a
    )

    # goalkeeper: looked up from dictionary using (match, team) key
    # .get() returns the second argument as a default if key not found
    po['goalkeeper'] = po.apply(
        lambda row: GOALKEEPER_MAP.get(
            (match_label, row['pucking_team']), 'unknown'
        ),
        axis=1
    )

    # retained_bool: True/False version of retained for calculations
    # .map() with a dict replaces string values with booleans
    # NaN (from 'unknown') stays as NaN — pandas handles this correctly
    # in mean() calculations (ignores NaN by default)
    po['retained_bool'] = po['retained'].map({'yes': True, 'no': False})

    # game_state: binned score differential
    po['game_state'] = po['score_diff'].apply(get_game_state)

    # net_score_value: numeric value for Expected Points model
    # Empty string → NaN (not yet logged), which is correct —
    # we don't want to assume 0 for unlogged rows
    po['net_score_value'] = po['poss_outcome'].map(POSS_OUTCOME_VALUE)

    # outcome_logged: flag for whether poss_outcome has been filled
    # Use this to filter to model-ready rows: po[po['outcome_logged']]
    po['outcome_logged'] = po['poss_outcome'] != ''

    print(f"  ✓ Derived columns added")
    print(f"  ✓ Possession outcome logged: "
          f"{po['outcome_logged'].sum()} of {len(po)} rows")

    # ── STEP 7: Sort and re-sequence IDs ─────────────────────
    po = po.sort_values(['half', 'minute']).reset_index(drop=True)
    # reset_index(drop=True) resets the row numbers after sorting
    # drop=True means don't keep the old index as a column
    po['id'] = range(1, len(po) + 1)

    # ── STEP 8: Final validation report ──────────────────────
    print(f"\n  CLEANED SUMMARY")
    print(f"  {'─'*45}")
    print(f"  Rows:               {len(po)}")
    print(f"  Teams:              {po['pucking_team'].unique()}")
    print(f"  Minute range:       {po['minute'].min():.0f} – {po['minute'].max():.0f}")
    print(f"  Unknown zones:      {(po['target_zone']=='unknown').sum()}")
    print(f"  Unknown delivery:   {(po['delivery']=='unknown').sum()}")
    print(f"  Unknown retained:   {(po['retained']=='unknown').sum()}")
    print(f"  Retention rate:     {po['retained_bool'].mean()*100:.1f}%")
    print(f"  {'─'*45}")

    return po


# =============================================================================
# FREES CLEANING FUNCTION
# =============================================================================

def clean_frees(path, match_label, team_a, team_b):
    """
    Loads and cleans a raw frees/65s CSV export from the tagger.

    Simpler than puckouts — no touch chain to handle.
    The main tasks are:
      1. Standardise team names and data types
      2. Handle missing values
      3. Assign designated free-takers from dictionary
      4. Derive scored and possession flags
      5. Sort and re-sequence

    Parameters
    ----------
    path        : str  — path to the raw CSV
    match_label : str  — match identifier
    team_a      : str  — Team A name
    team_b      : str  — Team B name

    Returns
    -------
    pandas.DataFrame — cleaned frees data
    """

    print(f"\n{'='*55}")
    print(f"  Cleaning frees: {match_label}")
    print(f"{'='*55}")

    fr = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(fr)} rows from {path}")

    # ── Standardise team names ────────────────────────────────
    for col in ['shooting_team', 'team_a', 'team_b']:
        fr[col] = fr[col].str.strip().str.title()

    # ── Fix data types ────────────────────────────────────────
    fr['minute'] = pd.to_numeric(fr['minute'], errors='coerce')
    fr['half']   = pd.to_numeric(fr['half'],   errors='coerce')

    bad_minutes = fr[fr['minute'].isna() | (fr['minute'] > 100)]
    if len(bad_minutes) > 0:
        print(f"\n  ⚠ {len(bad_minutes)} rows with suspect minute values:")
        print(bad_minutes[['id','minute','shooting_team','outcome']].to_string())
    else:
        print("  ✓ All minute values look valid")

    # ── Standardise missing values ────────────────────────────
    # Same logic as puckouts: NaN → 'unknown' for fields that
    # should have been filled during tagging
    categorical_cols = ['set_piece_type', 'position_zone', 'attempt_type',
                        'striking_side', 'under_pressure', 'outcome']
    for col in categorical_cols:
        if col in fr.columns:
            fr[col] = fr[col].fillna('unknown').replace('', 'unknown')

    # Shooter field: NaN means unknown, not that no one took it
    # Keep as empty string — can be populated via FREE_TAKER_MAP
    fr['shooter'] = fr['shooter'].fillna('').astype(str)

    fr['notes'] = fr['notes'].fillna('')

    # ── Assign designated free-takers ────────────────────────
    # For teams with a known designated taker, assign them
    # to all frees for that team in this match.
    # This can be overridden per-row if needed.
    for (ml, team), player in FREE_TAKER_MAP.items():
        if ml == match_label:
            mask = fr['shooting_team'] == team
            fr.loc[mask, 'shooter'] = player
            print(f"  ✓ Assigned {player} to {mask.sum()} {team} frees")

    # ── Add match context ─────────────────────────────────────
    fr['match_label'] = match_label
    fr['opposition']  = np.where(
        fr['shooting_team'] == team_a, team_b, team_a
    )
    fr['game_state'] = fr['score_diff'].apply(get_game_state)

    # ── Derive scored and possession flags ────────────────────
    # scored: True if the attempt resulted in a score (point or goal)
    # Used for conversion rate calculations
    fr['scored'] = fr['outcome'].isin(['point', 'goal'])

    # possession: True if the shooting team kept the ball
    # Includes scores (retained via set piece) and explicit retentions
    fr['possession'] = fr['outcome'].isin(['point', 'goal', 'retained'])

    # score_value: numeric value of the score (0, 1, or 3)
    fr['score_value'] = fr['outcome'].map(
        {'point': 1, 'goal': 3}
    ).fillna(0)
    # fillna(0) here is correct — non-scoring outcomes are genuinely
    # worth 0 points, not missing data

    # ── Sort and re-sequence ──────────────────────────────────
    fr = fr.sort_values(['half', 'minute']).reset_index(drop=True)
    fr['id'] = range(1, len(fr) + 1)

    # ── Final validation ──────────────────────────────────────
    print(f"\n  CLEANED SUMMARY")
    print(f"  {'─'*45}")
    print(f"  Rows:               {len(fr)}")
    print(f"  Teams:              {fr['shooting_team'].unique()}")
    print(f"  Types:              {fr['set_piece_type'].value_counts().to_dict()}")
    print(f"  Overall conversion: {fr['scored'].mean()*100:.1f}%")
    for team in fr['shooting_team'].unique():
        sub = fr[fr['shooting_team']==team]
        print(f"  {team}: {sub['scored'].mean()*100:.1f}% "
              f"({sub['scored'].sum()}/{len(sub)})")
    print(f"  {'─'*45}")

    return fr


# =============================================================================
# RUN CLEANING
# =============================================================================

if __name__ == '__main__':

    # Clean both files
    po_clean = clean_puckouts(PO_RAW_PATH, MATCH_LABEL, TEAM_A, TEAM_B)
    fr_clean = clean_frees(FR_RAW_PATH, MATCH_LABEL, TEAM_A, TEAM_B)

    # Save processed files
    po_clean.to_csv(PO_OUT_PATH, index=False)
    fr_clean.to_csv(FR_OUT_PATH, index=False)
    print(f"\n  ✓ Puckouts saved to {PO_OUT_PATH}")
    print(f"  ✓ Frees saved to    {FR_OUT_PATH}")

    # ── COMBINING MULTIPLE MATCHES ────────────────────────────
    #
    # When you're ready to build a combined dataset across all matches,
    # run this section instead of (or after) the single-match cleaning above.
    #
    # Uncomment and edit the list of matches:

    # import glob
    #
    # all_po, all_fr = [], []
    #
    # MATCHES = [
    #     ('2025_AIF_COR_TIP',  'Cork',      'Tipperary'),
    #     ('2025_AISF_KIK_TIP', 'Kilkenny',  'Tipperary'),
    #     # add more matches here as you tag them
    # ]
    #
    # for match_label, team_a, team_b in MATCHES:
    #     po_path = f'data/raw/puckouts/puckouts_{match_label}.csv'
    #     fr_path = f'data/raw/frees/frees_{match_label}.csv'
    #
    #     po_clean = clean_puckouts(po_path, match_label, team_a, team_b)
    #     fr_clean = clean_frees(fr_path, match_label, team_a, team_b)
    #
    #     all_po.append(po_clean)
    #     all_fr.append(fr_clean)
    #
    # # pd.concat stacks DataFrames vertically
    # # ignore_index=True resets the row numbers across the combined file
    # po_combined = pd.concat(all_po, ignore_index=True)
    # fr_combined = pd.concat(all_fr, ignore_index=True)
    #
    # # Re-sequence IDs across the combined dataset
    # po_combined['id'] = range(1, len(po_combined) + 1)
    # fr_combined['id'] = range(1, len(fr_combined) + 1)
    #
    # po_combined.to_csv('data/combined/puckouts_combined.csv', index=False)
    # fr_combined.to_csv('data/combined/frees_combined.csv', index=False)
    # print(f"\n  ✓ Combined: {len(po_combined)} puckouts, {len(fr_combined)} frees")
