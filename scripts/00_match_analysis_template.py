# ================================================================
# HURLING MATCH ANALYSIS — REUSABLE TEMPLATE
# 00_match_analysis_template.py
#
# HOW TO USE:
#   1. Copy this file and rename it for your match
#      e.g. 03_eda_2025_QF_LIM_WAT.py
#   2. Edit CELL 1 (Config) only — team names, colours, file paths
#   3. Run all cells
#
# Libraries: Plotly (interactive) · Seaborn (statistical)
#            Matplotlib (pitch maps only) · Light theme throughout
# ================================================================


# ── CELL 1: CONFIG — edit this block for each match ─────────────
# ════════════════════════════════════════════════════════════════

MATCH_LABEL  = '2025_AISF_KIK_TIP'         # used in file names
MATCH_TITLE  = '2025 All-Ireland Semi-Final' # used in chart titles
TEAM_A       = 'Kilkenny'
TEAM_B       = 'Tipperary'
RESULT       = 'Kilkenny 0-30  Tipperary 4-20  (Tipp win by 2pts)'

# Team colours — use hex codes
COLOUR_A     = '#1a1a2e'   # Kilkenny black
COLOUR_B     = '#003DA5'   # Tipperary blue

# File paths — update to GitHub raw URLs once pushed, or keep as local paths
# Example GitHub:
# BASE = 'https://raw.githubusercontent.com/YOUR_USER/hurling-match-data/main/data/processed/'
# PO_PATH = BASE + f'puckouts/{MATCH_LABEL}.csv'
# FR_PATH = BASE + f'frees/{MATCH_LABEL}.csv'
PO_PATH = f'data/processed/puckouts/{MATCH_LABEL}.csv'
FR_PATH = f'data/processed/frees/{MATCH_LABEL}.csv'

# Free-takers — set to None if no designated taker, or a dict for multiple
# Format: { 'team_name': 'player_name' }
# Set to {} to skip free-taker assignment entirely
FREE_TAKERS  = {
    'Kilkenny': 'TJ Reid',
    # 'Tipperary': 'Jason Forde',  # uncomment if known
}

# Analytical context note — appears in final summary cell
CONTEXT_NOTE = (
    "Match outcome driven by 4 Tipperary goals from defensive breakdowns. "
    "Kilkenny outperformed Tipperary on all puckout and set-piece process metrics. "
    "Scoreline not fully reflective of possession dominance."
)

# ════════════════════════════════════════════════════════════════
# ── Everything below this line is reusable — do not edit ────────
# ════════════════════════════════════════════════════════════════


# ── CELL 2: Setup & Load ────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Shared palette (non-team colours) ───────────────────────────
C_GREEN  = '#2d8a4e'
C_RED    = '#e05050'
C_GOLD   = '#f5c842'
C_GREY   = '#888888'
C_GRASS  = '#e8f4e8'
C_LIGHT  = '#f7f7f7'
TEAMS    = [TEAM_A, TEAM_B]
COLOURS  = [COLOUR_A, COLOUR_B]

# ── Seaborn light theme ──────────────────────────────────────────
sns.set_theme(style='whitegrid', font='sans-serif')

# ── Plotly base layout ───────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template='plotly_white',
    font=dict(family='Arial', size=12, color='#333333'),
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(t=60, b=40, l=50, r=30),
)

# ── Load data ────────────────────────────────────────────────────
po = pd.read_csv(PO_PATH)
fr = pd.read_csv(FR_PATH)

# ── Derived fields ───────────────────────────────────────────────
ZONE_TO_TYPE = {
    'DL':'short','DC':'short','DR':'short',
    'ML':'medium','MC':'medium','MR':'medium',
    'AL':'long','AC':'long','AR':'long'
}
GS_ORDER = ['Losing 4+','Losing 1-3','Level','Winning 1-3','Winning 4+']

def gs_bin(d):
    if d <= -4:   return 'Losing 4+'
    elif d <= -1: return 'Losing 1-3'
    elif d == 0:  return 'Level'
    elif d <= 3:  return 'Winning 1-3'
    else:         return 'Winning 4+'

po['type']          = po['target_zone'].map(ZONE_TO_TYPE)
po['retained_bool'] = po['retained'] == 'yes'
po['minute']        = pd.to_numeric(po['minute'], errors='coerce')
po['game_state']    = po['score_diff'].apply(gs_bin)

fr['scored']  = fr['outcome'].isin(['point','goal'])
fr['minute']  = pd.to_numeric(fr['minute'], errors='coerce')
fr['shooter'] = fr['shooter'].astype(str)

# Assign designated free-takers
for team, player in FREE_TAKERS.items():
    fr.loc[fr['shooting_team']==team, 'shooter'] = player

# ── Convenience subsets ──────────────────────────────────────────
sub_po = {t: po[po['pucking_team']==t] for t in TEAMS}
sub_fr = {t: fr[fr['shooting_team']==t] for t in TEAMS}

print(f"Match:  {MATCH_TITLE}  |  {MATCH_LABEL}")
print(f"Result: {RESULT}")
print(f"\nPuckouts loaded: {len(po)}  |  Frees loaded: {len(fr)}")
print(f"\n{TEAM_A}: {len(sub_po[TEAM_A])} puckouts  "
      f"ret: {sub_po[TEAM_A]['retained_bool'].mean()*100:.0f}%")
print(f"{TEAM_B}: {len(sub_po[TEAM_B])} puckouts  "
      f"ret: {sub_po[TEAM_B]['retained_bool'].mean()*100:.0f}%")


# ── CELL 3: Match Summary Scorecard ─────────────────────────────

def safe_pct(series, condition, denom=None):
    s = series[condition] if denom is None else series
    d = len(s) if denom is None else denom
    return round(s.mean()*100) if d > 0 else 0

metrics = [
    'Puckouts', 'Retention %', 'Short %', 'Medium %', 'Long %',
    'Short Retention %', 'Long Retention %',
    'Broken Deliveries %', 'Free Attempts', 'Free Conversion %'
]

def team_metrics(team):
    p  = sub_po[team]
    f  = sub_fr[team]
    n  = len(p)
    return [
        n,
        round(p['retained_bool'].mean()*100),
        round((p['type']=='short').mean()*100),
        round((p['type']=='medium').mean()*100),
        round((p['type']=='long').mean()*100),
        round(p[p['type']=='short']['retained_bool'].mean()*100) if (p['type']=='short').any() else '—',
        round(p[p['type']=='long']['retained_bool'].mean()*100)  if (p['type']=='long').any()  else '—',
        round((p['delivery']=='B').mean()*100),
        len(f),
        round(f['scored'].mean()*100) if len(f) > 0 else '—',
    ]

vals_a = team_metrics(TEAM_A)
vals_b = team_metrics(TEAM_B)

fig = go.Figure(data=[go.Table(
    columnwidth=[220, 110, 110],
    header=dict(
        values=['<b>Metric</b>', f'<b>{TEAM_A}</b>', f'<b>{TEAM_B}</b>'],
        fill_color=[C_LIGHT, COLOUR_A, COLOUR_B],
        font=dict(color=['#333333','white','white'], size=13),
        align='center', height=36,
    ),
    cells=dict(
        values=[metrics, vals_a, vals_b],
        fill_color=[
            C_LIGHT,
            ['#f0f0f0' if i%2==0 else 'white' for i in range(len(metrics))],
            ['#f0f0f0' if i%2==0 else 'white' for i in range(len(metrics))],
        ],
        font=dict(color='#333333', size=12),
        align=['left','center','center'],
        height=32,
    )
)])
fig.update_layout(
    title=f'<b>{MATCH_TITLE} — Process Metrics Summary</b>',
    title_font_size=16, height=420, **PLOTLY_LAYOUT
)
fig.show()


# ── CELL 4: Puckout Strategy — Plotly ───────────────────────────

types     = ['short','medium','long']
type_pcts = {t: [(sub_po[t]['type']==ty).mean()*100 for ty in types] for t in TEAMS}
ret_rates = {t: [sub_po[t][sub_po[t]['type']==ty]['retained_bool'].mean()*100
                 if (sub_po[t]['type']==ty).any() else 0
                 for ty in types] for t in TEAMS}

fig = make_subplots(rows=1, cols=2,
    subplot_titles=('<b>Puckout Type Distribution</b>',
                    '<b>Retention Rate by Type</b>'),
    horizontal_spacing=0.12)

for i, (team, col) in enumerate(zip(TEAMS, COLOURS)):
    fig.add_trace(go.Bar(
        name=team,
        x=[t.title() for t in types],
        y=[round(v,1) for v in type_pcts[team]],
        marker_color=col,
        text=[f'{v:.0f}%' for v in type_pcts[team]],
        textposition='outside',
        offsetgroup=i,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        name=team,
        x=[t.title() for t in types],
        y=[round(v,1) for v in ret_rates[team]],
        mode='lines+markers+text',
        line=dict(color=col, width=2.5),
        marker=dict(size=10, color=col),
        text=[f'{v:.0f}%' for v in ret_rates[team]],
        textposition='top center',
        showlegend=False,
    ), row=1, col=2)

fig.update_layout(
    barmode='group',
    yaxis=dict(title='% of puckouts', range=[0,85]),
    yaxis2=dict(title='Retention %', range=[0,120]),
    legend=dict(orientation='h', yanchor='bottom', y=1.08,
                x=0.5, xanchor='center'),
    title=f'<b>{MATCH_TITLE} — Puckout Strategy</b>',
    title_font_size=16, height=420, **PLOTLY_LAYOUT
)
fig.show()


# ── CELL 5: Delivery Code Analysis — Seaborn ───────────────────

DEL_ORDER  = ['C','P','B','T','X']
DEL_LABELS = ['Clean','Pressured','Broken','Turnover','Failed']
DEL_COLS   = [C_GREEN,'#f5a623',C_GREY,C_RED,'#c0392b']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('white')

# Stacked bar — delivery distribution
ax = axes[0]
bottom = np.zeros(len(TEAMS))
for code, label, col in zip(DEL_ORDER, DEL_LABELS, DEL_COLS):
    vals = np.array([(sub_po[t]['delivery']==code).mean()*100 for t in TEAMS])
    ax.bar(TEAMS, vals, bottom=bottom, color=col, alpha=0.88,
           edgecolor='white', linewidth=0.8, label=label)
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax.text(xi, b+v/2, f'{v:.0f}%',
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white')
    bottom += vals
ax.set_ylabel('% of puckouts', fontsize=10)
ax.set_title('Delivery Code Distribution', fontsize=12,
             fontweight='bold', pad=10)
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.set_facecolor('white'); ax.spines[['top','right']].set_visible(False)

# Dot plot — retention by delivery code
ax2 = axes[1]
for team, col, marker in zip(TEAMS, COLOURS, ['o','s']):
    xs, ys = [], []
    for code, label in zip(DEL_ORDER, DEL_LABELS):
        s = sub_po[team][sub_po[team]['delivery']==code]
        if len(s) >= 2:
            xs.append(label)
            ys.append(s['retained_bool'].mean()*100)
    ax2.scatter(xs, ys, color=col, s=100, marker=marker,
                zorder=5, label=team, alpha=0.9)
ax2.axhline(po['retained_bool'].mean()*100, color=C_GREY,
            linestyle='--', linewidth=1, alpha=0.6, label='Overall avg')
ax2.set_ylabel('Retention %', fontsize=10)
ax2.set_title('Retention Rate by Delivery Code', fontsize=12,
              fontweight='bold', pad=10)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.set_facecolor('white'); ax2.spines[['top','right']].set_visible(False)
ax2.set_ylim(-5, 110)

plt.suptitle(f'{MATCH_TITLE} — Delivery Quality',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{MATCH_LABEL}_delivery.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()


# ── CELL 6: Zone Maps — Matplotlib ──────────────────────────────

PO_ZONES = {
    'AL':(0,0),'AC':(0,1),'AR':(0,2),
    'ML':(1,0),'MC':(1,1),'MR':(1,2),
    'DL':(2,0),'DC':(2,1),'DR':(2,2),
}
rg_cmap = LinearSegmentedColormap.from_list('rg',
    ['#e05050','#f5c842','#2d8a4e'])

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.patch.set_facecolor('white')

for row_i, (team, col_t) in enumerate(zip(TEAMS, COLOURS)):
    sub   = sub_po[team]
    total = len(sub)
    zone_counts = sub['target_zone'].value_counts()
    zone_ret    = {z: sub[sub['target_zone']==z]['retained_bool'].mean()
                   if len(sub[sub['target_zone']==z]) > 0 else np.nan
                   for z in PO_ZONES}

    freq_cmap = LinearSegmentedColormap.from_list(
        'freq', ['#f0f0f0','#a8d8a8', col_t])

    for col_i, (data_dict, title_suffix, cmap) in enumerate([
        ({z: zone_counts.get(z,0)/total for z in PO_ZONES},
         'Zone Frequency', freq_cmap),
        (zone_ret, 'Retention Rate by Zone', rg_cmap)
    ]):
        ax = axes[row_i][col_i]
        ax.set_facecolor(C_GRASS)
        ax.set_xlim(0,3); ax.set_ylim(0,3); ax.set_aspect('equal')

        for zone, (row, col) in PO_ZONES.items():
            val = data_dict.get(zone, np.nan)
            colour = cmap(val) if not (isinstance(val,float) and np.isnan(val)) else '#dddddd'
            ax.add_patch(patches.Rectangle(
                (col, 2-row), 1, 1,
                facecolor=colour, alpha=0.9,
                edgecolor='white', linewidth=2))
            if col_i == 0:
                cnt = zone_counts.get(zone, 0)
                lbl = f"{zone}\n{cnt} ({val*100:.0f}%)"
            else:
                lbl = f"{zone}\n{val*100:.0f}%" if not (isinstance(val,float) and np.isnan(val)) else f"{zone}\nn/a"
            ax.text(col+0.5, 2-row+0.5, lbl,
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color='#333333')

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{team} — {title_suffix}', fontsize=11,
                     fontweight='bold', color=col_t, pad=8)
        ax.text(0.02, -0.05, '← Opp Goal', transform=ax.transAxes,
                fontsize=7, color=C_GREY)
        ax.text(0.98, -0.05, 'Own Goal →', transform=ax.transAxes,
                fontsize=7, color=C_GREY, ha='right')

plt.suptitle(f'{MATCH_TITLE} — Zone Analysis',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{MATCH_LABEL}_zones.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()


# ── CELL 7: Game State Analysis — Plotly ───────────────────────

gs_rows = []
for team in TEAMS:
    sub = sub_po[team]
    for gs in GS_ORDER:
        s = sub[sub['game_state']==gs]
        if len(s) > 0:
            gs_rows.append({
                'Team': team, 'Game State': gs, 'Count': len(s),
                'Short %':     round((s['type']=='short').mean()*100, 1),
                'Long %':      round((s['type']=='long').mean()*100, 1),
                'Retention %': round(s['retained_bool'].mean()*100, 1),
            })
df_gs = pd.DataFrame(gs_rows)

fig = make_subplots(rows=1, cols=3,
    subplot_titles=('<b>Short % by Game State</b>',
                    '<b>Long % by Game State</b>',
                    '<b>Retention % by Game State</b>'),
    horizontal_spacing=0.1)

for col_i, metric in enumerate(['Short %','Long %','Retention %'], 1):
    for team, col in zip(TEAMS, COLOURS):
        sub_df = df_gs[df_gs['Team']==team].copy()
        sub_df['_order'] = sub_df['Game State'].map(
            {g:i for i,g in enumerate(GS_ORDER)})
        sub_df = sub_df.sort_values('_order')
        fig.add_trace(go.Scatter(
            x=sub_df['Game State'], y=sub_df[metric],
            mode='lines+markers+text',
            name=team,
            line=dict(color=col, width=2.5),
            marker=dict(size=9, color=col),
            text=[f'{v}%' for v in sub_df[metric]],
            textposition='top center',
            showlegend=(col_i==1),
        ), row=1, col=col_i)

fig.update_layout(
    title=f'<b>{MATCH_TITLE} — Game State Strategy</b>',
    title_font_size=16,
    legend=dict(orientation='h', yanchor='bottom', y=1.1,
                x=0.5, xanchor='center'),
    height=430, **PLOTLY_LAYOUT
)
for i in range(1, 4):
    fig.update_yaxes(range=[0, 110], row=1, col=i)
fig.show()


# ── CELL 8: Frees Analysis — Plotly ────────────────────────────

fig = make_subplots(rows=1, cols=3,
    subplot_titles=('<b>Score Conversion %</b>',
                    '<b>Outcome Breakdown</b>',
                    '<b>Attempt Type Conversion</b>'),
    horizontal_spacing=0.12)

# Panel 1: conversion
for i, (team, col) in enumerate(zip(TEAMS, COLOURS)):
    conv = sub_fr[team]['scored'].mean()*100
    fig.add_trace(go.Bar(
        x=[team], y=[conv], marker_color=col,
        text=[f'{conv:.0f}%'], textposition='outside',
        name=team, showlegend=False,
    ), row=1, col=1)

# Panel 2: outcomes stacked
OUT_ORDER  = ['point','goal','wide','short','saved','cleared','retained','lost']
OUT_COLS   = {
    'point':C_GREEN,'goal':C_GOLD,'wide':C_RED,'short':C_RED,
    'saved':'#e08830','cleared':'#e08830','retained':'#5090e0','lost':C_GREY
}
for outcome in OUT_ORDER:
    vals = [len(sub_fr[t][sub_fr[t]['outcome']==outcome]) for t in TEAMS]
    if sum(vals) > 0:
        fig.add_trace(go.Bar(
            name=outcome.title(), x=TEAMS, y=vals,
            marker_color=OUT_COLS.get(outcome, C_GREY),
            showlegend=True,
        ), row=1, col=2)

# Panel 3: attempt type
att_types = fr['attempt_type'].value_counts().index.tolist()
for i, (team, col) in enumerate(zip(TEAMS, COLOURS)):
    sub = sub_fr[team]
    convs = [sub[sub['attempt_type']==a]['scored'].mean()*100
             if len(sub[sub['attempt_type']==a]) > 0 else 0
             for a in att_types]
    ns    = [len(sub[sub['attempt_type']==a]) for a in att_types]
    fig.add_trace(go.Bar(
        name=team,
        x=[a.title() for a in att_types], y=convs,
        marker_color=col,
        text=[f'{c:.0f}%<br>n={n}' for c,n in zip(convs,ns)],
        textposition='outside',
        offsetgroup=i, showlegend=False,
    ), row=1, col=3)

fig.update_layout(
    barmode='group',
    title=f'<b>{MATCH_TITLE} — Frees & Set Pieces</b>',
    title_font_size=16,
    legend=dict(orientation='v', x=1.02, y=0.5),
    height=430, **PLOTLY_LAYOUT
)
fig.update_yaxes(range=[0,110], row=1, col=1)
fig.update_yaxes(range=[0,110], row=1, col=3)
fig.show()


# ── CELL 9: Free-Taker Breakdown — Seaborn ─────────────────────
# Only runs if FREE_TAKERS is populated

if FREE_TAKERS:
    for team, player in FREE_TAKERS.items():
        player_fr = fr[fr['shooter']==player]
        if len(player_fr) == 0:
            print(f"No frees found for {player}")
            continue

        zone_summary = player_fr.groupby('position_zone').agg(
            attempts=('scored','count'),
            scored=('scored','sum')
        ).assign(
            conversion=lambda x: x.scored/x.attempts*100
        ).reset_index().sort_values('attempts', ascending=False)

        fig2, ax = plt.subplots(figsize=(9, 4))
        col_t = COLOUR_A if team==TEAM_A else COLOUR_B
        sns.barplot(data=zone_summary, x='position_zone',
                    y='conversion', color=col_t, alpha=0.85, ax=ax)
        ax.axhline(player_fr['scored'].mean()*100, color=C_GREY,
                   linestyle='--', linewidth=1.2,
                   label=f'Avg: {player_fr["scored"].mean()*100:.0f}%')
        ax.set_xlabel('Zone', fontsize=10)
        ax.set_ylabel('Conversion %', fontsize=10)
        ax.set_title(f'{player} ({team}) — Free Conversion by Zone',
                     fontsize=12, fontweight='bold')
        ax.set_facecolor('white'); ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(0, 115); ax.legend(fontsize=9)
        for bar, (_, row) in zip(ax.patches, zone_summary.iterrows()):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+2,
                    f"{int(row['scored'])}/{int(row['attempts'])}",
                    ha='center', fontsize=9, color='#333333')
        plt.tight_layout()
        plt.savefig(f'{MATCH_LABEL}_{player.replace(" ","_")}_zones.png',
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"\n{player}: {len(player_fr)} attempts  "
              f"conversion: {player_fr['scored'].mean()*100:.0f}%")


# ── CELL 10: Rolling Retention Timeline — Plotly ───────────────

fig = go.Figure()
for team, col, dash in zip(TEAMS, COLOURS, ['solid','dash']):
    sub = sub_po[team].sort_values('minute').reset_index(drop=True)
    if len(sub) >= 5:
        sub['rolling'] = sub['retained_bool'].rolling(7, min_periods=3).mean()*100
        fig.add_trace(go.Scatter(
            x=sub['minute'], y=sub['rolling'].round(1),
            mode='lines', name=team,
            line=dict(color=col, width=2.5, dash=dash),
            hovertemplate='Min %{x}  Retention: %{y:.0f}%<extra></extra>',
        ))

fig.add_vline(x=35, line_dash='dot', line_color=C_GREY,
              annotation_text='Half Time',
              annotation_position='top right')
fig.update_layout(
    title=f'<b>{MATCH_TITLE} — Rolling Puckout Retention (7-puckout window)</b>',
    title_font_size=16,
    xaxis=dict(title='Minute'),
    yaxis=dict(title='Retention % (rolling)', range=[0, 110]),
    legend=dict(orientation='h', yanchor='bottom', y=1.05,
                x=0.5, xanchor='center'),
    height=380, **PLOTLY_LAYOUT
)
fig.show()


# ── CELL 11: Final Summary ──────────────────────────────────────

W = 62
print("=" * W)
print(f"  {MATCH_TITLE.upper()}")
print(f"  {RESULT}")
print("=" * W)

print(f"\n{'PUCKOUTS':─<{W}}")
print(f"{'Metric':<36} {TEAM_A:>12} {TEAM_B:>12}")
print("─" * W)
po_rows = [
    ("Puckouts",             len(sub_po[TEAM_A]),  len(sub_po[TEAM_B])),
    ("Overall retention",    f"{sub_po[TEAM_A]['retained_bool'].mean()*100:.0f}%",
                             f"{sub_po[TEAM_B]['retained_bool'].mean()*100:.0f}%"),
    ("Short",                (sub_po[TEAM_A]['type']=='short').sum(),
                             (sub_po[TEAM_B]['type']=='short').sum()),
    ("Short retention",      f"{sub_po[TEAM_A][sub_po[TEAM_A]['type']=='short']['retained_bool'].mean()*100:.0f}%",
                             f"{sub_po[TEAM_B][sub_po[TEAM_B]['type']=='short']['retained_bool'].mean()*100:.0f}%"),
    ("Long",                 (sub_po[TEAM_A]['type']=='long').sum(),
                             (sub_po[TEAM_B]['type']=='long').sum()),
    ("Long retention",       f"{sub_po[TEAM_A][sub_po[TEAM_A]['type']=='long']['retained_bool'].mean()*100:.0f}%",
                             f"{sub_po[TEAM_B][sub_po[TEAM_B]['type']=='long']['retained_bool'].mean()*100:.0f}%"),
    ("Broken deliveries",    f"{(sub_po[TEAM_A]['delivery']=='B').sum()} "
                             f"({(sub_po[TEAM_A]['delivery']=='B').mean()*100:.0f}%)",
                             f"{(sub_po[TEAM_B]['delivery']=='B').sum()} "
                             f"({(sub_po[TEAM_B]['delivery']=='B').mean()*100:.0f}%)"),
    ("Top zone",             sub_po[TEAM_A]['target_zone'].mode()[0],
                             sub_po[TEAM_B]['target_zone'].mode()[0]),
]
for label, a, b in po_rows:
    print(f"{label:<36} {str(a):>12} {str(b):>12}")

print(f"\n{'FREES & SET PIECES':─<{W}}")
print(f"{'Metric':<36} {TEAM_A:>12} {TEAM_B:>12}")
print("─" * W)
fr_rows = [
    ("Attempts",         len(sub_fr[TEAM_A]),  len(sub_fr[TEAM_B])),
    ("Conversion",       f"{sub_fr[TEAM_A]['scored'].mean()*100:.0f}%",
                         f"{sub_fr[TEAM_B]['scored'].mean()*100:.0f}%"),
    ("Points",           (sub_fr[TEAM_A]['outcome']=='point').sum(),
                         (sub_fr[TEAM_B]['outcome']=='point').sum()),
    ("Wides",            (sub_fr[TEAM_A]['outcome']=='wide').sum(),
                         (sub_fr[TEAM_B]['outcome']=='wide').sum()),
    ("Retained",         (sub_fr[TEAM_A]['outcome']=='retained').sum(),
                         (sub_fr[TEAM_B]['outcome']=='retained').sum()),
]
for label, a, b in fr_rows:
    print(f"{label:<36} {str(a):>12} {str(b):>12}")

if CONTEXT_NOTE:
    print(f"\n{'CONTEXT':─<{W}}")
    # Word-wrap the context note
    words = CONTEXT_NOTE.split()
    line, lines = '', []
    for w in words:
        if len(line)+len(w)+1 <= W-2:
            line += (' ' if line else '') + w
        else:
            lines.append(line); line = w
    if line: lines.append(line)
    for l in lines:
        print(f"  {l}")

print("=" * W)
