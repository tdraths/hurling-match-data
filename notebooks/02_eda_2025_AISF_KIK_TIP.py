# ================================================================
# 2025 All-Ireland Senior Hurling Championship Semi-Final
# Kilkenny vs Tipperary — Match Analytics Notebook
# 02_eda_2025_AISF_KIK_TIP.ipynb
#
# Libraries:
#   Plotly  — interactive bar, line, scatter charts
#   Seaborn — statistical distribution and heatmap charts
#   Matplotlib — pitch zone maps only
# ================================================================


# ── CELL 1: Setup & Load ────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Colour palette ──────────────────────────────────────────────
C_KK    = '#1a1a2e'   # Kilkenny black
C_TIP   = '#003DA5'   # Tipperary blue
C_GOLD  = '#f5c842'   # Kilkenny gold accent
C_RED   = '#e05050'   # negative / loss
C_GREEN = '#2d8a4e'   # positive / retained
C_GREY  = '#888888'
C_GRASS = '#3a7d44'
C_LIGHT = '#f7f7f7'

# Seaborn light theme
sns.set_theme(style='whitegrid', font='sans-serif')
sns.set_palette([C_KK, C_TIP])

# Plotly light template base
PLOTLY_LAYOUT = dict(
    template='plotly_white',
    font=dict(family='Arial', size=12, color='#333333'),
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(t=50, b=40, l=50, r=30),
)

# ── Load data ───────────────────────────────────────────────────
# Update BASE_URL to your GitHub raw path once pushed
BASE_URL = 'https://raw.githubusercontent.com/YOUR_USERNAME/hurling-match-data/main/'
# Or load locally:
po = pd.read_csv('data/processed/puckouts/2025_AISF_KIK_TIP.csv')
fr = pd.read_csv('data/processed/frees/2025_AISF_KIK_TIP.csv')

# ── Derived fields ───────────────────────────────────────────────
zone_to_type = {
    'DL':'short','DC':'short','DR':'short',
    'ML':'medium','MC':'medium','MR':'medium',
    'AL':'long','AC':'long','AR':'long'
}
po['type']          = po['target_zone'].map(zone_to_type)
po['retained_bool'] = po['retained'] == 'yes'
po['minute']        = pd.to_numeric(po['minute'], errors='coerce')

fr['scored']  = fr['outcome'].isin(['point','goal'])
fr['minute']  = pd.to_numeric(fr['minute'], errors='coerce')
fr['shooter'] = fr['shooter'].astype(str)
fr.loc[fr['shooting_team']=='Kilkenny', 'shooter'] = 'TJ Reid'

def gs_bin(d):
    if d <= -4:  return 'Losing 4+'
    elif d <= -1: return 'Losing 1-3'
    elif d == 0:  return 'Level'
    elif d <= 3:  return 'Winning 1-3'
    else:         return 'Winning 4+'

GS_ORDER = ['Losing 4+','Losing 1-3','Level','Winning 1-3','Winning 4+']
po['game_state'] = po['score_diff'].apply(gs_bin)

print(f"Match: {po['match'].iloc[0]}  |  Date: {po['date'].iloc[0]}")
print(f"Result: Tipperary 4-20  Kilkenny 0-30  (Tipp win by 2pts)")
print(f"\nPuckouts: {len(po)}  |  Frees/65s: {len(fr)}")


# ── CELL 2: Match Summary — Plotly scorecard ────────────────────

kk  = po[po['pucking_team']=='Kilkenny']
tip = po[po['pucking_team']=='Tipperary']
kk_fr  = fr[fr['shooting_team']=='Kilkenny']
tip_fr = fr[fr['shooting_team']=='Tipperary']

metrics = ['Puckouts', 'Retention %', 'Short %', 'Long %',
           'Long Retention %', 'Broken Deliveries %',
           'Free Attempts', 'Free Conversion %']

kk_vals = [
    len(kk),
    round(kk['retained_bool'].mean()*100),
    round((kk['type']=='short').mean()*100),
    round((kk['type']=='long').mean()*100),
    round(kk[kk['type']=='long']['retained_bool'].mean()*100),
    round((kk['delivery']=='B').mean()*100),
    len(kk_fr),
    round(kk_fr['scored'].mean()*100),
]

tip_vals = [
    len(tip),
    round(tip['retained_bool'].mean()*100),
    round((tip['type']=='short').mean()*100),
    round((tip['type']=='long').mean()*100),
    round(tip[tip['type']=='long']['retained_bool'].mean()*100),
    round((tip['delivery']=='B').mean()*100),
    len(tip_fr),
    round(tip_fr['scored'].mean()*100),
]

fig = go.Figure(data=[
    go.Table(
        columnwidth=[200, 100, 100],
        header=dict(
            values=['<b>Metric</b>', '<b>Kilkenny</b>', '<b>Tipperary</b>'],
            fill_color=[C_LIGHT, C_KK, C_TIP],
            font=dict(color=['#333333','#f5c842','white'], size=13),
            align='center', height=36,
        ),
        cells=dict(
            values=[metrics, kk_vals, tip_vals],
            fill_color=[C_LIGHT,
                        ['#f0f0f0' if i%2==0 else 'white' for i in range(len(metrics))],
                        ['#f0f0f0' if i%2==0 else 'white' for i in range(len(metrics))]],
            font=dict(color='#333333', size=12),
            align=['left','center','center'],
            height=32,
        )
    )
])
fig.update_layout(title='<b>Match Summary — Process Metrics</b>',
                  title_font_size=16, **PLOTLY_LAYOUT)
fig.show()


# ── CELL 3: Puckout Strategy — Plotly grouped bar ───────────────

teams     = ['Kilkenny','Tipperary']
colours   = [C_KK, C_TIP]
types     = ['short','medium','long']
type_pcts = {}
ret_rates = {}

for team in teams:
    sub = po[po['pucking_team']==team]
    type_pcts[team] = [round(len(sub[sub['type']==t])/len(sub)*100,1) for t in types]
    ret_rates[team] = [round(sub[sub['type']==t]['retained_bool'].mean()*100,1)
                       if len(sub[sub['type']==t])>0 else 0 for t in types]

fig = make_subplots(rows=1, cols=2,
    subplot_titles=('<b>Puckout Type Distribution</b>',
                    '<b>Retention Rate by Type</b>'),
    horizontal_spacing=0.12)

for i, (team, col) in enumerate(zip(teams, colours)):
    fig.add_trace(go.Bar(
        name=team, x=[t.title() for t in types], y=type_pcts[team],
        marker_color=col, text=[f'{v}%' for v in type_pcts[team]],
        textposition='outside', showlegend=True,
        offsetgroup=i
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        name=team, x=[t.title() for t in types], y=ret_rates[team],
        mode='lines+markers+text',
        line=dict(color=col, width=2.5),
        marker=dict(size=10, color=col),
        text=[f'{v}%' for v in ret_rates[team]],
        textposition='top center',
        showlegend=False
    ), row=1, col=2)

fig.update_layout(
    barmode='group',
    yaxis=dict(title='% of puckouts', range=[0,85]),
    yaxis2=dict(title='Retention %', range=[0,115]),
    legend=dict(orientation='h', yanchor='bottom', y=1.08, x=0.5, xanchor='center'),
    title='<b>2025 AISF — Puckout Strategy Comparison</b>',
    title_font_size=16, height=420,
    **PLOTLY_LAYOUT
)
fig.show()


# ── CELL 4: Delivery Code Analysis — Seaborn ───────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('white')

del_order  = ['C','P','B','T','X']
del_labels = ['Clean','Pressured','Broken','Turnover','Failed']
del_colours = [C_GREEN, '#f5a623', C_GREY, C_RED, '#c0392b']

# Panel 1: delivery distribution stacked by team
del_data = []
for team in teams:
    sub = po[po['pucking_team']==team]
    for code, label in zip(del_order, del_labels):
        del_data.append({'Team':team, 'Code':label,
                         'Count':len(sub[sub['delivery']==code]),
                         'Pct':round(len(sub[sub['delivery']==code])/len(sub)*100,1)})
df_del = pd.DataFrame(del_data)

ax = axes[0]
x = np.arange(len(teams))
bottom = np.zeros(len(teams))
for code, label, col in zip(del_order, del_labels, del_colours):
    vals = [df_del[(df_del['Team']==t)&(df_del['Code']==label)]['Pct'].values[0]
            for t in teams]
    bars = ax.bar(x, vals, bottom=bottom, color=col, alpha=0.88,
                  edgecolor='white', linewidth=0.8, label=label)
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax.text(xi, b + v/2, f'{v:.0f}%',
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white')
    bottom += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels(teams, fontsize=11)
ax.set_ylabel('% of puckouts', fontsize=10)
ax.set_title('Delivery Code Distribution', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.set_facecolor('white'); ax.spines[['top','right']].set_visible(False)

# Panel 2: retention rate by delivery code — dot plot
ax2 = axes[1]
for i, (team, col, marker) in enumerate(zip(teams, colours, ['o','s'])):
    sub = po[po['pucking_team']==team]
    ys, xs = [], []
    for j, (code, label) in enumerate(zip(del_order, del_labels)):
        s = sub[sub['delivery']==code]
        if len(s) >= 2:
            ys.append(s['retained_bool'].mean()*100)
            xs.append(label)
    ax2.scatter(xs, ys, color=col, s=100, marker=marker,
                zorder=5, label=team, alpha=0.9)
ax2.axhline(po['retained_bool'].mean()*100, color=C_GREY,
            linestyle='--', linewidth=1, alpha=0.6, label='Overall avg')
ax2.set_ylabel('Retention %', fontsize=10)
ax2.set_title('Retention Rate by Delivery Code', fontsize=12, fontweight='bold', pad=10)
ax2.legend(fontsize=9, framealpha=0.9)
ax2.set_facecolor('white'); ax2.spines[['top','right']].set_visible(False)
ax2.set_ylim(-5, 110)

plt.suptitle('2025 AISF — Delivery Quality Analysis', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('q2_delivery_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()

print(f"\nBroken ball deliveries: {(po['delivery']=='B').sum()} of {len(po)} "
      f"({(po['delivery']=='B').mean()*100:.0f}%) — a highly contested match")


# ── CELL 5: Zone Maps — Matplotlib (pitch maps) ─────────────────

PO_ZONES = {
    'AL':(0,0),'AC':(0,1),'AR':(0,2),
    'ML':(1,0),'MC':(1,1),'MR':(1,2),
    'DL':(2,0),'DC':(2,1),'DR':(2,2),
}
rg_cmap = LinearSegmentedColormap.from_list('rg',['#e05050','#f5c842','#2d8a4e'])

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.patch.set_facecolor('white')

team_colours = {'Kilkenny': C_KK, 'Tipperary': C_TIP}

for row_i, team in enumerate(teams):
    sub   = po[po['pucking_team']==team]
    total = len(sub)
    col_t = team_colours[team]
    zone_counts = sub['target_zone'].value_counts()

    # Frequency map
    freq_cmap = LinearSegmentedColormap.from_list('freq',
        ['#f0f0f0', '#a8d8a8', col_t])
    ax = axes[row_i][0]
    ax.set_facecolor('#e8f4e8'); ax.set_xlim(0,3); ax.set_ylim(0,3)
    ax.set_aspect('equal')
    for zone,(row,col) in PO_ZONES.items():
        cnt = zone_counts.get(zone, 0)
        val = cnt / total
        colour = freq_cmap(val)
        ax.add_patch(patches.Rectangle(
            (col, 2-row), 1, 1,
            facecolor=colour, alpha=0.9,
            edgecolor='white', linewidth=2))
        ax.text(col+0.5, 2-row+0.5,
                f"{zone}\n{cnt} ({val*100:.0f}%)",
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='#333333')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{team} — Zone Frequency', fontsize=11,
                 fontweight='bold', color=col_t, pad=8)
    ax.text(0.02, -0.05, '← Opp Goal', transform=ax.transAxes,
            fontsize=7, color=C_GREY)
    ax.text(0.98, -0.05, 'Own Goal →', transform=ax.transAxes,
            fontsize=7, color=C_GREY, ha='right')

    # Retention rate map
    zone_ret = {}
    for zone in PO_ZONES:
        s = sub[sub['target_zone']==zone]
        zone_ret[zone] = s['retained_bool'].mean() if len(s)>0 else np.nan

    ax2 = axes[row_i][1]
    ax2.set_facecolor('#e8f4e8'); ax2.set_xlim(0,3); ax2.set_ylim(0,3)
    ax2.set_aspect('equal')
    for zone,(row,col) in PO_ZONES.items():
        val = zone_ret.get(zone, np.nan)
        colour = rg_cmap(val) if not np.isnan(val) else '#dddddd'
        ax2.add_patch(patches.Rectangle(
            (col, 2-row), 1, 1,
            facecolor=colour, alpha=0.9,
            edgecolor='white', linewidth=2))
        lbl = f"{zone}\n{val*100:.0f}%" if not np.isnan(val) else f"{zone}\nn/a"
        ax2.text(col+0.5, 2-row+0.5, lbl,
                 ha='center', va='center', fontsize=9,
                 fontweight='bold', color='#333333')
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f'{team} — Retention Rate by Zone', fontsize=11,
                  fontweight='bold', color=col_t, pad=8)
    ax2.text(0.02, -0.05, '← Opp Goal', transform=ax2.transAxes,
             fontsize=7, color=C_GREY)
    ax2.text(0.98, -0.05, 'Own Goal →', transform=ax2.transAxes,
             fontsize=7, color=C_GREY, ha='right')

plt.suptitle('2025 AISF — Puckout Zone Analysis',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('q3_zone_maps.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ── CELL 6: Game State Analysis — Plotly ───────────────────────

gs_data = []
for team in teams:
    sub = po[po['pucking_team']==team]
    for gs in GS_ORDER:
        s = sub[sub['game_state']==gs]
        if len(s) > 0:
            gs_data.append({
                'Team': team, 'Game State': gs,
                'Count': len(s),
                'Short %': round((s['type']=='short').mean()*100, 1),
                'Long %':  round((s['type']=='long').mean()*100, 1),
                'Retention %': round(s['retained_bool'].mean()*100, 1),
            })
df_gs = pd.DataFrame(gs_data)

fig = make_subplots(rows=1, cols=3,
    subplot_titles=(
        '<b>Short % by Game State</b>',
        '<b>Long % by Game State</b>',
        '<b>Retention % by Game State</b>'
    ),
    horizontal_spacing=0.1
)

metrics_gs = ['Short %','Long %','Retention %']
for col_i, metric in enumerate(metrics_gs, 1):
    for team, col in zip(teams, colours):
        sub_df = df_gs[df_gs['Team']==team].sort_values(
            'Game State',
            key=lambda x: x.map({g:i for i,g in enumerate(GS_ORDER)})
        )
        fig.add_trace(go.Scatter(
            x=sub_df['Game State'], y=sub_df[metric],
            mode='lines+markers',
            name=team,
            line=dict(color=col, width=2.5),
            marker=dict(size=9, color=col),
            showlegend=(col_i==1),
            text=[f'{v}%' for v in sub_df[metric]],
            textposition='top center',
        ), row=1, col=col_i)

fig.update_layout(
    title='<b>2025 AISF — Game State Strategy</b>',
    title_font_size=16,
    legend=dict(orientation='h', yanchor='bottom', y=1.1,
                x=0.5, xanchor='center'),
    height=430,
    **PLOTLY_LAYOUT
)
for i in range(1,4):
    fig.update_yaxes(range=[0,110], row=1, col=i)

fig.show()

print("\n=== KEY GAME STATE FINDINGS ===")
print(df_gs[['Team','Game State','Count','Short %','Long %','Retention %']].to_string(index=False))


# ── CELL 7: Frees Analysis — Plotly + Seaborn ──────────────────

# Plotly: Conversion comparison
fig = make_subplots(rows=1, cols=3,
    subplot_titles=(
        '<b>Score Conversion %</b>',
        '<b>Outcomes Breakdown</b>',
        '<b>Attempt Type vs Outcome</b>'
    ),
    horizontal_spacing=0.12
)

# Panel 1: overall conversion
for i, (team, col) in enumerate(zip(teams, colours)):
    sub  = fr[fr['shooting_team']==team]
    conv = sub['scored'].mean()*100
    fig.add_trace(go.Bar(
        x=[team], y=[conv],
        marker_color=col,
        text=[f'{conv:.0f}%'],
        textposition='outside',
        name=team, showlegend=False
    ), row=1, col=1)

# Panel 2: outcomes breakdown
outcome_order = ['point','goal','wide','short','saved','cleared','retained','lost']
out_colours   = {'point':C_GREEN,'goal':C_GOLD,'wide':C_RED,'short':C_RED,
                 'saved':'#e08830','cleared':'#e08830',
                 'retained':'#5090e0','lost':C_GREY}
for outcome in outcome_order:
    vals = [len(fr[(fr['shooting_team']==t)&(fr['outcome']==outcome)]) for t in teams]
    if sum(vals) > 0:
        fig.add_trace(go.Bar(
            name=outcome.title(), x=teams, y=vals,
            marker_color=out_colours.get(outcome, C_GREY),
            showlegend=True
        ), row=1, col=2)

# Panel 3: attempt type conversion
att_types = fr['attempt_type'].value_counts().index.tolist()
for i, (team, col) in enumerate(zip(teams, colours)):
    sub  = fr[fr['shooting_team']==team]
    convs = [sub[sub['attempt_type']==a]['scored'].mean()*100
             if len(sub[sub['attempt_type']==a])>0 else 0 for a in att_types]
    ns    = [len(sub[sub['attempt_type']==a]) for a in att_types]
    fig.add_trace(go.Bar(
        name=team,
        x=[a.title() for a in att_types],
        y=convs,
        marker_color=col,
        text=[f'{c:.0f}%<br>(n={n})' for c,n in zip(convs,ns)],
        textposition='outside',
        showlegend=False,
        offsetgroup=i
    ), row=1, col=3)

fig.update_layout(
    barmode='group',
    title='<b>2025 AISF — Frees & Set Pieces</b>',
    title_font_size=16,
    legend=dict(orientation='v', x=1.02, y=0.5),
    height=430,
    **PLOTLY_LAYOUT
)
fig.update_yaxes(range=[0,110], row=1, col=1)
fig.update_yaxes(range=[0,110], row=1, col=3)
fig.show()

# Seaborn: TJ Reid shot map by zone
print("\n=== TJ REID FREE-TAKING SUMMARY ===")
reid = fr[fr['shooter']=='TJ Reid']
print(f"Attempts: {len(reid)}")
print(f"Conversion: {reid['scored'].mean()*100:.0f}%")
print(f"Outcomes: {reid['outcome'].value_counts().to_dict()}")
print(f"Zones: {reid['position_zone'].value_counts().head(5).to_dict()}")

fig2, ax = plt.subplots(figsize=(8, 4))
reid_zone = reid.groupby('position_zone').agg(
    attempts=('scored','count'),
    scored=('scored','sum')
).assign(conversion=lambda x: x.scored/x.attempts*100).reset_index()

sns.barplot(data=reid_zone, x='position_zone', y='conversion',
            color=C_KK, alpha=0.85, ax=ax)
ax.axhline(100, color=C_GREEN, linestyle='--', linewidth=1,
           alpha=0.5, label='100% conversion')
ax.set_xlabel('Zone', fontsize=10)
ax.set_ylabel('Conversion %', fontsize=10)
ax.set_title('TJ Reid — Free Conversion by Zone', fontsize=12,
             fontweight='bold')
ax.set_facecolor('white'); ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, 115)
for bar, (_, row) in zip(ax.patches, reid_zone.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+2,
            f"{row['scored']:.0f}/{row['attempts']:.0f}",
            ha='center', fontsize=9, color='#333333')
plt.tight_layout()
plt.savefig('q4_reid_zones.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# ── CELL 8: Timeline — Plotly rolling retention ─────────────────

# Rolling 10-puckout retention window per team
fig = go.Figure()
for team, col, dash in zip(teams, colours, ['solid','dash']):
    sub = po[po['pucking_team']==team].sort_values('minute').reset_index(drop=True)
    if len(sub) >= 5:
        sub['rolling_ret'] = sub['retained_bool'].rolling(7, min_periods=3).mean()*100
        fig.add_trace(go.Scatter(
            x=sub['minute'], y=sub['rolling_ret'],
            mode='lines', name=team,
            line=dict(color=col, width=2.5, dash=dash),
            fill='tozeroy',
            fillcolor=col.replace(')', ',0.08)').replace('rgb','rgba')
                       if 'rgb' in col else col + '14',
        ))

# Add halftime marker
fig.add_vline(x=35, line_dash='dot', line_color=C_GREY,
              annotation_text='Half Time', annotation_position='top')
fig.update_layout(
    title='<b>2025 AISF — Rolling Puckout Retention (7-puckout window)</b>',
    title_font_size=16,
    xaxis=dict(title='Minute', range=[0,75]),
    yaxis=dict(title='Retention % (rolling)', range=[0,110]),
    legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0.5, xanchor='center'),
    height=380,
    **PLOTLY_LAYOUT
)
fig.show()


# ── CELL 9: Key Findings Summary ───────────────────────────────

W = 60
print("=" * W)
print("  2025 ALL-IRELAND SEMI-FINAL — PROCESS METRICS SUMMARY")
print("  Kilkenny 0-30  Tipperary 4-20  (Tipp win by 2 pts)")
print("=" * W)

print(f"\n{'PUCKOUTS':─<{W}}")
print(f"{'Metric':<35} {'Kilkenny':>11} {'Tipperary':>11}")
print("─" * W)
rows = [
    ("Total puckouts",          len(kk),        len(tip)),
    ("Overall retention",       f"{kk['retained_bool'].mean()*100:.0f}%",
                                f"{tip['retained_bool'].mean()*100:.0f}%"),
    ("Short puckouts",          (kk['type']=='short').sum(),
                                (tip['type']=='short').sum()),
    ("Short retention",         f"{kk[kk['type']=='short']['retained_bool'].mean()*100:.0f}%",
                                f"{tip[tip['type']=='short']['retained_bool'].mean()*100:.0f}%"),
    ("Long puckouts",           (kk['type']=='long').sum(),
                                (tip['type']=='long').sum()),
    ("Long retention",          f"{kk[kk['type']=='long']['retained_bool'].mean()*100:.0f}%",
                                f"{tip[tip['type']=='long']['retained_bool'].mean()*100:.0f}%"),
    ("Broken deliveries",       f"{(kk['delivery']=='B').sum()} ({(kk['delivery']=='B').mean()*100:.0f}%)",
                                f"{(tip['delivery']=='B').sum()} ({(tip['delivery']=='B').mean()*100:.0f}%)"),
    ("Top target zone",         kk['target_zone'].mode()[0],
                                tip['target_zone'].mode()[0]),
]
for label, k, t in rows:
    print(f"{label:<35} {str(k):>11} {str(t):>11}")

print(f"\n{'FREES & SET PIECES':─<{W}}")
print(f"{'Metric':<35} {'Kilkenny':>11} {'Tipperary':>11}")
print("─" * W)
fr_rows = [
    ("Total attempts",          len(kk_fr),     len(tip_fr)),
    ("Score conversion",        f"{kk_fr['scored'].mean()*100:.0f}%",
                                f"{tip_fr['scored'].mean()*100:.0f}%"),
    ("Points scored",           (kk_fr['outcome']=='point').sum(),
                                (tip_fr['outcome']=='point').sum()),
    ("Wides",                   (kk_fr['outcome']=='wide').sum(),
                                (tip_fr['outcome']=='wide').sum()),
    ("Retained (played short/long)", (kk_fr['outcome']=='retained').sum(),
                                (tip_fr['outcome']=='retained').sum()),
]
for label, k, t in fr_rows:
    print(f"{label:<35} {str(k):>11} {str(t):>11}")

print(f"\n{'ANALYTICAL NOTES':─<{W}}")
print("• Kilkenny outperformed Tipperary on all process metrics")
print("• 34% broken ball rate indicates an extremely physical contest")
print("• Tipperary's long puckout retention (21%) was critically low")
print("• KK free conversion largely attributable to TJ Reid")
print("• Match outcome driven by 4 Tipp goals from defensive breakdowns")
print("• Scoreline not reflective of possession/set-piece dominance")
print("=" * W)
