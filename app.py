import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Prediction v1.0",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Prediction")
st.markdown("Simple model using actual performance data")

# ========== SIMPLE CONSTANTS ==========
MAX_GOALS_CALC = 6

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

def factorial_cache(n):
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    """Poisson probability mass function"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

@st.cache_data(ttl=3600)
def load_league_data(league_name):
    try:
        file_path = f"leagues/{league_name.lower().replace(' ', '_')}.csv"
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare home and away data"""
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    for df_part in [home_data, away_data]:
        if len(df_part) > 0:
            df_part['matches'] = df_part['matches'].astype(float)
            df_part['goals_for_pm'] = df_part['gf'] / df_part['matches']
            df_part['goals_against_pm'] = df_part['ga'] / df_part['matches']
            df_part['goals_vs_xg_pm'] = df_part['goals_vs_xg'] / df_part['matches']
            df_part['goals_allowed_vs_xga_pm'] = df_part['goals_allowed_vs_xga'] / df_part['matches']
            df_part['xg_pm'] = df_part['xg'] / df_part['matches']
            df_part['xga_pm'] = df_part['xga'] / df_part['matches']
    
    return home_data.set_index('team'), away_data.set_index('team')

# ========== SIMPLE PREDICTION LOGIC ==========

class SimplePredictor:
    """Simple prediction using actual goals scored/allowed"""
    
    def __init__(self):
        pass
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Simple prediction based on actual performance"""
        
        # Use ACTUAL goals per game, not xG
        home_attack = home_stats['goals_for_pm']
        home_defense = home_stats['goals_against_pm']
        away_attack = away_stats['goals_for_pm']
        away_defense = away_stats['goals_against_pm']
        
        # Simple average method
        home_expected = (home_attack + away_defense) / 2 * 1.1  # Home advantage
        away_expected = (away_attack + home_defense) / 2 * 0.9  # Away disadvantage
        
        # Ensure reasonable values
        home_expected = max(0.2, min(4.0, home_expected))
        away_expected = max(0.2, min(4.0, away_expected))
        
        # Calculate probabilities using Poisson
        score_probabilities = []
        max_goals = min(MAX_GOALS_CALC, int(home_expected + away_expected) + 3)
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (poisson_pmf(home_goals, home_expected) * 
                       poisson_pmf(away_goals, away_expected))
                if prob > 0.0001:
                    score_probabilities.append({
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'probability': prob
                    })
        
        most_likely = max(score_probabilities, key=lambda x: x['probability'])
        most_likely_score = f"{most_likely['home_goals']}-{most_likely['away_goals']}"
        
        home_win_prob = sum(p['probability'] for p in score_probabilities 
                           if p['home_goals'] > p['away_goals'])
        draw_prob = sum(p['probability'] for p in score_probabilities 
                       if p['home_goals'] == p['away_goals'])
        away_win_prob = sum(p['probability'] for p in score_probabilities 
                           if p['home_goals'] < p['away_goals'])
        
        over_2_5_prob = sum(p['probability'] for p in score_probabilities 
                           if p['home_goals'] + p['away_goals'] > 2.5)
        under_2_5_prob = 1 - over_2_5_prob
        
        btts_prob = sum(p['probability'] for p in score_probabilities 
                       if p['home_goals'] > 0 and p['away_goals'] > 0)
        
        top_scores = sorted(score_probabilities, key=lambda x: x['probability'], reverse=True)[:5]
        
        # Determine winner
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            winner = home_team
            winner_prob = home_win_prob
            winner_type = "HOME"
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            winner = away_team
            winner_prob = away_win_prob
            winner_type = "AWAY"
        else:
            winner = "DRAW"
            winner_prob = draw_prob
            winner_type = "DRAW"
        
        # Simple confidence based on probability
        winner_confidence = int(winner_prob * 100)
        if winner_confidence >= 60:
            conf_category = "HIGH"
        elif winner_confidence >= 50:
            conf_category = "MEDIUM"
        else:
            conf_category = "LOW"
        
        # Totals direction
        if over_2_5_prob > under_2_5_prob:
            totals_dir = "OVER"
            totals_prob = over_2_5_prob
        else:
            totals_dir = "UNDER"
            totals_prob = under_2_5_prob
        
        totals_confidence = int(totals_prob * 100)
        if totals_confidence >= 70:
            totals_conf_cat = "HIGH"
        elif totals_confidence >= 60:
            totals_conf_cat = "MEDIUM"
        else:
            totals_conf_cat = "LOW"
        
        # Simple insights
        insights = []
        
        # Finishing insight
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        if home_finish > 0.2:
            insights.append(f"⚡ {home_team} scores more than expected (+{home_finish:.2f}/game)")
        elif home_finish < -0.2:
            insights.append(f"⚡ {home_team} wastes chances ({home_finish:.2f}/game below xG)")
        
        if away_finish > 0.2:
            insights.append(f"⚡ {away_team} scores more than expected (+{away_finish:.2f}/game)")
        elif away_finish < -0.2:
            insights.append(f"⚡ {away_team} wastes chances ({away_finish:.2f}/game below xG)")
        
        # Defense insight
        home_def_qual = home_stats['goals_allowed_vs_xga_pm']
        away_def_qual = away_stats['goals_allowed_vs_xga_pm']
        
        if home_def_qual < -0.2:
            insights.append(f"🛡️ {home_team} has good defense (-{abs(home_def_qual):.2f} goals/game below expected)")
        elif home_def_qual > 0.2:
            insights.append(f"🛡️ {home_team} has poor defense (+{home_def_qual:.2f} goals/game above expected)")
        
        if away_def_qual < -0.2:
            insights.append(f"🛡️ {away_team} has good defense (-{abs(away_def_qual):.2f} goals/game below expected)")
        elif away_def_qual > 0.2:
            insights.append(f"🛡️ {away_team} has poor defense (+{away_def_qual:.2f} goals/game above expected)")
        
        return {
            'winner': {
                'team': winner,
                'type': winner_type,
                'probability': winner_prob,
                'confidence': conf_category,
                'confidence_score': winner_confidence
            },
            'totals': {
                'direction': totals_dir,
                'probability': totals_prob,
                'confidence': totals_conf_cat,
                'confidence_score': totals_confidence
            },
            'probabilities': {
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob,
                'over_2_5': over_2_5_prob,
                'under_2_5': under_2_5_prob,
                'btts': btts_prob
            },
            'expected_goals': {
                'home': home_expected,
                'away': away_expected,
                'total': home_expected + away_expected
            },
            'most_likely_score': most_likely_score,
            'top_scores': [(f"{s['home_goals']}-{s['away_goals']}", s['probability']) for s in top_scores],
            'insights': insights[:4]
        }

# ========== STREAMLIT UI ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie", "RFPL"]
    selected_league = st.selectbox("Select League", leagues)
    
    df = load_league_data(selected_league)
    
    if df is not None:
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        if len(home_stats_df) > 0 and len(away_stats_df) > 0:
            home_teams = sorted(home_stats_df.index.unique())
            away_teams = sorted(away_stats_df.index.unique())
            common_teams = sorted(list(set(home_teams) & set(away_teams)))
            
            if len(common_teams) == 0:
                st.error("No teams with complete home and away data")
                st.stop()
            
            home_team = st.selectbox("Home Team", common_teams)
            away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
            
            if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
                calculate_btn = True
            else:
                calculate_btn = False
        else:
            st.error("Could not prepare team data")
            st.stop()

if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Generate Prediction'")
    st.stop()

try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"Team data error: {e}")
    st.stop()

# Generate prediction
st.header(f"🎯 {home_team} vs {away_team}")
st.caption(f"League: {selected_league}")

predictor = SimplePredictor()
prediction = predictor.predict_match(home_team, away_team, home_stats, away_stats)

# ========== DISPLAY RESULTS ==========

# Main prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction['winner']
    winner_prob = winner_pred['probability']
    winner_conf = winner_pred['confidence_score']
    
    if winner_pred['type'] == "HOME":
        icon = "🏠"
        color = "#22C55E" if winner_conf >= 60 else "#F59E0B" if winner_conf >= 50 else "#EF4444"
    elif winner_pred['type'] == "AWAY":
        icon = "✈️"
        color = "#22C55E" if winner_conf >= 60 else "#F59E0B" if winner_conf >= 50 else "#EF4444"
    else:
        icon = "🤝"
        color = "#F59E0B"
    
    st.markdown(f"""
    <div style="background-color: #1F2937; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0; border: 2px solid {color};">
        <h3 style="color: white; margin: 0;">PREDICTED WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {icon} {winner_pred['team']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_prob*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            Confidence: {winner_pred['confidence']} ({winner_conf}/100)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    totals_pred = prediction['totals']
    direction = totals_pred['direction']
    probability = totals_pred['probability']
    conf_score = totals_pred['confidence_score']
    
    if direction == "OVER":
        color = "#22C55E" if conf_score >= 70 else "#F59E0B" if conf_score >= 60 else "#EF4444"
    else:
        color = "#EF4444" if conf_score >= 70 else "#F59E0B" if conf_score >= 60 else "#22C55E"
    
    st.markdown(f"""
    <div style="background-color: #1F2937; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0; border: 2px solid {color};">
        <h3 style="color: white; margin: 0;">TOTAL GOALS</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {direction} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {probability*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            Confidence: {totals_pred['confidence']} ({conf_score}/100)
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== INSIGHTS ==========
if prediction['insights']:
    st.subheader("📊 Team Analysis")
    for insight in prediction['insights']:
        st.write(f"• {insight}")

# ========== PROBABILITIES ==========
st.subheader("🎲 Probabilities")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(f"🏠 {home_team} Win", f"{prediction['probabilities']['home_win']*100:.1f}%")

with col2:
    st.metric("🤝 Draw", f"{prediction['probabilities']['draw']*100:.1f}%")

with col3:
    st.metric(f"✈️ {away_team} Win", f"{prediction['probabilities']['away_win']*100:.1f}%")

with col4:
    st.metric("Both Teams Score", f"{prediction['probabilities']['btts']*100:.1f}%")

# ========== MOST LIKELY SCORES ==========
st.subheader("🎯 Most Likely Scores")
scores_cols = st.columns(5)
for idx, (score, prob) in enumerate(prediction['top_scores'][:5]):
    with scores_cols[idx]:
        st.metric(f"{score}", f"{prob*100:.1f}%")

# ========== EXPECTED GOALS ==========
st.subheader("⚽ Expected Goals")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(f"{home_team}", f"{prediction['expected_goals']['home']:.2f}")

with col2:
    st.metric(f"{away_team}", f"{prediction['expected_goals']['away']:.2f}")

with col3:
    total_xg = prediction['expected_goals']['total']
    st.metric("Total", f"{total_xg:.2f}")

# ========== SIMPLE BETTING RECOMMENDATION ==========
st.divider()
st.subheader("💰 Betting Suggestion")

winner_prob = prediction['winner']['probability']
winner_conf = prediction['winner']['confidence_score']
totals_prob = prediction['totals']['probability']
totals_conf = prediction['totals']['confidence_score']

if totals_prob >= 0.65 and totals_conf >= 65:
    st.success(f"**Consider betting {prediction['totals']['direction']} 2.5 goals** ({totals_prob*100:.1f}% probability)")
elif winner_prob >= 0.55 and winner_conf >= 55:
    st.info(f"**Consider betting {prediction['winner']['team']} to win** ({winner_prob*100:.1f}% probability)")
else:
    st.warning("**No clear betting opportunity** - probabilities too close")

# ========== EXPORT ==========
st.divider()
st.subheader("📤 Export")

report = f"""
FOOTBALL PREDICTION
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

PREDICTIONS:
Winner: {prediction['winner']['team']} ({prediction['winner']['probability']*100:.1f}%)
Total Goals: {prediction['totals']['direction']} 2.5 ({prediction['totals']['probability']*100:.1f}%)
Most Likely Score: {prediction['most_likely_score']}

PROBABILITIES:
{home_team} Win: {prediction['probabilities']['home_win']*100:.1f}%
Draw: {prediction['probabilities']['draw']*100:.1f}%
{away_team} Win: {prediction['probabilities']['away_win']*100:.1f}%
OVER 2.5: {prediction['probabilities']['over_2_5']*100:.1f}%
UNDER 2.5: {prediction['probabilities']['under_2_5']*100:.1f}%
Both Teams Score: {prediction['probabilities']['btts']*100:.1f}%

EXPECTED GOALS:
{home_team}: {prediction['expected_goals']['home']:.2f}
{away_team}: {prediction['expected_goals']['away']:.2f}
Total: {prediction['expected_goals']['total']:.2f}
"""

st.code(report, language="text")

if st.button("📥 Download Report", use_container_width=True):
    st.download_button(
        label="Click to download",
        data=report,
        file_name=f"{home_team}_vs_{away_team}.txt",
        mime="text/plain"
    )
