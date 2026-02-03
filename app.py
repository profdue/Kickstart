import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Football xG Predictor Pro+",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚽ Football Match Predictor Pro+")
st.markdown("""
    Advanced dual-model prediction system with xG regression + defensive gap analysis.
    Combines statistical modeling with defensive matchup intelligence.
""")

# Constants
MAX_GOALS = 8
REG_BASE_FACTOR = 0.75
REG_MATCH_THRESHOLD = 5
MAX_REGRESSION = 0.3

# Match type thresholds
SUPPRESSION_THRESHOLD = 3
VOLATILITY_THRESHOLD = 3

# Defensive gap thresholds
STRONG_OVER_THRESHOLD = 1.0
STRONG_UNDER_THRESHOLD = -1.0

# Initialize session state
if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

def factorial_cache(n):
    """Cache factorial calculations for performance"""
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    """Calculate Poisson probability manually"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

@st.cache_data(ttl=3600)
def load_league_data(league_name):
    """Load league data from CSV with caching"""
    try:
        file_map = {
            "premier_league": "premier_league.csv",
            "bundesliga": "bundesliga.csv",
            "serie_a": "serie_a.csv",
            "laliga": "laliga.csv",
            "ligue_1": "ligue_1.csv",
            "eredivisie": "eredivisie.csv"
        }
        
        actual_filename = file_map.get(league_name, f"{league_name}.csv")
        file_path = f"leagues/{actual_filename}"
        
        df = pd.read_csv(file_path)
        
        required_cols = ['team', 'venue', 'matches', 'xg', 'xga', 'goals_vs_xg']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
            return None
            
        return df
    except FileNotFoundError:
        st.error(f"⚠️ League file not found: leagues/{actual_filename}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare home and away stats from the data"""
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    home_stats = home_data.set_index('team')
    away_stats = away_data.set_index('team')
    
    return home_stats, away_stats

def calculate_league_baselines(df):
    """Calculate league average xGA for defensive gap analysis"""
    home_xga_per_match = df[df['venue'] == 'home']['xga'] / df[df['venue'] == 'home']['matches']
    away_xga_per_match = df[df['venue'] == 'away']['xga'] / df[df['venue'] == 'away']['matches']
    
    all_xga_per_match = pd.concat([home_xga_per_match, away_xga_per_match])
    
    league_avg_xga = all_xga_per_match.mean()
    league_std_xga = all_xga_per_match.std()
    
    return league_avg_xga, league_std_xga

class DefensiveGapModel:
    """Supporting logic model specializing in Over/Under prediction"""
    
    def __init__(self, league_avg_xga, league_std_xga):
        self.league_avg_xga = league_avg_xga
        self.league_std_xga = league_std_xga
    
    def analyze_match(self, home_stats, away_stats):
        """Analyze match for defensive gap signal"""
        home_xga_per_match = home_stats['xga'] / max(home_stats['matches'], 1)
        away_xga_per_match = away_stats['xga'] / max(away_stats['matches'], 1)
        
        home_def_score = (home_xga_per_match - self.league_avg_xga) / max(self.league_std_xga, 0.1)
        away_def_score = (away_xga_per_match - self.league_avg_xga) / max(self.league_std_xga, 0.1)
        
        match_gap = home_def_score + away_def_score
        
        if match_gap > STRONG_OVER_THRESHOLD:
            signal = "STRONG_OVER"
            confidence = "HIGH"
            explanation = f"Both teams have weak defenses ({match_gap:.2f} std above league avg)"
        elif match_gap < STRONG_UNDER_THRESHOLD:
            signal = "STRONG_UNDER"
            confidence = "HIGH"
            explanation = f"Both teams have strong defenses ({abs(match_gap):.2f} std below league avg)"
        elif abs(match_gap) > 0.5:
            signal = "MILD_OVER" if match_gap > 0 else "MILD_UNDER"
            confidence = "MEDIUM"
            explanation = f"Defensive matchup leans {'Over' if match_gap > 0 else 'Under'}"
        else:
            signal = "NEUTRAL"
            confidence = "LOW"
            explanation = "Mixed defensive matchup - no clear signal"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'gap_score': match_gap,
            'home_def_score': home_def_score,
            'away_def_score': away_def_score,
            'explanation': explanation
        }
    
    def get_bet_recommendation(self, defensive_signal, main_over_prob, main_under_prob):
        """Generate consensus betting recommendation"""
        main_over_threshold = 60
        main_under_threshold = 60
        
        main_signal = "OVER" if main_over_prob > main_over_threshold else "UNDER" if main_under_prob > main_under_threshold else "NEUTRAL"
        
        dg_signal = defensive_signal['signal']
        dg_confidence = defensive_signal['confidence']
        
        if dg_signal in ["STRONG_OVER", "MILD_OVER"]:
            dg_direction = "OVER"
        elif dg_signal in ["STRONG_UNDER", "MILD_UNDER"]:
            dg_direction = "UNDER"
        else:
            dg_direction = "NEUTRAL"
        
        if main_signal == "NEUTRAL" and dg_direction == "NEUTRAL":
            return "NO_BET", "Both models uncertain", "⚪"
        
        if main_signal == dg_direction:
            if dg_confidence == "HIGH":
                return "STRONG_BET", f"Strong consensus on {main_signal}", "✅"
            else:
                return "MODERATE_BET", f"Consensus on {main_signal} (medium confidence)", "🟡"
        
        elif main_signal == "NEUTRAL" and dg_direction != "NEUTRAL":
            if dg_confidence == "HIGH":
                return "CONSIDER_BET", f"Defensive model strongly suggests {dg_direction}", "🔵"
            else:
                return "WEAK_BET", f"Defensive model suggests {dg_direction}", "⚪"
        
        elif dg_direction == "NEUTRAL" and main_signal != "NEUTRAL":
            return "MAIN_MODEL_ONLY", f"Main model suggests {main_signal} (no defensive signal)", "🟠"
        
        else:
            if dg_confidence == "HIGH":
                return "CONFLICT_AVOID", f"Strong conflict: Main {main_signal} vs Defensive {dg_direction}", "❌"
            else:
                return "WEAK_CONFLICT", f"Mild conflict: Main {main_signal} vs Defensive {dg_direction}", "🟡"

def calculate_regression_factors(home_team_stats, away_team_stats, regression_factor):
    """Calculate attack regression factors with asymmetric capping"""
    home_matches = home_team_stats['matches']
    away_matches = away_team_stats['matches']
    
    if home_matches >= REG_MATCH_THRESHOLD:
        home_base_reg = (home_team_stats['goals_vs_xg'] / home_matches) * regression_factor
    else:
        home_base_reg = 0
    
    if away_matches >= REG_MATCH_THRESHOLD:
        away_base_reg = (away_team_stats['goals_vs_xg'] / away_matches) * regression_factor
    else:
        away_base_reg = 0
    
    home_wins = home_team_stats.get('wins', 0)
    away_wins = away_team_stats.get('wins', 0)
    
    home_win_rate = home_wins / max(home_matches, 1)
    away_win_rate = away_wins / max(away_matches, 1)
    
    if home_win_rate > 0.6:
        home_attack_reg = max(min(home_base_reg, MAX_REGRESSION), -MAX_REGRESSION)
    elif home_win_rate < 0.3:
        home_attack_reg = max(min(home_base_reg, MAX_REGRESSION * 0.5), -MAX_REGRESSION * 0.5)
    else:
        home_attack_reg = max(min(home_base_reg, MAX_REGRESSION * 0.75), -MAX_REGRESSION * 0.75)
    
    if away_win_rate > 0.6:
        away_attack_reg = max(min(away_base_reg, MAX_REGRESSION), -MAX_REGRESSION)
    elif away_win_rate < 0.3:
        away_attack_reg = max(min(away_base_reg, MAX_REGRESSION * 0.5), -MAX_REGRESSION * 0.5)
    else:
        away_attack_reg = max(min(away_base_reg, MAX_REGRESSION * 0.75), -MAX_REGRESSION * 0.75)
    
    return home_attack_reg, away_attack_reg

def calculate_expected_goals(home_stats, away_stats, home_attack_reg, away_attack_reg):
    """Calculate expected goals for both teams"""
    home_xg_per_match = home_stats['xg'] / max(home_stats['matches'], 1)
    away_xga_per_match = away_stats['xga'] / max(away_stats['matches'], 1)
    
    away_xg_per_match = away_stats['xg'] / max(away_stats['matches'], 1)
    home_xga_per_match = home_stats['xga'] / max(home_stats['matches'], 1)
    
    home_expected = np.sqrt(home_xg_per_match * away_xga_per_match) * (1 + home_attack_reg)
    away_expected = np.sqrt(away_xg_per_match * home_xga_per_match) * (1 + away_attack_reg)
    
    home_expected = max(home_expected, 0.3)
    away_expected = max(away_expected, 0.3)
    
    home_expected = min(home_expected, 4.0)
    away_expected = min(away_expected, 4.0)
    
    return home_expected, away_expected

def create_probability_matrix(home_lam, away_lam, max_goals=MAX_GOALS):
    """Create probability matrix for all score combinations"""
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_home = poisson_pmf(i, home_lam)
            prob_away = poisson_pmf(j, away_lam)
            prob_matrix[i, j] = prob_home * prob_away
    
    return prob_matrix

def calculate_outcome_probabilities(prob_matrix):
    """Calculate home win, draw, and away win probabilities"""
    home_win = np.sum(np.triu(prob_matrix, k=1))
    draw = np.sum(np.diag(prob_matrix))
    away_win = np.sum(np.tril(prob_matrix, k=-1))
    
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    
    return home_win, draw, away_win

def calculate_betting_markets(prob_matrix):
    """Calculate betting market probabilities"""
    over_25 = 0
    under_25 = 0
    
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            total_goals = i + j
            prob = prob_matrix[i, j]
            
            if total_goals > 2.5:
                over_25 += prob
            else:
                under_25 += prob
    
    btts_yes = 0
    btts_no = 0
    
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            prob = prob_matrix[i, j]
            
            if i >= 1 and j >= 1:
                btts_yes += prob
            else:
                btts_no += prob
    
    return over_25, under_25, btts_yes, btts_no

def create_correct_outcome_chart(home_win_prob, draw_prob, away_win_prob, home_team, away_team):
    """Create properly ordered outcome chart using Plotly"""
    categories = [f'{home_team} Win', 'Draw', f'{away_team} Win']
    probabilities = [home_win_prob, draw_prob, away_win_prob]
    
    # Colors for each outcome
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=probabilities,
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Match Outcome Probabilities',
        xaxis_title='Outcome',
        yaxis_title='Probability',
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        showlegend=False,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_expected_goals_chart(home_xg, away_xg, home_team, away_team):
    """Create expected goals chart using Plotly"""
    teams = [home_team, away_team]
    xg_values = [home_xg, away_xg]
    
    fig = go.Figure(data=[
        go.Bar(
            x=teams,
            y=xg_values,
            text=[f'{xg:.2f}' for xg in xg_values],
            textposition='auto',
            marker_color=['#1f77b4', '#ff7f0e'],
            hovertemplate='<b>%{x}</b><br>Expected Goals: %{y:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Expected Goals',
        xaxis_title='Team',
        yaxis_title='Expected Goals',
        showlegend=False,
        height=300,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

# ========== SIDEBAR CONTROLS ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie"]
    selected_league = st.selectbox("Select League", leagues)
    
    league_to_file = {
        "Premier League": "premier_league",
        "Bundesliga": "bundesliga",
        "Serie A": "serie_a",
        "La Liga": "laliga",
        "Ligue 1": "ligue_1",
        "Eredivisie": "eredivisie"
    }
    
    league_key = league_to_file[selected_league]
    df = load_league_data(league_key)
    
    if df is not None:
        league_avg_xga, league_std_xga = calculate_league_baselines(df)
        
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        available_home_teams = sorted(home_stats_df.index.unique())
        available_away_teams = sorted(away_stats_df.index.unique())
        common_teams = sorted(list(set(available_home_teams) & set(available_away_teams)))
        
        if len(common_teams) == 0:
            st.error("❌ No teams with both home and away data available")
        else:
            home_team = st.selectbox("Home Team", common_teams)
            away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
            
            regression_factor = st.slider(
                "Regression Factor",
                min_value=0.0,
                max_value=2.0,
                value=REG_BASE_FACTOR,
                step=0.05,
                help="Adjust how much to regress team performance to mean"
            )
            
            with st.expander("⚙️ Dual Model Settings"):
                enable_defensive_model = st.checkbox("Enable Defensive Gap Model", value=True,
                    help="Use independent defensive analysis for Over/Under validation")
                show_consensus = st.checkbox("Show Consensus Analysis", value=True)
            
            calculate_btn = st.button("🎯 Calculate Predictions", type="primary", use_container_width=True)
            
            st.divider()
            st.subheader("📊 Display Options")
            show_matrix = st.checkbox("Show Score Probability Matrix", value=False)

# ========== MAIN CONTENT ==========
if df is None:
    st.warning("📁 Please add league CSV files to the 'leagues' folder")
    st.info("""
    **Required CSV files:**
    - `premier_league.csv`, `bundesliga.csv`, `serie a.csv`
    - `laliga.csv`, `ligue_1.csv`, `eredivisie.csv` (optional)
    """)
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Calculate Predictions' to start")
    
    with st.expander("📋 Preview of Loaded Data"):
        st.dataframe(df.head(10))
    st.stop()

try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"❌ Team data not found: {e}")
    st.stop()

# ========== PHASE 1: MAIN MODEL CALCULATIONS ==========
st.header(f"📊 {home_team} vs {away_team}")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader(f"🏠 {home_team} (Home)")
    st.metric("Matches", int(home_stats['matches']))
    if 'wins' in home_stats:
        st.metric("Wins", int(home_stats['wins']))
    home_xg_per_match = home_stats['xg'] / max(home_stats['matches'], 1)
    home_xga_per_match = home_stats['xga'] / max(home_stats['matches'], 1)
    st.metric("xG/match", f"{home_xg_per_match:.2f}")
    st.metric("xGA/match", f"{home_xga_per_match:.2f}")

with col2:
    st.subheader(f"✈️ {away_team} (Away)")
    st.metric("Matches", int(away_stats['matches']))
    if 'wins' in away_stats:
        st.metric("Wins", int(away_stats['wins']))
    away_xg_per_match = away_stats['xg'] / max(away_stats['matches'], 1)
    away_xga_per_match = away_stats['xga'] / max(away_stats['matches'], 1)
    st.metric("xG/match", f"{away_xg_per_match:.2f}")
    st.metric("xGA/match", f"{away_xga_per_match:.2f}")

with col3:
    home_attack_reg, away_attack_reg = calculate_regression_factors(
        home_stats, away_stats, regression_factor
    )
    
    home_xg, away_xg = calculate_expected_goals(
        home_stats, away_stats, home_attack_reg, away_attack_reg
    )
    
    st.subheader("🎯 Expected Goals (Main Model)")
    
    # Use Plotly for correct visualization
    xg_chart = create_expected_goals_chart(home_xg, away_xg, home_team, away_team)
    st.plotly_chart(xg_chart, use_container_width=True)
    
    col_xg1, col_xg2, col_xg3 = st.columns(3)
    with col_xg1:
        st.metric("Home xG", f"{home_xg:.2f}")
    with col_xg2:
        st.metric("Away xG", f"{away_xg:.2f}")
    with col_xg3:
        total_xg = home_xg + away_xg
        st.metric("Total xG", f"{total_xg:.2f}")
    
    if total_xg > 2.6:
        st.success(f"📈 Over bias: Total xG = {total_xg:.2f} > 2.6")
    elif total_xg < 2.3:
        st.info(f"📉 Under bias: Total xG = {total_xg:.2f} < 2.3")

# ========== PHASE 2: PROBABILITY CALCULATIONS ==========
st.divider()
st.header("📈 Probability Calculations")

prob_matrix = create_probability_matrix(home_xg, away_xg)
home_win_prob, draw_prob, away_win_prob = calculate_outcome_probabilities(prob_matrix)
over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob = calculate_betting_markets(prob_matrix)

# ========== PHASE 3: DEFENSIVE GAP MODEL ==========
if enable_defensive_model:
    st.divider()
    st.header("🛡️ Defensive Gap Analysis (Supporting Model)")
    
    defensive_model = DefensiveGapModel(league_avg_xga, league_std_xga)
    defensive_analysis = defensive_model.analyze_match(home_stats, away_stats)
    
    col_def1, col_def2, col_def3 = st.columns(3)
    
    with col_def1:
        signal = defensive_analysis['signal']
        if "OVER" in signal:
            st.success(f"**Signal:** {signal}")
        elif "UNDER" in signal:
            st.info(f"**Signal:** {signal}")
        else:
            st.warning(f"**Signal:** {signal}")
        
        st.metric("Gap Score", f"{defensive_analysis['gap_score']:.2f}")
    
    with col_def2:
        st.metric("Home Defense", f"{defensive_analysis['home_def_score']:.2f}σ",
                 delta="Strong" if defensive_analysis['home_def_score'] < 0 else "Weak")
    
    with col_def3:
        st.metric("Away Defense", f"{defensive_analysis['away_def_score']:.2f}σ",
                 delta="Strong" if defensive_analysis['away_def_score'] < 0 else "Weak")
    
    st.info(defensive_analysis['explanation'])

# ========== PHASE 4: CONSENSUS ANALYSIS ==========
if enable_defensive_model and show_consensus:
    st.divider()
    st.header("✅ Consensus Analysis")
    
    bet_rec, bet_reason, bet_icon = defensive_model.get_bet_recommendation(
        defensive_analysis, 
        over_25_prob * 100, 
        under_25_prob * 100
    )
    
    col_cons1, col_cons2, col_cons3 = st.columns([1, 2, 1])
    
    with col_cons1:
        st.subheader("Main Model")
        st.metric("Over 2.5", f"{over_25_prob*100:.1f}%")
        st.metric("Under 2.5", f"{under_25_prob*100:.1f}%")
    
    with col_cons2:
        st.subheader("Consensus")
        
        if bet_rec == "STRONG_BET":
            st.success(f"🎯 **STRONG BET SIGNAL** {bet_icon}")
        elif bet_rec == "MODERATE_BET":
            st.info(f"📊 **MODERATE BET** {bet_icon}")
        elif bet_rec == "CONSIDER_BET":
            st.info(f"🤔 **CONSIDER BET** {bet_icon}")
        elif bet_rec == "MAIN_MODEL_ONLY":
            st.warning(f"⚠️ **MAIN MODEL ONLY** {bet_icon}")
        elif bet_rec == "CONFLICT_AVOID":
            st.error(f"❌ **AVOID BET** {bet_icon}")
        else:
            st.warning(f"⚪ **NO CLEAR SIGNAL** {bet_icon}")
        
        st.write(bet_reason)

# ========== LAYER 5: SCORE PROBABILITIES ==========
with st.expander("🎯 Most Likely Scores", expanded=True):
    score_probs = []
    for i in range(min(6, prob_matrix.shape[0])):
        for j in range(min(6, prob_matrix.shape[1])):
            prob = prob_matrix[i, j]
            if prob > 0.001:
                score_probs.append(((i, j), prob))
    
    score_probs.sort(key=lambda x: x[1], reverse=True)
    
    cols = st.columns(5)
    for idx, ((home_goals, away_goals), prob) in enumerate(score_probs[:5]):
        with cols[idx]:
            st.metric(
                label=f"{home_goals}-{away_goals}",
                value=f"{prob*100:.1f}%",
                delta="Most Likely" if idx == 0 else None
            )
    
    if score_probs:
        most_likely_score, most_likely_prob = score_probs[0]
        st.success(f"**Most Likely Score:** {most_likely_score[0]}-{most_likely_score[1]} ({(most_likely_prob*100):.1f}%)")

# ========== LAYER 6: OUTCOME PROBABILITIES ==========
with st.expander("📊 Match Outcome Probabilities", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"{home_team} Win", f"{home_win_prob*100:.1f}%")
        st.progress(home_win_prob)
    
    with col2:
        st.metric("Draw", f"{draw_prob*100:.1f}%")
        st.progress(draw_prob)
    
    with col3:
        st.metric(f"{away_team} Win", f"{away_win_prob*100:.1f}%")
        st.progress(away_win_prob)
    
    # Use Plotly for correct visualization
    outcome_chart = create_correct_outcome_chart(home_win_prob, draw_prob, away_win_prob, home_team, away_team)
    st.plotly_chart(outcome_chart, use_container_width=True)

# ========== LAYER 7: BETTING MARKETS ==========
with st.expander("💰 Betting Markets", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Over/Under 2.5 Goals")
        st.metric("Over 2.5", f"{over_25_prob*100:.1f}%")
        st.progress(over_25_prob)
        st.metric("Under 2.5", f"{under_25_prob*100:.1f}%")
        st.progress(under_25_prob)
    
    with col2:
        st.subheader("Both Teams to Score")
        st.metric("Yes", f"{btts_yes_prob*100:.1f}%")
        st.progress(btts_yes_prob)
        st.metric("No", f"{btts_no_prob*100:.1f}%")
        st.progress(btts_no_prob)

# ========== OUTPUT FORMATS ==========
st.divider()
st.header("📤 Export & Share")

summary = f"""
⚽ DUAL-MODEL PREDICTION: {home_team} vs {away_team}
League: {selected_league}

📊 MAIN MODEL (xG Regression):
• Expected Goals: {home_team} {home_xg:.2f} - {away_team} {away_xg:.2f}
• Total xG: {home_xg + away_xg:.2f}
• Over 2.5: {over_25_prob*100:.1f}%
• Under 2.5: {under_25_prob*100:.1f}%

🛡️ DEFENSIVE GAP MODEL:
• Signal: {defensive_analysis['signal'] if enable_defensive_model else 'N/A'}
• Confidence: {defensive_analysis['confidence'] if enable_defensive_model else 'N/A'}
• Gap Score: {defensive_analysis['gap_score']:.2f} if enable_defensive_model else 'N/A'

✅ CONSENSUS ANALYSIS:
• Recommendation: {bet_rec if enable_defensive_model else 'N/A'}
• Reason: {bet_reason if enable_defensive_model else 'N/A'}

📈 Most Likely Score: {score_probs[0][0][0] if score_probs else 'N/A'}-{score_probs[0][0][1] if score_probs else 'N/A'} ({(score_probs[0][1]*100 if score_probs else 0):.1f}%)

🏆 Outcome Probabilities:
• {home_team} Win: {home_win_prob*100:.1f}%
• Draw: {draw_prob*100:.1f}%
• {away_team} Win: {away_win_prob*100:.1f}%

📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Regression Factor: {regression_factor}
"""

st.code(summary, language="text")

col_export1, col_export2 = st.columns(2)

with col_export1:
    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name=f"prediction_{home_team}_vs_{away_team}.txt",
        mime="text/plain"
    )

# ========== FOOTER ==========
st.divider()
footer_text = f"⚡ Dual-model prediction system with xG regression + defensive gap analysis"
if enable_defensive_model:
    footer_text += f" | Consensus: {bet_rec}"
footer_text += f" | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
st.caption(footer_text)
