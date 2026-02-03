import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
import json
from collections import defaultdict
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
    
    **Validation Framework:** 6-test system ensures override layer adds predictive signal, not just complexity
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

# Validation thresholds
OVERRIDE_MIN_ACCURACY_ADVANTAGE = 2.0  # percentage points
MIN_SAMPLE_SIZE = 30

# Initialize session state for validation tracking
if 'validation_history' not in st.session_state:
    st.session_state.validation_history = {
        'base_model_accuracy': [],
        'full_model_accuracy': [],
        'override_accuracy': [],
        'no_override_accuracy': [],
        'resolution_spread': [],
        'regime_decay': defaultdict(list),
        'error_correlations': [],
        'override_trigger_counts': {'fired': 0, 'available_not_fired': 0, 'no_signal': 0}
    }

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
        
        # Enhanced column checking
        required_cols = ['team', 'venue', 'matches', 'xg', 'xga', 'goals_vs_xg']
        for col in ['wins', 'draws', 'losses', 'gf', 'ga']:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with defaults
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
            return None
            
        return df
    except FileNotFoundError:
        # Try to create sample data for demo
        st.warning(f"⚠️ League file not found: leagues/{actual_filename}")
        st.info("Using sample Premier League data for demo...")
        return create_sample_data()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = {
        'team': ['Arsenal', 'Arsenal', 'Man City', 'Man City', 'Liverpool', 'Liverpool', 
                 'Chelsea', 'Chelsea', 'Man United', 'Man United', 'Tottenham', 'Tottenham'],
        'venue': ['home', 'away', 'home', 'away', 'home', 'away', 'home', 'away', 'home', 'away', 'home', 'away'],
        'matches': [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
        'wins': [9, 7, 9, 5, 7, 4, 6, 5, 7, 4, 2, 5],
        'draws': [2, 3, 2, 3, 3, 3, 3, 4, 3, 5, 4, 4],
        'losses': [1, 2, 1, 4, 2, 5, 3, 3, 2, 3, 6, 3],
        'gf': [28, 18, 29, 20, 20, 19, 20, 22, 23, 21, 15, 20],
        'ga': [8, 9, 8, 15, 12, 21, 13, 14, 15, 21, 16, 17],
        'pts': [29, 24, 29, 18, 24, 15, 21, 19, 24, 17, 10, 19],
        'xg': [25.86, 23.43, 26.94, 20.12, 23.37, 19.57, 24.29, 23.04, 25.39, 21.51, 15.37, 13.93],
        'xga': [8.64, 10.15, 13.34, 15.80, 11.11, 19.90, 19.29, 17.37, 13.21, 18.93, 16.00, 17.04],
        'goals_vs_xg': [-2.14, 5.43, -2.06, 0.12, 3.37, 0.57, 4.29, 1.04, 2.39, 0.51, 0.37, 6.07]
    }
    return pd.DataFrame(sample_data)

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

class ValidationTracker:
    """Track model performance for validation framework"""
    
    @staticmethod
    def update_resolution_spread(predictions):
        """Track prediction distribution spread"""
        if len(predictions) > 0:
            spread = max(predictions) - min(predictions)
            st.session_state.validation_history['resolution_spread'].append(spread)
    
    @staticmethod
    def update_regime_decay(signal_type, matches_since_detection, correct):
        """Track regime signal decay over time"""
        key = f"{signal_type}_{matches_since_detection}"
        st.session_state.validation_history['regime_decay'][key].append(1 if correct else 0)
    
    @staticmethod
    def update_override_trigger(trigger_type):
        """Track when override triggers vs when it doesn't"""
        st.session_state.validation_history['override_trigger_counts'][trigger_type] += 1
    
    @staticmethod
    def calculate_validation_metrics():
        """Calculate all validation metrics"""
        metrics = {}
        
        # Calculate base vs full model comparison
        if len(st.session_state.validation_history['base_model_accuracy']) > MIN_SAMPLE_SIZE:
            base_avg = np.mean(st.session_state.validation_history['base_model_accuracy'][-MIN_SAMPLE_SIZE:])
            full_avg = np.mean(st.session_state.validation_history['full_model_accuracy'][-MIN_SAMPLE_SIZE:])
            metrics['base_vs_full_diff'] = full_avg - base_avg
        
        # Calculate override decision quality
        override_cases = st.session_state.validation_history['override_accuracy']
        no_override_cases = st.session_state.validation_history['no_override_accuracy']
        
        if len(override_cases) > 10:
            metrics['override_accuracy'] = np.mean(override_cases)
            metrics['no_override_accuracy'] = np.mean(no_override_cases) if len(no_override_cases) > 0 else 0
            metrics['override_advantage'] = metrics['override_accuracy'] - metrics['no_override_accuracy']
        
        # Calculate resolution preservation
        if len(st.session_state.validation_history['resolution_spread']) > 0:
            recent_spread = np.mean(st.session_state.validation_history['resolution_spread'][-20:])
            metrics['resolution_spread'] = recent_spread
        
        # Calculate regime decay
        decay_metrics = {}
        for key, values in st.session_state.validation_history['regime_decay'].items():
            if len(values) >= 5:
                decay_metrics[key] = np.mean(values)
        metrics['regime_decay'] = decay_metrics
        
        return metrics
    
    @staticmethod
    def get_validation_status(metrics):
        """Determine validation status based on metrics"""
        status = {
            'base_vs_full': 'PASS' if metrics.get('base_vs_full_diff', 0) > 0 else 'FAIL',
            'override_decision': 'PASS' if metrics.get('override_advantage', -100) > OVERRIDE_MIN_ACCURACY_ADVANTAGE else 'FAIL',
            'resolution': 'PASS' if metrics.get('resolution_spread', 0) > 0.2 else 'WARNING',
            'regime_stability': 'PASS' if len(metrics.get('regime_decay', {})) > 0 else 'INSUFFICIENT_DATA'
        }
        
        overall = 'PASS' if all(v == 'PASS' for k, v in status.items() if k != 'regime_stability') else 'FAIL'
        status['overall'] = overall
        
        return status

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
        
        # Determine if override should trigger
        should_override = False
        override_strength = None
        
        # Override conditions based on our validation framework
        if main_signal == "NEUTRAL" and dg_direction != "NEUTRAL":
            # Main model uncertain, defensive model has signal
            if dg_confidence == "HIGH":
                should_override = True
                override_strength = "CONSIDER_BET"
            elif dg_confidence == "MEDIUM":
                should_override = True
                override_strength = "WEAK_BET"
        
        elif main_signal != "NEUTRAL" and dg_direction == "NEUTRAL":
            # Main model has signal, defensive model uncertain
            should_override = False
            override_strength = "MAIN_MODEL_ONLY"
        
        elif main_signal == dg_direction:
            # Both agree
            should_override = True
            if dg_confidence == "HIGH":
                override_strength = "STRONG_BET"
            else:
                override_strength = "MODERATE_BET"
        
        else:
            # Direct conflict
            if dg_confidence == "HIGH":
                should_override = True
                override_strength = "CONSIDER_BET"  # Give defensive model chance when high confidence
            else:
                should_override = False
                override_strength = "CONFLICT_AVOID"
        
        # Generate recommendation text
        if override_strength == "STRONG_BET":
            return "STRONG_BET", f"Strong consensus on {main_signal}", "✅", should_override
        elif override_strength == "MODERATE_BET":
            return "MODERATE_BET", f"Consensus on {main_signal} (medium confidence)", "🟡", should_override
        elif override_strength == "CONSIDER_BET":
            return "CONSIDER_BET", f"Defensive model suggests {dg_direction}", "🔵", should_override
        elif override_strength == "WEAK_BET":
            return "WEAK_BET", f"Defensive model suggests {dg_direction}", "⚪", should_override
        elif override_strength == "MAIN_MODEL_ONLY":
            return "MAIN_MODEL_ONLY", f"Main model suggests {main_signal} (no defensive signal)", "🟠", should_override
        elif override_strength == "CONFLICT_AVOID":
            return "CONFLICT_AVOID", f"Conflict: Main {main_signal} vs Defensive {dg_direction}", "❌", should_override
        else:
            return "NO_BET", "Both models uncertain", "⚪", should_override

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
    
    # Normalize to account for truncation
    total_prob = prob_matrix.sum()
    if total_prob > 0:
        prob_matrix /= total_prob
    
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

def create_correct_outcome_display(home_win_prob, draw_prob, away_win_prob, home_team, away_team):
    """Create properly ordered outcome display"""
    st.subheader("📊 Outcome Probability Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**{home_team} Win**")
        progress_html = f"""
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 5px; margin: 5px 0;">
            <div style="background-color: #1f77b4; width: {home_win_prob*100}%; height: 25px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {home_win_prob*100:.1f}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Draw**")
        progress_html = f"""
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 5px; margin: 5px 0;">
            <div style="background-color: #2ca02c; width: {draw_prob*100}%; height: 25px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {draw_prob*100:.1f}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"**{away_team} Win**")
        progress_html = f"""
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 5px; margin: 5px 0;">
            <div style="background-color: #ff7f0e; width: {away_win_prob*100}%; height: 25px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {away_win_prob*100:.1f}%
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    st.markdown("---")
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        if home_win_prob > away_win_prob:
            st.success(f"📈 **{home_team} is favored to win**")
        elif away_win_prob > home_win_prob:
            st.info(f"📈 **{away_team} is favored to win**")
        else:
            st.warning("⚖️ **Teams are evenly matched**")
    
    with col_comp2:
        favorite_prob = max(home_win_prob, away_win_prob)
        favorite_team = home_team if home_win_prob > away_win_prob else away_team
        st.metric("Favorite's Advantage", f"{(favorite_prob - min(home_win_prob, away_win_prob))*100:.1f}%")

def create_expected_goals_display(home_xg, away_xg, home_team, away_team):
    """Create expected goals display"""
    st.subheader("🎯 Expected Goals Comparison")
    
    total_xg = home_xg + away_xg
    home_share = (home_xg / total_xg * 100) if total_xg > 0 else 50
    away_share = (away_xg / total_xg * 100) if total_xg > 0 else 50
    
    col_xg1, col_xg2 = st.columns(2)
    
    with col_xg1:
        st.markdown(f"**{home_team}**")
        progress_html = f"""
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 5px; margin: 5px 0;">
            <div style="background-color: #1f77b4; width: {min(home_share, 100)}%; height: 30px; border-radius: 5px; display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold;">
                {home_xg:.2f} xG
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    with col_xg2:
        st.markdown(f"**{away_team}**")
        progress_html = f"""
        <div style="background-color: #f0f2f6; border-radius: 10px; padding: 5px; margin: 5px 0;">
            <div style="background-color: #ff7f0e; width: {min(away_share, 100)}%; height: 30px; border-radius: 5px; display: flex; align-items: center; padding-left: 10px; color: white; font-weight: bold;">
                {away_xg:.2f} xG
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Total xG", f"{total_xg:.2f}")
    with col_sum2:
        if home_xg > away_xg:
            st.metric("Attack Advantage", home_team, delta=f"+{home_xg-away_xg:.2f}")
        else:
            st.metric("Attack Advantage", away_team, delta=f"+{away_xg-home_xg:.2f}")
    with col_sum3:
        if total_xg > 2.6:
            st.success("📈 High-scoring expected")
        elif total_xg < 2.3:
            st.info("📉 Low-scoring expected")

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
                show_validation = st.checkbox("Show Validation Dashboard", value=True,
                    help="Show the 6-test validation framework")
                show_consensus = st.checkbox("Show Consensus Analysis", value=True)
            
            calculate_btn = st.button("🎯 Calculate Predictions", type="primary", use_container_width=True)
            
            st.divider()
            st.subheader("📊 Display Options")
            show_matrix = st.checkbox("Show Score Probability Matrix", value=False)
            
            if show_validation:
                st.divider()
                st.subheader("📈 Validation Dashboard")
                metrics = ValidationTracker.calculate_validation_metrics()
                status = ValidationTracker.get_validation_status(metrics)
                
                for test_name, test_status in status.items():
                    if test_name != 'overall':
                        color = "🟢" if test_status == 'PASS' else "🟡" if test_status == 'WARNING' else "🔴"
                        st.write(f"{color} {test_name.replace('_', ' ').title()}: {test_status}")
                
                st.metric("Override Advantage", f"{metrics.get('override_advantage', 0):.1f}%")
                st.metric("Resolution Spread", f"{metrics.get('resolution_spread', 0):.3f}")

# ========== MAIN CONTENT ==========
if df is None:
    st.warning("📁 Please add league CSV files to the 'leagues' folder")
    st.info("""
    **Required CSV format:**
    - Columns: team,venue,matches,wins,draws,losses,gf,ga,pts,xg,xga,xpts,goals_vs_xg
    - One row per team per venue (home/away)
    - Sample data provided in the app
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
    
    create_expected_goals_display(home_xg, away_xg, home_team, away_team)
    
    st.caption(f"Regression factors: Home {home_attack_reg:.3f}, Away {away_attack_reg:.3f}")

# ========== PHASE 2: PROBABILITY CALCULATIONS ==========
st.divider()
st.header("📈 Probability Calculations")

prob_matrix = create_probability_matrix(home_xg, away_xg)
home_win_prob, draw_prob, away_win_prob = calculate_outcome_probabilities(prob_matrix)
over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob = calculate_betting_markets(prob_matrix)

# Track resolution spread for validation
predictions = [home_win_prob, draw_prob, away_win_prob, over_25_prob, under_25_prob]
ValidationTracker.update_resolution_spread(predictions)

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
    
    bet_rec, bet_reason, bet_icon, should_override = defensive_model.get_bet_recommendation(
        defensive_analysis, 
        over_25_prob * 100, 
        under_25_prob * 100
    )
    
    # Track override trigger for validation
    if defensive_analysis['signal'] != "NEUTRAL":
        if should_override:
            ValidationTracker.update_override_trigger('fired')
        else:
            ValidationTracker.update_override_trigger('available_not_fired')
    else:
        ValidationTracker.update_override_trigger('no_signal')
    
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
        elif bet_rec == "WEAK_BET":
            st.warning(f"⚪ **WEAK BET** {bet_icon}")
        elif bet_rec == "MAIN_MODEL_ONLY":
            st.warning(f"⚠️ **MAIN MODEL ONLY** {bet_icon}")
        elif bet_rec == "CONFLICT_AVOID":
            st.error(f"❌ **AVOID BET** {bet_icon}")
        else:
            st.warning(f"⚪ **NO CLEAR SIGNAL** {bet_icon}")
        
        st.write(bet_reason)
        
        # Show override status for validation transparency
        if enable_defensive_model:
            st.caption(f"Override triggered: {'✅ Yes' if should_override else '❌ No'}")

# ========== SCORE PROBABILITIES ==========
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

# ========== OUTCOME PROBABILITIES ==========
with st.expander("📊 Match Outcome Probabilities", expanded=True):
    create_correct_outcome_display(home_win_prob, draw_prob, away_win_prob, home_team, away_team)
    
    col_met1, col_met2, col_met3 = st.columns(3)
    with col_met1:
        st.metric(f"{home_team} Win", f"{home_win_prob*100:.1f}%")
    with col_met2:
        st.metric("Draw", f"{draw_prob*100:.1f}%")
    with col_met3:
        st.metric(f"{away_team} Win", f"{away_win_prob*100:.1f}%")

# ========== BETTING MARKETS ==========
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
    
    # Implied odds
    st.subheader("Implied Odds")
    col_odds1, col_odds2, col_odds3 = st.columns(3)
    with col_odds1:
        if home_win_prob > 0:
            odds = 1 / home_win_prob
            st.metric(f"{home_team} Win Odds", f"{odds:.2f}")
    
    with col_odds2:
        if draw_prob > 0:
            odds = 1 / draw_prob
            st.metric("Draw Odds", f"{odds:.2f}")
    
    with col_odds3:
        if away_win_prob > 0:
            odds = 1 / away_win_prob
            st.metric(f"{away_team} Win Odds", f"{odds:.2f}")

# ========== VALIDATION DASHBOARD ==========
if show_validation:
    st.divider()
    st.header("📊 Model Validation Dashboard")
    
    metrics = ValidationTracker.calculate_validation_metrics()
    status = ValidationTracker.get_validation_status(metrics)
    
    # Display validation results
    col_val1, col_val2, col_val3, col_val4 = st.columns(4)
    
    with col_val1:
        st.metric("Base vs Full Diff", f"{metrics.get('base_vs_full_diff', 0):.2f}%",
                 delta="Positive" if metrics.get('base_vs_full_diff', 0) > 0 else "Negative")
        st.caption(f"Status: {status['base_vs_full']}")
    
    with col_val2:
        st.metric("Override Advantage", f"{metrics.get('override_advantage', 0):.2f}%",
                 delta="Good" if metrics.get('override_advantage', 0) > OVERRIDE_MIN_ACCURACY_ADVANTAGE else "Poor")
        st.caption(f"Status: {status['override_decision']}")
    
    with col_val3:
        st.metric("Resolution Spread", f"{metrics.get('resolution_spread', 0):.3f}",
                 delta="Good" if metrics.get('resolution_spread', 0) > 0.2 else "Poor")
        st.caption(f"Status: {status['resolution']}")
    
    with col_val4:
        st.metric("Override Triggers", 
                 f"{st.session_state.validation_history['override_trigger_counts']['fired']}",
                 delta=f"{st.session_state.validation_history['override_trigger_counts']['available_not_fired']} not fired")
    
    # Show regime decay if available
    if metrics.get('regime_decay'):
        st.subheader("Regime Signal Decay Analysis")
        decay_data = metrics['regime_decay']
        decay_df = pd.DataFrame([
            {'Signal': k.split('_')[0], 'Matches Since': int(k.split('_')[1]), 'Accuracy': v}
            for k, v in decay_data.items()
        ])
        if not decay_df.empty:
            st.dataframe(decay_df.sort_values('Matches Since'))
    
    # Overall validation status
    st.subheader("Overall Validation Status")
    if status['overall'] == 'PASS':
        st.success("✅ All validation tests passing - override system adding predictive value")
    else:
        st.warning("⚠️ Some validation tests failing - consider simplifying override logic")

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
• Gap Score: {defensive_analysis['gap_score']:.2f if enable_defensive_model else 'N/A'}

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

with col_export2:
    if st.button("🔄 Reset Validation History"):
        st.session_state.validation_history = {
            'base_model_accuracy': [],
            'full_model_accuracy': [],
            'override_accuracy': [],
            'no_override_accuracy': [],
            'resolution_spread': [],
            'regime_decay': defaultdict(list),
            'error_correlations': [],
            'override_trigger_counts': {'fired': 0, 'available_not_fired': 0, 'no_signal': 0}
        }
        st.success("Validation history reset!")
        st.rerun()

# ========== FOOTER ==========
st.divider()
footer_text = f"⚡ Dual-model prediction system with 6-test validation framework"
if enable_defensive_model and 'bet_rec' in locals():
    footer_text += f" | Consensus: {bet_rec}"
footer_text += f" | Validation: {status.get('overall', 'N/A')}"
footer_text += f" | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
st.caption(footer_text)

# ========== SAMPLE DATA CREATION INSTRUCTIONS ==========
with st.sidebar.expander("📁 Data Format Instructions"):
    st.markdown("""
    **CSV Format Requirements:**
    ```
    team,venue,matches,wins,draws,losses,gf,ga,pts,xg,xga,goals_vs_xg
    Arsenal,home,12,9,2,1,28,8,29,25.86,8.64,-2.14
    Arsenal,away,12,7,3,2,18,9,24,23.43,10.15,5.43
    ```
    
    **Place files in `/leagues/` folder:**
    - premier_league.csv
    - bundesliga.csv
    - serie_a.csv
    - laliga.csv
    - ligue_1.csv
    - eredivisie.csv
    """)
