import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
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
    Advanced xG prediction system with defensive confirmation layer.
    **Confirmation layer modulates confidence in predictions - does NOT override them.**
""")

# Constants
MAX_GOALS = 8
REG_BASE_FACTOR = 0.75
REG_MATCH_THRESHOLD = 5
MAX_REGRESSION = 0.3

# Defensive gap thresholds
STRONG_OVER_THRESHOLD = 1.0
STRONG_UNDER_THRESHOLD = -1.0

# Confidence thresholds
CONFIDENCE_HIGH_THRESHOLD = 0.7
CONFIDENCE_MEDIUM_THRESHOLD = 0.55

# Initialize session state for validation tracking
if 'validation_history' not in st.session_state:
    st.session_state.validation_history = {
        'main_model_accuracy': [],
        'confidence_calibration': defaultdict(list),
        'resolution_spread': [],
        'agreement_tracking': defaultdict(list),
        'match_count': 0
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
                df[col] = 0
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
            return None
            
        return df
    except FileNotFoundError:
        st.warning(f"⚠️ League file not found: leagues/{actual_filename}")
        # Create sample data from the CSV provided in the prompt
        sample_data = {
            'team': ['Arsenal', 'Arsenal', 'Chelsea', 'Chelsea', 'Manchester City', 'Manchester City',
                     'Liverpool', 'Liverpool', 'Manchester United', 'Manchester United', 'Tottenham', 'Tottenham'],
            'venue': ['home', 'away', 'home', 'away', 'home', 'away', 'home', 'away', 'home', 'away', 'home', 'away'],
            'matches': [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
            'wins': [9, 7, 6, 5, 9, 5, 7, 4, 7, 4, 2, 5],
            'draws': [2, 3, 3, 4, 2, 3, 3, 3, 3, 5, 4, 4],
            'losses': [1, 2, 3, 3, 1, 4, 2, 5, 2, 3, 6, 3],
            'gf': [28, 18, 20, 22, 29, 20, 20, 19, 23, 21, 15, 20],
            'ga': [8, 9, 13, 14, 8, 15, 12, 21, 15, 21, 16, 17],
            'pts': [29, 24, 21, 19, 29, 18, 24, 15, 24, 17, 10, 19],
            'xg': [25.86, 23.43, 24.29, 23.04, 26.94, 20.12, 23.37, 19.57, 25.39, 21.51, 15.37, 13.93],
            'xga': [8.64, 10.15, 19.29, 17.37, 13.34, 15.80, 11.11, 19.90, 13.21, 18.93, 16.00, 17.04],
            'goals_vs_xg': [-2.14, 5.43, 4.29, 1.04, -2.06, 0.12, 3.37, 0.57, 2.39, 0.51, 0.37, 6.07]
        }
        return pd.DataFrame(sample_data)
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

class ValidationTracker:
    """Track confidence calibration for the confirmation layer"""
    
    @staticmethod
    def update_confidence_calibration(confidence_level, prediction_correct):
        """Track how well confidence levels predict accuracy"""
        st.session_state.validation_history['confidence_calibration'][confidence_level].append(
            1 if prediction_correct else 0
        )
    
    @staticmethod
    def update_resolution_spread(predictions):
        """Track prediction distribution spread"""
        if len(predictions) > 0:
            spread = np.std(predictions)  # Standard deviation of predictions
            st.session_state.validation_history['resolution_spread'].append(spread)
    
    @staticmethod
    def update_agreement_tracking(main_prediction, confirmation_signal, correct):
        """Track agreement between main model and confirmation layer"""
        # Determine if they agree (both suggest same direction)
        if main_prediction >= 0.5:  # Main predicts Over
            agreement = "AGREE" if "OVER" in confirmation_signal else "DISAGREE" if "UNDER" in confirmation_signal else "NEUTRAL"
        else:  # Main predicts Under
            agreement = "AGREE" if "UNDER" in confirmation_signal else "DISAGREE" if "OVER" in confirmation_signal else "NEUTRAL"
        
        st.session_state.validation_history['agreement_tracking'][agreement].append(
            1 if correct else 0
        )
    
    @staticmethod
    def increment_match_count():
        """Increment total match count"""
        st.session_state.validation_history['match_count'] += 1
    
    @staticmethod
    def calculate_validation_metrics():
        """Calculate all validation metrics"""
        metrics = {}
        
        # Calculate confidence calibration
        calibration_data = {}
        for confidence_level, results in st.session_state.validation_history['confidence_calibration'].items():
            if len(results) >= 3:  # Minimum sample
                calibration_data[confidence_level] = np.mean(results)
        metrics['confidence_calibration'] = calibration_data
        
        # Calculate resolution
        if len(st.session_state.validation_history['resolution_spread']) > 0:
            metrics['resolution_spread'] = np.mean(st.session_state.validation_history['resolution_spread'][-20:])
        
        # Calculate agreement performance
        agreement_performance = {}
        for agreement_type, results in st.session_state.validation_history['agreement_tracking'].items():
            if len(results) >= 3:
                agreement_performance[agreement_type] = np.mean(results)
        metrics['agreement_performance'] = agreement_performance
        
        # Calculate overall stats
        metrics['total_matches'] = st.session_state.validation_history['match_count']
        
        return metrics
    
    @staticmethod
    def get_validation_status(metrics):
        """Determine validation status based on metrics"""
        status = {
            'confidence_calibration': 'INSUFFICIENT_DATA',
            'resolution': 'PASS' if metrics.get('resolution_spread', 0) > 0.1 else 'WARNING',
            'agreement_tracking': 'INSUFFICIENT_DATA'
        }
        
        # Check confidence calibration
        calibration = metrics.get('confidence_calibration', {})
        if calibration:
            # Ideally, HIGH confidence should have >65% accuracy, MEDIUM 55-65%, LOW <55%
            valid_calibration = True
            for level, accuracy in calibration.items():
                if level == 'HIGH' and accuracy < 0.6:
                    valid_calibration = False
                elif level == 'MEDIUM' and (accuracy < 0.5 or accuracy > 0.7):
                    valid_calibration = False
            
            status['confidence_calibration'] = 'PASS' if valid_calibration else 'NEEDS_CALIBRATION'
        
        # Check agreement tracking
        agreement = metrics.get('agreement_performance', {})
        if agreement:
            # AGREE should have higher accuracy than DISAGREE
            agree_acc = agreement.get('AGREE', 0)
            disagree_acc = agreement.get('DISAGREE', 0)
            if agree_acc > disagree_acc and len(agreement) >= 2:
                status['agreement_tracking'] = 'PASS'
            else:
                status['agreement_tracking'] = 'NEEDS_IMPROVEMENT'
        
        # Overall status
        if all(v == 'PASS' for v in status.values() if v != 'INSUFFICIENT_DATA'):
            status['overall'] = 'PASS'
        elif any(v == 'NEEDS_CALIBRATION' for v in status.values()):
            status['overall'] = 'NEEDS_CALIBRATION'
        else:
            status['overall'] = 'INSUFFICIENT_DATA'
        
        return status

class DefensiveConfirmationModel:
    """Confirmation layer that assesses confidence in main model predictions"""
    
    def __init__(self, league_avg_xga, league_std_xga):
        self.league_avg_xga = league_avg_xga
        self.league_std_xga = league_std_xga
    
    def analyze_defensive_gap(self, home_stats, away_stats):
        """Analyze match for defensive gap - returns information only"""
        home_xga_per_match = home_stats['xga'] / max(home_stats['matches'], 1)
        away_xga_per_match = away_stats['xga'] / max(away_stats['matches'], 1)
        
        home_def_score = (home_xga_per_match - self.league_avg_xga) / max(self.league_std_xga, 0.1)
        away_def_score = (away_xga_per_match - self.league_avg_xga) / max(self.league_std_xga, 0.1)
        
        match_gap = home_def_score + away_def_score
        
        if match_gap > STRONG_OVER_THRESHOLD:
            signal = "STRONG_OVER"
            confidence = "HIGH"
            explanation = f"Both teams have weak defenses ({match_gap:.2f}σ above league avg)"
        elif match_gap < STRONG_UNDER_THRESHOLD:
            signal = "STRONG_UNDER"
            confidence = "HIGH"
            explanation = f"Both teams have strong defenses ({abs(match_gap):.2f}σ below league avg)"
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
    
    def assess_confidence(self, main_over_prob, defensive_signal):
        """Assess confidence level based on agreement between main model and confirmation"""
        # Main model direction
        main_direction = "OVER" if main_over_prob > 50 else "UNDER"
        
        # Defensive signal direction
        if "OVER" in defensive_signal['signal']:
            defensive_direction = "OVER"
        elif "UNDER" in defensive_signal['signal']:
            defensive_direction = "UNDER"
        else:
            defensive_direction = "NEUTRAL"
        
        # Determine confidence level
        if defensive_direction == "NEUTRAL":
            confidence_level = "MEDIUM"
            reason = "Confirmation layer neutral"
        elif main_direction == defensive_direction:
            if defensive_signal['confidence'] == "HIGH":
                confidence_level = "HIGH"
                reason = f"Strong confirmation for {main_direction}"
            else:
                confidence_level = "MEDIUM"
                reason = f"Moderate confirmation for {main_direction}"
        else:
            if defensive_signal['confidence'] == "HIGH":
                confidence_level = "LOW"
                reason = f"Strong defensive signal contradicts main {main_direction}"
            else:
                confidence_level = "MEDIUM"
                reason = f"Mild defensive disagreement with main {main_direction}"
        
        return confidence_level, reason

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
    st.subheader("📊 Match Outcome Probabilities")
    
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
        advantage = (favorite_prob - min(home_win_prob, away_win_prob)) * 100
        st.metric("Favorite's Advantage", f"{advantage:.1f}%")

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
            
            with st.expander("⚙️ Confirmation Layer Settings"):
                enable_confirmation_layer = st.checkbox("Enable Defensive Confirmation", value=True,
                    help="Use defensive analysis to assess confidence in predictions")
                show_validation = st.checkbox("Show Validation Dashboard", value=True,
                    help="Show confidence calibration tracking")
            
            calculate_btn = st.button("🎯 Calculate Predictions", type="primary", use_container_width=True)
            
            st.divider()
            st.subheader("📊 Display Options")
            show_matrix = st.checkbox("Show Score Probability Matrix", value=False)
            
            if show_validation:
                st.divider()
                st.subheader("📈 Validation Dashboard")
                metrics = ValidationTracker.calculate_validation_metrics()
                status = ValidationTracker.get_validation_status(metrics)
                
                st.write("**Confidence Calibration:**")
                for level, accuracy in metrics.get('confidence_calibration', {}).items():
                    st.write(f"  {level}: {accuracy:.1%} accuracy")
                
                st.write(f"**Resolution Spread:** {metrics.get('resolution_spread', 0):.3f}")
                st.write(f"**Total Matches Analyzed:** {metrics.get('total_matches', 0)}")
                
                st.write("**Validation Status:**")
                for test_name, test_status in status.items():
                    if test_name != 'overall':
                        color = "🟢" if test_status == 'PASS' else "🟡" if test_status in ['WARNING', 'NEEDS_CALIBRATION', 'NEEDS_IMPROVEMENT'] else "⚪"
                        st.write(f"{color} {test_name.replace('_', ' ').title()}: {test_status}")

# ========== MAIN CONTENT ==========
if df is None:
    st.warning("📁 Please add league CSV files to the 'leagues' folder")
    st.info("""
    **Required CSV format:**
    - Columns: team,venue,matches,wins,draws,losses,gf,ga,pts,xg,xga,goals_vs_xg
    - One row per team per venue (home/away)
    - Using sample data for demonstration
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

# ========== PHASE 2: MAIN MODEL PREDICTIONS ==========
st.divider()
st.header("📈 Main Model Predictions")

prob_matrix = create_probability_matrix(home_xg, away_xg)
home_win_prob, draw_prob, away_win_prob = calculate_outcome_probabilities(prob_matrix)
over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob = calculate_betting_markets(prob_matrix)

# Track resolution spread for validation
predictions = [home_win_prob, draw_prob, away_win_prob, over_25_prob, under_25_prob]
ValidationTracker.update_resolution_spread(predictions)

# Display main predictions
col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    st.subheader("Over/Under 2.5 Goals")
    st.metric("Over 2.5", f"{over_25_prob*100:.1f}%")
    st.progress(over_25_prob)
    st.metric("Under 2.5", f"{under_25_prob*100:.1f}%")
    st.progress(under_25_prob)

with col_pred2:
    st.subheader("Both Teams to Score")
    st.metric("Yes", f"{btts_yes_prob*100:.1f}%")
    st.progress(btts_yes_prob)
    st.metric("No", f"{btts_no_prob*100:.1f}%")
    st.progress(btts_no_prob)

# ========== PHASE 3: CONFIRMATION LAYER ==========
if enable_confirmation_layer:
    st.divider()
    st.header("🛡️ Defensive Confirmation Layer")
    
    confirmation_model = DefensiveConfirmationModel(league_avg_xga, league_std_xga)
    defensive_analysis = confirmation_model.analyze_defensive_gap(home_stats, away_stats)
    
    # Assess confidence in main prediction
    confidence_level, confidence_reason = confirmation_model.assess_confidence(
        over_25_prob * 100, defensive_analysis
    )
    
    # Display confirmation analysis
    col_conf1, col_conf2, col_conf3 = st.columns(3)
    
    with col_conf1:
        signal = defensive_analysis['signal']
        if "OVER" in signal:
            st.success(f"**Defensive Signal:** {signal}")
        elif "UNDER" in signal:
            st.info(f"**Defensive Signal:** {signal}")
        else:
            st.warning(f"**Defensive Signal:** {signal}")
        
        gap_score = defensive_analysis['gap_score']
        st.metric("Gap Score", f"{gap_score:.2f}")
    
    with col_conf2:
        home_def_score = defensive_analysis['home_def_score']
        st.metric(f"{home_team} Defense", f"{home_def_score:.2f}σ",
                 delta="Strong" if home_def_score < 0 else "Weak")
    
    with col_conf3:
        away_def_score = defensive_analysis['away_def_score']
        st.metric(f"{away_team} Defense", f"{away_def_score:.2f}σ",
                 delta="Strong" if away_def_score < 0 else "Weak")
    
    st.info(defensive_analysis['explanation'])
    
    # Display confidence assessment
    st.subheader("🔍 Confidence Assessment")
    
    if confidence_level == "HIGH":
        st.success(f"**Confidence Level: HIGH** 🎯")
        st.write(f"*{confidence_reason}*")
        st.info("Main model prediction has strong defensive confirmation")
    elif confidence_level == "MEDIUM":
        st.warning(f"**Confidence Level: MEDIUM** ⚠️")
        st.write(f"*{confidence_reason}*")
        st.info("Proceed with caution - defensive context is neutral or mildly conflicting")
    else:  # LOW
        st.error(f"**Confidence Level: LOW** 🚨")
        st.write(f"*{confidence_reason}*")
        st.warning("Strong defensive signal contradicts main prediction - exercise high caution")
    
    # Note: No override - prediction remains unchanged
    st.caption("ℹ️ **Note:** Confirmation layer modulates confidence only - main prediction unchanged")

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

# ========== VALIDATION DASHBOARD ==========
if show_validation:
    st.divider()
    st.header("📊 Validation Dashboard")
    
    metrics = ValidationTracker.calculate_validation_metrics()
    status = ValidationTracker.get_validation_status(metrics)
    
    # Display validation results
    col_val1, col_val2, col_val3 = st.columns(3)
    
    with col_val1:
        st.subheader("Confidence Calibration")
        calibration = metrics.get('confidence_calibration', {})
        if calibration:
            for level, accuracy in calibration.items():
                st.metric(f"{level} Confidence", f"{accuracy:.1%}")
        else:
            st.info("Insufficient data")
    
    with col_val2:
        st.subheader("Model Resolution")
        resolution = metrics.get('resolution_spread', 0)
        st.metric("Spread", f"{resolution:.3f}")
        if resolution > 0.15:
            st.success("Good differentiation")
        elif resolution > 0.1:
            st.warning("Moderate differentiation")
        else:
            st.error("Low differentiation")
    
    with col_val3:
        st.subheader("Agreement Performance")
        agreement = metrics.get('agreement_performance', {})
        if agreement:
            for agree_type, accuracy in agreement.items():
                st.metric(agree_type, f"{accuracy:.1%}")
        else:
            st.info("Insufficient data")
    
    # Overall validation status
    st.subheader("Overall Validation Status")
    if status['overall'] == 'PASS':
        st.success("✅ All validation tests passing - confirmation layer properly calibrated")
    elif status['overall'] == 'NEEDS_CALIBRATION':
        st.warning("⚠️ Some validation tests need calibration - confidence levels may not match actual accuracy")
    else:
        st.info("📊 Insufficient data for validation - continue accumulating predictions")

# ========== OUTPUT FORMATS ==========
st.divider()
st.header("📤 Export & Share")

# Fix the formatting issue by separating conditional logic
if enable_confirmation_layer:
    gap_display = f"{defensive_analysis['gap_score']:.2f}"
    signal_display = defensive_analysis['signal']
    confidence_display = defensive_analysis['confidence']
    confirmation_summary = f"""
    🛡️ DEFENSIVE CONFIRMATION LAYER:
    • Signal: {signal_display}
    • Confidence: {confidence_display}
    • Gap Score: {gap_display}
    • Confidence Level: {confidence_level}
    • Reason: {confidence_reason}
    """
else:
    confirmation_summary = "    🛡️ DEFENSIVE CONFIRMATION LAYER: Disabled"

summary = f"""
⚽ FOOTBALL MATCH PREDICTION: {home_team} vs {away_team}
League: {selected_league}

📊 MAIN MODEL PREDICTIONS:
• Expected Goals: {home_team} {home_xg:.2f} - {away_team} {away_xg:.2f}
• Total xG: {home_xg + away_xg:.2f}
• Over 2.5 Goals: {over_25_prob*100:.1f}%
• Under 2.5 Goals: {under_25_prob*100:.1f}%
• Both Teams to Score: {btts_yes_prob*100:.1f}%

{confirmation_summary}

📈 Most Likely Score: {score_probs[0][0][0] if score_probs else 'N/A'}-{score_probs[0][0][1] if score_probs else 'N/A'} ({(score_probs[0][1]*100 if score_probs else 0):.1f}%)

🏆 Match Outcome Probabilities:
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
            'main_model_accuracy': [],
            'confidence_calibration': defaultdict(list),
            'resolution_spread': [],
            'agreement_tracking': defaultdict(list),
            'match_count': 0
        }
        st.success("Validation history reset!")
        st.rerun()

# ========== FOOTER ==========
st.divider()
footer_text = f"⚡ xG prediction system with defensive confirmation layer"
if enable_confirmation_layer and 'confidence_level' in locals():
    footer_text += f" | Confidence: {confidence_level}"
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
