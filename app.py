import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
from collections import defaultdict, deque
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Football xG Predictor Pro+",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚽ Football xG Predictor Pro+")
st.markdown("""
    Advanced xG prediction system with operational confidence assessment.
    **Main model provides predictions → Confirmation layer guides trust → Actionable recommendations**
""")

# Constants
MAX_GOALS = 8
REG_BASE_FACTOR = 0.75
REG_MATCH_THRESHOLD = 5
MAX_REGRESSION = 0.3

# Defensive gap thresholds
STRONG_OVER_THRESHOLD = 1.0
STRONG_UNDER_THRESHOLD = -1.0

# Action thresholds
ACTION_HIGH_THRESHOLD = 0.65    # 65%+ probability
ACTION_MEDIUM_THRESHOLD = 0.55  # 55-65% probability
ACTION_STAKE_REDUCTION = 0.5    # Reduce stake by 50% for LOW confidence

# Initialize session state for validation tracking
if 'validation_history' not in st.session_state:
    st.session_state.validation_history = {
        'main_model_accuracy': deque(maxlen=100),
        'confidence_calibration': defaultdict(lambda: deque(maxlen=50)),
        'resolution_history': deque(maxlen=100),
        'agreement_tracking': defaultdict(lambda: deque(maxlen=50)),
        'action_recommendations': defaultdict(lambda: deque(maxlen=50)),
        'match_count': 0,
        'prediction_history': []
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
    """Track confidence calibration and provide operational insights"""
    
    @staticmethod
    def update_prediction_history(prediction_data):
        """Store prediction history for operational insights"""
        st.session_state.validation_history['prediction_history'].append(prediction_data)
        if len(st.session_state.validation_history['prediction_history']) > 200:
            st.session_state.validation_history['prediction_history'] = st.session_state.validation_history['prediction_history'][-200:]
    
    @staticmethod
    def update_confidence_calibration(confidence_level, prediction_correct):
        """Track how well confidence levels predict accuracy"""
        st.session_state.validation_history['confidence_calibration'][confidence_level].append(
            1 if prediction_correct else 0
        )
    
    @staticmethod
    def calculate_resolution_metrics(predictions):
        """Calculate actionable resolution metrics"""
        if not predictions:
            return {}
        
        pred_array = np.array(predictions)
        
        # Distance from 50% (decisiveness metric)
        distance_from_50 = np.mean(np.abs(pred_array - 0.5))
        
        # Spread of predictions (standard deviation)
        spread = np.std(pred_array)
        
        # Percentage of predictions with clear signal (>55% or <45%)
        clear_signals = np.sum((pred_array > 0.55) | (pred_array < 0.45)) / len(pred_array)
        
        # Categorization
        if distance_from_50 > 0.15:
            decisiveness = "HIGH"
        elif distance_from_50 > 0.1:
            decisiveness = "MODERATE"
        else:
            decisiveness = "LOW"
        
        return {
            'decisiveness': decisiveness,
            'distance_from_50': distance_from_50,
            'spread': spread,
            'clear_signals_pct': clear_signals,
            'prediction_range': f"{pred_array.min():.1%} - {pred_array.max():.1%}"
        }
    
    @staticmethod
    def update_action_recommendation(action_type, correct):
        """Track performance of action recommendations"""
        st.session_state.validation_history['action_recommendations'][action_type].append(
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
        
        # Calculate confidence calibration with actionable insights
        calibration_data = {}
        calibration_samples = {}
        for confidence_level, results in st.session_state.validation_history['confidence_calibration'].items():
            if len(results) >= 10:  # Minimum sample for meaningful stats
                accuracy = np.mean(results)
                calibration_data[confidence_level] = {
                    'accuracy': accuracy,
                    'samples': len(results),
                    'reliability': 'HIGH' if len(results) >= 30 else 'MODERATE' if len(results) >= 15 else 'LOW'
                }
        
        metrics['confidence_calibration'] = calibration_data
        
        # Calculate resolution metrics from prediction history
        if st.session_state.validation_history['prediction_history']:
            all_predictions = []
            for pred_data in st.session_state.validation_history['prediction_history']:
                if 'over_25_prob' in pred_data:
                    all_predictions.append(pred_data['over_25_prob'])
                if 'home_win_prob' in pred_data:
                    all_predictions.append(pred_data['home_win_prob'])
                    all_predictions.append(pred_data['draw_prob'])
                    all_predictions.append(pred_data['away_win_prob'])
            
            metrics['resolution'] = ValidationTracker.calculate_resolution_metrics(all_predictions)
        
        # Calculate agreement performance
        agreement_performance = {}
        for agreement_type, results in st.session_state.validation_history['agreement_tracking'].items():
            if len(results) >= 10:
                agreement_performance[agreement_type] = {
                    'accuracy': np.mean(results),
                    'samples': len(results)
                }
        
        metrics['agreement_performance'] = agreement_performance
        
        # Calculate action recommendation performance
        action_performance = {}
        for action_type, results in st.session_state.validation_history['action_recommendations'].items():
            if len(results) >= 5:
                action_performance[action_type] = {
                    'accuracy': np.mean(results),
                    'samples': len(results)
                }
        
        metrics['action_performance'] = action_performance
        
        # Overall stats
        metrics['total_matches'] = st.session_state.validation_history['match_count']
        metrics['total_predictions'] = len(st.session_state.validation_history['prediction_history'])
        
        return metrics
    
    @staticmethod
    def get_validation_status(metrics):
        """Determine validation status with operational context"""
        status = {
            'confidence_calibration': {'status': 'INSUFFICIENT_DATA', 'details': 'Need more samples'},
            'resolution': {'status': 'INSUFFICIENT_DATA', 'details': 'Need more predictions'},
            'action_recommendations': {'status': 'INSUFFICIENT_DATA', 'details': 'Need more samples'}
        }
        
        # Check confidence calibration
        calibration = metrics.get('confidence_calibration', {})
        if calibration:
            all_good = True
            details = []
            for level, data in calibration.items():
                accuracy = data['accuracy']
                samples = data['samples']
                reliability = data['reliability']
                
                # Check if calibration makes sense
                if level == 'HIGH' and accuracy < 0.65:
                    all_good = False
                    details.append(f"HIGH confidence only {accuracy:.1%} accurate")
                elif level == 'MEDIUM' and (accuracy < 0.5 or accuracy > 0.7):
                    all_good = False
                    details.append(f"MEDIUM confidence {accuracy:.1%} (should be 50-70%)")
                elif level == 'LOW' and accuracy > 0.55:
                    all_good = False
                    details.append(f"LOW confidence {accuracy:.1%} (should be <55%)")
            
            status['confidence_calibration'] = {
                'status': 'PASS' if all_good else 'NEEDS_CALIBRATION',
                'details': ', '.join(details) if details else 'Well calibrated',
                'reliability': 'HIGH' if all(cal['reliability'] == 'HIGH' for cal in calibration.values()) else 'MODERATE'
            }
        
        # Check resolution
        resolution = metrics.get('resolution', {})
        if resolution:
            decisiveness = resolution.get('decisiveness', 'LOW')
            distance = resolution.get('distance_from_50', 0)
            
            if decisiveness == 'HIGH':
                status_details = f"Strong model decisiveness ({distance:.3f} from 50%)"
            elif decisiveness == 'MODERATE':
                status_details = f"Moderate decisiveness ({distance:.3f} from 50%)"
            else:
                status_details = f"Low decisiveness ({distance:.3f} from 50%) - model plays safe"
            
            status['resolution'] = {
                'status': 'PASS' if decisiveness in ['HIGH', 'MODERATE'] else 'WARNING',
                'details': status_details,
                'decisiveness': decisiveness
            }
        
        # Check action recommendations
        actions = metrics.get('action_performance', {})
        if actions:
            avg_accuracy = np.mean([data['accuracy'] for data in actions.values()])
            total_samples = sum([data['samples'] for data in actions.values()])
            
            if avg_accuracy > 0.55 and total_samples >= 20:
                status['action_recommendations'] = {
                    'status': 'PASS',
                    'details': f"Recommendations {avg_accuracy:.1%} accurate ({total_samples} samples)",
                    'accuracy': avg_accuracy
                }
            elif total_samples >= 10:
                status['action_recommendations'] = {
                    'status': 'MODERATE',
                    'details': f"Early data: {avg_accuracy:.1%} accuracy ({total_samples} samples)",
                    'accuracy': avg_accuracy
                }
        
        # Overall status
        status_items = [data['status'] for data in status.values()]
        if all(s == 'PASS' for s in status_items if s != 'INSUFFICIENT_DATA'):
            status['overall'] = 'PASS'
        elif any(s == 'NEEDS_CALIBRATION' for s in status_items):
            status['overall'] = 'NEEDS_CALIBRATION'
        elif any(s == 'WARNING' for s in status_items):
            status['overall'] = 'WARNING'
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
        main_prob = main_over_prob / 100  # Convert to decimal
        if main_prob > 0.5:
            main_direction = "OVER"
            main_strength = main_prob - 0.5
        else:
            main_direction = "UNDER"
            main_strength = 0.5 - main_prob
        
        # Defensive signal direction
        if "OVER" in defensive_signal['signal']:
            defensive_direction = "OVER"
            defensive_strength = defensive_signal['confidence']  # HIGH/MEDIUM/LOW
        elif "UNDER" in defensive_signal['signal']:
            defensive_direction = "UNDER"
            defensive_strength = defensive_signal['confidence']
        else:
            defensive_direction = "NEUTRAL"
            defensive_strength = "LOW"
        
        # Determine confidence level
        if defensive_direction == "NEUTRAL":
            confidence_level = "MEDIUM"
            reason = "Confirmation layer neutral - standard confidence"
            confidence_score = 0.5
        
        elif main_direction == defensive_direction:
            # Agreement - boost confidence
            if defensive_strength == "HIGH":
                confidence_level = "HIGH"
                reason = f"Strong confirmation for {main_direction}"
                confidence_score = 0.8 + min(0.15, main_strength * 0.3)
            else:
                confidence_level = "MEDIUM"
                reason = f"Moderate confirmation for {main_direction}"
                confidence_score = 0.6 + min(0.15, main_strength * 0.3)
        
        else:
            # Disagreement - reduce confidence
            if defensive_strength == "HIGH":
                confidence_level = "LOW"
                reason = f"Strong defensive signal contradicts main {main_direction}"
                confidence_score = 0.3 - min(0.1, main_strength * 0.2)
            else:
                confidence_level = "MEDIUM"
                reason = f"Mild defensive disagreement with main {main_direction}"
                confidence_score = 0.5 - min(0.1, main_strength * 0.2)
        
        return confidence_level, reason, confidence_score
    
    def get_action_recommendation(self, main_over_prob, confidence_level, confidence_score, defensive_signal):
        """Generate actionable betting recommendations based on confidence"""
        main_prob = main_over_prob / 100
        
        # Base recommendation from main model
        base_recommendation = "OVER" if main_prob > 0.5 else "UNDER"
        base_strength = abs(main_prob - 0.5)  # How far from 50%
        
        # Map confidence to action
        if confidence_level == "HIGH":
            action = "NORMAL_STAKE"
            stake_multiplier = 1.0
            advice = f"Full confidence in {base_recommendation}"
            
        elif confidence_level == "MEDIUM":
            if confidence_score > 0.55:  # Leaning positive
                action = "NORMAL_STAKE"
                stake_multiplier = 1.0
                advice = f"Standard play on {base_recommendation}"
            else:  # Leaning negative
                action = "REDUCED_STAKE"
                stake_multiplier = ACTION_STAKE_REDUCTION
                advice = f"Caution advised on {base_recommendation}"
                
        else:  # LOW confidence
            if base_strength > 0.1:  # Strong main signal despite low confidence
                action = "REDUCED_STAKE"
                stake_multiplier = ACTION_STAKE_REDUCTION
                advice = f"Reduced stake on {base_recommendation} due to conflict"
            else:  # Weak main signal + low confidence
                action = "AVOID"
                stake_multiplier = 0.0
                advice = f"Avoid bet - conflicting signals ({base_recommendation} vs defensive {defensive_signal['signal'].split('_')[1]})"
        
        # Special case: Very strong defensive signal may suggest alternative
        if defensive_signal['confidence'] == "HIGH" and confidence_level == "LOW":
            alternative = defensive_signal['signal'].split('_')[1]
            if base_strength < 0.08:  # Very weak main signal
                advice = f"Consider {alternative} instead - strong defensive signal outweighs weak main prediction"
        
        return {
            'action': action,
            'stake_multiplier': stake_multiplier,
            'advice': advice,
            'confidence_score': confidence_score,
            'base_recommendation': base_recommendation
        }

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
                show_action_recommendations = st.checkbox("Show Action Recommendations", value=True,
                    help="Show specific betting actions based on confidence")
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
                calibration = metrics.get('confidence_calibration', {})
                if calibration:
                    for level, data in calibration.items():
                        acc = data['accuracy']
                        samples = data['samples']
                        rel = data['reliability']
                        st.write(f"  {level}: {acc:.1%} ({samples} samples, {rel} reliability)")
                else:
                    st.write("  Insufficient data")
                
                resolution = metrics.get('resolution', {})
                if resolution:
                    st.write(f"**Model Decisiveness:** {resolution.get('decisiveness', 'N/A')}")
                    st.write(f"**Avg distance from 50%:** {resolution.get('distance_from_50', 0):.3f}")
                    st.write(f"**Prediction range:** {resolution.get('prediction_range', 'N/A')}")
                
                st.write("**Overall Status:**")
                overall = status.get('overall', 'INSUFFICIENT_DATA')
                if overall == 'PASS':
                    st.success("✅ System properly calibrated")
                elif overall == 'WARNING':
                    st.warning("⚠️ Some warnings - check details")
                elif overall == 'NEEDS_CALIBRATION':
                    st.error("❌ Needs calibration")
                else:
                    st.info("📊 Collecting more data")

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

# Store prediction for validation
prediction_data = {
    'home_team': home_team,
    'away_team': away_team,
    'home_win_prob': home_win_prob,
    'draw_prob': draw_prob,
    'away_win_prob': away_win_prob,
    'over_25_prob': over_25_prob,
    'under_25_prob': under_25_prob,
    'btts_yes_prob': btts_yes_prob,
    'btts_no_prob': btts_no_prob,
    'timestamp': datetime.now()
}
ValidationTracker.update_prediction_history(prediction_data)
ValidationTracker.increment_match_count()

# Display main predictions
col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    st.subheader("Over/Under 2.5 Goals")
    
    # Calculate decisiveness of prediction
    over_strength = abs(over_25_prob - 0.5)
    if over_strength > 0.15:
        decisiveness = "STRONG"
        color = "green"
    elif over_strength > 0.08:
        decisiveness = "MODERATE"
        color = "orange"
    else:
        decisiveness = "WEAK"
        color = "gray"
    
    st.metric("Over 2.5", f"{over_25_prob*100:.1f}%", 
              delta=f"{decisiveness} signal" if decisiveness != "WEAK" else None)
    st.progress(over_25_prob)
    
    st.metric("Under 2.5", f"{under_25_prob*100:.1f}%")
    st.progress(under_25_prob)
    
    st.caption(f"Prediction strength: {over_strength:.3f} from 50% ({decisiveness.lower()})")

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
    confidence_level, confidence_reason, confidence_score = confirmation_model.assess_confidence(
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
        st.metric("Gap Score", f"{gap_score:.2f}σ")
    
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
    
    col_conf_assess1, col_conf_assess2 = st.columns([2, 1])
    
    with col_conf_assess1:
        if confidence_level == "HIGH":
            st.success(f"**Confidence Level: HIGH** 🎯 (Score: {confidence_score:.2f})")
            st.write(f"*{confidence_reason}*")
            st.info("Main model prediction has strong defensive confirmation")
        elif confidence_level == "MEDIUM":
            st.warning(f"**Confidence Level: MEDIUM** ⚠️ (Score: {confidence_score:.2f})")
            st.write(f"*{confidence_reason}*")
            st.info("Proceed with standard caution - defensive context is neutral or mildly conflicting")
        else:  # LOW
            st.error(f"**Confidence Level: LOW** 🚨 (Score: {confidence_score:.2f})")
            st.write(f"*{confidence_reason}*")
            st.warning("Strong defensive signal contradicts main prediction - exercise high caution")
    
    with col_conf_assess2:
        # Confidence score visualization
        st.metric("Confidence Score", f"{confidence_score:.2f}")
        st.progress(confidence_score)
        if confidence_score > 0.7:
            st.caption("High confidence")
        elif confidence_score > 0.5:
            st.caption("Moderate confidence")
        else:
            st.caption("Low confidence")
    
    # Action Recommendations
    if show_action_recommendations:
        st.subheader("🎯 Action Recommendations")
        
        action_recommendation = confirmation_model.get_action_recommendation(
            over_25_prob * 100, confidence_level, confidence_score, defensive_analysis
        )
        
        col_action1, col_action2, col_action3 = st.columns(3)
        
        with col_action1:
            if action_recommendation['action'] == "NORMAL_STAKE":
                st.success("**Action:** Normal Stake ✅")
                st.metric("Stake Multiplier", "1.0x")
            elif action_recommendation['action'] == "REDUCED_STAKE":
                st.warning("**Action:** Reduced Stake ⚠️")
                st.metric("Stake Multiplier", f"{action_recommendation['stake_multiplier']:.1f}x")
            else:
                st.error("**Action:** Avoid ❌")
                st.metric("Stake Multiplier", "0.0x")
        
        with col_action2:
            base_rec = action_recommendation['base_recommendation']
            if base_rec == "OVER":
                st.metric("Base Recommendation", "OVER 2.5", 
                         delta=f"{over_25_prob*100:.1f}%")
            else:
                st.metric("Base Recommendation", "UNDER 2.5",
                         delta=f"{under_25_prob*100:.1f}%")
        
        with col_action3:
            st.metric("Confidence Impact", 
                     f"{-((over_strength * 100) * (1 - confidence_score)):.1f}%",
                     delta="Reduction" if confidence_score < 0.7 else "Neutral")
        
        st.info(f"**Advice:** {action_recommendation['advice']}")
        
        # Decision rationale
        with st.expander("📋 Decision Rationale"):
            st.write(f"""
            1. **Main Model Prediction:** {over_25_prob*100:.1f}% Over 2.5 goals
            2. **Defensive Signal:** {defensive_analysis['signal']} ({defensive_analysis['confidence']} confidence)
            3. **Agreement Assessment:** {'Agree' if confidence_level == 'HIGH' else 'Partial' if confidence_level == 'MEDIUM' else 'Disagree'}
            4. **Confidence Score:** {confidence_score:.2f} (translates to {action_recommendation['action'].replace('_', ' ').lower()})
            5. **Recommendation:** {action_recommendation['advice']}
            
            **Key Insight:** When main model and defensive confirmation disagree (confidence LOW), 
            the system recommends caution (reduced stake or avoid) rather than overriding the prediction.
            """)
    
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
    st.header("📊 Operational Validation Dashboard")
    
    metrics = ValidationTracker.calculate_validation_metrics()
    status = ValidationTracker.get_validation_status(metrics)
    
    # Display validation results
    col_val1, col_val2, col_val3 = st.columns(3)
    
    with col_val1:
        st.subheader("Confidence Calibration")
        calibration = metrics.get('confidence_calibration', {})
        if calibration:
            for level, data in calibration.items():
                acc = data['accuracy']
                samples = data['samples']
                rel = data['reliability']
                
                if level == 'HIGH' and acc >= 0.65:
                    st.success(f"{level}: {acc:.1%} ✅")
                elif level == 'MEDIUM' and 0.5 <= acc <= 0.7:
                    st.info(f"{level}: {acc:.1%} ⚠️")
                elif level == 'LOW' and acc <= 0.55:
                    st.warning(f"{level}: {acc:.1%} 🚨")
                else:
                    st.error(f"{level}: {acc:.1%} ❌")
                
                st.caption(f"{samples} samples, {rel} reliability")
        else:
            st.info("Insufficient data")
    
    with col_val2:
        st.subheader("Model Resolution")
        resolution = metrics.get('resolution', {})
        if resolution:
            decisiveness = resolution.get('decisiveness', 'LOW')
            distance = resolution.get('distance_from_50', 0)
            pred_range = resolution.get('prediction_range', 'N/A')
            
            if decisiveness == 'HIGH':
                st.success(f"**Decisiveness:** HIGH ✅")
                st.metric("Distance from 50%", f"{distance:.3f}")
            elif decisiveness == 'MODERATE':
                st.warning(f"**Decisiveness:** MODERATE ⚠️")
                st.metric("Distance from 50%", f"{distance:.3f}")
            else:
                st.error(f"**Decisiveness:** LOW ❌")
                st.metric("Distance from 50%", f"{distance:.3f}")
            
            st.caption(f"Prediction range: {pred_range}")
        else:
            st.info("Insufficient data")
    
    with col_val3:
        st.subheader("Action Performance")
        actions = metrics.get('action_performance', {})
        if actions:
            avg_accuracy = np.mean([data['accuracy'] for data in actions.values()])
            total_samples = sum([data['samples'] for data in actions.values()])
            
            if avg_accuracy > 0.55:
                st.success(f"**Accuracy:** {avg_accuracy:.1%} ✅")
            elif avg_accuracy > 0.5:
                st.warning(f"**Accuracy:** {avg_accuracy:.1%} ⚠️")
            else:
                st.error(f"**Accuracy:** {avg_accuracy:.1%} ❌")
            
            st.metric("Total Samples", total_samples)
            
            # Show breakdown
            with st.expander("Breakdown"):
                for action_type, data in actions.items():
                    st.write(f"{action_type}: {data['accuracy']:.1%} ({data['samples']})")
        else:
            st.info("Insufficient data")
    
    # Overall validation status with operational guidance
    st.subheader("Overall Operational Status")
    
    overall_status = status.get('overall', 'INSUFFICIENT_DATA')
    
    if overall_status == 'PASS':
        st.success("""
        ✅ **System Properly Calibrated**
        
        **Operational Guidance:**
        - Confidence levels accurately reflect prediction reliability
        - Model shows good decisiveness in predictions
        - Action recommendations have proven accuracy
        - **Proceed with normal decision-making**
        """)
    
    elif overall_status == 'WARNING':
        st.warning("""
        ⚠️ **System Has Warnings**
        
        **Operational Guidance:**
        - Some metrics need attention (check details above)
        - Model may be too cautious or too aggressive
        - **Proceed with increased caution**
        - Consider reducing stake sizes until calibration improves
        """)
    
    elif overall_status == 'NEEDS_CALIBRATION':
        st.error("""
        ❌ **System Needs Calibration**
        
        **Operational Guidance:**
        - Confidence levels don't match actual accuracy
        - Action recommendations underperforming
        - **Avoid significant decisions**
        - Use system for informational purposes only
        - Collect more data for recalibration
        """)
    
    else:
        st.info("""
        📊 **Insufficient Data for Validation**
        
        **Operational Guidance:**
        - System needs more predictions to establish reliability
        - **Use with caution** - early stage
        - Track your own results to validate system performance
        - Recommendations based on theoretical framework, not empirical evidence
        """)

# ========== OUTPUT FORMATS ==========
st.divider()
st.header("📤 Export & Share")

# Fix the formatting issue by separating conditional logic
if enable_confirmation_layer:
    gap_display = f"{defensive_analysis['gap_score']:.2f}"
    signal_display = defensive_analysis['signal']
    confidence_display = defensive_analysis['confidence']
    
    if show_action_recommendations and 'action_recommendation' in locals():
        action_display = action_recommendation['action'].replace('_', ' ')
        advice_display = action_recommendation['advice']
        action_summary = f"""
    • Recommended Action: {action_display}
    • Stake Multiplier: {action_recommendation['stake_multiplier']:.1f}x
    • Advice: {advice_display}
        """
    else:
        action_summary = ""
    
    confirmation_summary = f"""
    🛡️ DEFENSIVE CONFIRMATION LAYER:
    • Signal: {signal_display}
    • Confidence: {confidence_display}
    • Gap Score: {gap_display}
    • Confidence Level: {confidence_level} (Score: {confidence_score:.2f})
    • Reason: {confidence_reason}
    {action_summary}
    """
else:
    confirmation_summary = "    🛡️ DEFENSIVE CONFIRMATION LAYER: Disabled"

# Get validation insights
metrics = ValidationTracker.calculate_validation_metrics()
resolution = metrics.get('resolution', {})
decisiveness = resolution.get('decisiveness', 'N/A')
distance_from_50 = resolution.get('distance_from_50', 0)

summary = f"""
⚽ FOOTBALL MATCH PREDICTION: {home_team} vs {away_team}
League: {selected_league}

📊 MAIN MODEL PREDICTIONS:
• Expected Goals: {home_team} {home_xg:.2f} - {away_team} {away_xg:.2f}
• Total xG: {home_xg + away_xg:.2f}
• Over 2.5 Goals: {over_25_prob*100:.1f}% (Strength: {abs(over_25_prob-0.5):.3f})
• Under 2.5 Goals: {under_25_prob*100:.1f}%
• Both Teams to Score: {btts_yes_prob*100:.1f}%

{confirmation_summary}

📈 MODEL CHARACTERISTICS:
• Decisiveness: {decisiveness}
• Avg distance from 50%: {distance_from_50:.3f}
• Most predictions in range: {resolution.get('prediction_range', 'N/A')}

🎯 Most Likely Score: {score_probs[0][0][0] if score_probs else 'N/A'}-{score_probs[0][0][1] if score_probs else 'N/A'} ({(score_probs[0][1]*100 if score_probs else 0):.1f}%)

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
            'main_model_accuracy': deque(maxlen=100),
            'confidence_calibration': defaultdict(lambda: deque(maxlen=50)),
            'resolution_history': deque(maxlen=100),
            'agreement_tracking': defaultdict(lambda: deque(maxlen=50)),
            'action_recommendations': defaultdict(lambda: deque(maxlen=50)),
            'match_count': 0,
            'prediction_history': []
        }
        st.success("Validation history reset!")
        st.rerun()

# ========== FOOTER ==========
st.divider()
footer_text = f"⚡ xG prediction system with operational confidence assessment"
if enable_confirmation_layer and 'confidence_level' in locals():
    footer_text += f" | Confidence: {confidence_level}"
if show_action_recommendations and 'action_recommendation' in locals():
    footer_text += f" | Action: {action_recommendation['action']}"
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
