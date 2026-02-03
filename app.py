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
    page_title="Unified Football xG Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚽ Unified Football xG Predictor")
st.markdown("""
    **Intelligent xG prediction system with match type classification and asymmetric correction.**
    *Goals happen when attacks overcome defenses, not when defenses exist in isolation.*
""")

# Constants
MAX_GOALS = 8
REG_BASE_FACTOR = 0.75
MAX_REGRESSION = 0.3
MAX_CORRECTION = 0.3
MIN_PROBABILITY = 0.1
MAX_PROBABILITY = 0.9

# Match Type Thresholds
ELITE_DEFENSE_THRESHOLD = -1.0
STRONG_DEFENSE_THRESHOLD = -0.5
ELITE_ATTACK_THRESHOLD = 1.0
WEAK_DEFENSE_THRESHOLD = 0.8
ATTACK_DOM_DEFENSE_THRESHOLD = 0.5

# Confidence Thresholds
LOW_CONFIDENCE_THRESHOLD = 0.05  # 5% from 50%
MEDIUM_CONFIDENCE_THRESHOLD = 0.15  # 15% from 50%
HIGH_CONFIDENCE_THRESHOLD = 0.25  # 25% from 50%

# Override Protection
OVERRIDE_BASE_HIGH = 0.65
OVERRIDE_BASE_LOW = 0.35
OVERRIDE_LIMIT_DEFENSE = 0.10
OVERRIDE_LIMIT_ATTACK = 0.10

# Initialize session state
if 'validation_history' not in st.session_state:
    st.session_state.validation_history = {
        'prediction_history': deque(maxlen=200),
        'match_type_distribution': defaultdict(int),
        'correction_effectiveness': defaultdict(lambda: deque(maxlen=50)),
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
    """Calculate league average statistics for normalization"""
    # Calculate per-match averages
    home_xg_per_match = df[df['venue'] == 'home']['xg'] / df[df['venue'] == 'home']['matches']
    away_xg_per_match = df[df['venue'] == 'away']['xg'] / df[df['venue'] == 'away']['matches']
    home_xga_per_match = df[df['venue'] == 'home']['xga'] / df[df['venue'] == 'home']['matches']
    away_xga_per_match = df[df['venue'] == 'away']['xga'] / df[df['venue'] == 'away']['matches']
    
    # Combine home and away
    all_xg_per_match = pd.concat([home_xg_per_match, away_xg_per_match])
    all_xga_per_match = pd.concat([home_xga_per_match, away_xga_per_match])
    
    # Calculate league averages and standard deviations
    league_avg_xg = all_xg_per_match.mean()
    league_std_xg = all_xg_per_match.std()
    league_avg_xga = all_xga_per_match.mean()
    league_std_xga = all_xga_per_match.std()
    
    return {
        'avg_xg': league_avg_xg,
        'std_xg': league_std_xg,
        'avg_xga': league_avg_xga,
        'std_xga': league_std_xga
    }

def calculate_team_scores(team_stats, league_baselines):
    """Calculate Team Attack Score and Team Defense Score"""
    # Get per-match stats
    matches = team_stats['matches']
    xg_per_match = team_stats['xg'] / max(matches, 1)
    xga_per_match = team_stats['xga'] / max(matches, 1)
    
    # Calculate scores (z-scores relative to league)
    attack_score = (xg_per_match - league_baselines['avg_xg']) / max(league_baselines['std_xg'], 0.1)
    defense_score = (xga_per_match - league_baselines['avg_xga']) / max(league_baselines['std_xga'], 0.1)
    
    # Calculate regression factor (goals vs xG)
    goals_vs_xg_per_match = team_stats['goals_vs_xg'] / max(matches, 1)
    
    return {
        'attack_score': attack_score,
        'defense_score': defense_score,
        'regression_factor': min(max(goals_vs_xg_per_match, -MAX_REGRESSION), MAX_REGRESSION),
        'xg_per_match': xg_per_match,
        'xga_per_match': xga_per_match
    }

class MatchTypeClassifier:
    """Classify matches into 4 types based on attack/defense scores"""
    
    @staticmethod
    def classify_match(home_scores, away_scores):
        """Classify match into one of 4 types"""
        
        home_attack = home_scores['attack_score']
        home_defense = home_scores['defense_score']
        away_attack = away_scores['attack_score']
        away_defense = away_scores['defense_score']
        
        # TYPE A: ELITE DEFENSIVE SHOWDOWN
        # IF (home_defense_score < -1.0 AND away_defense_score < -0.5)
        # AND (home_attack_score < 1.0 AND away_attack_score < 1.0)
        if (home_defense < ELITE_DEFENSE_THRESHOLD and away_defense < STRONG_DEFENSE_THRESHOLD and
            home_attack < ELITE_ATTACK_THRESHOLD and away_attack < ELITE_ATTACK_THRESHOLD):
            match_type = "DEFENSIVE_TACTICAL"
            explanation = "Both teams have elite/strong defenses with non-elite attacks"
        
        # TYPE B: ATTACK DOMINANCE
        # IF (home_attack_score > 1.0 AND away_defense_score > 0.5)
        # OR (away_attack_score > 1.0 AND home_defense_score > 0.5)
        elif ((home_attack > ELITE_ATTACK_THRESHOLD and away_defense > ATTACK_DOM_DEFENSE_THRESHOLD) or
              (away_attack > ELITE_ATTACK_THRESHOLD and home_defense > ATTACK_DOM_DEFENSE_THRESHOLD)):
            match_type = "ATTACK_DOMINANCE"
            explanation = "Elite attack facing weak defense"
        
        # TYPE C: DEFENSIVE CATASTROPHE
        # IF (home_defense_score > 1.0 AND away_defense_score > 0.8)
        elif (home_defense > ELITE_ATTACK_THRESHOLD and away_defense > WEAK_DEFENSE_THRESHOLD):
            match_type = "DEFENSIVE_WEAKNESS"
            explanation = "Both teams have very weak defenses"
        
        # TYPE D: STANDARD
        else:
            match_type = "STANDARD"
            explanation = "Standard match with balanced characteristics"
        
        return {
            'match_type': match_type,
            'explanation': explanation,
            'home_attack': home_attack,
            'home_defense': home_defense,
            'away_attack': away_attack,
            'away_defense': away_defense
        }

class BasePredictionEngine:
    """Generate base predictions using Poisson distribution"""
    
    @staticmethod
    def calculate_expected_goals(home_scores, away_scores, league_baselines):
        """Calculate expected goals for both teams"""
        
        # Get per-match values
        home_attack = home_scores['xg_per_match']
        home_defense = home_scores['xga_per_match']
        away_attack = away_scores['xg_per_match']
        away_defense = away_scores['xga_per_match']
        
        # Calculate expected goals using the formula:
        # home_expected = (home_attack * away_defense) / league_avg_xG
        # away_expected = (away_attack * home_defense) / league_avg_xG
        home_expected = (home_attack * away_defense) / max(league_baselines['avg_xg'], 0.1)
        away_expected = (away_attack * home_defense) / max(league_baselines['avg_xg'], 0.1)
        
        # Apply regression factors (capped at ±30%)
        home_final = home_expected * (1 + min(MAX_REGRESSION, home_scores['regression_factor']))
        away_final = away_expected * (1 + min(MAX_REGRESSION, away_scores['regression_factor']))
        
        # Apply minimum and maximum bounds
        home_final = max(min(home_final, 4.0), 0.3)
        away_final = max(min(away_final, 4.0), 0.3)
        
        return home_final, away_final
    
    @staticmethod
    def calculate_base_probability(total_expected_goals):
        """Calculate base probability of Over 2.5 goals using Poisson"""
        # Calculate probability of total goals > 2.5
        prob_0_goals = poisson_pmf(0, total_expected_goals)
        prob_1_goal = poisson_pmf(1, total_expected_goals)
        prob_2_goals = poisson_pmf(2, total_expected_goals)
        
        prob_under_25 = prob_0_goals + prob_1_goal + prob_2_goals
        prob_over_25 = 1 - prob_under_25
        
        return prob_over_25

class IntelligentCorrectionSystem:
    """Apply match type specific corrections to base predictions"""
    
    @staticmethod
    def calculate_correction(match_classification, base_prob, home_scores, away_scores):
        """Calculate type-specific correction"""
        
        match_type = match_classification['match_type']
        home_defense = match_classification['home_defense']
        away_defense = match_classification['away_defense']
        home_attack = match_classification['home_attack']
        away_attack = match_classification['away_attack']
        
        correction = 0.0
        
        # RULE 1: DEFENSIVE TACTICAL matches
        if match_type == "DEFENSIVE_TACTICAL":
            # Main model overestimates scoring in elite defensive matchups
            # Correction = -20% to -30% (scale with defense extremity)
            defense_extremity = abs(min(home_defense, away_defense))
            correction_range = (-0.30, -0.20)
            correction = correction_range[0] + (defense_extremity * (correction_range[1] - correction_range[0]))
            
            # Apply override protection
            if base_prob > OVERRIDE_BASE_HIGH:
                correction = max(correction, -OVERRIDE_LIMIT_DEFENSE)
        
        # RULE 2: ATTACK DOMINANCE matches
        elif match_type == "ATTACK_DOMINANCE":
            # Main model underestimates elite attacks
            # Correction = +10% to +25% (scale with attack dominance)
            attack_extremity = max(home_attack if home_attack > ELITE_ATTACK_THRESHOLD else 0,
                                 away_attack if away_attack > ELITE_ATTACK_THRESHOLD else 0)
            correction_range = (0.10, 0.25)
            correction = correction_range[0] + ((attack_extremity - ELITE_ATTACK_THRESHOLD) * 0.1)
            correction = min(max(correction, correction_range[0]), correction_range[1])
            
            # Apply override protection
            if base_prob < OVERRIDE_BASE_LOW:
                correction = min(correction, OVERRIDE_LIMIT_ATTACK)
        
        # RULE 3: DEFENSIVE WEAKNESS matches
        elif match_type == "DEFENSIVE_WEAKNESS":
            # Main model underestimates terrible defenses
            # Correction = +15% to +30% (scale with defensive weakness)
            defense_weakness = max(home_defense, away_defense)
            correction_range = (0.15, 0.30)
            correction = correction_range[0] + ((defense_weakness - WEAK_DEFENSE_THRESHOLD) * 0.1)
            correction = min(max(correction, correction_range[0]), correction_range[1])
        
        # RULE 4: STANDARD matches
        else:
            # Trust main model, confirmation only for confidence
            correction = 0.0
        
        return correction
    
    @staticmethod
    def apply_correction(base_prob, correction):
        """Apply correction with bounds"""
        final_prob = base_prob + correction
        
        # Clamp between MIN_PROBABILITY and MAX_PROBABILITY
        final_prob = max(min(final_prob, MAX_PROBABILITY), MIN_PROBABILITY)
        
        return final_prob

class ConfidenceValidator:
    """Determine confidence level based on prediction strength"""
    
    @staticmethod
    def calculate_confidence(final_prob):
        """Calculate confidence level based on distance from 50%"""
        distance_from_50 = abs(final_prob - 0.5)
        
        if distance_from_50 < LOW_CONFIDENCE_THRESHOLD:
            confidence = "LOW"
            confidence_score = 0.3
        elif distance_from_50 < MEDIUM_CONFIDENCE_THRESHOLD:
            confidence = "MEDIUM"
            confidence_score = 0.6
        else:
            confidence = "HIGH"
            confidence_score = 0.8
        
        # Adjust confidence based on extremity
        if distance_from_50 > HIGH_CONFIDENCE_THRESHOLD:
            confidence = "VERY HIGH"
            confidence_score = 0.95
        
        return confidence, confidence_score, distance_from_50

class UnifiedPredictionSystem:
    """Main unified prediction system that orchestrates all components"""
    
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
        self.classifier = MatchTypeClassifier()
        self.base_engine = BasePredictionEngine()
        self.correction_system = IntelligentCorrectionSystem()
        self.confidence_validator = ConfidenceValidator()
    
    def predict(self, home_team, away_team, home_stats, away_stats):
        """Generate unified prediction for a match"""
        
        # PHASE 1: Calculate team scores
        home_scores = calculate_team_scores(home_stats, self.league_baselines)
        away_scores = calculate_team_scores(away_stats, self.league_baselines)
        
        # PHASE 2: Classify match type
        match_classification = self.classifier.classify_match(home_scores, away_scores)
        
        # PHASE 3: Generate base prediction
        home_xg, away_xg = self.base_engine.calculate_expected_goals(
            home_scores, away_scores, self.league_baselines
        )
        total_xg = home_xg + away_xg
        base_prob = self.base_engine.calculate_base_probability(total_xg)
        
        # PHASE 4: Apply intelligent correction
        correction = self.correction_system.calculate_correction(
            match_classification, base_prob, home_scores, away_scores
        )
        final_prob = self.correction_system.apply_correction(base_prob, correction)
        
        # PHASE 5: Determine confidence
        confidence, confidence_score, distance_from_50 = self.confidence_validator.calculate_confidence(final_prob)
        
        # PHASE 6: Create final prediction
        direction = "OVER" if final_prob > 0.5 else "UNDER"
        
        # Generate rationale
        rationale = self._generate_rationale(
            match_classification, base_prob, correction, final_prob, confidence
        )
        
        # Create probability matrix for additional insights
        prob_matrix = create_probability_matrix(home_xg, away_xg)
        home_win_prob, draw_prob, away_win_prob = calculate_outcome_probabilities(prob_matrix)
        over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob = calculate_betting_markets(prob_matrix)
        
        # Store for validation
        self._store_prediction(
            home_team, away_team, match_classification['match_type'],
            base_prob, final_prob, correction, confidence
        )
        
        return {
            # Core prediction
            'final_probability': final_prob,
            'direction': direction,
            'match_type': match_classification['match_type'],
            'confidence': confidence,
            'confidence_score': confidence_score,
            'correction_applied': correction,
            'rationale': rationale,
            
            # Base model details
            'base_probability': base_prob,
            'expected_goals': {
                'home': home_xg,
                'away': away_xg,
                'total': total_xg
            },
            
            # Team scores
            'team_scores': {
                'home': home_scores,
                'away': away_scores
            },
            
            # Match classification
            'classification': match_classification,
            
            # Additional probabilities
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'over_25_prob': over_25_prob,
            'under_25_prob': under_25_prob,
            'btts_yes_prob': btts_yes_prob,
            'btts_no_prob': btts_no_prob,
            
            # Metrics
            'distance_from_50': distance_from_50
        }
    
    def _generate_rationale(self, classification, base_prob, correction, final_prob, confidence):
        """Generate detailed rationale for the prediction"""
        
        match_type = classification['match_type']
        explanation = classification['explanation']
        
        if match_type == "DEFENSIVE_TACTICAL":
            return f"{match_type}: {explanation}. Base model overestimates scoring by {abs(correction*100):.1f}% in elite defensive matchups."
        elif match_type == "ATTACK_DOMINANCE":
            return f"{match_type}: {explanation}. Base model underestimates scoring by {correction*100:.1f}% when elite attacks face weak defenses."
        elif match_type == "DEFENSIVE_WEAKNESS":
            return f"{match_type}: {explanation}. Base model underestimates scoring by {correction*100:.1f}% when both defenses are weak."
        else:
            return f"{match_type}: {explanation}. Trusting base model with {confidence} confidence."
    
    def _store_prediction(self, home_team, away_team, match_type, base_prob, final_prob, correction, confidence):
        """Store prediction for validation tracking"""
        
        st.session_state.validation_history['match_type_distribution'][match_type] += 1
        st.session_state.validation_history['match_count'] += 1
        
        prediction_data = {
            'home_team': home_team,
            'away_team': away_team,
            'match_type': match_type,
            'base_prob': base_prob,
            'final_prob': final_prob,
            'correction': correction,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        st.session_state.validation_history['prediction_history'].append(prediction_data)

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

def create_team_score_display(team_scores, team_name, is_home=True):
    """Create display for team attack/defense scores"""
    bg_color = "#1f77b4" if is_home else "#ff7f0e"
    
    attack_score = team_scores['attack_score']
    defense_score = team_scores['defense_score']
    
    attack_label = "Elite" if attack_score > 1.0 else "Above Avg" if attack_score > 0 else "Below Avg" if attack_score > -1.0 else "Weak"
    defense_label = "Elite" if defense_score < -1.0 else "Strong" if defense_score < -0.5 else "Avg" if defense_score < 0.5 else "Weak" if defense_score < 1.0 else "Very Weak"
    
    attack_color = "green" if attack_score > 0.5 else "orange" if attack_score > -0.5 else "red"
    defense_color = "green" if defense_score < -0.5 else "orange" if defense_score < 0.5 else "red"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Attack Score:**")
        st.markdown(f"""
        <div style="background-color: {bg_color}; color: white; padding: 10px; border-radius: 5px;">
            <div style="font-size: 20px; font-weight: bold;">{attack_score:.2f}</div>
            <div style="color: {attack_color};">{attack_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Defense Score:**")
        st.markdown(f"""
        <div style="background-color: {bg_color}; color: white; padding: 10px; border-radius: 5px;">
            <div style="font-size: 20px; font-weight: bold;">{defense_score:.2f}</div>
            <div style="color: {defense_color};">{defense_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    return attack_label, defense_label

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
        league_baselines = calculate_league_baselines(df)
        
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        available_home_teams = sorted(home_stats_df.index.unique())
        available_away_teams = sorted(away_stats_df.index.unique())
        common_teams = sorted(list(set(available_home_teams) & set(available_away_teams)))
        
        if len(common_teams) == 0:
            st.error("❌ No teams with both home and away data available")
        else:
            home_team = st.selectbox("Home Team", common_teams)
            away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
            
            st.divider()
            st.subheader("🎯 Display Options")
            show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
            show_validation = st.checkbox("Show Validation Dashboard", value=True)
            
            calculate_btn = st.button("🎯 Generate Unified Prediction", type="primary", use_container_width=True)
            
            if show_validation:
                st.divider()
                st.subheader("📈 Validation Dashboard")
                
                total_matches = st.session_state.validation_history['match_count']
                match_types = st.session_state.validation_history['match_type_distribution']
                
                if total_matches > 0:
                    st.write(f"**Total Predictions:** {total_matches}")
                    st.write("**Match Type Distribution:**")
                    for match_type, count in match_types.items():
                        percentage = (count / total_matches) * 100
                        st.write(f"  {match_type}: {count} ({percentage:.1f}%)")
                else:
                    st.info("No predictions generated yet")

# ========== MAIN CONTENT ==========
if df is None:
    st.warning("📁 Please add league CSV files to the 'leagues' folder")
    st.info("""
    **Required CSV format:**
    - Columns: team,venue,matches,wins,draws,losses,gf,ga,pts,xg,xga,goals_vs_xg
    - One row per team per venue (home/away)
    - Using sample data for demonstration
    """)
    
    with st.expander("📋 Preview of Loaded Data"):
        st.dataframe(df.head(10))
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Generate Unified Prediction' to start")
    
    # Show league baselines
    st.subheader("📊 League Baselines")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg xG/match", f"{league_baselines['avg_xg']:.2f}")
    with col2:
        st.metric("Std xG", f"{league_baselines['std_xg']:.2f}")
    with col3:
        st.metric("Avg xGA/match", f"{league_baselines['avg_xga']:.2f}")
    with col4:
        st.metric("Std xGA", f"{league_baselines['std_xga']:.2f}")
    
    st.stop()

try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"❌ Team data not found: {e}")
    st.stop()

# ========== PHASE 1: UNIFIED PREDICTION SYSTEM ==========
st.header(f"🎯 {home_team} vs {away_team}")

# Initialize prediction system
prediction_system = UnifiedPredictionSystem(league_baselines)

# Generate prediction
prediction = prediction_system.predict(home_team, away_team, home_stats, away_stats)

# ========== PHASE 2: DISPLAY PREDICTION ==========
# Main prediction card
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    final_prob = prediction['final_probability']
    direction = prediction['direction']
    confidence = prediction['confidence']
    
    # Determine card color based on confidence and direction
    if confidence == "VERY HIGH":
        card_color = "#4CAF50" if direction == "OVER" else "#FF5722"
    elif confidence == "HIGH":
        card_color = "#8BC34A" if direction == "OVER" else "#FF9800"
    elif confidence == "MEDIUM":
        card_color = "#FFC107" if direction == "OVER" else "#FFA726"
    else:
        card_color = "#FFEB3B" if direction == "OVER" else "#FFCC80"
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
        <h1 style="color: white; margin: 0;">{direction} 2.5</h1>
        <div style="font-size: 48px; font-weight: bold; color: white; margin: 10px 0;">{final_prob*100:.1f}%</div>
        <div style="font-size: 18px; color: white;">Confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)

# Match type and rationale
st.subheader("📋 Prediction Analysis")

col_info1, col_info2 = st.columns(2)

with col_info1:
    match_type = prediction['match_type']
    type_colors = {
        "DEFENSIVE_TACTICAL": "#2196F3",
        "ATTACK_DOMINANCE": "#4CAF50",
        "DEFENSIVE_WEAKNESS": "#FF5722",
        "STANDARD": "#FFC107"
    }
    
    st.markdown(f"""
    <div style="background-color: {type_colors.get(match_type, '#9E9E9E')}; 
                color: white; padding: 15px; border-radius: 10px;">
        <h3 style="margin: 0;">Match Type: {match_type}</h3>
        <p style="margin: 10px 0 0 0;">{prediction['classification']['explanation']}</p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    correction = prediction['correction_applied']
    base_prob = prediction['base_probability']
    
    st.markdown("**Correction Analysis**")
    
    if abs(correction) > 0.01:
        if correction > 0:
            st.success(f"📈 **Applied +{correction*100:.1f}% correction**")
            st.write(f"Base model: {base_prob*100:.1f}% → Final: {final_prob*100:.1f}%")
        else:
            st.info(f"📉 **Applied {correction*100:.1f}% correction**")
            st.write(f"Base model: {base_prob*100:.1f}% → Final: {final_prob*100:.1f}%")
    else:
        st.write("🔄 **No correction applied**")
        st.write(f"Trusting base model: {final_prob*100:.1f}%")

# Display rationale
st.info(f"**Rationale:** {prediction['rationale']}")

# ========== PHASE 3: DETAILED ANALYSIS ==========
if show_detailed_analysis:
    st.divider()
    st.header("🔍 Detailed Analysis")
    
    # Team Scores
    col_team1, col_team2 = st.columns(2)
    
    with col_team1:
        st.subheader(f"🏠 {home_team} Analysis")
        home_scores = prediction['team_scores']['home']
        create_team_score_display(home_scores, home_team, True)
        
        # Additional stats
        st.write("**Additional Stats:**")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.metric("xG/match", f"{home_scores['xg_per_match']:.2f}")
        with col_h2:
            st.metric("xGA/match", f"{home_scores['xga_per_match']:.2f}")
    
    with col_team2:
        st.subheader(f"✈️ {away_team} Analysis")
        away_scores = prediction['team_scores']['away']
        create_team_score_display(away_scores, away_team, False)
        
        # Additional stats
        st.write("**Additional Stats:**")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.metric("xG/match", f"{away_scores['xg_per_match']:.2f}")
        with col_a2:
            st.metric("xGA/match", f"{away_scores['xga_per_match']:.2f}")
    
    # Expected Goals Comparison
    st.subheader("🎯 Expected Goals")
    
    col_xg1, col_xg2, col_xg3 = st.columns(3)
    
    with col_xg1:
        home_xg = prediction['expected_goals']['home']
        st.metric(f"{home_team} xG", f"{home_xg:.2f}")
    
    with col_xg2:
        away_xg = prediction['expected_goals']['away']
        st.metric(f"{away_team} xG", f"{away_xg:.2f}")
    
    with col_xg3:
        total_xg = prediction['expected_goals']['total']
        st.metric("Total xG", f"{total_xg:.2f}")
        
        # Scoring expectation
        if total_xg > 3.0:
            st.success("Very high scoring expected")
        elif total_xg > 2.5:
            st.info("High scoring expected")
        elif total_xg > 2.0:
            st.warning("Moderate scoring expected")
        else:
            st.error("Low scoring expected")
    
    # Base vs Final Comparison
    st.subheader("🔄 Model Comparison")
    
    col_base1, col_base2, col_base3 = st.columns(3)
    
    with col_base1:
        st.metric("Base Model", f"{prediction['base_probability']*100:.1f}%")
    
    with col_base2:
        correction_display = f"{prediction['correction_applied']*100:+.1f}%"
        st.metric("Correction Applied", correction_display)
    
    with col_base3:
        st.metric("Final Prediction", f"{prediction['final_probability']*100:.1f}%",
                 delta=f"{prediction['direction']}")
    
    # Confidence Metrics
    st.subheader("📊 Confidence Metrics")
    
    col_conf1, col_conf2, col_conf3 = st.columns(3)
    
    with col_conf1:
        distance = prediction['distance_from_50']
        st.metric("Distance from 50%", f"{distance:.3f}")
        
        if distance > 0.2:
            st.success("Strong signal")
        elif distance > 0.1:
            st.warning("Moderate signal")
        else:
            st.error("Weak signal")
    
    with col_conf2:
        confidence_score = prediction['confidence_score']
        st.metric("Confidence Score", f"{confidence_score:.2f}")
        st.progress(confidence_score)
    
    with col_conf3:
        st.metric("Prediction Strength", prediction['confidence'])
        
        if confidence == "VERY HIGH" or confidence == "HIGH":
            st.success("High reliability")
        elif confidence == "MEDIUM":
            st.warning("Moderate reliability")
        else:
            st.error("Low reliability")
    
    # Match Outcome Probabilities
    st.subheader("🏆 Match Outcome Probabilities")
    
    col_out1, col_out2, col_out3 = st.columns(3)
    
    with col_out1:
        st.metric(f"{home_team} Win", f"{prediction['home_win_prob']*100:.1f}%")
        st.progress(prediction['home_win_prob'])
    
    with col_out2:
        st.metric("Draw", f"{prediction['draw_prob']*100:.1f}%")
        st.progress(prediction['draw_prob'])
    
    with col_out3:
        st.metric(f"{away_team} Win", f"{prediction['away_win_prob']*100:.1f}%")
        st.progress(prediction['away_win_prob'])
    
    # Additional Betting Markets
    st.subheader("💰 Additional Markets")
    
    col_mkt1, col_mkt2 = st.columns(2)
    
    with col_mkt1:
        st.metric("Both Teams to Score", f"{prediction['btts_yes_prob']*100:.1f}%")
        st.progress(prediction['btts_yes_prob'])
    
    with col_mkt2:
        st.metric("Clean Sheet Probability", f"{prediction['btts_no_prob']*100:.1f}%")
        st.progress(prediction['btts_no_prob'])
    
    # Most Likely Scores
    st.subheader("🎯 Most Likely Scores")
    
    prob_matrix = create_probability_matrix(prediction['expected_goals']['home'], 
                                          prediction['expected_goals']['away'])
    
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

# ========== PHASE 4: VALIDATION DASHBOARD ==========
if show_validation and st.session_state.validation_history['match_count'] > 0:
    st.divider()
    st.header("📊 Validation Dashboard")
    
    total_matches = st.session_state.validation_history['match_count']
    match_types = st.session_state.validation_history['match_type_distribution']
    predictions = st.session_state.validation_history['prediction_history']
    
    # Summary Statistics
    col_val1, col_val2, col_val3 = st.columns(3)
    
    with col_val1:
        st.metric("Total Predictions", total_matches)
    
    with col_val2:
        avg_correction = np.mean([p['correction'] for p in predictions]) * 100
        st.metric("Avg Correction", f"{avg_correction:+.1f}%")
    
    with col_val3:
        over_predictions = sum(1 for p in predictions if p['final_prob'] > 0.5)
        over_percentage = (over_predictions / total_matches) * 100
        st.metric("Over Predictions", f"{over_percentage:.1f}%")
    
    # Match Type Distribution
    st.subheader("📈 Match Type Distribution")
    
    type_df = pd.DataFrame({
        'Match Type': list(match_types.keys()),
        'Count': list(match_types.values()),
        'Percentage': [(count / total_matches) * 100 for count in match_types.values()]
    })
    
    if not type_df.empty:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.dataframe(type_df.style.format({'Percentage': '{:.1f}%'}))
        
        with col_chart2:
            # Simple bar chart using matplotlib
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = [type_colors.get(t, '#9E9E9E') for t in type_df['Match Type']]
            bars = ax.bar(type_df['Match Type'], type_df['Count'], color=colors)
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom')
            
            ax.set_ylabel('Count')
            ax.set_title('Match Type Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Correction Analysis
    st.subheader("🔄 Correction Analysis")
    
    corrections = [p['correction'] * 100 for p in predictions]
    
    if corrections:
        col_corr1, col_corr2, col_corr3 = st.columns(3)
        
        with col_corr1:
            avg_corr = np.mean(corrections)
            st.metric("Average Correction", f"{avg_corr:+.1f}%")
        
        with col_corr2:
            max_corr = np.max(corrections)
            st.metric("Maximum Correction", f"{max_corr:+.1f}%")
        
        with col_corr3:
            min_corr = np.min(corrections)
            st.metric("Minimum Correction", f"{min_corr:+.1f}%")
        
        # Correction distribution
        st.write("**Correction Distribution:**")
        
        # Create bins for correction sizes
        correction_bins = {
            "Large Negative (< -20%)": len([c for c in corrections if c < -20]),
            "Moderate Negative (-20% to -5%)": len([c for c in corrections if -20 <= c < -5]),
            "Small (-5% to 5%)": len([c for c in corrections if -5 <= c <= 5]),
            "Moderate Positive (5% to 20%)": len([c for c in corrections if 5 < c <= 20]),
            "Large Positive (> 20%)": len([c for c in corrections if c > 20])
        }
        
        for label, count in correction_bins.items():
            percentage = (count / len(corrections)) * 100
            st.write(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Recent Predictions
    st.subheader("📋 Recent Predictions")
    
    if len(predictions) > 0:
        recent_predictions = predictions[-10:]  # Last 10 predictions
        display_data = []
        
        for pred in recent_predictions:
            display_data.append({
                'Home': pred['home_team'],
                'Away': pred['away_team'],
                'Type': pred['match_type'],
                'Base %': f"{pred['base_prob']*100:.1f}",
                'Final %': f"{pred['final_prob']*100:.1f}",
                'Correction': f"{pred['correction']*100:+.1f}",
                'Confidence': pred['confidence']
            })
        
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)

# ========== PHASE 5: EXPORT PREDICTION ==========
st.divider()
st.header("📤 Export Prediction")

# Create comprehensive summary
summary = f"""
⚽ UNIFIED FOOTBALL PREDICTION
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 FINAL PREDICTION
{direction} 2.5 Goals: {final_prob*100:.1f}%
Confidence: {confidence} (Score: {prediction['confidence_score']:.2f})

📋 MATCH ANALYSIS
Match Type: {match_type}
Rationale: {prediction['rationale']}

⚽ EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

📊 TEAM SCORES (vs League Average)
{home_team} Attack: {prediction['team_scores']['home']['attack_score']:.2f}σ
{home_team} Defense: {prediction['team_scores']['home']['defense_score']:.2f}σ
{away_team} Attack: {prediction['team_scores']['away']['attack_score']:.2f}σ
{away_team} Defense: {prediction['team_scores']['away']['defense_score']:.2f}σ

🔄 MODEL COMPARISON
Base Model Probability: {prediction['base_probability']*100:.1f}%
Applied Correction: {prediction['correction_applied']*100:+.1f}%
Correction Rationale: {prediction['rationale']}

🏆 ADDITIONAL PROBABILITIES
{home_team} Win: {prediction['home_win_prob']*100:.1f}%
Draw: {prediction['draw_prob']*100:.1f}%
{away_team} Win: {prediction['away_win_prob']*100:.1f}%
Both Teams to Score: {prediction['btts_yes_prob']*100:.1f}%

🎯 KEY INSIGHTS
• Distance from 50%: {prediction['distance_from_50']:.3f}
• Prediction Strength: {prediction['confidence']}
• Match Type: {match_type}

🔧 SYSTEM PARAMETERS
League Avg xG: {league_baselines['avg_xg']:.2f}
League Avg xGA: {league_baselines['avg_xga']:.2f}
Max Correction: ±{MAX_CORRECTION*100:.0f}%
Max Regression: ±{MAX_REGRESSION*100:.0f}%

---
Generated by Unified Football xG Predictor
"Goals happen when attacks overcome defenses"
"""

st.code(summary, language="text")

# Export buttons
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name=f"unified_prediction_{home_team}_vs_{away_team}.txt",
        mime="text/plain"
    )

with col_exp2:
    if st.button("🔄 Reset Validation Data"):
        st.session_state.validation_history = {
            'prediction_history': deque(maxlen=200),
            'match_type_distribution': defaultdict(int),
            'correction_effectiveness': defaultdict(lambda: deque(maxlen=50)),
            'match_count': 0
        }
        st.success("Validation data reset!")
        st.rerun()

# ========== FOOTER ==========
st.divider()
footer_text = f"🎯 Unified Prediction: {direction} 2.5 ({final_prob*100:.1f}%) | Match Type: {match_type} | Confidence: {confidence}"
st.caption(footer_text)

# ========== DATA FORMAT INSTRUCTIONS ==========
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
    
    **Match Type Classifications:**
    1. **DEFENSIVE_TACTICAL**: Elite defenses, non-elite attacks
    2. **ATTACK_DOMINANCE**: Elite attack vs weak defense
    3. **DEFENSIVE_WEAKNESS**: Both teams have weak defenses
    4. **STANDARD**: All other matches
    """)
