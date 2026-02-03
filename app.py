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
    page_title="⚽ Football Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚽ Football Intelligence Engine")
st.markdown("""
    **Intelligent Prediction with Goal Source Analysis**
    *Goals happen when capable attacks overcome relevant defenses.*
""")

# ========== UNIVERSAL CONSTANTS ==========
MAX_GOALS = 8
MAX_REGRESSION = 0.3
MIN_PROBABILITY = 0.1
MAX_PROBABILITY = 0.9

# ========== INTELLIGENCE THRESHOLDS ==========
# Defense thresholds
DEFENSE_ELITE = -1.0
DEFENSE_STRONG = -0.5
DEFENSE_GOOD = -0.3
DEFENSE_AVERAGE = 0.0
DEFENSE_WEAK = 0.5
DEFENSE_VERY_WEAK = 1.0

# Attack thresholds
ATTACK_ELITE_PLUS = 1.5
ATTACK_ELITE = 1.0
ATTACK_ABOVE_AVG = 0.5
ATTACK_AVERAGE = 0.0

# Dominance thresholds
DOMINANCE_THRESHOLD = 0.65  # >65% of goals from one team
HIGH_XG_THRESHOLD = 2.4     # Minimum total xG for dominance classification

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "VERY_HIGH": 0.25,
    "HIGH": 0.15,
    "MEDIUM": 0.08,
    "LOW": 0.05
}

# Initialize session state
if 'football_intelligence' not in st.session_state:
    st.session_state.football_intelligence = {
        'prediction_history': deque(maxlen=500),
        'match_patterns': defaultdict(int),
        'correction_applied': defaultdict(int),
        'total_matches': 0
    }

if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

# ========== CORE FUNCTIONS ==========
def factorial_cache(n):
    """Cache factorial calculations"""
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    """Calculate Poisson probability"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

@st.cache_data(ttl=3600)
def load_league_data(league_name):
    """Load league data from CSV"""
    try:
        file_map = {
            "Premier League": "premier_league.csv",
            "Bundesliga": "bundesliga.csv",
            "Serie A": "serie_a.csv",
            "La Liga": "laliga.csv",
            "Ligue 1": "ligue_1.csv",
            "Eredivisie": "eredivisie.csv"
        }
        
        filename = file_map.get(league_name, f"{league_name.lower().replace(' ', '_')}.csv")
        file_path = f"leagues/{filename}"
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required = ['team', 'venue', 'matches', 'xg', 'xga', 'goals_vs_xg']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare home and away stats"""
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    return home_data.set_index('team'), away_data.set_index('team')

def calculate_league_baselines(df):
    """Calculate league statistics for normalization"""
    home_xg = df[df['venue'] == 'home']['xg'] / df[df['venue'] == 'home']['matches']
    away_xg = df[df['venue'] == 'away']['xg'] / df[df['venue'] == 'away']['matches']
    home_xga = df[df['venue'] == 'home']['xga'] / df[df['venue'] == 'home']['matches']
    away_xga = df[df['venue'] == 'away']['xga'] / df[df['venue'] == 'away']['matches']
    
    all_xg = pd.concat([home_xg, away_xg])
    all_xga = pd.concat([home_xga, away_xga])
    
    return {
        'avg_xg': all_xg.mean(),
        'std_xg': max(all_xg.std(), 0.1),
        'avg_xga': all_xga.mean(),
        'std_xga': max(all_xga.std(), 0.1)
    }

# ========== INTELLIGENCE CLASSES ==========
class TeamIntelligence:
    """Analyze team strengths and weaknesses"""
    
    @staticmethod
    def analyze_team(team_stats, league_baselines):
        """Deep team analysis"""
        matches = team_stats['matches']
        
        xg_per_match = team_stats['xg'] / max(matches, 1)
        xga_per_match = team_stats['xga'] / max(matches, 1)
        goals_vs_xg_per_match = team_stats['goals_vs_xg'] / max(matches, 1)
        
        attack_score = (xg_per_match - league_baselines['avg_xg']) / league_baselines['std_xg']
        defense_score = (xga_per_match - league_baselines['avg_xga']) / league_baselines['std_xga']
        
        regression_factor = min(max(goals_vs_xg_per_match, -MAX_REGRESSION), MAX_REGRESSION)
        
        defense_tier = TeamIntelligence._classify_defense(defense_score)
        attack_tier = TeamIntelligence._classify_attack(attack_score)
        
        return {
            'attack_score': attack_score,
            'defense_score': defense_score,
            'attack_tier': attack_tier,
            'defense_tier': defense_tier,
            'regression_factor': regression_factor,
            'xg_per_match': xg_per_match,
            'xga_per_match': xga_per_match
        }
    
    @staticmethod
    def _classify_defense(score):
        """Classify defense tier"""
        if score < DEFENSE_ELITE:
            return "ELITE"
        elif score < DEFENSE_STRONG:
            return "STRONG"
        elif score < DEFENSE_GOOD:
            return "GOOD"
        elif score < DEFENSE_AVERAGE:
            return "AVERAGE"
        elif score < DEFENSE_WEAK:
            return "WEAK"
        else:
            return "VERY_WEAK"
    
    @staticmethod
    def _classify_attack(score):
        """Classify attack tier"""
        if score > ATTACK_ELITE_PLUS:
            return "ELITE_PLUS"
        elif score > ATTACK_ELITE:
            return "ELITE"
        elif score > ATTACK_ABOVE_AVG:
            return "ABOVE_AVG"
        elif score > ATTACK_AVERAGE:
            return "AVERAGE"
        else:
            return "BELOW_AVG"

class MatchIntelligence:
    """Analyze match dynamics with goal source analysis"""
    
    @staticmethod
    def analyze_match(home_analysis, away_analysis, home_xg, away_xg):
        """Intelligent match classification"""
        
        total_xg = home_xg + away_xg
        
        # ========== GOAL SOURCE ANALYSIS ==========
        if total_xg > 0.1:
            home_scoring_share = home_xg / total_xg
            away_scoring_share = away_xg / total_xg
            dominant_share = max(home_scoring_share, away_scoring_share)
            dominant_team = "HOME" if home_scoring_share > away_scoring_share else "AWAY"
        else:
            dominant_share = 0.5
            dominant_team = None
        
        is_dominance_match = (dominant_share > DOMINANCE_THRESHOLD and 
                             total_xg > HIGH_XG_THRESHOLD)
        
        # ========== MATCH TYPE CLASSIFICATION ==========
        home_def = home_analysis['defense_tier']
        away_def = away_analysis['defense_tier']
        home_att = home_analysis['attack_tier']
        away_att = away_analysis['attack_tier']
        
        # 1. DOMINANCE MATCH (One team supplies majority of scoring)
        if is_dominance_match:
            match_type = "DOMINANCE"
            if dominant_team == "HOME":
                explanation = f"Home dominance: {home_team} supplies {dominant_share:.0%} of expected goals"
            else:
                explanation = f"Away dominance: {away_team} supplies {dominant_share:.0%} of expected goals"
        
        # 2. ATTACK DOMINANCE (Elite attack vs Weak defense)
        elif ((home_att in ["ELITE", "ELITE_PLUS"] and away_def in ["WEAK", "VERY_WEAK"]) or
              (away_att in ["ELITE", "ELITE_PLUS"] and home_def in ["WEAK", "VERY_WEAK"])):
            match_type = "ATTACK_DOMINANCE"
            explanation = "Elite attack exploits weak defense"
        
        # 3. GENUINE DEFENSIVE TACTICAL (Strong defense vs Capable attack)
        elif ((home_def in ["ELITE", "STRONG"] and away_att in ["ABOVE_AVG", "ELITE", "ELITE_PLUS"]) or
              (away_def in ["ELITE", "STRONG"] and home_att in ["ABOVE_AVG", "ELITE", "ELITE_PLUS"])):
            match_type = "DEFENSIVE_TACTICAL"
            explanation = "Strong defense faces capable attack"
        
        # 4. FALSE DEFENSIVE (Strong defense vs Weak attack - irrelevant)
        elif ((home_def in ["ELITE", "STRONG", "GOOD"] and away_att in ["BELOW_AVG", "AVERAGE"]) or
              (away_def in ["ELITE", "STRONG", "GOOD"] and home_att in ["BELOW_AVG", "AVERAGE"])):
            match_type = "FALSE_DEFENSIVE"
            explanation = "Strong defense irrelevant against weak attack"
        
        # 5. DEFENSIVE CATASTROPHE
        elif (home_def == "VERY_WEAK" and away_def == "VERY_WEAK"):
            match_type = "DEFENSIVE_CATASTROPHE"
            explanation = "Both defenses incompetent"
        
        # 6. STANDARD
        else:
            match_type = "STANDARD"
            explanation = "Balanced matchup"
        
        return {
            'match_type': match_type,
            'explanation': explanation,
            'is_dominance_match': is_dominance_match,
            'dominant_share': dominant_share,
            'dominant_team': dominant_team,
            'home_def_tier': home_def,
            'away_def_tier': away_def,
            'home_att_tier': home_att,
            'away_att_tier': away_att,
            'total_xg': total_xg,
            'home_xg': home_xg,
            'away_xg': away_xg
        }

class PredictionEngine:
    """Generate base predictions"""
    
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
    
    def predict_expected_goals(self, home_analysis, away_analysis):
        """Calculate expected goals"""
        home_attack = home_analysis['xg_per_match']
        home_defense = home_analysis['xga_per_match']
        away_attack = away_analysis['xg_per_match']
        away_defense = away_analysis['xga_per_match']
        
        home_expected = (home_attack * away_defense) / max(self.league_baselines['avg_xg'], 0.1)
        away_expected = (away_attack * home_defense) / max(self.league_baselines['avg_xg'], 0.1)
        
        home_final = home_expected * (1 + home_analysis['regression_factor'])
        away_final = away_expected * (1 + away_analysis['regression_factor'])
        
        home_final = max(min(home_final, 4.0), 0.3)
        away_final = max(min(away_final, 4.0), 0.3)
        
        return home_final, away_final
    
    def calculate_base_probability(self, total_xg):
        """Base probability from Poisson"""
        prob_under = (poisson_pmf(0, total_xg) + 
                     poisson_pmf(1, total_xg) + 
                     poisson_pmf(2, total_xg))
        return 1 - prob_under

class CorrectionIntelligence:
    """Apply intelligent corrections with proper gating"""
    
    @staticmethod
    def calculate_correction(match_analysis, base_prob):
        """Calculate correction with proper logic gates"""
        
        match_type = match_analysis['match_type']
        correction = 0.0
        rationale = []
        correction_type = "NONE"
        
        # ========== CORRECTION MATRIX ==========
        corrections = {
            "DOMINANCE": 0.0,           # Trust base model
            "ATTACK_DOMINANCE": 0.15,   # +15% for elite vs weak
            "DEFENSIVE_TACTICAL": -0.20, # -20% for genuine defensive
            "FALSE_DEFENSIVE": 0.0,     # No correction - irrelevant
            "DEFENSIVE_CATASTROPHE": 0.25, # +25% for terrible defenses
            "STANDARD": 0.0             # Trust base model
        }
        
        # Get base correction
        if match_type in corrections:
            correction = corrections[match_type]
            correction_type = match_type
            
            # Generate rationale
            if match_type == "DOMINANCE":
                rationale.append(f"Dominance match: {match_analysis['dominant_team']} supplies {match_analysis['dominant_share']:.0%} of goals")
                rationale.append("Trust base model for dominance patterns")
            
            elif match_type == "ATTACK_DOMINANCE":
                rationale.append("Elite attack vs weak defense: +15% correction")
            
            elif match_type == "DEFENSIVE_TACTICAL":
                rationale.append("Genuine defensive context: -20% correction")
            
            elif match_type == "FALSE_DEFENSIVE":
                rationale.append("Strong defense irrelevant against weak attack")
                rationale.append("No correction applied")
            
            elif match_type == "DEFENSIVE_CATASTROPHE":
                rationale.append("Both defenses terrible: +25% correction")
            
            else:
                rationale.append("Standard match: trust base model")
        
        # ========== ELITE ATTACK OVERRIDE ==========
        # If dominant team has elite attack in dominance match
        if match_type == "DOMINANCE" and match_analysis['dominant_team']:
            if match_analysis['dominant_team'] == "HOME":
                att_tier = match_analysis['home_att_tier']
            else:
                att_tier = match_analysis['away_att_tier']
            
            if att_tier in ["ELITE", "ELITE_PLUS"]:
                rationale.append(f"Dominant team has {att_tier} attack - can produce Overs alone")
                correction = 0.0  # Definitely trust base model
        
        # ========== EXTREME WEAK ATTACK ADJUSTMENT ==========
        # If weak attack is extremely weak (< -1.0σ), reduce defensive correction
        if match_type == "DEFENSIVE_TACTICAL":
            # Check if the "capable attack" is actually not that capable
            home_att_score = match_analysis.get('home_attack_score', 0)
            away_att_score = match_analysis.get('away_attack_score', 0)
            
            if (match_analysis['home_def_tier'] in ["ELITE", "STRONG"] and away_att_score < -1.0) or \
               (match_analysis['away_def_tier'] in ["ELITE", "STRONG"] and home_att_score < -1.0):
                correction = -0.10  # Reduced from -20% to -10%
                rationale.append("Attack not truly capable - reduced defensive correction")
        
        # Apply override protection
        correction = CorrectionIntelligence._apply_override_protection(correction, base_prob)
        
        return correction, rationale, correction_type
    
    @staticmethod
    def _apply_override_protection(correction, base_prob):
        """Protect against over-correction"""
        # If base is very high (>70%) and we're applying negative correction
        if base_prob > 0.70 and correction < 0:
            max_negative = -0.15  # Limit to -15%
            correction = max(correction, max_negative)
        
        # If base is very low (<30%) and we're applying positive correction
        elif base_prob < 0.30 and correction > 0:
            max_positive = 0.15  # Limit to +15%
            correction = min(correction, max_positive)
        
        return correction

class ConfidenceIntelligence:
    """Assess prediction confidence"""
    
    @staticmethod
    def assess_confidence(final_prob, match_type, correction):
        """Calculate confidence level"""
        distance = abs(final_prob - 0.5)
        
        # Base confidence from distance
        if distance > CONFIDENCE_THRESHOLDS["VERY_HIGH"]:
            confidence = "VERY HIGH"
            score = 0.9
        elif distance > CONFIDENCE_THRESHOLDS["HIGH"]:
            confidence = "HIGH"
            score = 0.75
        elif distance > CONFIDENCE_THRESHOLDS["MEDIUM"]:
            confidence = "MEDIUM"
            score = 0.6
        elif distance > CONFIDENCE_THRESHOLDS["LOW"]:
            confidence = "LOW"
            score = 0.4
        else:
            confidence = "VERY LOW"
            score = 0.25
        
        # Adjust based on match type and correction
        if match_type in ["DOMINANCE", "ATTACK_DOMINANCE", "DEFENSIVE_CATASTROPHE"]:
            score = min(score * 1.1, 0.95)  # Boost for clear patterns
        
        if abs(correction) > 0.15:
            score = min(score * 1.15, 0.98)  # Boost for strong corrections
        
        return confidence, score, distance

# ========== MAIN ENGINE ==========
class FootballIntelligenceEngine:
    """Main prediction engine"""
    
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
        self.team_intel = TeamIntelligence()
        self.match_intel = MatchIntelligence()
        self.prediction_engine = PredictionEngine(league_baselines)
        self.correction_intel = CorrectionIntelligence()
        self.confidence_intel = ConfidenceIntelligence()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate complete prediction"""
        
        # Team analysis
        home_analysis = self.team_intel.analyze_team(home_stats, self.league_baselines)
        away_analysis = self.team_intel.analyze_team(away_stats, self.league_baselines)
        
        # Expected goals
        home_xg, away_xg = self.prediction_engine.predict_expected_goals(home_analysis, away_analysis)
        total_xg = home_xg + away_xg
        
        # Base probability
        base_prob = self.prediction_engine.calculate_base_probability(total_xg)
        
        # Match intelligence (WITH goal source analysis)
        match_analysis = self.match_intel.analyze_match(home_analysis, away_analysis, home_xg, away_xg)
        
        # Intelligent correction
        correction, correction_rationale, correction_type = self.correction_intel.calculate_correction(
            match_analysis, base_prob
        )
        
        # Apply correction
        final_prob = base_prob + correction
        final_prob = max(min(final_prob, MAX_PROBABILITY), MIN_PROBABILITY)
        
        # Confidence assessment
        confidence, confidence_score, distance = self.confidence_intel.assess_confidence(
            final_prob, match_analysis['match_type'], correction
        )
        
        # Direction
        direction = "OVER" if final_prob > 0.5 else "UNDER"
        
        # Generate rationale
        rationale = FootballIntelligenceEngine._generate_rationale(
            match_analysis, correction, correction_rationale, 
            base_prob, final_prob, confidence, home_xg, away_xg
        )
        
        # Additional probabilities
        prob_matrix = FootballIntelligenceEngine._create_probability_matrix(home_xg, away_xg)
        home_win, draw, away_win = FootballIntelligenceEngine._calculate_outcomes(prob_matrix)
        over_25, under_25, btts_yes, btts_no = FootballIntelligenceEngine._calculate_markets(prob_matrix)
        
        # Store for tracking
        FootballIntelligenceEngine._store_prediction(
            home_team, away_team, match_analysis['match_type'],
            base_prob, final_prob, correction, correction_type, confidence
        )
        
        return {
            'final_probability': final_prob,
            'direction': direction,
            'match_type': match_analysis['match_type'],
            'confidence': confidence,
            'confidence_score': confidence_score,
            'correction': correction,
            'correction_type': correction_type,
            'correction_rationale': correction_rationale,
            'rationale': rationale,
            'base_probability': base_prob,
            'expected_goals': {
                'home': home_xg,
                'away': away_xg,
                'total': total_xg
            },
            'home_analysis': home_analysis,
            'away_analysis': away_analysis,
            'match_analysis': match_analysis,
            'home_win_prob': home_win,
            'draw_prob': draw,
            'away_win_prob': away_win,
            'over_25_prob': over_25,
            'under_25_prob': under_25,
            'btts_yes_prob': btts_yes,
            'btts_no_prob': btts_no,
            'distance_from_50': distance
        }
    
    @staticmethod
    def _create_probability_matrix(home_lam, away_lam):
        """Create probability matrix"""
        prob_matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        
        for i in range(MAX_GOALS + 1):
            for j in range(MAX_GOALS + 1):
                prob_matrix[i, j] = poisson_pmf(i, home_lam) * poisson_pmf(j, away_lam)
        
        total = prob_matrix.sum()
        if total > 0:
            prob_matrix /= total
        
        return prob_matrix
    
    @staticmethod
    def _calculate_outcomes(prob_matrix):
        """Calculate match outcomes"""
        home_win = np.sum(np.triu(prob_matrix, k=1))
        draw = np.sum(np.diag(prob_matrix))
        away_win = np.sum(np.tril(prob_matrix, k=-1))
        
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total
        
        return home_win, draw, away_win
    
    @staticmethod
    def _calculate_markets(prob_matrix):
        """Calculate betting markets"""
        over_25 = under_25 = 0
        btts_yes = btts_no = 0
        
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                prob = prob_matrix[i, j]
                total_goals = i + j
                
                if total_goals > 2.5:
                    over_25 += prob
                else:
                    under_25 += prob
                
                if i >= 1 and j >= 1:
                    btts_yes += prob
                else:
                    btts_no += prob
        
        return over_25, under_25, btts_yes, btts_no
    
    @staticmethod
    def _generate_rationale(match_analysis, correction, correction_rationale, 
                           base_prob, final_prob, confidence, home_xg, away_xg):
        """Generate prediction rationale"""
        rationale_parts = []
        
        rationale_parts.append(f"**{match_analysis['match_type']}**: {match_analysis['explanation']}")
        
        if correction_rationale:
            for reason in correction_rationale:
                rationale_parts.append(f"**{reason}**")
        
        if abs(correction) > 0.01:
            rationale_parts.append(f"Base model adjusted by {correction*100:+.1f}% ({base_prob*100:.1f}% → {final_prob*100:.1f}%)")
        else:
            rationale_parts.append("Trusting base model with no correction")
        
        rationale_parts.append(f"**Confidence**: {confidence}")
        
        # Goal source context
        if match_analysis['is_dominance_match']:
            rationale_parts.append(f"**Goal Source**: {match_analysis['dominant_team']} supplies {match_analysis['dominant_share']:.0%} of expected goals")
        
        return " ".join(rationale_parts)
    
    @staticmethod
    def _store_prediction(home_team, away_team, match_type, base_prob, final_prob, 
                         correction, correction_type, confidence):
        """Store prediction for tracking"""
        intelligence = st.session_state.football_intelligence
        
        intelligence['total_matches'] += 1
        intelligence['match_patterns'][match_type] += 1
        intelligence['correction_applied'][correction_type] += 1
        
        prediction_data = {
            'home_team': home_team,
            'away_team': away_team,
            'match_type': match_type,
            'base_prob': base_prob,
            'final_prob': final_prob,
            'correction': correction,
            'correction_type': correction_type,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        intelligence['prediction_history'].append(prediction_data)

# ========== STREAMLIT UI ==========
# Sidebar
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie"]
    selected_league = st.selectbox("Select League", leagues)
    
    df = load_league_data(selected_league)
    
    if df is not None:
        league_baselines = calculate_league_baselines(df)
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        home_teams = sorted(home_stats_df.index.unique())
        away_teams = sorted(away_stats_df.index.unique())
        common_teams = sorted(list(set(home_teams) & set(away_teams)))
        
        if len(common_teams) == 0:
            st.error("No teams with complete data")
            st.stop()
        
        home_team = st.selectbox("Home Team", common_teams)
        away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
        
        st.divider()
        
        if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
            calculate_btn = True
        else:
            calculate_btn = False

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Generate Prediction'")
    
    # Show league baselines
    st.subheader("📊 League Baselines")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg xG", f"{league_baselines['avg_xg']:.2f}")
    with col2:
        st.metric("Std xG", f"{league_baselines['std_xg']:.2f}")
    with col3:
        st.metric("Avg xGA", f"{league_baselines['avg_xga']:.2f}")
    with col4:
        st.metric("Std xGA", f"{league_baselines['std_xga']:.2f}")
    
    st.stop()

try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"Team data error: {e}")
    st.stop()

# Generate prediction
st.header(f"🎯 {home_team} vs {away_team}")

engine = FootballIntelligenceEngine(league_baselines)
prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)

# Display prediction
final_prob = prediction['final_probability']
direction = prediction['direction']
confidence = prediction['confidence']
match_type = prediction['match_type']

# Prediction card
if direction == "OVER":
    card_color = "#14532D" if confidence == "VERY HIGH" else "#166534" if confidence == "HIGH" else "#365314"
    text_color = "#22C55E" if confidence == "VERY HIGH" else "#4ADE80" if confidence == "HIGH" else "#84CC16"
else:
    card_color = "#7F1D1D" if confidence == "VERY HIGH" else "#991B1B" if confidence == "HIGH" else "#78350F"
    text_color = "#EF4444" if confidence == "VERY HIGH" else "#F87171" if confidence == "HIGH" else "#F59E0B"

st.markdown(f"""
<div style="background-color: {card_color}; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
    <h1 style="color: {text_color}; margin: 0;">{direction} 2.5</h1>
    <div style="font-size: 48px; font-weight: bold; color: white; margin: 10px 0;">
        {final_prob*100:.1f}%
    </div>
    <div style="font-size: 18px; color: white;">
        Confidence: {confidence} | Match Type: {match_type}
    </div>
</div>
""", unsafe_allow_html=True)

# Rationale
st.info(f"**Rationale**: {prediction['rationale']}")

# Detailed analysis
with st.expander("🔍 Detailed Analysis"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team}")
        home_analysis = prediction['home_analysis']
        st.write(f"**Attack**: {home_analysis['attack_tier']} ({home_analysis['attack_score']:.2f}σ)")
        st.write(f"**Defense**: {home_analysis['defense_tier']} ({home_analysis['defense_score']:.2f}σ)")
        st.write(f"**xG/match**: {home_analysis['xg_per_match']:.2f}")
    
    with col2:
        st.subheader(f"✈️ {away_team}")
        away_analysis = prediction['away_analysis']
        st.write(f"**Attack**: {away_analysis['attack_tier']} ({away_analysis['attack_score']:.2f}σ)")
        st.write(f"**Defense**: {away_analysis['defense_tier']} ({away_analysis['defense_score']:.2f}σ)")
        st.write(f"**xG/match**: {away_analysis['xg_per_match']:.2f}")
    
    st.subheader("📊 Model Details")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Base Probability", f"{prediction['base_probability']*100:.1f}%")
    
    with col4:
        correction = prediction['correction']
        if correction > 0:
            st.success(f"Correction: +{correction*100:.1f}%")
        elif correction < 0:
            st.info(f"Correction: {correction*100:.1f}%")
        else:
            st.metric("Correction", "0.0%")
    
    with col5:
        st.metric("Final", f"{final_prob*100:.1f}%", delta=direction)
    
    st.subheader("🎯 Expected Goals")
    col6, col7, col8 = st.columns(3)
    
    with col6:
        st.metric(f"{home_team} xG", f"{prediction['expected_goals']['home']:.2f}")
    
    with col7:
        st.metric(f"{away_team} xG", f"{prediction['expected_goals']['away']:.2f}")
    
    with col8:
        total_xg = prediction['expected_goals']['total']
        st.metric("Total xG", f"{total_xg:.2f}")
        
        # Scoring expectation
        if match_type == "DOMINANCE":
            st.info(f"Dominance match: {prediction['match_analysis']['dominant_team']} supplies {prediction['match_analysis']['dominant_share']:.0%} of goals")
        elif match_type == "DEFENSIVE_TACTICAL":
            st.warning("Genuine defensive context")
        elif match_type == "FALSE_DEFENSIVE":
            st.info("Strong defense irrelevant against weak attack")
    
    st.subheader("💰 Additional Markets")
    col9, col10, col11, col12 = st.columns(4)
    
    with col9:
        st.metric(f"{home_team} Win", f"{prediction['home_win_prob']*100:.1f}%")
    
    with col10:
        st.metric("Draw", f"{prediction['draw_prob']*100:.1f}%")
    
    with col11:
        st.metric(f"{away_team} Win", f"{prediction['away_win_prob']*100:.1f}%")
    
    with col12:
        st.metric("Both Teams Score", f"{prediction['btts_yes_prob']*100:.1f}%")

# System tracking
if st.session_state.football_intelligence['total_matches'] > 0:
    with st.expander("📈 System Tracking"):
        intel = st.session_state.football_intelligence
        
        st.write(f"**Total Predictions**: {intel['total_matches']}")
        
        if intel['match_patterns']:
            st.write("**Match Type Distribution**:")
            for match_type, count in intel['match_patterns'].items():
                percentage = (count / intel['total_matches']) * 100
                st.write(f"- {match_type}: {count} ({percentage:.1f}%)")
        
        if intel['correction_applied']:
            st.write("**Corrections Applied**:")
            for corr_type, count in intel['correction_applied'].items():
                if count > 0:
                    st.write(f"- {corr_type}: {count}")

# Export
st.divider()
st.subheader("📤 Export Prediction")

report = f"""
⚽ FOOTBALL INTELLIGENCE PREDICTION
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 PREDICTION
{direction} 2.5 Goals: {final_prob*100:.1f}%
Confidence: {confidence}
Match Type: {match_type}

🧠 ANALYSIS
{prediction['rationale']}

📊 EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

🔄 MODEL DETAILS
Base Probability: {prediction['base_probability']*100:.1f}%
Correction Applied: {prediction['correction']*100:+.1f}%
Correction Type: {prediction['correction_type']}

---
Generated by Football Intelligence Engine
"""

st.code(report, language="text")

st.download_button(
    label="📥 Download Report",
    data=report,
    file_name=f"prediction_{home_team}_vs_{away_team}.txt",
    mime="text/plain"
)

# Footer
st.divider()
st.caption(f"Football Intelligence Engine | Total Predictions: {st.session_state.football_intelligence['total_matches']}")
