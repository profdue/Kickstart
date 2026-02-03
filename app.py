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
st.title("🧠 Football Intelligence Engine")
st.markdown("""
    **The Smart System: Defense dictates, attack negotiates.**
    *Goals happen when elite attacks overcome incompetent defenses. Everything else is low-scoring.*
""")

# ========== UNIVERSAL CONSTANTS ==========
MAX_GOALS = 8
MAX_REGRESSION = 0.3
MIN_PROBABILITY = 0.1
MAX_PROBABILITY = 0.9

# ========== FOOTBALL INTELLIGENCE THRESHOLDS ==========
# DEFENSE BOUNDARIES (σ scores)
DEFENSE_ELITE = -1.0      # Top 15% - Dominates games
DEFENSE_STRONG = -0.5     # Top 30% - Controls games  
DEFENSE_GOOD = -0.3       # Top 40% - Competent
DEFENSE_AVERAGE = 0.0     # League average
DEFENSE_WEAK = 0.5        # Bottom 30% - Vulnerable
DEFENSE_VERY_WEAK = 1.0   # Bottom 15% - Terrible

# ATTACK BOUNDARIES (σ scores)
ATTACK_ELITE_PLUS = 1.5   # Top 5% - Exceptional
ATTACK_ELITE = 1.0        # Top 15% - Elite
ATTACK_ABOVE_AVG = 0.5    # Top 30% - Above average
ATTACK_AVERAGE = 0.0      # League average

# ========== INTELLIGENT CORRECTION MATRIX ==========
# DEFENSIVE CORRECTIONS (Strong, asymmetric)
DEFENSE_CORRECTIONS = {
    ("ELITE", "ELITE"): -0.35,
    ("ELITE", "STRONG"): -0.30,
    ("ELITE", "GOOD"): -0.28,
    ("ELITE", "AVERAGE"): -0.25,
    ("ELITE", "WEAK"): -0.20,
    ("STRONG", "STRONG"): -0.25,
    ("STRONG", "GOOD"): -0.22,
    ("STRONG", "AVERAGE"): -0.20,
    ("STRONG", "WEAK"): -0.15,
    ("GOOD", "GOOD"): -0.18,
    ("GOOD", "AVERAGE"): -0.15,
    ("GOOD", "WEAK"): -0.10,
}

# ATTACK CORRECTIONS (Weaker, conditional)
ATTACK_CORRECTIONS = {
    ("ELITE_PLUS", "VERY_WEAK"): 0.25,
    ("ELITE_PLUS", "WEAK"): 0.20,
    ("ELITE", "VERY_WEAK"): 0.20,
    ("ELITE", "WEAK"): 0.15,
    ("ELITE", "AVERAGE"): 0.10,
    ("ABOVE_AVG", "VERY_WEAK"): 0.10,
}

# ========== CONFIDENCE THRESHOLDS ==========
CONFIDENCE_THRESHOLDS = {
    "VERY_HIGH": 0.25,
    "HIGH": 0.15,
    "MEDIUM": 0.08,
    "LOW": 0.05,
}

# ========== INITIALIZE SESSION STATE ==========
if 'football_intelligence' not in st.session_state:
    st.session_state.football_intelligence = {
        'prediction_history': deque(maxlen=500),
        'match_patterns': defaultdict(lambda: {'count': 0}),
        'total_matches': 0,
    }

if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

# ========== CORE MATHEMATICS ==========
def factorial_cache(n):
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

# ========== DATA LOADING ==========
@st.cache_data(ttl=3600)
def load_league_data(league_name):
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
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    return home_data.set_index('team'), away_data.set_index('team')

def calculate_league_baselines(df):
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

# ========== FOOTBALL INTELLIGENCE CLASSES ==========
class TeamIntelligence:
    @staticmethod
    def analyze_team(team_stats, league_baselines):
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
            'xga_per_match': xga_per_match,
            'goals_vs_xg_per_match': goals_vs_xg_per_match,
        }
    
    @staticmethod
    def _classify_defense(score):
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
    @staticmethod
    def analyze_match(home_analysis, away_analysis):
        home_def_tier = home_analysis['defense_tier']
        away_def_tier = away_analysis['defense_tier']
        home_att_tier = home_analysis['attack_tier']
        away_att_tier = away_analysis['attack_tier']
        
        defensive_dominance = abs(home_analysis['defense_score'] + away_analysis['defense_score'])
        
        match_type, explanation = MatchIntelligence._classify_match_type(
            home_def_tier, away_def_tier, home_att_tier, away_att_tier
        )
        
        return {
            'match_type': match_type,
            'explanation': explanation,
            'defensive_dominance': defensive_dominance,
            'home_def_tier': home_def_tier,
            'away_def_tier': away_def_tier,
            'home_att_tier': home_att_tier,
            'away_att_tier': away_att_tier,
            'defense_gradient': home_analysis['defense_score'] - away_analysis['defense_score'],
            'attack_gradient': home_analysis['attack_score'] - away_analysis['attack_score']
        }
    
    @staticmethod
    def _classify_match_type(home_def, away_def, home_att, away_att):
        # DEFENSIVE TACTICAL: Any competent defense present
        if (home_def in ["ELITE", "STRONG", "GOOD"] or 
            away_def in ["ELITE", "STRONG", "GOOD"]):
            # Unless BOTH attacks are elite
            if not (home_att in ["ELITE", "ELITE_PLUS"] and 
                   away_att in ["ELITE", "ELITE_PLUS"]):
                return "DEFENSIVE_TACTICAL", "Competent defense dictates tempo"
        
        # ATTACK DOMINANCE: Elite attack vs Weak defense
        if ((home_att in ["ELITE", "ELITE_PLUS"] and away_def in ["WEAK", "VERY_WEAK"]) or
            (away_att in ["ELITE", "ELITE_PLUS"] and home_def in ["WEAK", "VERY_WEAK"])):
            return "ATTACK_DOMINANCE", "Elite attack exploits weak defense"
        
        # DEFENSIVE CATASTROPHE: Both defenses terrible
        if home_def in ["VERY_WEAK"] and away_def in ["VERY_WEAK"]:
            return "DEFENSIVE_CATASTROPHE", "Both defenses incompetent"
        
        return "STANDARD", "Balanced matchup"

class PredictionIntelligence:
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
    
    def predict_expected_goals(self, home_analysis, away_analysis):
        home_attack = home_analysis['xg_per_match']
        home_defense = home_analysis['xga_per_match']
        away_attack = away_analysis['xg_per_match']
        away_defense = away_analysis['xga_per_match']
        
        home_base = (home_attack * away_defense) / max(self.league_baselines['avg_xg'], 0.1)
        away_base = (away_attack * home_defense) / max(self.league_baselines['avg_xg'], 0.1)
        
        home_final = home_base * (1 + home_analysis['regression_factor'])
        away_final = away_base * (1 + away_analysis['regression_factor'])
        
        # Apply bounds
        home_final = max(min(home_final, 4.0), 0.3)
        away_final = max(min(away_final, 4.0), 0.3)
        
        return home_final, away_final
    
    def calculate_base_probability(self, total_xg):
        prob_0 = poisson_pmf(0, total_xg)
        prob_1 = poisson_pmf(1, total_xg)
        prob_2 = poisson_pmf(2, total_xg)
        prob_under = prob_0 + prob_1 + prob_2
        return 1 - prob_under

class CorrectionIntelligence:
    @staticmethod
    def calculate_intelligent_correction(match_analysis, base_prob, home_analysis, away_analysis):
        match_type = match_analysis['match_type']
        home_def_tier = match_analysis['home_def_tier']
        away_def_tier = match_analysis['away_def_tier']
        home_att_tier = match_analysis['home_att_tier']
        away_att_tier = match_analysis['away_att_tier']
        
        correction = 0.0
        rationale = []
        correction_type = "NONE"
        
        if match_type == "DEFENSIVE_TACTICAL":
            correction, sub_rationale = CorrectionIntelligence._calculate_defensive_correction(
                home_def_tier, away_def_tier, home_att_tier, away_att_tier
            )
            rationale.extend(sub_rationale)
            correction_type = "DEFENSIVE"
        
        elif match_type == "ATTACK_DOMINANCE":
            correction, sub_rationale = CorrectionIntelligence._calculate_attack_correction(
                home_att_tier, away_att_tier, home_def_tier, away_def_tier
            )
            rationale.extend(sub_rationale)
            correction_type = "ATTACK"
        
        elif match_type == "DEFENSIVE_CATASTROPHE":
            correction = 0.25
            rationale.append("Both defenses incompetent - high scoring guaranteed")
            correction_type = "CATASTROPHE"
        
        # Apply override protection
        if correction_type == "DEFENSIVE" and correction < 0 and base_prob < 0.40:
            correction = max(correction, -0.10)
            rationale.append("Limited override (low base probability)")
        
        correction = max(min(correction, 0.35), -0.35)
        
        return correction, rationale, correction_type
    
    @staticmethod
    def _calculate_defensive_correction(home_def, away_def, home_att, away_att):
        correction = 0.0
        rationale = []
        
        defense_tiers = {"ELITE": 4, "STRONG": 3, "GOOD": 2, "AVERAGE": 1, "WEAK": 0, "VERY_WEAK": -1}
        home_def_value = defense_tiers.get(home_def, 0)
        away_def_value = defense_tiers.get(away_def, 0)
        
        stronger_def = home_def if home_def_value > away_def_value else away_def
        weaker_def = away_def if home_def_value > away_def_value else home_def
        
        key = (stronger_def, weaker_def)
        if key in DEFENSE_CORRECTIONS:
            correction = DEFENSE_CORRECTIONS[key]
            rationale.append(f"{stronger_def} defense vs {weaker_def} defense: {correction*100:.0f}% reduction")
        else:
            if stronger_def == "ELITE":
                correction = -0.25
            elif stronger_def == "STRONG":
                correction = -0.20
            elif stronger_def == "GOOD":
                correction = -0.15
            rationale.append(f"{stronger_def} defense present: {correction*100:.0f}% reduction")
        
        if home_att in ["BELOW_AVG", "AVERAGE"] and away_att in ["BELOW_AVG", "AVERAGE"]:
            correction -= 0.05
            rationale.append("Both attacks non-elite: additional -5%")
        
        return correction, rationale
    
    @staticmethod
    def _calculate_attack_correction(home_att, away_att, home_def, away_def):
        correction = 0.0
        rationale = []
        
        if home_att in ["ELITE", "ELITE_PLUS"] and away_def in ["WEAK", "VERY_WEAK"]:
            elite_att = home_att
            weak_def = away_def
            attacker = "Home"
        else:
            elite_att = away_att
            weak_def = home_def
            attacker = "Away"
        
        key = (elite_att, weak_def)
        if key in ATTACK_CORRECTIONS:
            correction = ATTACK_CORRECTIONS[key]
            rationale.append(f"{attacker} {elite_att} attack vs {weak_def} defense: +{correction*100:.0f}%")
        else:
            if elite_att == "ELITE_PLUS":
                correction = 0.20
            elif elite_att == "ELITE":
                correction = 0.15
            rationale.append(f"{attacker} {elite_att} attack: +{correction*100:.0f}%")
        
        return correction, rationale

class ConfidenceIntelligence:
    @staticmethod
    def assess_confidence(final_prob, match_type, correction, base_prob):
        distance = abs(final_prob - 0.5)
        
        if distance > CONFIDENCE_THRESHOLDS["VERY_HIGH"]:
            confidence = "VERY HIGH"
            score = 0.95
        elif distance > CONFIDENCE_THRESHOLDS["HIGH"]:
            confidence = "HIGH"
            score = 0.80
        elif distance > CONFIDENCE_THRESHOLDS["MEDIUM"]:
            confidence = "MEDIUM"
            score = 0.65
        elif distance > CONFIDENCE_THRESHOLDS["LOW"]:
            confidence = "LOW"
            score = 0.45
        else:
            confidence = "VERY LOW"
            score = 0.30
        
        if match_type in ["DEFENSIVE_TACTICAL", "DEFENSIVE_CATASTROPHE"]:
            score = min(score * 1.1, 0.95)
            if abs(correction) > 0.15:
                confidence = f"{confidence} (Defensive Context)"
        
        if abs(correction) > 0.20:
            score = min(score * 1.15, 0.98)
            confidence = f"{confidence} (Strong Signal)"
        
        if (base_prob > 0.6 and final_prob < 0.4) or (base_prob < 0.4 and final_prob > 0.6):
            score *= 0.8
            confidence = f"{confidence} (Model Conflict)"
        
        return confidence, min(score, 0.99), distance

# ========== FOOTBALL INTELLIGENCE ENGINE ==========
class FootballIntelligenceEngine:
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
        self.team_intel = TeamIntelligence()
        self.match_intel = MatchIntelligence()
        self.prediction_intel = PredictionIntelligence(league_baselines)
        self.correction_intel = CorrectionIntelligence()
        self.confidence_intel = ConfidenceIntelligence()
    
    def analyze_match(self, home_team, away_team, home_stats, away_stats):
        home_analysis = self.team_intel.analyze_team(home_stats, self.league_baselines)
        away_analysis = self.team_intel.analyze_team(away_stats, self.league_baselines)
        
        match_analysis = self.match_intel.analyze_match(home_analysis, away_analysis)
        
        home_xg, away_xg = self.prediction_intel.predict_expected_goals(home_analysis, away_analysis)
        total_xg = home_xg + away_xg
        base_prob = self.prediction_intel.calculate_base_probability(total_xg)
        
        correction, correction_rationale, correction_type = self.correction_intel.calculate_intelligent_correction(
            match_analysis, base_prob, home_analysis, away_analysis
        )
        
        final_prob = base_prob + correction
        final_prob = max(min(final_prob, MAX_PROBABILITY), MIN_PROBABILITY)
        
        confidence, confidence_score, distance = self.confidence_intel.assess_confidence(
            final_prob, match_analysis['match_type'], correction, base_prob
        )
        
        direction = "OVER" if final_prob > 0.5 else "UNDER"
        
        rationale = self._generate_intelligent_rationale(
            match_analysis, correction, correction_rationale, correction_type,
            base_prob, final_prob, confidence
        )
        
        prob_matrix = self._create_probability_matrix(home_xg, away_xg)
        home_win, draw, away_win = self._calculate_outcome_probabilities(prob_matrix)
        over_25, under_25, btts_yes, btts_no = self._calculate_betting_markets(prob_matrix)
        
        self._store_intelligence(
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
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': total_xg},
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
            'distance_from_50': distance,
            'prediction_strength': "STRONG" if abs(correction) > 0.15 else "MODERATE" if abs(correction) > 0.05 else "WEAK"
        }
    
    def _create_probability_matrix(self, home_lam, away_lam):
        prob_matrix = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        for i in range(MAX_GOALS + 1):
            for j in range(MAX_GOALS + 1):
                prob_home = poisson_pmf(i, home_lam)
                prob_away = poisson_pmf(j, away_lam)
                prob_matrix[i, j] = prob_home * prob_away
        total = prob_matrix.sum()
        if total > 0:
            prob_matrix /= total
        return prob_matrix
    
    def _calculate_outcome_probabilities(self, prob_matrix):
        home_win = np.sum(np.triu(prob_matrix, k=1))
        draw = np.sum(np.diag(prob_matrix))
        away_win = np.sum(np.tril(prob_matrix, k=-1))
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total
        return home_win, draw, away_win
    
    def _calculate_betting_markets(self, prob_matrix):
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
    
    def _generate_intelligent_rationale(self, match_analysis, correction, correction_rationale, 
                                       correction_type, base_prob, final_prob, confidence):
        rationale = []
        rationale.append(f"**{match_analysis['match_type']}**: {match_analysis['explanation']}")
        
        if abs(correction) > 0.01:
            if correction_type == "DEFENSIVE":
                rationale.append(f"**Defensive Intelligence**: {', '.join(correction_rationale)}")
            elif correction_type == "ATTACK":
                rationale.append(f"**Attack Intelligence**: {', '.join(correction_rationale)}")
            else:
                rationale.append(f"**Match Context**: {', '.join(correction_rationale)}")
            
            rationale.append(f"Base model adjusted by {correction*100:+.1f}% ({base_prob*100:.1f}% → {final_prob*100:.1f}%)")
        else:
            rationale.append("**Base Model Trusted**: No significant football intelligence override")
        
        rationale.append(f"**Confidence**: {confidence}")
        
        if match_analysis['match_type'] == "DEFENSIVE_TACTICAL" and final_prob < 0.5:
            rationale.append("**Football Wisdom**: Strong defenses create low-scoring games")
        elif match_analysis['match_type'] == "ATTACK_DOMINANCE" and final_prob > 0.5:
            rationale.append("**Football Wisdom**: Elite attacks exploit weak defenses")
        
        return " ".join(rationale)
    
    def _store_intelligence(self, home_team, away_team, match_type, base_prob, 
                           final_prob, correction, correction_type, confidence):
        intelligence = st.session_state.football_intelligence
        intelligence['total_matches'] += 1
        
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
        pattern_key = f"{match_type}_{correction_type}"
        intelligence['match_patterns'][pattern_key]['count'] += 1

# ========== STREAMLIT UI COMPONENTS ==========
def create_prediction_display_safe(prediction, home_team, away_team):
    """Safe display without HTML issues"""
    final_prob = prediction['final_probability']
    direction = prediction['direction']
    confidence = prediction['confidence']
    match_type = prediction['match_type']
    strength = prediction['prediction_strength']
    
    # Use Streamlit components instead of HTML
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Match type badge
        type_colors = {
            "DEFENSIVE_TACTICAL": "blue",
            "ATTACK_DOMINANCE": "green",
            "DEFENSIVE_CATASTROPHE": "orange",
            "STANDARD": "gray"
        }
        
        st.markdown(f"**Match Type: ** :{type_colors.get(match_type, 'gray')}[{match_type.replace('_', ' ')}]")
        
        # Main prediction
        if direction == "OVER":
            st.success(f"# {direction} 2.5 Goals")
        else:
            st.error(f"# {direction} 2.5 Goals")
        
        # Probability
        st.markdown(f"## {final_prob*100:.1f}%")
        
        # Confidence
        if "VERY HIGH" in confidence or "HIGH" in confidence:
            st.success(f"**Confidence**: {confidence}")
        elif "MEDIUM" in confidence:
            st.warning(f"**Confidence**: {confidence}")
        else:
            st.info(f"**Confidence**: {confidence}")
        
        # Strength
        st.caption(f"**Signal Strength**: {strength}")

def create_team_display_safe(team_analysis, team_name, is_home=True):
    """Safe team display"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Attack", 
                 f"{team_analysis['attack_tier'].replace('_', ' ')}",
                 f"{team_analysis['attack_score']:.2f}σ")
    
    with col2:
        st.metric("Defense",
                 f"{team_analysis['defense_tier']}",
                 f"{team_analysis['defense_score']:.2f}σ")
    
    with col3:
        perf = team_analysis['goals_vs_xg_per_match']
        st.metric("Performance",
                 f"{perf:+.2f}/match",
                 f"{team_analysis['xg_per_match']:.2f} xG")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Match Configuration")
    
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
        
        if st.button("🧠 Generate Intelligent Prediction", type="primary", use_container_width=True):
            calculate_btn = True
        else:
            calculate_btn = False

# ========== MAIN CONTENT ==========
if df is None:
    st.error("Please add CSV files to the leagues/ folder")
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Configure match and click 'Generate Intelligent Prediction'")
    
    # Show league baselines
    st.subheader("📊 League Intelligence Baseline")
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

# ========== GENERATE PREDICTION ==========
try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"Team data error: {e}")
    st.stop()

st.header(f"🧠 {home_team} vs {away_team}")

# Initialize intelligence engine
intelligence_engine = FootballIntelligenceEngine(league_baselines)

# Generate intelligent prediction
with st.spinner("🧠 Applying football intelligence..."):
    prediction = intelligence_engine.analyze_match(home_team, away_team, home_stats, away_stats)

# ========== DISPLAY PREDICTION ==========
create_prediction_display_safe(prediction, home_team, away_team)

# Display rationale
with st.expander("📖 Intelligence Rationale", expanded=True):
    st.markdown(prediction['rationale'])

# ========== INTELLIGENCE ANALYSIS ==========
st.divider()
st.header("🔍 Deep Intelligence Analysis")

# Team Intelligence
col_team1, col_team2 = st.columns(2)

with col_team1:
    st.subheader(f"🏠 {home_team} Intelligence")
    create_team_display_safe(prediction['home_analysis'], home_team, True)

with col_team2:
    st.subheader(f"✈️ {away_team} Intelligence")
    create_team_display_safe(prediction['away_analysis'], away_team, False)

# Matchup Intelligence
st.subheader("⚔️ Matchup Intelligence")

col_match1, col_match2, col_match3 = st.columns(3)

with col_match1:
    diff = prediction['match_analysis']['defense_gradient']
    st.metric("Defense Advantage", 
             f"{home_team if diff < -0.5 else away_team if diff > 0.5 else 'Balanced'}",
             f"{diff:.2f}σ")

with col_match2:
    diff = prediction['match_analysis']['attack_gradient']
    st.metric("Attack Advantage",
             f"{home_team if diff > 0.5 else away_team if diff < -0.5 else 'Balanced'}",
             f"{diff:.2f}σ")

with col_match3:
    dom = prediction['match_analysis']['defensive_dominance']
    if dom > 2.0:
        label = "Extreme Defensive"
    elif dom > 1.0:
        label = "Strong Defensive"
    else:
        label = "Normal"
    st.metric("Defensive Context", label, f"{dom:.2f}σ")

# Model Intelligence
st.subheader("🔄 Model Intelligence")

col_model1, col_model2, col_model3 = st.columns(3)

with col_model1:
    st.metric("Base Model", f"{prediction['base_probability']*100:.1f}%")

with col_model2:
    correction = prediction['correction']
    if correction > 0:
        st.success(f"Intelligent Correction")
        st.metric("Correction", f"+{correction*100:.1f}%")
    elif correction < 0:
        st.info(f"Intelligent Correction")
        st.metric("Correction", f"{correction*100:.1f}%")
    else:
        st.metric("Correction", "0.0%")

with col_model3:
    final = prediction['final_probability']
    base = prediction['base_probability']
    change = final - base
    st.metric("Final Prediction", f"{final*100:.1f}%",
             delta=f"{'OVER' if final > 0.5 else 'UNDER'} (Δ{change*100:+.1f}%)")

# Expected Goals
st.subheader("🎯 Expected Goals Intelligence")

col_xg1, col_xg2, col_xg3 = st.columns(3)

with col_xg1:
    st.metric(f"{home_team} xG", f"{prediction['expected_goals']['home']:.2f}")

with col_xg2:
    st.metric(f"{away_team} xG", f"{prediction['expected_goals']['away']:.2f}")

with col_xg3:
    total_xg = prediction['expected_goals']['total']
    st.metric("Total xG", f"{total_xg:.2f}")
    
    if total_xg > 3.5:
        if prediction['match_type'] == "DEFENSIVE_TACTICAL":
            st.warning("⚠️ High xG but defensive context")
        else:
            st.success("🎯 Very high scoring expected")
    elif total_xg > 2.8:
        st.info("📈 High scoring expected")
    elif total_xg > 2.2:
        st.write("📊 Moderate scoring expected")
    else:
        st.warning("📉 Low scoring expected")

# Additional Markets
st.subheader("💰 Market Intelligence")

col_mkt1, col_mkt2, col_mkt3, col_mkt4 = st.columns(4)

with col_mkt1:
    st.metric("Home Win", f"{prediction['home_win_prob']*100:.1f}%")

with col_mkt2:
    st.metric("Draw", f"{prediction['draw_prob']*100:.1f}%")

with col_mkt3:
    st.metric("Away Win", f"{prediction['away_win_prob']*100:.1f}%")

with col_mkt4:
    st.metric("Both Teams Score", f"{prediction['btts_yes_prob']*100:.1f}%")

# ========== EXPORT ==========
st.divider()
st.header("📤 Export Intelligence Report")

report = f"""
⚽ FOOTBALL INTELLIGENCE REPORT
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 INTELLIGENT PREDICTION
{prediction['direction']} 2.5 Goals: {prediction['final_probability']*100:.1f}%
Confidence: {prediction['confidence']}
Match Type: {prediction['match_type']}

🧠 INTELLIGENCE ANALYSIS
{prediction['rationale']}

📊 TEAM INTELLIGENCE
{home_team}: Attack {prediction['home_analysis']['attack_tier']} ({prediction['home_analysis']['attack_score']:.2f}σ), Defense {prediction['home_analysis']['defense_tier']} ({prediction['home_analysis']['defense_score']:.2f}σ)
{away_team}: Attack {prediction['away_analysis']['attack_tier']} ({prediction['away_analysis']['attack_score']:.2f}σ), Defense {prediction['away_analysis']['defense_tier']} ({prediction['away_analysis']['defense_score']:.2f}σ)

⚽ EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

🔄 MODEL INTELLIGENCE
Base Probability: {prediction['base_probability']*100:.1f}%
Intelligent Correction: {prediction['correction']*100:+.1f}%
"""

st.code(report, language="text")

st.download_button(
    label="📥 Download Intelligence Report",
    data=report,
    file_name=f"football_intelligence_{home_team}_vs_{away_team}.txt",
    mime="text/plain"
)

# ========== FOOTER ==========
st.divider()
st.caption(f"🧠 Football Intelligence Engine | Total Matches Analyzed: {st.session_state.football_intelligence['total_matches']}")
