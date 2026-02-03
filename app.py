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
    # Format: (stronger_defense_tier, weaker_defense_tier): correction
    ("ELITE", "ELITE"): -0.35,      # 0-0, 1-0 games
    ("ELITE", "STRONG"): -0.30,     # Elite dominates strong
    ("ELITE", "GOOD"): -0.28,       # Elite controls good
    ("ELITE", "AVERAGE"): -0.25,    # Elite shuts down average
    ("ELITE", "WEAK"): -0.20,       # Elite limits weak
    ("STRONG", "STRONG"): -0.25,    # Defensive stalemate
    ("STRONG", "GOOD"): -0.22,      # Strong over good
    ("STRONG", "AVERAGE"): -0.20,   # STRONG vs AVERAGE (Udinese-Roma case)
    ("STRONG", "WEAK"): -0.15,      # Strong contains weak
    ("GOOD", "GOOD"): -0.18,        # Dual competence
    ("GOOD", "AVERAGE"): -0.15,     # Good over average
    ("GOOD", "WEAK"): -0.10,        # Good limits weak
}

# ATTACK CORRECTIONS (Weaker, conditional)
ATTACK_CORRECTIONS = {
    # Only apply when attack truly ELITE AND defense truly WEAK
    ("ELITE_PLUS", "VERY_WEAK"): 0.25,   # Exceptional vs Terrible
    ("ELITE_PLUS", "WEAK"): 0.20,        # Exceptional vs Weak
    ("ELITE", "VERY_WEAK"): 0.20,        # Elite vs Terrible
    ("ELITE", "WEAK"): 0.15,             # Elite vs Weak (Marseille-Rennes)
    ("ELITE", "AVERAGE"): 0.10,          # Elite vs Average
    ("ABOVE_AVG", "VERY_WEAK"): 0.10,    # Above avg vs Terrible
}

# ========== CONFIDENCE THRESHOLDS ==========
CONFIDENCE_THRESHOLDS = {
    "VERY_HIGH": 0.25,    # >25% from 50% - Clear signal
    "HIGH": 0.15,         # 15-25% - Strong signal
    "MEDIUM": 0.08,       # 8-15% - Moderate signal  
    "LOW": 0.05,          # 5-8% - Weak signal
}

# ========== OVERRIDE PROTECTION ==========
OVERRIDE_PROTECTION = {
    "DEFENSE_MAX": 0.15,   # Never override by more than 15% for defense
    "ATTACK_MAX": 0.10,    # Never override by more than 10% for attack
    "BASE_HIGH": 0.70,     # Base >70% - limit defensive override
    "BASE_LOW": 0.30,      # Base <30% - limit attack override
}

# ========== INITIALIZE SESSION STATE ==========
if 'football_intelligence' not in st.session_state:
    st.session_state.football_intelligence = {
        'prediction_history': deque(maxlen=500),
        'match_patterns': defaultdict(lambda: {
            'count': 0,
            'avg_goals': deque(maxlen=100),
            'success_rate': deque(maxlen=100)
        }),
        'defensive_success': defaultdict(lambda: deque(maxlen=100)),
        'attack_success': defaultdict(lambda: deque(maxlen=100)),
        'total_matches': 0,
        'performance_metrics': {
            'defensive_correction_accuracy': deque(maxlen=200),
            'attack_correction_accuracy': deque(maxlen=200),
            'base_model_accuracy': deque(maxlen=200)
        }
    }

if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

# ========== CORE MATHEMATICS ==========
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

# ========== DATA LOADING ==========
@st.cache_data(ttl=3600)
def load_league_data(league_name):
    """Load and validate league data"""
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
    # Per-match averages
    home_xg = df[df['venue'] == 'home']['xg'] / df[df['venue'] == 'home']['matches']
    away_xg = df[df['venue'] == 'away']['xg'] / df[df['venue'] == 'away']['matches']
    home_xga = df[df['venue'] == 'home']['xga'] / df[df['venue'] == 'home']['matches']
    away_xga = df[df['venue'] == 'away']['xga'] / df[df['venue'] == 'away']['matches']
    
    # Combine
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
    """Intelligent team analysis with football context"""
    
    @staticmethod
    def analyze_team(team_stats, league_baselines):
        """Deep analysis of team strengths and weaknesses"""
        matches = team_stats['matches']
        
        # Core metrics
        xg_per_match = team_stats['xg'] / max(matches, 1)
        xga_per_match = team_stats['xga'] / max(matches, 1)
        goals_vs_xg_per_match = team_stats['goals_vs_xg'] / max(matches, 1)
        
        # Standardized scores
        attack_score = (xg_per_match - league_baselines['avg_xg']) / league_baselines['std_xg']
        defense_score = (xga_per_match - league_baselines['avg_xga']) / league_baselines['std_xga']
        
        # Regression factor (performance vs expectation)
        regression_factor = min(max(goals_vs_xg_per_match, -MAX_REGRESSION), MAX_REGRESSION)
        
        # Determine tiers
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
            'raw_potential': xg_per_match - xga_per_match
        }
    
    @staticmethod
    def _classify_defense(score):
        """Classify defense based on universal thresholds"""
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
        """Classify attack based on universal thresholds"""
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
    """Intelligent match analysis and classification"""
    
    @staticmethod
    def analyze_match(home_analysis, away_analysis):
        """Deep analysis of match dynamics"""
        
        # Get tiers
        home_def_tier = home_analysis['defense_tier']
        away_def_tier = away_analysis['defense_tier']
        home_att_tier = home_analysis['attack_tier']
        away_att_tier = away_analysis['attack_tier']
        
        # Calculate defensive dominance
        defensive_dominance = abs(home_analysis['defense_score'] + away_analysis['defense_score'])
        
        # Determine match type based on football intelligence
        match_type, explanation = MatchIntelligence._classify_match_type(
            home_def_tier, away_def_tier, home_att_tier, away_att_tier
        )
        
        # Calculate match difficulty for each team
        home_matchup_difficulty = MatchIntelligence._calculate_matchup_difficulty(
            home_analysis, away_analysis
        )
        away_matchup_difficulty = MatchIntelligence._calculate_matchup_difficulty(
            away_analysis, home_analysis
        )
        
        return {
            'match_type': match_type,
            'explanation': explanation,
            'defensive_dominance': defensive_dominance,
            'home_def_tier': home_def_tier,
            'away_def_tier': away_def_tier,
            'home_att_tier': home_att_tier,
            'away_att_tier': away_att_tier,
            'home_matchup_difficulty': home_matchup_difficulty,
            'away_matchup_difficulty': away_matchup_difficulty,
            'defense_gradient': home_analysis['defense_score'] - away_analysis['defense_score'],
            'attack_gradient': home_analysis['attack_score'] - away_analysis['attack_score']
        }
    
    @staticmethod
    def _classify_match_type(home_def, away_def, home_att, away_att):
        """Intelligent match type classification"""
        
        # DEFENSIVE TACTICAL: Any competent defense present
        if (home_def in ["ELITE", "STRONG", "GOOD"] or 
            away_def in ["ELITE", "STRONG", "GOOD"]):
            
            # Unless BOTH attacks are elite (attack cancels defense)
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
        
        # STANDARD: Everything else
        return "STANDARD", "Balanced matchup - follow base probabilities"
    
    @staticmethod
    def _calculate_matchup_difficulty(team_analysis, opponent_analysis):
        """Calculate how difficult this matchup is for the team"""
        # Defense advantage: negative means opponent defense is better
        defense_advantage = opponent_analysis['defense_score'] - team_analysis['attack_score']
        
        # Attack advantage: negative means opponent attack is better
        attack_advantage = team_analysis['defense_score'] - opponent_analysis['attack_score']
        
        # Overall difficulty (lower = harder)
        difficulty = defense_advantage + attack_advantage
        return difficulty

class PredictionIntelligence:
    """Intelligent prediction engine with football context"""
    
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
    
    def predict_expected_goals(self, home_analysis, away_analysis):
        """Intelligent xG prediction with football context"""
        
        # Base calculation
        home_attack = home_analysis['xg_per_match']
        home_defense = home_analysis['xga_per_match']
        away_attack = away_analysis['xg_per_match']
        away_defense = away_analysis['xga_per_match']
        
        home_base = (home_attack * away_defense) / max(self.league_baselines['avg_xg'], 0.1)
        away_base = (away_attack * home_defense) / max(self.league_baselines['avg_xg'], 0.1)
        
        # Apply regression
        home_final = home_base * (1 + home_analysis['regression_factor'])
        away_final = away_base * (1 + away_analysis['regression_factor'])
        
        # Apply intelligent bounds
        home_final = self._apply_intelligent_bounds(home_final, home_analysis, away_analysis, is_home=True)
        away_final = self._apply_intelligent_bounds(away_final, away_analysis, home_analysis, is_home=False)
        
        return home_final, away_final
    
    def _apply_intelligent_bounds(self, xg, team_analysis, opponent_analysis, is_home):
        """Apply football-intelligent bounds to xG predictions"""
        
        # Base bounds
        min_xg = 0.3
        max_xg = 4.0
        
        # Defensive suppression: strong defense reduces opponent xG
        if opponent_analysis['defense_tier'] in ["ELITE", "STRONG"]:
            max_xg = min(max_xg, 2.5)  # Elite defenses rarely concede >2.5
        
        # Attack limitation: weak attack limited even against weak defense
        if team_analysis['attack_tier'] in ["BELOW_AVG", "AVERAGE"]:
            max_xg = min(max_xg, 2.0)
        
        # Home advantage boost (modest)
        if is_home and team_analysis['attack_tier'] in ["ELITE", "ELITE_PLUS"]:
            min_xg = max(min_xg, 1.0)
        
        # Apply bounds
        return max(min(xg, max_xg), min_xg)
    
    def calculate_base_probability(self, total_xg):
        """Base probability from Poisson"""
        prob_0 = poisson_pmf(0, total_xg)
        prob_1 = poisson_pmf(1, total_xg)
        prob_2 = poisson_pmf(2, total_xg)
        
        prob_under = prob_0 + prob_1 + prob_2
        return 1 - prob_under

class CorrectionIntelligence:
    """Intelligent correction system with football wisdom"""
    
    @staticmethod
    def calculate_intelligent_correction(match_analysis, base_prob, home_analysis, away_analysis):
        """Calculate correction based on football intelligence"""
        
        match_type = match_analysis['match_type']
        home_def_tier = match_analysis['home_def_tier']
        away_def_tier = match_analysis['away_def_tier']
        home_att_tier = match_analysis['home_att_tier']
        away_att_tier = match_analysis['away_att_tier']
        
        correction = 0.0
        rationale = []
        correction_type = "NONE"
        
        # DEFENSIVE CORRECTION (applies more often, stronger)
        if match_type == "DEFENSIVE_TACTICAL":
            correction, sub_rationale = CorrectionIntelligence._calculate_defensive_correction(
                home_def_tier, away_def_tier, home_att_tier, away_att_tier
            )
            rationale.extend(sub_rationale)
            correction_type = "DEFENSIVE"
        
        # ATTACK CORRECTION (applies rarely, weaker)
        elif match_type == "ATTACK_DOMINANCE":
            correction, sub_rationale = CorrectionIntelligence._calculate_attack_correction(
                home_att_tier, away_att_tier, home_def_tier, away_def_tier
            )
            rationale.extend(sub_rationale)
            correction_type = "ATTACK"
        
        # CATASTROPHE CORRECTION (both defenses terrible)
        elif match_type == "DEFENSIVE_CATASTROPHE":
            correction = 0.25  # +25% for terrible defenses
            rationale.append("Both defenses incompetent - high scoring guaranteed")
            correction_type = "CATASTROPHE"
        
        # Apply override protection
        correction = CorrectionIntelligence._apply_override_protection(
            correction, base_prob, correction_type
        )
        
        # Final bounds
        correction = max(min(correction, 0.35), -0.35)
        
        return correction, rationale, correction_type
    
    @staticmethod
    def _calculate_defensive_correction(home_def, away_def, home_att, away_att):
        """Calculate defensive correction with football intelligence"""
        correction = 0.0
        rationale = []
        
        # Determine stronger and weaker defenses
        defense_tiers = {"ELITE": 4, "STRONG": 3, "GOOD": 2, "AVERAGE": 1, "WEAK": 0, "VERY_WEAK": -1}
        
        home_def_value = defense_tiers.get(home_def, 0)
        away_def_value = defense_tiers.get(away_def, 0)
        
        stronger_def = home_def if home_def_value > away_def_value else away_def
        weaker_def = away_def if home_def_value > away_def_value else home_def
        
        # Look up correction
        key = (stronger_def, weaker_def)
        if key in DEFENSE_CORRECTIONS:
            correction = DEFENSE_CORRECTIONS[key]
            rationale.append(f"{stronger_def} defense vs {weaker_def} defense: {correction*100:.0f}% reduction")
        else:
            # Default defensive correction based on strongest defense
            if stronger_def == "ELITE":
                correction = -0.25
            elif stronger_def == "STRONG":
                correction = -0.20
            elif stronger_def == "GOOD":
                correction = -0.15
            rationale.append(f"{stronger_def} defense present: {correction*100:.0f}% reduction")
        
        # Additional penalty if both attacks are weak
        if home_att in ["BELOW_AVG", "AVERAGE"] and away_att in ["BELOW_AVG", "AVERAGE"]:
            correction -= 0.05
            rationale.append("Both attacks non-elite: additional -5%")
        
        return correction, rationale
    
    @staticmethod
    def _calculate_attack_correction(home_att, away_att, home_def, away_def):
        """Calculate attack correction with football intelligence"""
        correction = 0.0
        rationale = []
        
        # Determine elite attack and corresponding weak defense
        if home_att in ["ELITE", "ELITE_PLUS"] and away_def in ["WEAK", "VERY_WEAK"]:
            elite_att = home_att
            weak_def = away_def
            attacker = "Home"
        else:
            elite_att = away_att
            weak_def = home_def
            attacker = "Away"
        
        # Look up correction
        key = (elite_att, weak_def)
        if key in ATTACK_CORRECTIONS:
            correction = ATTACK_CORRECTIONS[key]
            rationale.append(f"{attacker} {elite_att} attack vs {weak_def} defense: +{correction*100:.0f}%")
        else:
            # Default attack correction
            if elite_att == "ELITE_PLUS":
                correction = 0.20
            elif elite_att == "ELITE":
                correction = 0.15
            rationale.append(f"{attacker} {elite_att} attack: +{correction*100:.0f}%")
        
        return correction, rationale
    
    @staticmethod
    def _apply_override_protection(correction, base_prob, correction_type):
        """Protect against over-correction"""
        
        if correction_type == "DEFENSIVE" and correction < 0:
            # Base already very low - limit defensive override
            if base_prob < 0.40:
                max_negative = -min(abs(correction), 0.10)
                if correction < max_negative:
                    correction = max_negative
            
            # Base already very high - still allow meaningful correction
            elif base_prob > OVERRIDE_PROTECTION["BASE_HIGH"]:
                max_negative = -min(abs(correction), OVERRIDE_PROTECTION["DEFENSE_MAX"])
                if correction < max_negative:
                    correction = max_negative
        
        elif correction_type == "ATTACK" and correction > 0:
            # Base already very high - limit attack override
            if base_prob > 0.60:
                max_positive = min(correction, 0.08)
                if correction > max_positive:
                    correction = max_positive
            
            # Base very low - still limit
            elif base_prob < OVERRIDE_PROTECTION["BASE_LOW"]:
                max_positive = min(correction, OVERRIDE_PROTECTION["ATTACK_MAX"])
                if correction > max_positive:
                    correction = max_positive
        
        return correction

class ConfidenceIntelligence:
    """Intelligent confidence assessment"""
    
    @staticmethod
    def assess_confidence(final_prob, match_type, correction, base_prob):
        """Assess prediction confidence with football context"""
        
        distance = abs(final_prob - 0.5)
        
        # Base confidence from distance
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
        
        # Adjust based on match type
        if match_type in ["DEFENSIVE_TACTICAL", "DEFENSIVE_CATASTROPHE"]:
            # Defensive matches more predictable
            score = min(score * 1.1, 0.95)
            if abs(correction) > 0.15:
                confidence = f"{confidence} (Defensive Context)"
        
        # Adjust based on correction magnitude
        if abs(correction) > 0.20:
            score = min(score * 1.15, 0.98)
            confidence = f"{confidence} (Strong Signal)"
        
        # Penalize when base and final disagree strongly
        if (base_prob > 0.6 and final_prob < 0.4) or (base_prob < 0.4 and final_prob > 0.6):
            score *= 0.8
            confidence = f"{confidence} (Model Conflict)"
        
        return confidence, min(score, 0.99), distance

# ========== FOOTBALL INTELLIGENCE ENGINE ==========
class FootballIntelligenceEngine:
    """Main intelligence engine that orchestrates everything"""
    
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
        self.team_intel = TeamIntelligence()
        self.match_intel = MatchIntelligence()
        self.prediction_intel = PredictionIntelligence(league_baselines)
        self.correction_intel = CorrectionIntelligence()
        self.confidence_intel = ConfidenceIntelligence()
    
    def analyze_match(self, home_team, away_team, home_stats, away_stats):
        """Complete intelligent match analysis"""
        
        # PHASE 1: Team Intelligence
        home_analysis = self.team_intel.analyze_team(home_stats, self.league_baselines)
        away_analysis = self.team_intel.analyze_team(away_stats, self.league_baselines)
        
        # PHASE 2: Match Intelligence
        match_analysis = self.match_intel.analyze_match(home_analysis, away_analysis)
        
        # PHASE 3: Base Prediction
        home_xg, away_xg = self.prediction_intel.predict_expected_goals(home_analysis, away_analysis)
        total_xg = home_xg + away_xg
        base_prob = self.prediction_intel.calculate_base_probability(total_xg)
        
        # PHASE 4: Intelligent Correction
        correction, correction_rationale, correction_type = self.correction_intel.calculate_intelligent_correction(
            match_analysis, base_prob, home_analysis, away_analysis
        )
        
        # Apply correction
        final_prob = base_prob + correction
        final_prob = max(min(final_prob, MAX_PROBABILITY), MIN_PROBABILITY)
        
        # PHASE 5: Confidence Intelligence
        confidence, confidence_score, distance = self.confidence_intel.assess_confidence(
            final_prob, match_analysis['match_type'], correction, base_prob
        )
        
        # PHASE 6: Additional Calculations
        direction = "OVER" if final_prob > 0.5 else "UNDER"
        
        # Generate comprehensive rationale
        rationale = self._generate_intelligent_rationale(
            match_analysis, correction, correction_rationale, correction_type,
            base_prob, final_prob, confidence
        )
        
        # Calculate additional probabilities
        prob_matrix = self._create_probability_matrix(home_xg, away_xg)
        home_win, draw, away_win = self._calculate_outcome_probabilities(prob_matrix)
        over_25, under_25, btts_yes, btts_no = self._calculate_betting_markets(prob_matrix)
        
        # Store for intelligence learning
        self._store_intelligence(
            home_team, away_team, match_analysis['match_type'],
            base_prob, final_prob, correction, correction_type, confidence,
            home_xg + away_xg  # Total goals for learning
        )
        
        return {
            # Core Prediction
            'final_probability': final_prob,
            'direction': direction,
            'match_type': match_analysis['match_type'],
            'confidence': confidence,
            'confidence_score': confidence_score,
            
            # Intelligence Analysis
            'correction': correction,
            'correction_type': correction_type,
            'correction_rationale': correction_rationale,
            'rationale': rationale,
            
            # Base Model
            'base_probability': base_prob,
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': total_xg},
            
            # Team Intelligence
            'home_analysis': home_analysis,
            'away_analysis': away_analysis,
            'match_analysis': match_analysis,
            
            # Additional Probabilities
            'home_win_prob': home_win,
            'draw_prob': draw,
            'away_win_prob': away_win,
            'over_25_prob': over_25,
            'under_25_prob': under_25,
            'btts_yes_prob': btts_yes,
            'btts_no_prob': btts_no,
            
            # Metrics
            'distance_from_50': distance,
            'prediction_strength': "STRONG" if abs(correction) > 0.15 else "MODERATE" if abs(correction) > 0.05 else "WEAK"
        }
    
    def _create_probability_matrix(self, home_lam, away_lam):
        """Create probability matrix"""
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
        """Calculate match outcome probabilities"""
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
        """Calculate betting market probabilities"""
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
        """Generate intelligent prediction rationale"""
        
        rationale = []
        
        # Match type context
        rationale.append(f"**{match_analysis['match_type']}**: {match_analysis['explanation']}")
        
        # Correction explanation
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
        
        # Confidence context
        rationale.append(f"**Confidence**: {confidence}")
        
        # Football wisdom
        if match_analysis['match_type'] == "DEFENSIVE_TACTICAL" and final_prob < 0.5:
            rationale.append("**Football Wisdom**: Strong defenses create low-scoring games")
        elif match_analysis['match_type'] == "ATTACK_DOMINANCE" and final_prob > 0.5:
            rationale.append("**Football Wisdom**: Elite attacks exploit weak defenses")
        
        return " ".join(rationale)
    
    def _store_intelligence(self, home_team, away_team, match_type, base_prob, 
                           final_prob, correction, correction_type, confidence, total_xg):
        """Store prediction for intelligence learning"""
        
        intelligence = st.session_state.football_intelligence
        
        # Store basic prediction
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
            'total_xg': total_xg,
            'timestamp': datetime.now()
        }
        
        intelligence['prediction_history'].append(prediction_data)
        
        # Store pattern for this match type
        pattern_key = f"{match_type}_{correction_type}"
        intelligence['match_patterns'][pattern_key]['count'] += 1
        
        # Learning: track when corrections work
        if abs(correction) > 0.10:
            correction_key = f"{correction_type}_{'HIGH' if abs(correction) > 0.20 else 'MEDIUM'}"
            
            if correction_type == "DEFENSIVE" and correction < 0:
                intelligence['defensive_success'][correction_key].append(1)  # Will be updated with actual results
            elif correction_type == "ATTACK" and correction > 0:
                intelligence['attack_success'][correction_key].append(1)

# ========== STREAMLIT UI COMPONENTS ==========
def create_team_intelligence_display(team_analysis, team_name, is_home=True):
    """Display team intelligence analysis"""
    
    colors = {
        "ELITE_PLUS": "#00FF00", "ELITE": "#90EE90", "ABOVE_AVG": "#ADFF2F",
        "AVERAGE": "#FFFF00", "BELOW_AVG": "#FFA500",
        "ELITE": "#00FF00", "STRONG": "#90EE90", "GOOD": "#ADFF2F",
        "AVERAGE": "#FFFF00", "WEAK": "#FFA500", "VERY_WEAK": "#FF4500"
    }
    
    bg_color = "#1E3A8A" if is_home else "#7C2D12"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        att_tier = team_analysis['attack_tier']
        att_score = team_analysis['attack_score']
        st.markdown(f"""
        <div style="background-color: {bg_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px; opacity: 0.9;">Attack Intelligence</div>
            <div style="font-size: 24px; font-weight: bold; color: {colors.get(att_tier, '#FFFFFF')};">
                {att_tier.replace('_', ' ')}
            </div>
            <div style="font-size: 16px; margin-top: 5px;">{att_score:.2f}σ</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">
                {team_analysis['xg_per_match']:.2f} xG/match
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        def_tier = team_analysis['defense_tier']
        def_score = team_analysis['defense_score']
        st.markdown(f"""
        <div style="background-color: {bg_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px; opacity: 0.9;">Defense Intelligence</div>
            <div style="font-size: 24px; font-weight: bold; color: {colors.get(def_tier, '#FFFFFF')};">
                {def_tier}
            </div>
            <div style="font-size: 16px; margin-top: 5px;">{def_score:.2f}σ</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">
                {team_analysis['xga_per_match']:.2f} xGA/match
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        perf = team_analysis['goals_vs_xg_per_match']
        perf_color = "#00FF00" if perf > 0 else "#FF4500" if perf < 0 else "#FFFF00"
        st.markdown(f"""
        <div style="background-color: {bg_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 14px; opacity: 0.9;">Performance vs Expectation</div>
            <div style="font-size: 24px; font-weight: bold; color: {perf_color};">
                {perf:+.2f}/match
            </div>
            <div style="font-size: 16px; margin-top: 5px;">{team_analysis['regression_factor']:.2f} reg factor</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">
                Raw Potential: {team_analysis['raw_potential']:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_prediction_display(prediction, home_team, away_team):
    """Display the intelligent prediction"""
    
    final_prob = prediction['final_probability']
    direction = prediction['direction']
    confidence = prediction['confidence']
    match_type = prediction['match_type']
    strength = prediction['prediction_strength']
    
    # Color logic
    if direction == "OVER":
        if confidence == "VERY HIGH":
            card_color = "#14532D"  # Dark green
            text_color = "#22C55E"  # Bright green
        elif confidence == "HIGH":
            card_color = "#166534"  # Green
            text_color = "#4ADE80"  # Light green
        else:
            card_color = "#365314"  # Olive
            text_color = "#84CC16"  # Lime
    else:  # UNDER
        if confidence == "VERY HIGH":
            card_color = "#7F1D1D"  # Dark red
            text_color = "#EF4444"  # Bright red
        elif confidence == "HIGH":
            card_color = "#991B1B"  # Red
            text_color = "#F87171"  # Light red
        else:
            card_color = "#78350F"  # Brown
            text_color = "#F59E0B"  # Amber
    
    # Match type indicator
    type_colors = {
        "DEFENSIVE_TACTICAL": "#3B82F6",  # Blue
        "ATTACK_DOMINANCE": "#10B981",    # Emerald
        "DEFENSIVE_CATASTROPHE": "#F59E0B", # Amber
        "STANDARD": "#6B7280"              # Gray
    }
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 30px; border-radius: 20px; 
                text-align: center; margin: 20px 0; position: relative; border: 3px solid {text_color};">
        
        <div style="position: absolute; top: 10px; left: 15px; background-color: {type_colors.get(match_type, '#6B7280')}; 
                    color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px;">
            {match_type.replace('_', ' ')}
        </div>
        
        <div style="position: absolute; top: 10px; right: 15px; background-color: rgba(255,255,255,0.2); 
                    color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px;">
            {strength} SIGNAL
        </div>
        
        <h1 style="color: {text_color}; margin: 30px 0 10px 0; font-size: 42px;">
            {direction} 2.5 GOALS
        </h1>
        
        <div style="font-size: 72px; font-weight: bold; color: white; margin: 10px 0;">
            {final_prob*100:.1f}%
        </div>
        
        <div style="font-size: 20px; color: white; margin: 10px 0;">
            Confidence: <span style="color: {text_color}; font-weight: bold;">{confidence}</span>
        </div>
        
        <div style="font-size: 16px; color: rgba(255,255,255,0.8); margin-top: 20px;">
            {home_team} vs {away_team}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1>🧠</h1>
        <h3>Football Intelligence</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Match Configuration")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie"]
    selected_league = st.selectbox("Select League", leagues, key="league_select")
    
    # Load data
    df = load_league_data(selected_league)
    
    if df is not None:
        league_baselines = calculate_league_baselines(df)
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        # Get available teams
        home_teams = sorted(home_stats_df.index.unique())
        away_teams = sorted(away_stats_df.index.unique())
        common_teams = sorted(list(set(home_teams) & set(away_teams)))
        
        if len(common_teams) == 0:
            st.error("No teams with complete data")
            st.stop()
        
        home_team = st.selectbox("Home Team", common_teams, key="home_select")
        away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team], key="away_select")
        
        st.divider()
        
        st.header("🎯 Intelligence Options")
        
        show_intelligence = st.checkbox("Show Detailed Intelligence", value=True)
        show_learning = st.checkbox("Show System Learning", value=False)
        
        st.divider()
        
        if st.button("🧠 Generate Intelligent Prediction", type="primary", use_container_width=True):
            calculate_btn = True
        else:
            calculate_btn = False

# ========== MAIN CONTENT ==========
if df is None:
    st.error("""
    ## 📁 Data Required
    
    Please place CSV files in the `leagues/` folder with format:
    
    Required columns: `team, venue, matches, xg, xga, goals_vs_xg`
    
    Example:
    ```
    team,venue,matches,xg,xga,goals_vs_xg
    Arsenal,home,12,25.86,8.64,-2.14
    Arsenal,away,12,23.43,10.15,5.43
    ```
    """)
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Configure match and click 'Generate Intelligent Prediction'")
    
    # Show league intelligence
    st.subheader("📊 League Intelligence Baseline")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg xG", f"{league_baselines['avg_xg']:.2f}", 
                 delta="Attack Baseline")
    with col2:
        st.metric("Std xG", f"{league_baselines['std_xg']:.2f}",
                 delta="Attack Variance")
    with col3:
        st.metric("Avg xGA", f"{league_baselines['avg_xga']:.2f}",
                 delta="Defense Baseline")
    with col4:
        st.metric("Std xGA", f"{league_baselines['std_xga']:.2f}",
                 delta="Defense Variance")
    
    st.divider()
    
    # Show football intelligence principles
    st.subheader("🧠 Football Intelligence Principles")
    
    principles = """
    1. **Defense Dictates**: Strong defenses create low-scoring games
    2. **Attack Negotiates**: Elite attacks only win against weak defenses  
    3. **Asymmetric Corrections**: Defense corrections stronger than attack
    4. **Competence Threshold**: Any defense better than average matters
    5. **Elite Rarity**: True elite attacks are rare (<15% of teams)
    """
    
    st.markdown(principles)
    
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
create_prediction_display(prediction, home_team, away_team)

# Display rationale
with st.expander("📖 Intelligence Rationale", expanded=True):
    st.markdown(f"**{prediction['rationale']}**")
    
    if prediction['correction_rationale']:
        st.markdown("### Correction Intelligence")
        for rationale in prediction['correction_rationale']:
            st.write(f"• {rationale}")

# ========== INTELLIGENCE ANALYSIS ==========
if show_intelligence:
    st.divider()
    st.header("🔍 Deep Intelligence Analysis")
    
    # Team Intelligence
    col_team1, col_team2 = st.columns(2)
    
    with col_team1:
        st.subheader(f"🏠 {home_team} Intelligence")
        create_team_intelligence_display(prediction['home_analysis'], home_team, True)
    
    with col_team2:
        st.subheader(f"✈️ {away_team} Intelligence")
        create_team_intelligence_display(prediction['away_analysis'], away_team, False)
    
    # Matchup Intelligence
    st.subheader("⚔️ Matchup Intelligence")
    
    col_match1, col_match2, col_match3 = st.columns(3)
    
    with col_match1:
        diff = prediction['match_analysis']['defense_gradient']
        if diff < -0.5:
            st.success(f"**Defense Advantage**: {home_team}")
            st.metric("Defense Gradient", f"{diff:.2f}σ")
        elif diff > 0.5:
            st.info(f"**Defense Advantage**: {away_team}")
            st.metric("Defense Gradient", f"{diff:.2f}σ")
        else:
            st.write("**Defense**: Balanced")
            st.metric("Defense Gradient", f"{diff:.2f}σ")
    
    with col_match2:
        diff = prediction['match_analysis']['attack_gradient']
        if diff < -0.5:
            st.success(f"**Attack Advantage**: {away_team}")
            st.metric("Attack Gradient", f"{diff:.2f}σ")
        elif diff > 0.5:
            st.info(f"**Attack Advantage**: {home_team}")
            st.metric("Attack Gradient", f"{diff:.2f}σ")
        else:
            st.write("**Attack**: Balanced")
            st.metric("Attack Gradient", f"{diff:.2f}σ")
    
    with col_match3:
        dom = prediction['match_analysis']['defensive_dominance']
        if dom > 2.0:
            st.error("**Extreme Defensive Matchup**")
            st.metric("Defensive Dominance", f"{dom:.2f}σ")
        elif dom > 1.0:
            st.warning("**Strong Defensive Matchup**")
            st.metric("Defensive Dominance", f"{dom:.2f}σ")
        else:
            st.write("**Defensive Context**: Normal")
            st.metric("Defensive Dominance", f"{dom:.2f}σ")
    
    # Model Intelligence
    st.subheader("🔄 Model Intelligence")
    
    col_model1, col_model2, col_model3 = st.columns(3)
    
    with col_model1:
        st.metric("Base Model", f"{prediction['base_probability']*100:.1f}%",
                 delta="Poisson Foundation")
    
    with col_model2:
        correction = prediction['correction']
        if correction > 0:
            st.success(f"Intelligent Correction", help="Attack intelligence applied")
            st.metric("Correction", f"+{correction*100:.1f}%")
        elif correction < 0:
            st.info(f"Intelligent Correction", help="Defensive intelligence applied")
            st.metric("Correction", f"{correction*100:.1f}%")
        else:
            st.metric("Correction", "0.0%", delta="No override needed")
    
    with col_model3:
        final = prediction['final_probability']
        base = prediction['base_probability']
        change = final - base
        st.metric("Final Prediction", f"{final*100:.1f}%",
                 delta=f"{'OVER' if final > 0.5 else 'UNDER'} (Δ{change*100:+.1f}%)")
    
    # Expected Goals with Intelligence
    st.subheader("🎯 Expected Goals Intelligence")
    
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
        
        # Scoring expectation with intelligence
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
    
    # Most Likely Scores
    st.subheader("🎯 Most Probable Outcomes")
    
    # Create simple score probabilities
    home_xg_val = prediction['expected_goals']['home']
    away_xg_val = prediction['expected_goals']['away']
    
    scores = []
    for i in range(4):
        for j in range(4):
            prob = poisson_pmf(i, home_xg_val) * poisson_pmf(j, away_xg_val)
            if prob > 0.01:
                scores.append(((i, j), prob))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    cols = st.columns(6)
    for idx, ((home_g, away_g), prob) in enumerate(scores[:6]):
        with cols[idx % 6]:
            st.metric(f"{home_g}-{away_g}", f"{prob*100:.1f}%",
                     delta="Most Likely" if idx == 0 else None)

# ========== SYSTEM LEARNING ==========
if show_learning and st.session_state.football_intelligence['total_matches'] > 0:
    st.divider()
    st.header("📈 System Intelligence Learning")
    
    intel = st.session_state.football_intelligence
    
    col_learn1, col_learn2, col_learn3 = st.columns(3)
    
    with col_learn1:
        st.metric("Total Matches Analyzed", intel['total_matches'])
    
    with col_learn2:
        patterns = len(intel['match_patterns'])
        st.metric("Patterns Learned", patterns)
    
    with col_learn3:
        corrections = sum(1 for p in intel['prediction_history'] if abs(p['correction']) > 0.05)
        correction_pct = (corrections / intel['total_matches']) * 100
        st.metric("Intelligence Applied", f"{correction_pct:.1f}%")
    
    # Show match type distribution
    if intel['match_patterns']:
        st.subheader("📊 Match Type Intelligence")
        
        pattern_data = []
        for pattern, data in intel['match_patterns'].items():
            if data['count'] > 0:
                pattern_data.append({
                    'Pattern': pattern,
                    'Count': data['count'],
                    'Frequency': (data['count'] / intel['total_matches']) * 100
                })
        
        if pattern_data:
            df_patterns = pd.DataFrame(pattern_data)
            st.dataframe(df_patterns.style.format({'Frequency': '{:.1f}%'}))

# ========== EXPORT ==========
st.divider()
st.header("📤 Export Intelligence Report")

# Create comprehensive report
report = f"""
⚽ FOOTBALL INTELLIGENCE REPORT
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 INTELLIGENT PREDICTION
{prediction['direction']} 2.5 Goals: {prediction['final_probability']*100:.1f}%
Confidence: {prediction['confidence']} (Score: {prediction['confidence_score']:.2f})
Prediction Strength: {prediction['prediction_strength']}
Match Type: {prediction['match_type']}

🧠 INTELLIGENCE ANALYSIS
{prediction['rationale']}

📊 TEAM INTELLIGENCE
{home_team}:
  Attack: {prediction['home_analysis']['attack_tier']} ({prediction['home_analysis']['attack_score']:.2f}σ)
  Defense: {prediction['home_analysis']['defense_tier']} ({prediction['home_analysis']['defense_score']:.2f}σ)
  Performance: {prediction['home_analysis']['goals_vs_xg_per_match']:+.2f}/match

{away_team}:
  Attack: {prediction['away_analysis']['attack_tier']} ({prediction['away_analysis']['attack_score']:.2f}σ)
  Defense: {prediction['away_analysis']['defense_tier']} ({prediction['away_analysis']['defense_score']:.2f}σ)
  Performance: {prediction['away_analysis']['goals_vs_xg_per_match']:+.2f}/match

⚽ EXPECTED GOALS (Intelligent)
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

🔄 MODEL INTELLIGENCE
Base Probability: {prediction['base_probability']*100:.1f}%
Intelligent Correction: {prediction['correction']*100:+.1f}%
Correction Type: {prediction['correction_type']}
Correction Rationale: {', '.join(prediction['correction_rationale'])}

🎯 ADDITIONAL PROBABILITIES
Match Outcome: {home_team} {prediction['home_win_prob']*100:.1f}% | Draw {prediction['draw_prob']*100:.1f}% | {away_team} {prediction['away_win_prob']*100:.1f}%
Both Teams to Score: {prediction['btts_yes_prob']*100:.1f}%

📈 KEY METRICS
Distance from 50%: {prediction['distance_from_50']:.3f}
Defensive Dominance: {prediction['match_analysis']['defensive_dominance']:.2f}σ
Defense Gradient: {prediction['match_analysis']['defense_gradient']:.2f}σ
Attack Gradient: {prediction['match_analysis']['attack_gradient']:.2f}σ

🔧 FOOTBALL INTELLIGENCE PRINCIPLES
1. Defense Dictates: Strong defenses create low-scoring games
2. Attack Negotiates: Elite attacks only win against weak defenses
3. Asymmetric Corrections: Defense > Attack in correction strength
4. Competence Threshold: Any defense better than average matters
5. Elite Rarity: True elite attacks are rare (<15% of teams)

---
Generated by Football Intelligence Engine
"Defense dictates, attack negotiates"
"""

st.code(report, language="text")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    st.download_button(
        label="📥 Download Intelligence Report",
        data=report,
        file_name=f"football_intelligence_{home_team}_vs_{away_team}.txt",
        mime="text/plain"
    )

with col_exp2:
    if st.button("🔄 Reset Intelligence Memory"):
        st.session_state.football_intelligence = {
            'prediction_history': deque(maxlen=500),
            'match_patterns': defaultdict(lambda: {'count': 0, 'avg_goals': deque(maxlen=100), 'success_rate': deque(maxlen=100)}),
            'defensive_success': defaultdict(lambda: deque(maxlen=100)),
            'attack_success': defaultdict(lambda: deque(maxlen=100)),
            'total_matches': 0,
            'performance_metrics': {
                'defensive_correction_accuracy': deque(maxlen=200),
                'attack_correction_accuracy': deque(maxlen=200),
                'base_model_accuracy': deque(maxlen=200)
            }
        }
        st.success("Intelligence memory reset!")
        st.rerun()

# ========== FOOTER ==========
st.divider()
footer = f"""
🧠 Football Intelligence Engine | {prediction['direction']} 2.5 ({prediction['final_probability']*100:.1f}%) | 
Match Type: {prediction['match_type']} | Confidence: {prediction['confidence']} | 
Correction: {prediction['correction']*100:+.1f}% | Total Matches: {st.session_state.football_intelligence['total_matches']}
"""
st.caption(footer)

# ========== DATA INFO ==========
with st.sidebar.expander("📁 Data Requirements"):
    st.markdown("""
    **CSV Format:**
    ```
    team,venue,matches,xg,xga,goals_vs_xg
    TeamName,home,12,25.86,8.64,-2.14
    TeamName,away,12,23.43,10.15,5.43
    ```
    
    **Intelligence Thresholds:**
    - Elite Defense: < -1.0σ (Top 15%)
    - Strong Defense: < -0.5σ (Top 30%)
    - Good Defense: < -0.3σ (Top 40%)
    - Elite Attack: > 1.0σ (Top 15%)
    - Elite+ Attack: > 1.5σ (Top 5%)
    
    **Football Wisdom:**
    1. Defense corrections: -10% to -35%
    2. Attack corrections: +10% to +25%
    3. Defense applies more often
    4. Attack requires elite vs weak
    """)
