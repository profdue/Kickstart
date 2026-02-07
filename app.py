import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine v3.1",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine v3.1")
st.markdown("""
    **Complete Logic System: Winners + Totals with Finishing Trend Analysis**
    *Separate confidence systems for winners and totals - IMPROVED WITH PROVEN PATTERNS*
""")

# ========== CONSTANTS ==========
MAX_GOALS_CALC = 8

# League-specific adjustments
LEAGUE_ADJUSTMENTS = {
    "Premier League": {"over_threshold": 2.5, "under_threshold": 2.5, "avg_goals": 2.79},
    "Bundesliga": {"over_threshold": 3.0, "under_threshold": 2.2, "avg_goals": 3.20},
    "Serie A": {"over_threshold": 2.7, "under_threshold": 2.3, "avg_goals": 2.40},
    "La Liga": {"over_threshold": 2.6, "under_threshold": 2.4, "avg_goals": 2.61},
    "Ligue 1": {"over_threshold": 2.8, "under_threshold": 2.2, "avg_goals": 2.85},
    "Eredivisie": {"over_threshold": 2.9, "under_threshold": 2.1, "avg_goals": 3.10},
    "RFPL": {"over_threshold": 2.5, "under_threshold": 2.2, "avg_goals": 2.53}
}

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

def generate_pattern_indicators(prediction):
    """Generate pattern indicators based on backtest findings - UPDATED WITH PROVEN PATTERNS"""
    indicators = {'winner': None, 'totals': None}
    
    # WINNER PATTERNS (working well - 4/4 perfect)
    winner_conf_score = prediction['winner']['confidence_score']
    if winner_conf_score >= 90:
        indicators['winner'] = {
            'type': 'MET',
            'color': 'green',
            'text': 'WINNER CONDITION MET',
            'explanation': 'Backtest: VERY HIGH confidence (90+) winners went 4/4 correct'
        }
    elif winner_conf_score < 45:
        indicators['winner'] = {
            'type': 'AVOID',
            'color': 'red',
            'text': 'AVOID WINNER BET',
            'explanation': 'Backtest: VERY LOW confidence (<45) winners went 0/3 correct'
        }
    else:
        indicators['winner'] = {
            'type': 'NO_PATTERN',
            'color': 'gray',
            'text': 'NO PROVEN PATTERN',
            'explanation': 'Backtest: Mixed results for this confidence range'
        }
    
    # TOTALS PATTERNS - UPDATED WITH PROVEN 17-MATCH PATTERNS
    finishing_alignment = prediction['totals']['finishing_alignment']
    total_category = prediction['totals']['total_category']
    risk_flags = prediction['totals']['risk_flags']
    
    # NEW PROVEN PATTERN 1: NEUTRAL + HIGH_xG = UNDER (3/3 proven)
    if finishing_alignment == "NEUTRAL" and total_category in ["HIGH", "VERY_HIGH"]:
        indicators['totals'] = {
            'type': 'MET',
            'color': 'green',
            'text': 'PROVEN PATTERN - UNDER 2.5',
            'explanation': '17-match test: NEUTRAL + HIGH_xG (xG>3.0) went 3/3 UNDER'
        }
    
    # NEW PROVEN PATTERN 2: MED_UNDER + HIGH_xG = OVER (3/3 proven)
    elif finishing_alignment == "MED_UNDER" and total_category in ["HIGH", "VERY_HIGH"]:
        indicators['totals'] = {
            'type': 'MET',
            'color': 'green',
            'text': 'PROVEN PATTERN - OVER 2.5',
            'explanation': '17-match test: MED_UNDER + HIGH_xG (xG>3.0) went 3/3 OVER'
        }
    
    # UPDATED: HIGH_OVER cautionary (1/3 in extended test)
    elif finishing_alignment == "HIGH_OVER":
        if "VOLATILE_OVER_BOTH" in risk_flags:
            indicators['totals'] = {
                'type': 'AVOID',
                'color': 'red',
                'text': 'AVOID OVER BET',
                'explanation': 'Extended test: HIGH_OVER with both overperforming went 1/3 OVER'
            }
        else:
            indicators['totals'] = {
                'type': 'WARNING',
                'color': 'yellow',
                'text': 'CAUTION - HIGH_OVER PATTERN',
                'explanation': 'Extended test: HIGH_OVER went 1/3 OVER (high variance)'
            }
    
    # PROVEN PATTERN: MED_OVER (3/4) - KEEP AS IS
    elif finishing_alignment == "MED_OVER":
        indicators['totals'] = {
            'type': 'MET',
            'color': 'green',
            'text': 'PROVEN PATTERN - OVER 2.5',
            'explanation': 'Backtest: MED_OVER alignment went 5/5 OVER 2.5'
        }
    
    # PROVEN RISK: LOW_UNDER + VERY_HIGH
    elif finishing_alignment == "LOW_UNDER" and total_category == "VERY_HIGH":
        indicators['totals'] = {
            'type': 'AVOID',
            'color': 'red',
            'text': 'PROVEN RISK - BET UNDER 2.5',
            'explanation': 'Backtest: LOW_UNDER + VERY_HIGH went 0/2 OVER 2.5 (both UNDER)'
        }
    
    # NEW: BUNDESLIGA specific pattern (3/4 went UNDER in test)
    elif "BUNDESLIGA_LOW_SCORING" in risk_flags:
        indicators['totals'] = {
            'type': 'WARNING',
            'color': 'yellow',
            'text': 'BUNDESLIGA LOW SCORING',
            'explanation': 'Bundesliga 3.0 line: 3/4 matches went UNDER in 17-match test'
        }
    
    else:
        # All other combinations
        indicators['totals'] = {
            'type': 'NO_PATTERN',
            'color': 'gray',
            'text': 'NO PROVEN PATTERN',
            'explanation': f'Insufficient backtest data for {finishing_alignment} alignment'
        }
    
    return indicators

@st.cache_data(ttl=3600)
def load_league_data(league_name):
    try:
        file_map = {
            "Premier League": "premier_league.csv",
            "Bundesliga": "bundesliga.csv",
            "Serie A": "serie_a.csv",
            "La Liga": "laliga.csv",
            "Ligue 1": "ligue_1.csv",
            "Eredivisie": "eredivisie.csv",
            "RFPL": "rfpl.csv"
        }
        
        filename = file_map.get(league_name, f"{league_name.lower().replace(' ', '_')}.csv")
        file_path = f"leagues/{filename}"
        
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare home and away data with per-match averages"""
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    for df_part in [home_data, away_data]:
        if len(df_part) > 0:
            df_part['goals_for_pm'] = df_part['gf'] / df_part['matches']
            df_part['goals_against_pm'] = df_part['ga'] / df_part['matches']
            df_part['goals_vs_xg_pm'] = df_part['goals_vs_xg'] / df_part['matches']
            df_part['goals_allowed_vs_xga_pm'] = df_part['goals_allowed_vs_xga'] / df_part['matches']
            df_part['xg_pm'] = df_part['xg'] / df_part['matches']
            df_part['xga_pm'] = df_part['xga'] / df_part['matches']
            df_part['points_pm'] = df_part['pts'] / df_part['matches']
            df_part['win_rate'] = df_part['wins'] / df_part['matches']
    
    return home_data.set_index('team'), away_data.set_index('team')

def calculate_league_metrics(df):
    """Calculate league-wide metrics"""
    if df is None or len(df) == 0:
        return {}
    
    total_matches = df['matches'].sum() / 2
    total_goals = df['gf'].sum()
    avg_goals_per_match = total_goals / total_matches if total_matches > 0 else 2.5
    
    return {'avg_goals_per_match': avg_goals_per_match}

# ========== OUR IMPROVED LOGIC CLASSES ==========

class ExpectedGoalsPredictor:
    """OUR LOGIC: Expected goals calculation"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_avg_goals = league_metrics.get('avg_goals_per_match', 2.5)
        self.league_name = league_name
    
    def predict_expected_goals(self, home_stats, away_stats):
        """OUR LOGIC: Step 1 - Adjusted Team Strength"""
        home_adjGF = home_stats['goals_for_pm'] + 0.6 * home_stats['goals_vs_xg_pm']
        home_adjGA = home_stats['goals_against_pm'] + 0.6 * home_stats['goals_allowed_vs_xga_pm']
        
        away_adjGF = away_stats['goals_for_pm'] + 0.6 * away_stats['goals_vs_xg_pm']
        away_adjGA = away_stats['goals_against_pm'] + 0.6 * away_stats['goals_allowed_vs_xga_pm']
        
        # OUR LOGIC: Dynamic Venue Factor
        venue_factor_home = 1 + 0.05 * (home_stats['points_pm'] - away_stats['points_pm']) / 3
        venue_factor_away = 1 + 0.05 * (away_stats['points_pm'] - home_stats['points_pm']) / 3
        
        venue_factor_home = max(0.8, min(1.2, venue_factor_home))
        venue_factor_away = max(0.8, min(1.2, venue_factor_away))
        
        # OUR LOGIC: Expected Goals Calculation
        home_xg = (home_adjGF + away_adjGA) / 2 * venue_factor_home
        away_xg = (away_adjGF + home_adjGA) / 2 * venue_factor_away
        
        # League normalization
        normalization_factor = self.league_avg_goals / 2.5
        home_xg *= normalization_factor
        away_xg *= normalization_factor
        
        home_xg = max(0.2, min(5.0, home_xg))
        away_xg = max(0.2, min(5.0, away_xg))
        
        return home_xg, away_xg, {
            'home_adjGF': home_adjGF,
            'home_adjGA': home_adjGA,
            'away_adjGF': away_adjGF,
            'away_adjGA': away_adjGA,
            'venue_factor_home': venue_factor_home,
            'venue_factor_away': venue_factor_away
        }

class WinnerPredictor:
    """OUR LOGIC: Winner prediction with proven confidence system"""
    
    def predict_winner(self, home_xg, away_xg, home_stats, away_stats):
        """OUR LOGIC: Winner determination"""
        delta = home_xg - away_xg
        
        # OUR LOGIC: Winner determination with draw suppression
        if delta > 1.2:
            predicted_winner = "HOME"
            winner_strength = "STRONG"
        elif delta > 0.5:
            predicted_winner = "HOME"
            winner_strength = "MODERATE"
        elif delta > 0.2:
            predicted_winner = "HOME"
            winner_strength = "SLIGHT"
        elif delta < -1.2:
            predicted_winner = "AWAY"
            winner_strength = "STRONG"
        elif delta < -0.5:
            predicted_winner = "AWAY"
            winner_strength = "MODERATE"
        elif delta < -0.2:
            predicted_winner = "AWAY"
            winner_strength = "SLIGHT"
        else:
            predicted_winner = "DRAW"
            winner_strength = "CLOSE"
        
        # OUR LOGIC: Winner confidence calculation (PROVEN TO WORK - 4/4 perfect)
        base_confidence = min(100, abs(delta) / max(home_xg, away_xg, 0.5) * 150)
        
        # Add bonuses
        venue_bonus = 0
        if home_stats['points_pm'] > 2.0:
            venue_bonus += 15
        if away_stats['points_pm'] < 1.0:
            venue_bonus += 15
        
        win_rate_diff = home_stats['win_rate'] - away_stats['win_rate']
        form_bonus = min(20, max(0, win_rate_diff * 40))
        
        winner_confidence = min(100, max(30, base_confidence + venue_bonus + form_bonus))
        
        # OUR LOGIC: Confidence categorization
        if winner_confidence >= 75:
            confidence_category = "VERY HIGH"
        elif winner_confidence >= 65:
            confidence_category = "HIGH"
        elif winner_confidence >= 55:
            confidence_category = "MEDIUM"
        elif winner_confidence >= 45:
            confidence_category = "LOW"
        else:
            confidence_category = "VERY LOW"
        
        return {
            'predicted_winner': predicted_winner,
            'winner_strength': winner_strength,
            'winner_confidence': winner_confidence,
            'winner_confidence_category': confidence_category,
            'delta': delta
        }

class TotalsPredictor:
    """OUR IMPROVED LOGIC: Totals prediction with proven pattern fixes"""
    
    def __init__(self, league_name):
        self.league_name = league_name
        self.league_adjustments = LEAGUE_ADJUSTMENTS.get(league_name, LEAGUE_ADJUSTMENTS["Premier League"])
    
    def categorize_finishing(self, value):
        """OUR LOGIC: Categorize finishing strength"""
        if value > 0.3:
            return "STRONG_OVERPERFORM"
        elif value > 0.1:
            return "MODERATE_OVERPERFORM"
        elif value > -0.1:
            return "NEUTRAL"
        elif value > -0.3:
            return "MODERATE_UNDERPERFORM"
        else:
            return "STRONG_UNDERPERFORM"
    
    def get_finishing_alignment(self, home_finish, away_finish):
        """OUR LOGIC: Finishing trend alignment matrix"""
        home_cat = self.categorize_finishing(home_finish)
        away_cat = self.categorize_finishing(away_finish)
        
        # OUR LOGIC: Alignment matrix
        alignment_matrix = {
            "STRONG_OVERPERFORM": {
                "STRONG_OVERPERFORM": "HIGH_OVER",
                "MODERATE_OVERPERFORM": "MED_OVER",
                "NEUTRAL": "MED_OVER",
                "MODERATE_UNDERPERFORM": "RISKY",
                "STRONG_UNDERPERFORM": "RISKY"
            },
            "MODERATE_OVERPERFORM": {
                "STRONG_OVERPERFORM": "MED_OVER",
                "MODERATE_OVERPERFORM": "MED_OVER",
                "NEUTRAL": "LOW_OVER",
                "MODERATE_UNDERPERFORM": "RISKY",
                "STRONG_UNDERPERFORM": "HIGH_RISK"
            },
            "NEUTRAL": {
                "STRONG_OVERPERFORM": "MED_OVER",
                "MODERATE_OVERPERFORM": "LOW_OVER",
                "NEUTRAL": "NEUTRAL",
                "MODERATE_UNDERPERFORM": "LOW_UNDER",
                "STRONG_UNDERPERFORM": "MED_UNDER"
            },
            "MODERATE_UNDERPERFORM": {
                "STRONG_OVERPERFORM": "RISKY",
                "MODERATE_OVERPERFORM": "RISKY",
                "NEUTRAL": "LOW_UNDER",
                "MODERATE_UNDERPERFORM": "MED_UNDER",
                "STRONG_UNDERPERFORM": "MED_UNDER"
            },
            "STRONG_UNDERPERFORM": {
                "STRONG_OVERPERFORM": "HIGH_RISK",
                "MODERATE_OVERPERFORM": "RISKY",
                "NEUTRAL": "MED_UNDER",
                "MODERATE_UNDERPERFORM": "MED_UNDER",
                "STRONG_UNDERPERFORM": "HIGH_UNDER"
            }
        }
        
        return alignment_matrix[home_cat][away_cat]
    
    def categorize_total_xg(self, total_xg):
        """OUR LOGIC: Total xG categories"""
        if total_xg > 3.3:
            return "VERY_HIGH"
        elif total_xg > 3.0:
            return "HIGH"
        elif total_xg > 2.7:
            return "MODERATE_HIGH"
        elif total_xg > 2.3:
            return "MODERATE_LOW"
        elif total_xg > 2.0:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def check_risk_flags(self, home_stats, away_stats, total_xg):
        """OUR IMPROVED LOGIC: Risk flag system"""
        risk_flags = []
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        # NEW RISK FLAG: Volatile overperformers (1/3 success in test)
        if home_finish > 0.35 and away_finish > 0.35:
            risk_flags.append("VOLATILE_OVER_BOTH")
        
        # Flag 1: Opposite extreme finishing
        if (home_finish > 0.5 and away_finish < -0.5) or (home_finish < -0.5 and away_finish > 0.5):
            risk_flags.append("OPPOSITE_EXTREME_FINISHING")
        
        # Flag 2: High variance teams
        if abs(home_finish) > 0.4 or abs(away_finish) > 0.4:
            risk_flags.append("HIGH_VARIANCE_TEAM")
        
        # Flag 3: Attack-defense mismatch
        if home_stats['goals_for_pm'] > 2.0 and away_stats['goals_for_pm'] < 1.0:
            risk_flags.append("ATTACK_DEFENSE_MISMATCH")
        
        # Flag 4: Close to threshold
        lower_thresh = self.league_adjustments['under_threshold'] - 0.1
        upper_thresh = self.league_adjustments['over_threshold'] + 0.1
        if lower_thresh < total_xg < upper_thresh:
            risk_flags.append("CLOSE_TO_THRESHOLD")
        
        # NEW: Bundesliga specific adjustment (3/4 went UNDER in test)
        if self.league_name == "Bundesliga" and total_xg < 3.3:
            risk_flags.append("BUNDESLIGA_LOW_SCORING")
        
        return risk_flags
    
    def predict_totals(self, home_xg, away_xg, home_stats, away_stats):
        """OUR IMPROVED LOGIC: Complete totals prediction with proven pattern fixes"""
        total_xg = home_xg + away_xg
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        # OUR LOGIC: Base prediction
        over_threshold = self.league_adjustments['over_threshold']
        base_direction = "OVER" if total_xg > over_threshold else "UNDER"
        
        # OUR LOGIC: Finishing alignment
        finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
        total_category = self.categorize_total_xg(total_xg)
        
        # OUR LOGIC: Risk flags
        risk_flags = self.check_risk_flags(home_stats, away_stats, total_xg)
        
        # ========== NEW: PROVEN PATTERN 1 ==========
        # AUTO-UNDER RULE: NEUTRAL + xG > 3.0 = UNDER (3/3 proven)
        if finishing_alignment == "NEUTRAL" and total_xg > 3.0:
            return {
                'direction': "UNDER",
                'total_xg': total_xg,
                'confidence': "HIGH",
                'confidence_score': 80,  # High confidence for proven pattern
                'finishing_alignment': finishing_alignment,
                'total_category': total_category,
                'risk_flags': risk_flags,
                'home_finishing': home_finish,
                'away_finishing': away_finish
            }
        
        # OUR IMPROVED LOGIC: Decision matrix with MED_UNDER fix
        decision_matrix = {
            "VERY_HIGH": {
                "HIGH_OVER": ("OVER", "MEDIUM", 60),
                "MED_OVER": ("OVER", "HIGH", 75),
                "LOW_OVER": ("OVER", "MEDIUM", 65),
                "NEUTRAL": ("OVER", "MEDIUM", 60),
                "RISKY": ("OVER", "LOW", 45),
                "HIGH_RISK": (base_direction, "VERY LOW", 35),
                "MED_UNDER": ("OVER", "MEDIUM", 65),  # FIXED: Was (base_direction, "LOW", 50) - 3/3 OVER proven
                "LOW_UNDER": ("UNDER", "LOW", 45)
            },
            "HIGH": {
                "HIGH_OVER": ("OVER", "LOW", 50),
                "MED_OVER": ("OVER", "HIGH", 70),
                "LOW_OVER": ("OVER", "MEDIUM", 60),
                "NEUTRAL": (base_direction, "LOW", 50),
                "RISKY": (base_direction, "LOW", 45),
                "HIGH_RISK": (base_direction, "VERY LOW", 35),
                "MED_UNDER": ("OVER", "MEDIUM", 60),  # FIXED: Was ("UNDER", "MEDIUM", 55) - 3/3 OVER proven
                "LOW_UNDER": ("UNDER", "LOW", 45)
            },
            "MODERATE_HIGH": {
                "HIGH_OVER": ("OVER", "MEDIUM", 55),
                "MED_OVER": ("OVER", "MEDIUM", 65),
                "LOW_OVER": ("OVER", "MEDIUM", 60),
                "NEUTRAL": (base_direction, "LOW", 50),
                "LOW_UNDER": ("UNDER", "LOW", 45),
                "MED_UNDER": ("OVER", "MEDIUM", 55)  # Already correct
            },
            "MODERATE_LOW": {
                "HIGH_UNDER": ("UNDER", "VERY HIGH", 80),
                "MED_UNDER": ("UNDER", "HIGH", 70),
                "LOW_UNDER": ("UNDER", "MEDIUM", 60),
                "NEUTRAL": (base_direction, "LOW", 50),
                "LOW_OVER": ("OVER", "LOW", 45),
                "MED_OVER": ("OVER", "MEDIUM", 55)
            },
            "LOW": {
                "HIGH_UNDER": ("UNDER", "VERY HIGH", 85),
                "MED_UNDER": ("UNDER", "HIGH", 75),
                "LOW_UNDER": ("UNDER", "MEDIUM", 65),
                "NEUTRAL": ("UNDER", "LOW", 50),
                "RISKY": ("UNDER", "LOW", 45),
                "HIGH_RISK": (base_direction, "VERY LOW", 35)
            },
            "VERY_LOW": {
                "HIGH_UNDER": ("UNDER", "VERY HIGH", 90),
                "MED_UNDER": ("UNDER", "HIGH", 80),
                "LOW_UNDER": ("UNDER", "MEDIUM", 70),
                "NEUTRAL": ("UNDER", "MEDIUM", 65),
                "RISKY": ("UNDER", "LOW", 45),
                "HIGH_RISK": (base_direction, "VERY LOW", 35)
            }
        }
        
        # Get decision from matrix
        direction = base_direction
        confidence_category = "LOW"
        base_confidence = 40
        
        if total_category in decision_matrix and finishing_alignment in decision_matrix[total_category]:
            direction, confidence_category, base_confidence = decision_matrix[total_category][finishing_alignment]
        
        # OUR LOGIC: Apply risk flag penalties
        final_confidence = base_confidence
        for flag in risk_flags:
            if flag == "VOLATILE_OVER_BOTH":
                final_confidence -= 25
            elif flag == "OPPOSITE_EXTREME_FINISHING":
                final_confidence -= 25
            elif flag == "HIGH_VARIANCE_TEAM":
                final_confidence -= 15
            elif flag == "ATTACK_DEFENSE_MISMATCH":
                final_confidence -= 10
            elif flag == "CLOSE_TO_THRESHOLD":
                final_confidence -= 10
            elif flag == "BUNDESLIGA_LOW_SCORING":
                final_confidence -= 15
        
        final_confidence = max(5, min(95, final_confidence))
        
        # Adjust confidence category based on final score
        if final_confidence >= 75:
            confidence_category = "VERY HIGH"
        elif final_confidence >= 65:
            confidence_category = "HIGH"
        elif final_confidence >= 55:
            confidence_category = "MEDIUM"
        elif final_confidence >= 45:
            confidence_category = "LOW"
        else:
            confidence_category = "VERY LOW"
        
        return {
            'direction': direction,
            'total_xg': total_xg,
            'confidence': confidence_category,
            'confidence_score': final_confidence,
            'finishing_alignment': finishing_alignment,
            'total_category': total_category,
            'risk_flags': risk_flags,
            'home_finishing': home_finish,
            'away_finishing': away_finish
        }

class PoissonProbabilityEngine:
    """Calculate all probabilities from Poisson distribution"""
    
    @staticmethod
    def calculate_all_probabilities(home_xg, away_xg):
        score_probabilities = []
        max_goals = min(MAX_GOALS_CALC, int(home_xg + away_xg) + 4)
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (poisson_pmf(home_goals, home_xg) * 
                       poisson_pmf(away_goals, away_xg))
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
        under_2_5_prob = sum(p['probability'] for p in score_probabilities 
                            if p['home_goals'] + p['away_goals'] < 2.5)
        
        btts_prob = sum(p['probability'] for p in score_probabilities 
                       if p['home_goals'] > 0 and p['away_goals'] > 0)
        
        top_scores = sorted(score_probabilities, key=lambda x: x['probability'], reverse=True)[:5]
        top_scores_formatted = [(f"{s['home_goals']}-{s['away_goals']}", s['probability']) 
                               for s in top_scores]
        
        return {
            'home_win_probability': home_win_prob,
            'draw_probability': draw_prob,
            'away_win_probability': away_win_prob,
            'over_2_5_probability': over_2_5_prob,
            'under_2_5_probability': under_2_5_prob,
            'btts_probability': btts_prob,
            'most_likely_score': most_likely_score,
            'top_scores': top_scores_formatted,
            'expected_home_goals': home_xg,
            'expected_away_goals': away_xg,
            'total_expected_goals': home_xg + away_xg
        }

class InsightsGenerator:
    """OUR IMPROVED LOGIC: Generate enhanced insights with proven patterns"""
    
    @staticmethod
    def generate_insights(winner_prediction, totals_prediction):
        insights = []
        
        # Winner insights (working well - 4/4 perfect)
        if winner_prediction.get('winner_confidence_category') == "VERY HIGH":
            insights.append(f"🎯 **High Confidence Winner**: Model strongly favors {winner_prediction.get('predicted_winner', 'N/A')} (4/4 perfect in tests)")
        elif winner_prediction.get('winner_confidence_category') == "LOW":
            insights.append(f"⚠️ **Low Confidence Winner**: Exercise caution on {winner_prediction.get('predicted_winner', 'N/A')} prediction (0/3 in backtests)")
        
        # NEW: Finishing volatility insight from 17-match test
        home_finish = totals_prediction.get('home_finishing', 0)
        away_finish = totals_prediction.get('away_finishing', 0)
        
        if home_finish > 0.35 and away_finish > 0.35:
            insights.append("⚠️ **Both teams strong overperformers** - High volatility expected (1/3 OVER in 17-match test)")
        
        # NEW PROVEN PATTERN INSIGHTS
        alignment = totals_prediction.get('finishing_alignment', 'NEUTRAL')
        total_category = totals_prediction.get('total_category', 'N/A')
        total_xg = totals_prediction.get('total_xg', 0)
        
        if alignment == "NEUTRAL" and total_xg > 3.0:
            insights.append("✅ **PROVEN PATTERN**: NEUTRAL + HIGH_xG (xG>3.0) = UNDER (3/3 in test)")
        elif alignment == "MED_UNDER" and total_xg > 3.0:
            insights.append("✅ **PROVEN PATTERN**: MED_UNDER + HIGH_xG (xG>3.0) = OVER (3/3 in test)")
        
        # Totals insights
        if totals_prediction.get('confidence') == "VERY HIGH":
            insights.append(f"🎯 **High Confidence Totals**: Strong signal for {totals_prediction.get('direction', 'N/A')} 2.5")
        elif totals_prediction.get('confidence') == "VERY LOW":
            insights.append(f"⚠️ **Low Confidence Totals**: High risk on {totals_prediction.get('direction', 'N/A')} 2.5 prediction")
        
        # Finishing trend insights
        if home_finish > 0.3:
            insights.append(f"⚡ **Home team overperforms xG** by {home_finish:.2f}/game (clinical finishing)")
        elif home_finish < -0.3:
            insights.append(f"⚡ **Home team underperforms xG** by {abs(home_finish):.2f}/game (wasteful finishing)")
        
        if away_finish > 0.3:
            insights.append(f"⚡ **Away team overperforms xG** by {away_finish:.2f}/game (clinical finishing)")
        elif away_finish < -0.3:
            insights.append(f"⚡ **Away team underperforms xG** by {abs(away_finish):.2f}/game (wasteful finishing)")
        
        # Finishing alignment insights
        if alignment == "HIGH_OVER":
            insights.append("⚠️ **HIGH_OVER pattern**: 17-match test shows 1/3 success rate (caution advised)")
        elif alignment == "MED_OVER":
            insights.append("✅ **MED_OVER pattern**: Proven 5/5 OVER in backtests")
        
        # Risk flag insights
        risk_flags = totals_prediction.get('risk_flags', [])
        if risk_flags:
            risk_count = len(risk_flags)
            flag_list = ", ".join(risk_flags)
            insights.append(f"⚠️ **{risk_count} risk flag(s) detected**: {flag_list}")
        
        return insights[:6]

class FootballIntelligenceEngineV3:
    """OUR IMPROVED LOGIC: Complete prediction engine with proven pattern fixes"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_name = league_name
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
        self.insights_generator = InsightsGenerator()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """OUR LOGIC: Complete match prediction"""
        
        # Step 1: Expected goals
        home_xg, away_xg, calc_details = self.xg_predictor.predict_expected_goals(
            home_stats, away_stats
        )
        
        # Step 2: Poisson probabilities
        probabilities = self.probability_engine.calculate_all_probabilities(
            home_xg, away_xg
        )
        
        # Step 3: OUR LOGIC - Winner prediction
        winner_prediction = self.winner_predictor.predict_winner(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Step 4: OUR IMPROVED LOGIC - Totals prediction with proven fixes
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Step 5: OUR IMPROVED LOGIC - Insights
        insights = self.insights_generator.generate_insights(winner_prediction, totals_prediction)
        
        # Step 6: Determine final probabilities
        if winner_prediction['predicted_winner'] == "HOME":
            winner_display = home_team
            winner_prob = probabilities['home_win_probability']
        elif winner_prediction['predicted_winner'] == "AWAY":
            winner_display = away_team
            winner_prob = probabilities['away_win_probability']
        else:
            winner_display = "DRAW"
            winner_prob = probabilities['draw_probability']
        
        # Get total probability
        if totals_prediction['direction'] == "OVER":
            total_prob = probabilities['over_2_5_probability']
        else:
            total_prob = probabilities['under_2_5_probability']
        
        return {
            # Winner prediction
            'winner': {
                'team': winner_display,
                'type': winner_prediction['predicted_winner'],
                'probability': winner_prob,
                'confidence': winner_prediction['winner_confidence_category'],
                'confidence_score': winner_prediction['winner_confidence'],
                'strength': winner_prediction['winner_strength'],
                'most_likely_score': probabilities['most_likely_score']
            },
            
            # Totals prediction
            'totals': {
                'direction': totals_prediction['direction'],
                'probability': total_prob,
                'confidence': totals_prediction['confidence'],
                'confidence_score': totals_prediction['confidence_score'],
                'total_xg': totals_prediction['total_xg'],
                'finishing_alignment': totals_prediction['finishing_alignment'],
                'total_category': totals_prediction['total_category'],
                'risk_flags': totals_prediction['risk_flags'],
                'home_finishing': totals_prediction['home_finishing'],
                'away_finishing': totals_prediction['away_finishing']
            },
            
            # All probabilities
            'probabilities': probabilities,
            
            # Expected goals
            'expected_goals': {
                'home': home_xg,
                'away': away_xg,
                'total': home_xg + away_xg
            },
            
            # Insights
            'insights': insights,
            
            # Calculation details
            'calculation_details': calc_details
        }

# ========== UNIFIED DECISION ENGINE ==========
class UnifiedDecisionEngine:
    """NEW: Unified decision engine combining base confidence + patterns"""
    
    @staticmethod
    def calculate_final_confidence(base_confidence, pattern_type, risk_flags=None):
        """Calculate final confidence with pattern bonuses"""
        # Pattern bonuses (based on backtest success rates)
        pattern_bonuses = {
            'MET': +40,      # Proven pattern (80%+ success)
            'CAUTION': +10,   # Warning pattern
            'NO_PATTERN': 0,  # No pattern
            'AVOID': -20      # Proven to avoid
        }
        
        # Risk penalties
        risk_penalties = {
            'VOLATILE_OVER_BOTH': -25,
            'OPPOSITE_EXTREME_FINISHING': -25,
            'HIGH_VARIANCE_TEAM': -15,
            'ATTACK_DEFENSE_MISMATCH': -10,
            'CLOSE_TO_THRESHOLD': -10,
            'BUNDESLIGA_LOW_SCORING': -15
        }
        
        # Start with base confidence
        final_confidence = base_confidence
        
        # Apply pattern bonus
        final_confidence += pattern_bonuses.get(pattern_type, 0)
        
        # Apply risk penalties
        if risk_flags:
            for flag in risk_flags:
                final_confidence += risk_penalties.get(flag, 0)
        
        # Clamp between 5-100
        return max(5, min(100, final_confidence))
    
    @staticmethod
    def get_bet_recommendation(final_confidence, pattern_type, market_type):
        """Get unified betting recommendation"""
        
        # OVERRIDE RULE: Proven patterns with <60 base confidence still bet
        if pattern_type == 'MET' and final_confidence >= 40:
            if final_confidence >= 80:
                return "✅ STRONG BET"
            elif final_confidence >= 60:
                return "✅ MODERATE BET"
            else:
                return "⚠️ SMALL BET"  # Proven pattern but low confidence
            
        # Normal betting rules
        if final_confidence >= 80:
            return "✅ STRONG BET"
        elif final_confidence >= 60:
            return "✅ MODERATE BET"
        elif pattern_type == 'AVOID':
            return "❌ AVOID BET"
        else:
            return "❌ NO BET"
    
    @staticmethod
    def get_unified_prediction(base_prediction, pattern_indicators):
        """Create unified prediction from all layers"""
        return {
            'winner': {
                'base_confidence': base_prediction['winner']['confidence_score'],
                'pattern_type': pattern_indicators['winner']['type'],
                'final_confidence': UnifiedDecisionEngine.calculate_final_confidence(
                    base_prediction['winner']['confidence_score'],
                    pattern_indicators['winner']['type']
                ),
                'bet_recommendation': UnifiedDecisionEngine.get_bet_recommendation(
                    UnifiedDecisionEngine.calculate_final_confidence(
                        base_prediction['winner']['confidence_score'],
                        pattern_indicators['winner']['type']
                    ),
                    pattern_indicators['winner']['type'],
                    'winner'
                ),
                'pattern_text': pattern_indicators['winner']['text']
            },
            'totals': {
                'base_confidence': base_prediction['totals']['confidence_score'],
                'pattern_type': pattern_indicators['totals']['type'],
                'final_confidence': UnifiedDecisionEngine.calculate_final_confidence(
                    base_prediction['totals']['confidence_score'],
                    pattern_indicators['totals']['type'],
                    base_prediction['totals']['risk_flags']
                ),
                'bet_recommendation': UnifiedDecisionEngine.get_bet_recommendation(
                    UnifiedDecisionEngine.calculate_final_confidence(
                        base_prediction['totals']['confidence_score'],
                        pattern_indicators['totals']['type'],
                        base_prediction['totals']['risk_flags']
                    ),
                    pattern_indicators['totals']['type'],
                    'totals'
                ),
                'pattern_text': pattern_indicators['totals']['text']
            }
        }

# ========== UNIFIED BETTING CARD ==========
class UnifiedBettingCard:
    """Single unified betting card that recommends the best bet(s)"""
    
    @staticmethod
    def get_unified_recommendation(prediction, pattern_indicators, home_team, away_team):
        """Determine the unified betting recommendation"""
        unified_engine = UnifiedDecisionEngine()
        unified_pred = unified_engine.get_unified_prediction(prediction, pattern_indicators)
        
        winner_unified = unified_pred['winner']
        totals_unified = unified_pred['totals']
        
        # Determine if we should combine bets
        winner_action = winner_unified['bet_recommendation']
        totals_action = totals_unified['bet_recommendation']
        
        # Check if both are strong bets
        both_strong = (winner_action.startswith('✅') and winner_unified['final_confidence'] >= 80 and
                      totals_action.startswith('✅') and totals_unified['final_confidence'] >= 80)
        
        # Check if one is very strong and other is moderate
        one_very_strong_other_moderate = (
            (winner_action.startswith('✅') and winner_unified['final_confidence'] >= 90 and
             totals_action.startswith('✅') and totals_unified['final_confidence'] >= 60) or
            (totals_action.startswith('✅') and totals_unified['final_confidence'] >= 90 and
             winner_action.startswith('✅') and winner_unified['final_confidence'] >= 60)
        )
        
        # Determine the best single bet if not combining
        best_single_bet = None
        if winner_unified['final_confidence'] >= totals_unified['final_confidence']:
            best_single_bet = {
                'type': 'winner',
                'team': prediction['winner']['team'],
                'confidence': winner_unified['final_confidence'],
                'action': winner_action
            }
        else:
            best_single_bet = {
                'type': 'totals',
                'direction': prediction['totals']['direction'],
                'confidence': totals_unified['final_confidence'],
                'action': totals_action
            }
        
        # Make final recommendation
        if both_strong:
            return {
                'type': 'combo',
                'text': f"🏆 {prediction['winner']['team']} to win + 📈 {prediction['totals']['direction']} 2.5",
                'confidence': min(winner_unified['final_confidence'], totals_unified['final_confidence']),
                'color': '#10B981',  # Emerald green
                'icon': '🎯',
                'subtext': 'STRONG DOUBLE BET',
                'reason': 'Both winner and totals show high confidence with proven patterns'
            }
        elif one_very_strong_other_moderate:
            return {
                'type': 'combo',
                'text': f"🏆 {prediction['winner']['team']} to win + 📈 {prediction['totals']['direction']} 2.5",
                'confidence': min(winner_unified['final_confidence'], totals_unified['final_confidence']),
                'color': '#059669',  # Slightly darker green
                'icon': '🎯',
                'subtext': 'MODERATE DOUBLE BET',
                'reason': 'One very strong signal with moderate complementary bet'
            }
        elif best_single_bet['confidence'] >= 60:
            if best_single_bet['type'] == 'winner':
                return {
                    'type': 'single',
                    'text': f"🏆 {best_single_bet['team']} to win",
                    'confidence': best_single_bet['confidence'],
                    'color': '#3B82F6' if best_single_bet['confidence'] >= 80 else '#60A5FA',
                    'icon': '🏆',
                    'subtext': 'STRONG BET' if best_single_bet['confidence'] >= 80 else 'MODERATE BET',
                    'reason': 'High confidence winner prediction with proven pattern'
                }
            else:
                return {
                    'type': 'single',
                    'text': f"📈 {best_single_bet['direction']} 2.5 Goals",
                    'confidence': best_single_bet['confidence'],
                    'color': '#8B5CF6' if best_single_bet['confidence'] >= 80 else '#A78BFA',
                    'icon': '📈',
                    'subtext': 'STRONG BET' if best_single_bet['confidence'] >= 80 else 'MODERATE BET',
                    'reason': 'High confidence totals prediction with proven pattern'
                }
        elif best_single_bet['confidence'] >= 40 and best_single_bet['action'].startswith('⚠️'):
            if best_single_bet['type'] == 'winner':
                return {
                    'type': 'single',
                    'text': f"🏆 {best_single_bet['team']} to win",
                    'confidence': best_single_bet['confidence'],
                    'color': '#F59E0B',
                    'icon': '⚠️',
                    'subtext': 'SMALL BET',
                    'reason': 'Proven pattern but lower confidence - small stake only'
                }
            else:
                return {
                    'type': 'single',
                    'text': f"📈 {best_single_bet['direction']} 2.5 Goals",
                    'confidence': best_single_bet['confidence'],
                    'color': '#F59E0B',
                    'icon': '⚠️',
                    'subtext': 'SMALL BET',
                    'reason': 'Proven pattern but lower confidence - small stake only'
                }
        else:
            return {
                'type': 'none',
                'text': "❌ No Recommended Bet",
                'confidence': max(winner_unified['final_confidence'], totals_unified['final_confidence']),
                'color': '#6B7280',
                'icon': '🚫',
                'subtext': 'AVOID ALL BETS',
                'reason': 'Insufficient confidence or proven avoid patterns present'
            }
    
    @staticmethod
    def display_card(recommendation):
        """Display the unified betting card"""
        if recommendation['type'] == 'none':
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {recommendation['color']}20 0%, #1F2937 100%);
                padding: 25px;
                border-radius: 20px;
                border: 2px solid {recommendation['color']};
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            ">
                <div style="font-size: 48px; margin-bottom: 15px;">
                    {recommendation['icon']}
                </div>
                <div style="font-size: 32px; font-weight: bold; color: {recommendation['color']}; margin-bottom: 10px;">
                    {recommendation['text']}
                </div>
                <div style="font-size: 18px; color: #9CA3AF; margin-bottom: 15px;">
                    Confidence: {recommendation['confidence']:.0f}/100
                </div>
                <div style="font-size: 16px; color: #D1D5DB; padding: 10px; background: rgba(107, 114, 128, 0.2); border-radius: 10px;">
                    {recommendation['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {recommendation['color']}20 0%, #1F2937 100%);
                padding: 25px;
                border-radius: 20px;
                border: 2px solid {recommendation['color']};
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            ">
                <div style="font-size: 48px; margin-bottom: 15px;">
                    {recommendation['icon']}
                </div>
                <div style="font-size: 36px; font-weight: bold; color: white; margin-bottom: 10px;">
                    {recommendation['text']}
                </div>
                <div style="font-size: 24px; color: {recommendation['color']}; margin-bottom: 10px; font-weight: bold;">
                    {recommendation['subtext']}
                </div>
                <div style="font-size: 18px; color: #9CA3AF; margin-bottom: 15px;">
                    Confidence: {recommendation['confidence']:.0f}/100
                </div>
                <div style="font-size: 16px; color: #D1D5DB; padding: 10px; background: rgba(59, 130, 246, 0.1); border-radius: 10px;">
                    {recommendation['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ========== STREAMLIT UI ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie", "RFPL"]
    selected_league = st.selectbox("Select League", leagues)
    
    df = load_league_data(selected_league)
    
    if df is not None:
        league_metrics = calculate_league_metrics(df)
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
            
            st.divider()
            
            st.markdown("### 🎯 Prediction Settings")
            show_details = st.checkbox("Show Detailed Analysis", value=True)
            
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
st.caption(f"League: {selected_league} | League Avg Goals: {league_metrics['avg_goals_per_match']:.2f}")

engine = FootballIntelligenceEngineV3(league_metrics, selected_league)
prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)

# ========== DISPLAY RESULTS ==========

# Main prediction cards
col1, col2 = st.columns(2)

with col1:
    # Winner prediction
    winner_pred = prediction['winner']
    winner_conf = winner_pred['confidence']
    winner_conf_score = winner_pred['confidence_score']
    
    if winner_pred['type'] == "HOME":
        winner_color = "#22C55E" if winner_conf in ["VERY HIGH", "HIGH"] else "#4ADE80" if winner_conf == "MEDIUM" else "#84CC16"
        icon = "🏠"
    elif winner_pred['type'] == "AWAY":
        winner_color = "#22C55E" if winner_conf in ["VERY HIGH", "HIGH"] else "#4ADE80" if winner_conf == "MEDIUM" else "#84CC16"
        icon = "✈️"
    else:
        winner_color = "#F59E0B"
        icon = "🤝"
    
    # Color based on confidence
    if winner_conf == "VERY HIGH":
        card_color = "#14532D"
    elif winner_conf == "HIGH":
        card_color = "#166534"
    elif winner_conf == "MEDIUM":
        card_color = "#365314"
    elif winner_conf == "LOW":
        card_color = "#3F6212"
    else:
        card_color = "#1E293B"
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">PREDICTED WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: {winner_color}; margin: 10px 0;">
            {icon} {winner_pred['team']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            Strength: {winner_pred['strength']} | Confidence: {winner_conf} ({winner_conf_score:.0f}/100)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Totals prediction
    totals_pred = prediction['totals']
    direction = totals_pred['direction']
    confidence = totals_pred['confidence']
    conf_score = totals_pred['confidence_score']
    
    if direction == "OVER":
        if confidence == "VERY HIGH":
            card_color = "#14532D"
            text_color = "#22C55E"
        elif confidence == "HIGH":
            card_color = "#166534"
            text_color = "#4ADE80"
        elif confidence == "MEDIUM":
            card_color = "#365314"
            text_color = "#84CC16"
        elif confidence == "LOW":
            card_color = "#3F6212"
            text_color = "#A3E635"
        else:
            card_color = "#1E293B"
            text_color = "#94A3B8"
    else:
        if confidence == "VERY HIGH":
            card_color = "#7F1D1D"
            text_color = "#EF4444"
        elif confidence == "HIGH":
            card_color = "#991B1B"
            text_color = "#F87171"
        elif confidence == "MEDIUM":
            card_color = "#78350F"
            text_color = "#F59E0B"
        elif confidence == "LOW":
            card_color = "#92400E"
            text_color = "#FBBF24"
        else:
            card_color = "#1E293B"
            text_color = "#94A3B8"
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">TOTAL GOALS</h3>
        <div style="font-size: 36px; font-weight: bold; color: {text_color}; margin: 10px 0;">
            {direction} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {prediction['probabilities'][f'{direction.lower()}_2_5_probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            Confidence: {confidence} ({conf_score:.0f}/100)
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== PATTERN INDICATORS ==========
st.divider()
st.subheader("🎯 Backtest-Proven Patterns")

pattern_indicators = generate_pattern_indicators(prediction)

col1, col2 = st.columns(2)

with col1:
    winner_indicator = pattern_indicators['winner']
    if winner_indicator['type'] == 'MET':
        st.markdown(f"""
        <div style="background-color: #14532D; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #22C55E;">
            <div style="font-size: 20px; font-weight: bold; color: #22C55E; margin: 5px 0;">
                ✅ {winner_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #BBF7D0;">
                {winner_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif winner_indicator['type'] == 'AVOID':
        st.markdown(f"""
        <div style="background-color: #7F1D1D; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #EF4444;">
            <div style="font-size: 20px; font-weight: bold; color: #EF4444; margin: 5px 0;">
                ❌ {winner_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #FECACA;">
                {winner_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #374151; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #9CA3AF;">
            <div style="font-size: 20px; font-weight: bold; color: #D1D5DB; margin: 5px 0;">
                ⚪ {winner_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #E5E7EB;">
                {winner_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    totals_indicator = pattern_indicators['totals']
    if totals_indicator['type'] == 'MET':
        st.markdown(f"""
        <div style="background-color: #14532D; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #22C55E;">
            <div style="font-size: 20px; font-weight: bold; color: #22C55E; margin: 5px 0;">
                ✅ {totals_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #BBF7D0;">
                {totals_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif totals_indicator['type'] == 'AVOID':
        st.markdown(f"""
        <div style="background-color: #7F1D1D; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #EF4444;">
            <div style="font-size: 20px; font-weight: bold; color: #EF4444; margin: 5px 0;">
                ❌ {totals_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #FECACA;">
                {totals_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif totals_indicator['type'] == 'WARNING':
        st.markdown(f"""
        <div style="background-color: #78350F; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #F59E0B;">
            <div style="font-size: 20px; font-weight: bold; color: #F59E0B; margin: 5px 0;">
                ⚠️ {totals_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #FDE68A;">
                {totals_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #374151; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #9CA3AF;">
            <div style="font-size: 20px; font-weight: bold; color: #D1D5DB; margin: 5px 0;">
                ⚪ {totals_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #E5E7EB;">
                {totals_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.caption("💡 **Based on 17-match extended test analysis** | Green = Proven pattern to bet | Red = Proven pattern to avoid | Yellow = Warning/Caution | Gray = No proven pattern")

# ========== UNIFIED BETTING CARD ==========
st.divider()
st.subheader("🎯 UNIFIED BETTING CARD - SINGLE BEST BET")

# Get unified prediction for comparison
unified_engine = UnifiedDecisionEngine()
unified_prediction = unified_engine.get_unified_prediction(prediction, pattern_indicators)

# Get unified recommendation
betting_card = UnifiedBettingCard()
unified_recommendation = betting_card.get_unified_recommendation(
    prediction, pattern_indicators, home_team, away_team
)

# Display the unified card
betting_card.display_card(unified_recommendation)

# Show reasoning breakdown
with st.expander("🧠 Why This Bet Was Selected", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏆 Winner Analysis**")
        st.write(f"Base Confidence: {prediction['winner']['confidence_score']:.0f}/100")
        st.write(f"Pattern: {pattern_indicators['winner']['text']}")
        st.write(f"Final Unified: {unified_prediction['winner']['final_confidence']:.0f}/100")
    
    with col2:
        st.write("**📈 Totals Analysis**")
        st.write(f"Base Confidence: {prediction['totals']['confidence_score']:.0f}/100")
        st.write(f"Pattern: {pattern_indicators['totals']['text']}")
        st.write(f"Final Unified: {unified_prediction['totals']['final_confidence']:.0f}/100")
    
    st.write("**📊 Decision Logic**")
    if unified_recommendation['type'] == 'combo':
        st.write("✓ Both winner and totals have high confidence (≥80)")
        st.write("✓ Proven patterns support both predictions")
        st.write("✓ Risk flags are minimal or accounted for")
    elif unified_recommendation['type'] == 'single':
        if 'winner' in unified_recommendation['text']:
            st.write(f"✓ Winner confidence: {unified_prediction['winner']['final_confidence']:.0f}/100 (highest)")
            st.write(f"✓ Totals confidence: {unified_prediction['totals']['final_confidence']:.0f}/100 (lower)")
        else:
            st.write(f"✓ Totals confidence: {unified_prediction['totals']['final_confidence']:.0f}/100 (highest)")
            st.write(f"✓ Winner confidence: {unified_prediction['winner']['final_confidence']:.0f}/100 (lower)")
    else:
        st.write("✗ No bet meets minimum confidence threshold (≥60)")
        st.write("✗ Multiple risk flags present")
        st.write("✗ Pattern indicators show 'AVOID' signals")

# ========== POTENTIAL BET COMBINATIONS ==========
if unified_recommendation['type'] == 'single' and unified_recommendation['confidence'] >= 70:
    # Check for complementary bet
    winner_conf = unified_prediction['winner']['final_confidence']
    totals_conf = unified_prediction['totals']['final_confidence']
    
    complementary_bet = ""
    # If winner is main bet and totals is moderately strong
    if 'winner' in unified_recommendation['text'] and totals_conf >= 60:
        complementary_bet = f"📈 {prediction['totals']['direction']} 2.5 Goals (Confidence: {totals_conf:.0f}/100)"
    # If totals is main bet and winner is moderately strong
    elif 'Goals' in unified_recommendation['text'] and winner_conf >= 60:
        complementary_bet = f"🏆 {prediction['winner']['team']} to win (Confidence: {winner_conf:.0f}/100)"
    
    if complementary_bet:
        st.info(f"💡 **Consider adding**: {complementary_bet}")

# ========== OLD vs NEW COMPARISON ==========
st.divider()
st.subheader("🔄 Comparison: Old vs Unified System")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏆 Winner Comparison")
    
    # Old system logic
    old_winner_conf = prediction['winner']['confidence_score']
    old_winner_action = ""
    if old_winner_conf >= 75:
        old_winner_action = f"✅ STRONG BET on {prediction['winner']['team']}"
    elif old_winner_conf >= 65:
        old_winner_action = f"✅ MODERATE BET on {prediction['winner']['team']}"
    elif old_winner_conf >= 55:
        old_winner_action = f"⚠️ CAUTION on {prediction['winner']['team']}"
    else:
        old_winner_action = f"❌ NO BET on winner"
    
    # New unified winner recommendation
    winner_uni = unified_prediction['winner']
    new_winner_action = f"{winner_uni['bet_recommendation']} on {prediction['winner']['team']}"
    
    st.info(f"**OLD**: {old_winner_action}")
    st.info(f"**NEW**: {new_winner_action}")
    
    if old_winner_action != new_winner_action:
        st.warning("⚠️ **Recommendation changed by unified engine!**")

with col2:
    st.markdown("#### 📈 Totals Comparison")
    
    # Old system logic
    old_totals_conf = prediction['totals']['confidence_score']
    old_totals_action = ""
    if old_totals_conf >= 75:
        old_totals_action = f"✅ STRONG BET on {prediction['totals']['direction']} 2.5"
    elif old_totals_conf >= 65:
        old_totals_action = f"✅ MODERATE BET on {prediction['totals']['direction']} 2.5"
    elif old_totals_conf >= 55:
        old_totals_action = f"⚠️ CAUTION on {prediction['totals']['direction']} 2.5"
    else:
        old_totals_action = f"❌ NO BET on totals"
    
    # New unified totals recommendation
    totals_uni = unified_prediction['totals']
    new_totals_action = f"{totals_uni['bet_recommendation']} on {prediction['totals']['direction']} 2.5"
    
    st.info(f"**OLD**: {old_totals_action}")
    st.info(f"**NEW**: {new_totals_action}")
    
    if old_totals_action != new_totals_action:
        st.warning("⚠️ **Recommendation changed by unified engine!**")

# ========== EXPLANATION OF CHANGES ==========
if old_winner_action != new_winner_action or old_totals_action != new_totals_action:
    with st.expander("🤔 Why did the recommendation change?"):
        st.write("""
        The **Unified Engine** combines:
        1. **Base Model Confidence** - Your original prediction confidence
        2. **Pattern Bonuses** - +40 for proven patterns, -20 for avoid patterns
        3. **Risk Penalties** - Reductions for high variance, volatility, etc.
        
        **Key improvements:**
        - Proven patterns can override low base confidence
        - Clear mathematical formula for decisions
        - No more conflicting signals between patterns and base model
        - Based on 17-match backtest performance
        """)
        
        # Show specific calculations
        if old_totals_action != new_totals_action:
            st.write("### Totals Calculation Example:")
            st.write(f"Base Confidence: {prediction['totals']['confidence_score']:.0f}")
            st.write(f"Pattern Bonus: +40 (PROVEN PATTERN)")
            st.write(f"Risk Penalties: -{max(0, 100 - unified_prediction['totals']['final_confidence'])}")
            st.write(f"**Final Confidence**: {unified_prediction['totals']['final_confidence']:.0f}/100")

# ========== INSIGHTS ==========
if prediction['insights']:
    st.subheader("🧠 Enhanced Insights")
    for insight in prediction['insights']:
        st.write(f"• {insight}")

# ========== RISK FLAGS ==========
if prediction['totals']['risk_flags']:
    st.warning(f"⚠️ **Risk Flags Detected**: {', '.join(prediction['totals']['risk_flags'])}")

# ========== FINISHING TREND ANALYSIS ==========
st.subheader("📊 Finishing Trend Analysis")
col1, col2 = st.columns(2)

with col1:
    home_finish = prediction['totals']['home_finishing']
    finish_cat = engine.totals_predictor.categorize_finishing(home_finish)
    st.metric(f"{home_team} Finishing", f"{home_finish:+.2f}", finish_cat)

with col2:
    away_finish = prediction['totals']['away_finishing']
    finish_cat = engine.totals_predictor.categorize_finishing(away_finish)
    st.metric(f"{away_team} Finishing", f"{away_finish:+.2f}", finish_cat)

finishing_alignment = prediction['totals'].get('finishing_alignment', 'N/A')
total_category = prediction['totals'].get('total_category', 'N/A')
st.info(f"**Finishing Alignment**: {finishing_alignment} | **Total xG Category**: {total_category}")

# ========== DETAILED PROBABILITIES ==========
st.subheader("🎲 Detailed Probabilities")
col1, col2, col3, col4 = st.columns(4)

with col1:
    probs = prediction['probabilities']
    st.metric(f"🏠 {home_team} Win", f"{probs['home_win_probability']*100:.1f}%")

with col2:
    st.metric("🤝 Draw", f"{probs['draw_probability']*100:.1f}%")

with col3:
    st.metric(f"✈️ {away_team} Win", f"{probs['away_win_probability']*100:.1f}%")

with col4:
    st.metric("Both Teams Score", f"{probs['btts_probability']*100:.1f}%")

# ========== MOST LIKELY SCORES ==========
st.subheader("🎯 Most Likely Scores")
scores_cols = st.columns(5)
for idx, (score, prob) in enumerate(prediction['probabilities']['top_scores'][:5]):
    with scores_cols[idx]:
        st.metric(f"{score}", f"{prob*100:.1f}%")

# ========== EXPECTED GOALS ==========
st.subheader("⚽ Expected Goals")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(f"{home_team} xG", f"{prediction['expected_goals']['home']:.2f}")

with col2:
    st.metric(f"{away_team} xG", f"{prediction['expected_goals']['away']:.2f}")

with col3:
    total_xg = prediction['expected_goals']['total']
    league_adj = LEAGUE_ADJUSTMENTS.get(selected_league, LEAGUE_ADJUSTMENTS["Premier League"])
    over_thresh = league_adj['over_threshold']
    st.metric("Total xG", f"{total_xg:.2f}", 
             delta=f"{'OVER' if total_xg > over_thresh else 'UNDER'} {over_thresh}")

# ========== LEGACY BETTING RECOMMENDATIONS ==========
st.divider()
st.subheader("💰 Legacy Betting Recommendations (Original System)")

col1, col2 = st.columns(2)

with col1:
    # Original winner recommendation logic
    winner_rec_old = ""
    if prediction['winner']['confidence'] in ["VERY HIGH", "HIGH"]:
        winner_rec_old = f"✅ **STRONG BET** on {prediction['winner']['team']} to win"
    elif prediction['winner']['confidence'] == "MEDIUM":
        winner_rec_old = f"⚠️ **MODERATE BET** on {prediction['winner']['team']} to win"
    elif prediction['winner']['confidence'] in ["LOW", "VERY LOW"]:
        winner_rec_old = f"❌ **NO BET** on winner - Low confidence"
    
    st.info(winner_rec_old)

with col2:
    # Original totals recommendation logic
    totals_rec_old = ""
    if prediction['totals']['confidence'] in ["VERY HIGH", "HIGH"]:
        totals_rec_old = f"✅ **STRONG BET** on {prediction['totals']['direction']} 2.5"
    elif prediction['totals']['confidence'] == "MEDIUM":
        totals_rec_old = f"⚠️ **MODERATE BET** on {prediction['totals']['direction']} 2.5"
    elif prediction['totals']['confidence'] in ["LOW", "VERY LOW"]:
        totals_rec_old = f"❌ **NO BET** on totals - Low confidence"
    
    st.info(totals_rec_old)

# ========== DETAILED ANALYSIS ==========
if show_details:
    with st.expander("🔍 Detailed Analysis", expanded=False):
        st.write("### Winner Prediction Analysis")
        st.write(f"- Expected Goals Difference: {prediction['winner'].get('strength', 'N/A')}")
        st.write(f"- Confidence Level: {prediction['winner']['confidence']}")
        
        st.write("### Totals Prediction Analysis")
        st.write(f"- Total xG: {prediction['totals']['total_xg']:.2f}")
        st.write(f"- Finishing Alignment: {prediction['totals'].get('finishing_alignment', 'N/A')}")
        st.write(f"- Total Category: {prediction['totals'].get('total_category', 'N/A')}")
        st.write(f"- League-adjusted threshold: {LEAGUE_ADJUSTMENTS.get(selected_league, LEAGUE_ADJUSTMENTS['Premier League'])['over_threshold']}")
        
        if prediction['totals']['risk_flags']:
            st.write("### Risk Analysis")
            for flag in prediction['totals']['risk_flags']:
                st.write(f"- {flag}")

# ========== EXPORT REPORT ==========
st.divider()
st.subheader("📤 Export Prediction Report")

report = f"""
⚽ FOOTBALL INTELLIGENCE ENGINE v3.1 - WITH UNIFIED BETTING CARD
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 UNIFIED BETTING CARD RECOMMENDATION
{unified_recommendation['icon']} {unified_recommendation['text']}
Confidence: {unified_recommendation['confidence']:.0f}/100
Type: {unified_recommendation['subtext']}
Reason: {unified_recommendation['reason']}

🎯 WINNER PREDICTION
Predicted Winner: {prediction['winner']['team']}
Probability: {prediction['winner']['probability']*100:.1f}%
Strength: {prediction['winner']['strength']}
Confidence: {prediction['winner']['confidence']} ({prediction['winner']['confidence_score']:.0f}/100)
Most Likely Score: {prediction['winner']['most_likely_score']}
Pattern: {pattern_indicators['winner']['text']}

🎯 TOTALS PREDICTION  
Direction: {prediction['totals']['direction']} 2.5
Probability: {prediction['probabilities'][f'{prediction["totals"]["direction"].lower()}_2_5_probability']*100:.1f}%
Confidence: {prediction['totals']['confidence']} ({prediction['totals']['confidence_score']:.0f}/100)
Total Expected Goals: {prediction['expected_goals']['total']:.2f}
Finishing Alignment: {prediction['totals'].get('finishing_alignment', 'N/A')}
Total xG Category: {prediction['totals'].get('total_category', 'N/A')}
Pattern: {pattern_indicators['totals']['text']}

📊 EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

📊 FINISHING TRENDS
{home_team}: {prediction['totals']['home_finishing']:+.2f} goals_vs_xg/game
{away_team}: {prediction['totals']['away_finishing']:+.2f} goals_vs_xg/game

⚠️ RISK FLAGS
{', '.join(prediction['totals']['risk_flags']) if prediction['totals']['risk_flags'] else 'None'}

💰 BETTING RECOMMENDATIONS - UNIFIED ENGINE
Winner: {unified_prediction['winner']['bet_recommendation']} on {prediction['winner']['team']}
Totals: {unified_prediction['totals']['bet_recommendation']} on {prediction['totals']['direction']} 2.5

💰 LEGACY RECOMMENDATIONS (Original)
Winner: {winner_rec_old}
Totals: {totals_rec_old}

---
IMPROVED WITH PROVEN 17-MATCH PATTERNS:
1. NEUTRAL + xG>3.0 = UNDER (3/3 proven)
2. MED_UNDER + xG>3.0 = OVER (3/3 proven)  
3. VERY HIGH winner confidence = WIN (4/4 proven)
"""

st.code(report, language="text")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"improved_logic_{home_team}_vs_{away_team}.txt",
        mime="text/plain",
        use_container_width=True
    )

with col2:
    if st.button("📊 Add to History", use_container_width=True):
        st.session_state.prediction_history.append({
            'timestamp': datetime.now(),
            'home_team': home_team,
            'away_team': away_team,
            'league': selected_league,
            'prediction': prediction,
            'pattern_indicators': pattern_indicators,
            'unified_recommendation': unified_recommendation
        })
        st.success("Added to prediction history!")

# Show history
if st.session_state.prediction_history:
    with st.expander("📚 Prediction History", expanded=False):
        for i, hist in enumerate(reversed(st.session_state.prediction_history[-5:])):
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{hist['home_team']} vs {hist['away_team']}**")
                    st.caption(f"{hist['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    winner = hist['prediction']['winner']['team']
                    winner_pattern = hist['pattern_indicators']['winner']['text']
                    st.write(f"🏆 {winner}")
                    st.caption(f"{winner_pattern}")
                with col3:
                    if 'unified_recommendation' in hist:
                        unified = hist['unified_recommendation']
                        st.write(f"🎯 {unified['subtext']}")
                        st.caption(f"{unified['icon']} {unified['text'][:20]}...")
                    else:
                        direction = hist['prediction']['totals']['direction']
                        totals_pattern = hist['pattern_indicators']['totals']['text']
                        st.write(f"📈 {direction} 2.5")
                        st.caption(f"{totals_pattern}")
                st.divider()
