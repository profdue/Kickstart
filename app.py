import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine v3.0",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine v3.0")
st.markdown("""
    **Complete Logic System: Winners + Totals with Finishing Trend Analysis**
    *Separate confidence systems for winners and totals*
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
    "Eredivisie": {"over_threshold": 2.9, "under_threshold": 2.1, "avg_goals": 3.10}
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

# ========== OUR NEW LOGIC CLASSES ==========

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
        
        # OUR LOGIC: Winner confidence calculation (PROVEN TO WORK)
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
    """OUR NEW LOGIC: Totals prediction with finishing trend analysis"""
    
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
        """OUR LOGIC: Risk flag system"""
        risk_flags = []
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
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
        
        return risk_flags
    
    def predict_totals(self, home_xg, away_xg, home_stats, away_stats):
        """OUR LOGIC: Complete totals prediction"""
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
        
        # OUR LOGIC: Decision matrix
        decision_matrix = {
            "VERY_HIGH": {
                "HIGH_OVER": ("OVER", "VERY HIGH", 85),
                "MED_OVER": ("OVER", "HIGH", 75),
                "LOW_OVER": ("OVER", "MEDIUM", 65),
                "NEUTRAL": ("OVER", "MEDIUM", 60),
                "RISKY": ("OVER", "LOW", 45),
                "HIGH_RISK": (base_direction, "VERY LOW", 35)
            },
            "HIGH": {
                "HIGH_OVER": ("OVER", "VERY HIGH", 80),
                "MED_OVER": ("OVER", "HIGH", 70),
                "LOW_OVER": ("OVER", "MEDIUM", 60),
                "NEUTRAL": (base_direction, "LOW", 50),
                "RISKY": (base_direction, "LOW", 45),
                "HIGH_RISK": (base_direction, "VERY LOW", 35)
            },
            "MODERATE_HIGH": {
                "HIGH_OVER": ("OVER", "HIGH", 75),
                "MED_OVER": ("OVER", "MEDIUM", 65),
                "LOW_OVER": ("OVER", "MEDIUM", 60),
                "NEUTRAL": (base_direction, "LOW", 50),
                "LOW_UNDER": ("UNDER", "LOW", 45),
                "MED_UNDER": ("UNDER", "MEDIUM", 55)
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
        if finishing_alignment in decision_matrix[total_category]:
            direction, confidence_category, base_confidence = decision_matrix[total_category][finishing_alignment]
        else:
            direction = base_direction
            confidence_category = "LOW"
            base_confidence = 40
        
        # OUR LOGIC: Apply risk flag penalties
        final_confidence = base_confidence
        for flag in risk_flags:
            if flag == "OPPOSITE_EXTREME_FINISHING":
                final_confidence -= 25
            elif flag == "HIGH_VARIANCE_TEAM":
                final_confidence -= 15
            elif flag == "ATTACK_DEFENSE_MISMATCH":
                final_confidence -= 10
            elif flag == "CLOSE_TO_THRESHOLD":
                final_confidence -= 10
        
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
    """OUR LOGIC: Generate enhanced insights"""
    
    @staticmethod
    def generate_insights(winner_prediction, totals_prediction):
        insights = []
        
        # Winner insights
        if winner_prediction['winner_confidence_category'] == "VERY HIGH":
            insights.append(f"🎯 **High Confidence Winner**: Model strongly favors {winner_prediction['predicted_winner']}")
        elif winner_prediction['winner_confidence_category'] == "LOW":
            insights.append(f"⚠️ **Low Confidence Winner**: Exercise caution on {winner_prediction['predicted_winner']} prediction")
        
        # Totals insights
        if totals_prediction['confidence'] == "VERY HIGH":
            insights.append(f"🎯 **High Confidence Totals**: Strong signal for {totals_prediction['direction']} 2.5")
        elif totals_prediction['confidence'] == "VERY LOW":
            insights.append(f"⚠️ **Low Confidence Totals**: High risk on {totals_prediction['direction']} 2.5 prediction")
        
        # Finishing trend insights
        home_finish = totals_prediction['home_finishing']
        away_finish = totals_prediction['away_finishing']
        
        if home_finish > 0.3:
            insights.append(f"⚡ **Home team overperforms xG** by {home_finish:.2f}/game (clinical finishing)")
        elif home_finish < -0.3:
            insights.append(f"⚡ **Home team underperforms xG** by {abs(home_finish):.2f}/game (wasteful finishing)")
        
        if away_finish > 0.3:
            insights.append(f"⚡ **Away team overperforms xG** by {away_finish:.2f}/game (clinical finishing)")
        elif away_finish < -0.3:
            insights.append(f"⚡ **Away team underperforms xG** by {abs(away_finish):.2f}/game (wasteful finishing)")
        
        # Finishing alignment insights
        alignment = totals_prediction['finishing_alignment']
        if alignment == "HIGH_OVER":
            insights.append("🔥 **Both teams show strong overperformance** - High probability of OVER")
        elif alignment == "HIGH_UNDER":
            insights.append("🛡️ **Both teams show strong underperformance** - High probability of UNDER")
        elif "RISKY" in alignment:
            insights.append("🎲 **Mixed finishing trends** - Higher variance expected")
        
        # Risk flag insights
        if totals_prediction['risk_flags']:
            risk_count = len(totals_prediction['risk_flags'])
            flag_list = ", ".join(totals_prediction['risk_flags'])
            insights.append(f"⚠️ **{risk_count} risk flag(s) detected**: {flag_list}")
        
        return insights[:5]

class FootballIntelligenceEngineV3:
    """OUR LOGIC: Complete prediction engine"""
    
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
        
        # Step 4: OUR LOGIC - Totals prediction
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Step 5: OUR LOGIC - Insights
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

# ========== STREAMLIT UI ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie"]
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

# Insights
if prediction['insights']:
    st.subheader("🧠 Enhanced Insights")
    for insight in prediction['insights']:
        st.write(f"• {insight}")

# Risk Flags
if prediction['totals']['risk_flags']:
    st.warning(f"⚠️ **Risk Flags Detected**: {', '.join(prediction['totals']['risk_flags'])}")

# Finishing Trend Analysis
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

st.info(f"**Finishing Alignment**: {prediction['totals']['finishing_alignment']} | **Total xG Category**: {prediction['totals']['total_category']}")

# Detailed Probabilities
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

# Most Likely Scores
st.subheader("🎯 Most Likely Scores")
scores_cols = st.columns(5)
for idx, (score, prob) in enumerate(prediction['probabilities']['top_scores'][:5]):
    with scores_cols[idx]:
        st.metric(f"{score}", f"{prob*100:.1f}%")

# Expected Goals
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

# Betting Recommendations
st.subheader("💰 Betting Recommendations")

winner_rec = ""
if prediction['winner']['confidence'] in ["VERY HIGH", "HIGH"]:
    winner_rec = f"✅ **STRONG BET** on {prediction['winner']['team']} to win"
elif prediction['winner']['confidence'] == "MEDIUM":
    winner_rec = f"⚠️ **MODERATE BET** on {prediction['winner']['team']} to win"
elif prediction['winner']['confidence'] in ["LOW", "VERY LOW"]:
    winner_rec = f"❌ **NO BET** on winner - Low confidence"

totals_rec = ""
if prediction['totals']['confidence'] in ["VERY HIGH", "HIGH"]:
    totals_rec = f"✅ **STRONG BET** on {prediction['totals']['direction']} 2.5"
elif prediction['totals']['confidence'] == "MEDIUM":
    totals_rec = f"⚠️ **MODERATE BET** on {prediction['totals']['direction']} 2.5"
elif prediction['totals']['confidence'] in ["LOW", "VERY LOW"]:
    totals_rec = f"❌ **NO BET** on totals - Low confidence"

col1, col2 = st.columns(2)
with col1:
    st.info(winner_rec)
with col2:
    st.info(totals_rec)

# Detailed Analysis
if show_details:
    with st.expander("🔍 Detailed Analysis", expanded=False):
        st.write("### Winner Prediction Analysis")
        st.write(f"- Expected Goals Difference: {prediction['winner'].get('strength', 'N/A')}")
        st.write(f"- Confidence Level: {prediction['winner']['confidence']}")
        
        st.write("### Totals Prediction Analysis")
        st.write(f"- Total xG: {prediction['totals']['total_xg']:.2f}")
        st.write(f"- Finishing Alignment: {prediction['totals']['finishing_alignment']}")
        st.write(f"- League-adjusted threshold: {LEAGUE_ADJUSTMENTS.get(selected_league, LEAGUE_ADJUSTMENTS['Premier League'])['over_threshold']}")
        
        if prediction['totals']['risk_flags']:
            st.write("### Risk Analysis")
            for flag in prediction['totals']['risk_flags']:
                st.write(f"- {flag}")

# Export Report
st.divider()
st.subheader("📤 Export Prediction Report")

report = f"""
⚽ FOOTBALL INTELLIGENCE ENGINE v3.0 - OUR COMPLETE LOGIC
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 WINNER PREDICTION
Predicted Winner: {prediction['winner']['team']}
Probability: {prediction['winner']['probability']*100:.1f}%
Strength: {prediction['winner']['strength']}
Confidence: {prediction['winner']['confidence']} ({prediction['winner']['confidence_score']:.0f}/100)
Most Likely Score: {prediction['winner']['most_likely_score']}

🎯 TOTALS PREDICTION  
Direction: {prediction['totals']['direction']} 2.5
Probability: {prediction['probabilities'][f'{prediction["totals"]["direction"].lower()}_2_5_probability']*100:.1f}%
Confidence: {prediction['totals']['confidence']} ({prediction['totals']['confidence_score']:.0f}/100)
Total Expected Goals: {prediction['expected_goals']['total']:.2f}
Finishing Alignment: {prediction['totals']['finishing_alignment']}
Total xG Category: {prediction['totals']['total_category']}

📊 EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

📊 FINISHING TRENDS
{home_team}: {prediction['totals']['home_finishing']:+.2f} goals_vs_xg/game
{away_team}: {prediction['totals']['away_finishing']:+.2f} goals_vs_xg/game

⚠️ RISK FLAGS
{', '.join(prediction['totals']['risk_flags']) if prediction['totals']['risk_flags'] else 'None'}

💰 BETTING RECOMMENDATIONS
Winner: {winner_rec.replace('✅', 'STRONG BET:').replace('⚠️', 'MODERATE BET:').replace('❌', 'NO BET:')}
Totals: {totals_rec.replace('✅', 'STRONG BET:').replace('⚠️', 'MODERATE BET:').replace('❌', 'NO BET:')}

---
Generated by Football Intelligence Engine v3.0
OUR COMPLETE LOGIC SYSTEM
"""

st.code(report, language="text")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"complete_logic_{home_team}_vs_{away_team}.txt",
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
            'prediction': prediction
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
                    st.write(f"🏆 {winner}")
                with col3:
                    direction = hist['prediction']['totals']['direction']
                    st.write(f"📈 {direction} 2.5")
                st.divider()
