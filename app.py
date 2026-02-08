import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine v4.0",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine v4.0")
st.markdown("""
    **CORRECTED LOGIC: Accounts for finishing tendencies and defensive performance**
    *No more relying on raw xG for predictions*
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
    """Generate pattern indicators based on corrected logic"""
    indicators = {'winner': None, 'totals': None}
    
    winner_pred = prediction['winner']
    winner_prob = winner_pred['probability']
    winner_conf = winner_pred['confidence_score']
    
    # WINNER PATTERNS
    if winner_prob > 0.6 and winner_conf >= 65:
        indicators['winner'] = {
            'type': 'MET',
            'color': 'green',
            'text': 'HIGH PROBABILITY WINNER',
            'explanation': f'{winner_pred["team"]} has {winner_prob*100:.1f}% win probability'
        }
    elif winner_prob < 0.35:
        indicators['winner'] = {
            'type': 'AVOID',
            'color': 'red',
            'text': 'LOW PROBABILITY WINNER',
            'explanation': f'Only {winner_prob*100:.1f}% win probability'
        }
    elif winner_pred.get('volatility_high'):
        indicators['winner'] = {
            'type': 'WARNING',
            'color': 'yellow',
            'text': 'HIGH VOLATILITY',
            'explanation': 'Unpredictable finishing tendencies'
        }
    else:
        indicators['winner'] = {
            'type': 'NO_PATTERN',
            'color': 'gray',
            'text': 'STANDARD MATCH',
            'explanation': f'{winner_prob*100:.1f}% win probability'
        }
    
    # TOTALS PATTERNS
    totals_pred = prediction['totals']
    totals_prob = totals_pred['probability']
    totals_conf = totals_pred['confidence_score']
    
    if totals_prob > 0.65 and totals_conf >= 65:
        indicators['totals'] = {
            'type': 'MET',
            'color': 'green',
            'text': 'HIGH PROBABILITY TOTALS',
            'explanation': f'{totals_pred["direction"]} 2.5 has {totals_prob*100:.1f}% probability'
        }
    elif totals_prob < 0.45:
        indicators['totals'] = {
            'type': 'AVOID',
            'color': 'red',
            'text': 'LOW PROBABILITY TOTALS',
            'explanation': f'Only {totals_prob*100:.1f}% probability'
        }
    elif abs(totals_prob - 0.5) < 0.1:
        indicators['totals'] = {
            'type': 'WARNING',
            'color': 'yellow',
            'text': 'CLOSE CALL',
            'explanation': f'{totals_prob*100:.1f}% probability - coin flip'
        }
    else:
        indicators['totals'] = {
            'type': 'NO_PATTERN',
            'color': 'gray',
            'text': 'STANDARD TOTALS',
            'explanation': f'{totals_prob*100:.1f}% probability for {totals_pred["direction"]} 2.5'
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
            # Per match calculations
            df_part['matches'] = df_part['matches'].astype(float)
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

# ========== CORRECTED LOGIC CLASSES ==========

class ExpectedGoalsPredictor:
    """CORRECTED: Accounts for finishing and defensive tendencies"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_avg_goals = league_metrics.get('avg_goals_per_match', 2.5)
        self.league_name = league_name
    
    def predict_expected_goals(self, home_stats, away_stats):
        """CORRECTED: Adjust xG based on actual performance"""
        
        # Base xG from data
        home_base_xg = home_stats['xg_pm']
        away_base_xg = away_stats['xg_pm']
        
        # Get finishing tendencies (PER MATCH)
        home_finishing = home_stats['goals_vs_xg_pm']  # e.g., +0.17, -0.07
        away_finishing = away_stats['goals_vs_xg_pm']
        
        # Get defensive performance (PER MATCH)
        home_defense = home_stats['goals_allowed_vs_xga_pm']  # Negative = good defense
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        # CORRECTED ADJUSTMENT: Apply finishing tendencies to OPPONENT'S defense
        # If opponent has bad defense, INCREASE our expected goals
        # If opponent has good defense, DECREASE our expected goals
        
        home_adjusted_xg = home_base_xg + home_finishing - away_defense
        away_adjusted_xg = away_base_xg + away_finishing - home_defense
        
        # Venue factor (small adjustment)
        venue_factor = 1.1  # Home advantage
        
        home_adjusted_xg *= venue_factor
        away_adjusted_xg /= venue_factor
        
        # League normalization
        normalization_factor = self.league_avg_goals / 2.5
        home_adjusted_xg *= normalization_factor
        away_adjusted_xg *= normalization_factor
        
        # Realistic bounds
        home_adjusted_xg = max(0.2, min(5.0, home_adjusted_xg))
        away_adjusted_xg = max(0.2, min(5.0, away_adjusted_xg))
        
        return home_adjusted_xg, away_adjusted_xg, {
            'home_base_xg': home_base_xg,
            'away_base_xg': away_base_xg,
            'home_finishing': home_finishing,
            'away_finishing': away_finishing,
            'home_defense': home_defense,
            'away_defense': away_defense,
            'home_adjusted_xg': home_adjusted_xg,
            'away_adjusted_xg': away_adjusted_xg
        }

class PoissonProbabilityEngine:
    """Calculate probabilities from Poisson distribution"""
    
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
            'total_expected_goals': home_xg + away_xg,
            'score_probabilities': score_probabilities
        }

class WinnerPredictor:
    """CORRECTED: Winner prediction based on probabilities"""
    
    def predict_winner(self, home_xg, away_xg, home_stats, away_stats, probabilities):
        """Determine winner based on highest probability"""
        
        home_win_prob = probabilities['home_win_probability']
        away_win_prob = probabilities['away_win_probability']
        draw_prob = probabilities['draw_probability']
        
        # Find highest probability
        max_prob = max(home_win_prob, away_win_prob, draw_prob)
        
        if max_prob == home_win_prob:
            predicted_winner = "HOME"
            win_probability = home_win_prob
            winner_team = "HOME_TEAM"
        elif max_prob == away_win_prob:
            predicted_winner = "AWAY"
            win_probability = away_win_prob
            winner_team = "AWAY_TEAM"
        else:
            predicted_winner = "DRAW"
            win_probability = draw_prob
            winner_team = "DRAW"
        
        # Determine strength based on probability margin
        probs = [home_win_prob, away_win_prob, draw_prob]
        probs_sorted = sorted(probs, reverse=True)
        margin = probs_sorted[0] - probs_sorted[1]
        
        # CORRECTED STRENGTH CATEGORIES
        if win_probability > 0.6 and margin > 0.2:
            strength = "STRONG"
        elif win_probability > 0.55 or margin > 0.15:
            strength = "MODERATE"
        elif win_probability > 0.51 or margin > 0.05:
            strength = "SLIGHT"
        else:
            strength = "CLOSE"
        
        # Base confidence = probability * 100
        base_confidence = win_probability * 100
        
        # Get finishing tendencies
        home_finishing = home_stats['goals_vs_xg_pm']
        away_finishing = away_stats['goals_vs_xg_pm']
        
        # Determine volatility
        volatility_high = False
        if abs(home_finishing) > 0.3 or abs(away_finishing) > 0.3:
            volatility_high = True
        
        # Apply finishing adjustments
        if predicted_winner == "HOME" and home_finishing > 0.2:
            base_confidence += 3  # Clinical finisher bonus
        elif predicted_winner == "AWAY" and away_finishing > 0.2:
            base_confidence += 3
        
        if predicted_winner == "HOME" and home_finishing < -0.2:
            base_confidence -= 3  # Wasteful finisher penalty
        elif predicted_winner == "AWAY" and away_finishing < -0.2:
            base_confidence -= 3
        
        # Volatility penalty
        if volatility_high:
            base_confidence -= 5
        
        # Defense quality adjustments
        home_defense = home_stats['goals_allowed_vs_xga_pm']
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        if predicted_winner == "HOME" and away_defense < -0.3:
            base_confidence -= 3  # Opponent has good defense
        elif predicted_winner == "AWAY" and home_defense < -0.3:
            base_confidence -= 3
        
        # Cap confidence
        final_confidence = min(95, max(5, base_confidence))
        
        # Confidence category
        if final_confidence >= 70:
            confidence_category = "HIGH"
        elif final_confidence >= 60:
            confidence_category = "MEDIUM"
        elif final_confidence >= 50:
            confidence_category = "LOW"
        else:
            confidence_category = "VERY LOW"
        
        # Add volatility to strength
        if volatility_high:
            strength = f"{strength}_VOLATILE"
        
        return {
            'predicted_winner': predicted_winner,
            'team': winner_team,
            'probability': win_probability,
            'strength': strength,
            'confidence': confidence_category,
            'confidence_score': final_confidence,
            'volatility_high': volatility_high,
            'home_finishing': home_finishing,
            'away_finishing': away_finishing,
            'home_defense_quality': home_defense,
            'away_defense_quality': away_defense,
            'margin': margin
        }

class TotalsPredictor:
    """CORRECTED: Totals prediction based on probabilities"""
    
    def __init__(self, league_name):
        self.league_name = league_name
        self.league_adjustments = LEAGUE_ADJUSTMENTS.get(league_name, LEAGUE_ADJUSTMENTS["Premier League"])
    
    def categorize_finishing(self, value):
        """Categorize finishing strength"""
        if value > 0.3:
            return "CLINICAL"
        elif value > 0.1:
            return "GOOD_FINISHER"
        elif value > -0.1:
            return "NEUTRAL"
        elif value > -0.3:
            return "WASTEFUL"
        else:
            return "VERY_WASTEFUL"
    
    def predict_totals(self, home_xg, away_xg, home_stats, away_stats, probabilities):
        """CORRECTED: Totals prediction"""
        total_xg = home_xg + away_xg
        
        # Get probabilities
        over_prob = probabilities['over_2_5_probability']
        under_prob = probabilities['under_2_5_probability']
        
        # Determine direction based on probability
        if over_prob > under_prob:
            base_direction = "OVER"
            base_probability = over_prob
        else:
            base_direction = "UNDER"
            base_probability = under_prob
        
        # Base confidence = probability * 100
        base_confidence = base_probability * 100
        
        # Get finishing tendencies
        home_finishing = home_stats['goals_vs_xg_pm']
        away_finishing = away_stats['goals_vs_xg_pm']
        
        # Categorize finishing
        home_finish_cat = self.categorize_finishing(home_finishing)
        away_finish_cat = self.categorize_finishing(away_finishing)
        
        # Apply finishing adjustments
        if base_direction == "OVER":
            # For OVER prediction, clinical finishing supports it
            if home_finish_cat in ["CLINICAL", "GOOD_FINISHER"]:
                base_confidence += 5
            if away_finish_cat in ["CLINICAL", "GOOD_FINISHER"]:
                base_confidence += 5
            
            # Wasteful finishing hurts OVER prediction
            if home_finish_cat in ["WASTEFUL", "VERY_WASTEFUL"]:
                base_confidence -= 5
            if away_finish_cat in ["WASTEFUL", "VERY_WASTEFUL"]:
                base_confidence -= 5
        else:  # UNDER prediction
            # Wasteful finishing supports UNDER
            if home_finish_cat in ["WASTEFUL", "VERY_WASTEFUL"]:
                base_confidence += 5
            if away_finish_cat in ["WASTEFUL", "VERY_WASTEFUL"]:
                base_confidence += 5
            
            # Clinical finishing hurts UNDER prediction
            if home_finish_cat in ["CLINICAL", "GOOD_FINISHER"]:
                base_confidence -= 5
            if away_finish_cat in ["CLINICAL", "GOOD_FINISHER"]:
                base_confidence -= 5
        
        # Defense quality adjustments
        home_defense = home_stats['goals_allowed_vs_xga_pm']
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        if base_direction == "OVER":
            # Bad defense supports OVER
            if home_defense > 0.2:  # Allows more than expected
                base_confidence += 3
            if away_defense > 0.2:
                base_confidence += 3
            
            # Good defense hurts OVER
            if home_defense < -0.2:  # Allows less than expected
                base_confidence -= 3
            if away_defense < -0.2:
                base_confidence -= 3
        else:  # UNDER
            # Good defense supports UNDER
            if home_defense < -0.2:
                base_confidence += 3
            if away_defense < -0.2:
                base_confidence += 3
            
            # Bad defense hurts UNDER
            if home_defense > 0.2:
                base_confidence -= 3
            if away_defense > 0.2:
                base_confidence -= 3
        
        # Check risk flags
        risk_flags = []
        
        # High finishing volatility
        if abs(home_finishing) > 0.3 or abs(away_finishing) > 0.3:
            risk_flags.append("HIGH_VARIANCE_FINISHING")
            base_confidence -= 3
        
        # Extreme defense mismatch
        if (home_defense > 0.5 and away_defense < -0.5) or (home_defense < -0.5 and away_defense > 0.5):
            risk_flags.append("EXTREME_DEFENSE_MISMATCH")
        
        # Close to threshold
        if 0.45 < base_probability < 0.55:
            risk_flags.append("CLOSE_TO_THRESHOLD")
            base_confidence -= 5
        
        # Cap confidence
        final_confidence = min(95, max(5, base_confidence))
        
        # Confidence category
        if final_confidence >= 70:
            confidence_category = "HIGH"
        elif final_confidence >= 60:
            confidence_category = "MEDIUM"
        elif final_confidence >= 50:
            confidence_category = "LOW"
        else:
            confidence_category = "VERY LOW"
        
        # Total xG category
        if total_xg > 3.3:
            total_category = "VERY_HIGH"
        elif total_xg > 3.0:
            total_category = "HIGH"
        elif total_xg > 2.7:
            total_category = "MODERATE_HIGH"
        elif total_xg > 2.3:
            total_category = "MODERATE_LOW"
        elif total_xg > 2.0:
            total_category = "LOW"
        else:
            total_category = "VERY_LOW"
        
        return {
            'direction': base_direction,
            'probability': base_probability,
            'confidence': confidence_category,
            'confidence_score': final_confidence,
            'total_xg': total_xg,
            'home_finishing': home_finishing,
            'away_finishing': away_finishing,
            'home_finish_category': home_finish_cat,
            'away_finish_category': away_finish_cat,
            'home_defense': home_defense,
            'away_defense': away_defense,
            'risk_flags': risk_flags
        }

class InsightsGenerator:
    """Generate insights based on corrected predictions"""
    
    @staticmethod
    def generate_insights(winner_prediction, totals_prediction):
        insights = []
        
        # Winner insights
        winner_prob = winner_prediction['probability']
        winner_strength = winner_prediction['strength']
        
        if winner_prob > 0.6:
            insights.append(f"🎯 **High Probability**: {winner_prediction['team']} has {winner_prob*100:.1f}% win probability")
        elif winner_prob < 0.4:
            insights.append(f"⚠️ **Low Probability**: {winner_prediction['team']} only has {winner_prob*100:.1f}% win probability")
        
        # Finishing insights
        home_finish = totals_prediction['home_finishing']
        away_finish = totals_prediction['away_finishing']
        home_finish_cat = totals_prediction['home_finish_category']
        away_finish_cat = totals_prediction['away_finish_category']
        
        if home_finish_cat == "CLINICAL":
            insights.append(f"⚡ **Home team clinical**: Overperforms xG by {home_finish:.2f}/game")
        elif home_finish_cat in ["WASTEFUL", "VERY_WASTEFUL"]:
            insights.append(f"⚡ **Home team wasteful**: Underperforms xG by {abs(home_finish):.2f}/game")
        
        if away_finish_cat == "CLINICAL":
            insights.append(f"⚡ **Away team clinical**: Overperforms xG by {away_finish:.2f}/game")
        elif away_finish_cat in ["WASTEFUL", "VERY_WASTEFUL"]:
            insights.append(f"⚡ **Away team wasteful**: Underperforms xG by {abs(away_finish):.2f}/game")
        
        # Defense insights
        home_def = totals_prediction['home_defense']
        away_def = totals_prediction['away_defense']
        
        if home_def < -0.3:
            insights.append(f"🛡️ **Home team good defense**: Allows {abs(home_def):.2f} fewer goals than expected")
        elif home_def > 0.3:
            insights.append(f"🛡️ **Home team poor defense**: Allows {home_def:.2f} more goals than expected")
        
        if away_def < -0.3:
            insights.append(f"🛡️ **Away team good defense**: Allows {abs(away_def):.2f} fewer goals than expected")
        elif away_def > 0.3:
            insights.append(f"🛡️ **Away team poor defense**: Allows {away_def:.2f} more goals than expected")
        
        # Risk flags
        risk_flags = totals_prediction.get('risk_flags', [])
        if risk_flags:
            insights.append(f"⚠️ **Risk factors**: {', '.join(risk_flags)}")
        
        return insights[:6]

class FootballIntelligenceEngineV4:
    """CORRECTED: All predictions account for finishing/defensive tendencies"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_name = league_name
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.probability_engine = PoissonProbabilityEngine()
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.insights_generator = InsightsGenerator()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Complete match prediction - corrected logic"""
        
        # Step 1: Expected goals with finishing/defense adjustments
        home_xg, away_xg, calc_details = self.xg_predictor.predict_expected_goals(
            home_stats, away_stats
        )
        
        # Step 2: Poisson probabilities
        probabilities = self.probability_engine.calculate_all_probabilities(
            home_xg, away_xg
        )
        
        # Step 3: Winner prediction
        winner_prediction = self.winner_predictor.predict_winner(
            home_xg, away_xg, home_stats, away_stats, probabilities
        )
        
        # Replace placeholder with actual team names
        if winner_prediction['predicted_winner'] == "HOME":
            winner_prediction['team'] = home_team
        elif winner_prediction['predicted_winner'] == "AWAY":
            winner_prediction['team'] = away_team
        else:
            winner_prediction['team'] = "DRAW"
        
        # Step 4: Totals prediction
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats, probabilities
        )
        
        # Step 5: Insights
        insights = self.insights_generator.generate_insights(winner_prediction, totals_prediction)
        
        return {
            # Winner prediction
            'winner': {
                'team': winner_prediction['team'],
                'type': winner_prediction['predicted_winner'],
                'probability': winner_prediction['probability'],
                'confidence': winner_prediction['confidence'],
                'confidence_score': winner_prediction['confidence_score'],
                'strength': winner_prediction['strength'],
                'most_likely_score': probabilities['most_likely_score'],
                'volatility_high': winner_prediction['volatility_high'],
                'home_finishing': winner_prediction['home_finishing'],
                'away_finishing': winner_prediction['away_finishing'],
                'margin': winner_prediction['margin']
            },
            
            # Totals prediction
            'totals': totals_prediction,
            
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

# ========== SIMPLE BETTING CARD ==========
class SimpleBettingCard:
    """Betting decisions based on corrected probabilities"""
    
    @staticmethod
    def get_recommendation(prediction, pattern_indicators):
        """Apply betting rules"""
        
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        winner_pattern = pattern_indicators['winner']
        totals_pattern = pattern_indicators['totals']
        
        # Helper: Evaluate a single market
        def evaluate_market(probability, confidence, pattern_type, pattern_exp):
            
            # RULE A: Pattern MET → BET (if probability ≥ 55% and confidence ≥ 55)
            if pattern_type == 'MET' and probability >= 0.55 and confidence >= 55:
                return True, f"✅ {pattern_exp}"
            
            # RULE B: Pattern AVOID → NO BET
            elif pattern_type == 'AVOID':
                return False, f"🚫 {pattern_exp}"
            
            # RULE C: Pattern NO_PATTERN → Check probability
            elif pattern_type == 'NO_PATTERN':
                if probability >= 0.6 and confidence >= 60:
                    return True, f"✅ {probability*100:.1f}% probability with good confidence"
                elif probability >= 0.55 and confidence >= 55:
                    return True, f"✅ {probability*100:.1f}% probability - acceptable"
                else:
                    return False, f"🚫 {probability*100:.1f}% probability too low"
            
            # RULE D: Pattern WARNING → Needs higher thresholds
            elif pattern_type == 'WARNING':
                if probability >= 0.65 and confidence >= 65:
                    return True, f"⚠️ WARNING but {probability*100:.1f}% probability with high confidence"
                else:
                    return False, f"⚠️ WARNING: {pattern_exp}"
            
            return False, "No decision"
        
        # Evaluate winner
        winner_bet, winner_reason = evaluate_market(
            winner_pred['probability'],
            winner_pred['confidence_score'],
            winner_pattern['type'],
            winner_pattern['explanation']
        )
        
        # Evaluate totals
        totals_bet, totals_reason = evaluate_market(
            totals_pred['probability'],
            totals_pred['confidence_score'],
            totals_pattern['type'],
            totals_pattern['explanation']
        )
        
        # Make final decision
        if winner_bet and totals_bet:
            min_conf = min(winner_pred['confidence_score'], totals_pred['confidence_score'])
            return {
                'type': 'combo',
                'text': f"🏆 {winner_pred['team']} to win + 📈 {totals_pred['direction']} 2.5",
                'confidence': min_conf,
                'color': '#10B981',
                'icon': '🎯',
                'subtext': 'DOUBLE BET',
                'reason': f'{winner_reason} | {totals_reason}'
            }
        
        elif winner_bet:
            return {
                'type': 'single',
                'text': f"🏆 {winner_pred['team']} to win",
                'confidence': winner_pred['confidence_score'],
                'color': '#3B82F6',
                'icon': '🏆',
                'subtext': 'WINNER BET',
                'reason': winner_reason
            }
        
        elif totals_bet:
            return {
                'type': 'single',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'confidence': totals_pred['confidence_score'],
                'color': '#8B5CF6',
                'icon': '📈',
                'subtext': 'TOTALS BET',
                'reason': totals_reason
            }
        
        else:
            max_conf = max(winner_pred['confidence_score'], totals_pred['confidence_score'])
            return {
                'type': 'none',
                'text': "🚫 No Recommended Bet",
                'confidence': max_conf,
                'color': '#6B7280',
                'icon': '🤔',
                'subtext': 'NO BET',
                'reason': f'{winner_reason} | {totals_reason}'
            }
    
    @staticmethod
    def display_card(recommendation):
        """Display the betting card"""
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
st.caption(f"League: {selected_league}")

engine = FootballIntelligenceEngineV4(league_metrics, selected_league)
prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)

# ========== DISPLAY RESULTS ==========

# Main prediction cards
col1, col2 = st.columns(2)

with col1:
    # Winner prediction
    winner_pred = prediction['winner']
    winner_prob = winner_pred['probability']
    winner_conf = winner_pred['confidence']
    winner_conf_score = winner_pred['confidence_score']
    
    if winner_pred['type'] == "HOME":
        winner_color = "#22C55E" if winner_conf in ["HIGH", "MEDIUM"] else "#F59E0B"
        icon = "🏠"
    elif winner_pred['type'] == "AWAY":
        winner_color = "#22C55E" if winner_conf in ["HIGH", "MEDIUM"] else "#F59E0B"
        icon = "✈️"
    else:
        winner_color = "#F59E0B"
        icon = "🤝"
    
    # Color based on confidence
    if winner_conf == "HIGH":
        card_color = "#14532D"
    elif winner_conf == "MEDIUM":
        card_color = "#166534"
    elif winner_conf == "LOW":
        card_color = "#365314"
    else:
        card_color = "#1E293B"
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">PREDICTED WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: {winner_color}; margin: 10px 0;">
            {icon} {winner_pred['team']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_prob*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            Strength: {winner_pred['strength']} | Confidence: {winner_conf} ({winner_conf_score:.0f}/100)
        </div>
        <div style="font-size: 14px; color: #BBF7D0; margin-top: 10px;">
            Margin: {winner_pred['margin']*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Totals prediction
    totals_pred = prediction['totals']
    direction = totals_pred['direction']
    probability = totals_pred['probability']
    confidence = totals_pred['confidence']
    conf_score = totals_pred['confidence_score']
    
    if direction == "OVER":
        if confidence == "HIGH":
            card_color = "#14532D"
            text_color = "#22C55E"
        elif confidence == "MEDIUM":
            card_color = "#166534"
            text_color = "#4ADE80"
        else:
            card_color = "#1E293B"
            text_color = "#94A3B8"
    else:
        if confidence == "HIGH":
            card_color = "#7F1D1D"
            text_color = "#EF4444"
        elif confidence == "MEDIUM":
            card_color = "#991B1B"
            text_color = "#F87171"
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
            {probability*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            Confidence: {confidence} ({conf_score:.0f}/100)
        </div>
        <div style="font-size: 14px; color: #BBF7D0; margin-top: 10px;">
            Expected Goals: {prediction['expected_goals']['total']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== PATTERN INDICATORS ==========
st.divider()
st.subheader("🎯 Pattern Analysis")

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
    elif winner_indicator['type'] == 'WARNING':
        st.markdown(f"""
        <div style="background-color: #78350F; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #F59E0B;">
            <div style="font-size: 20px; font-weight: bold; color: #F59E0B; margin: 5px 0;">
                ⚠️ {winner_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #FDE68A;">
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

st.caption("💡 **Patterns based on actual probabilities and team tendencies**")

# ========== BETTING CARD ==========
st.divider()
st.subheader("🎯 Betting Recommendation")

# Get recommendation
betting_card = SimpleBettingCard()
recommendation = betting_card.get_recommendation(prediction, pattern_indicators)

# Display the card
betting_card.display_card(recommendation)

# ========== INSIGHTS ==========
if prediction['insights']:
    st.subheader("🧠 Match Insights")
    for insight in prediction['insights']:
        st.write(f"• {insight}")

# ========== RISK FLAGS ==========
if prediction['totals']['risk_flags']:
    st.warning(f"⚠️ **Risk Flags Detected**: {', '.join(prediction['totals']['risk_flags'])}")

# ========== FINISHING ANALYSIS ==========
st.subheader("📊 Team Tendencies Analysis")
col1, col2 = st.columns(2)

with col1:
    home_finish = prediction['totals']['home_finishing']
    home_finish_cat = prediction['totals']['home_finish_category']
    home_def = prediction['totals']['home_defense']
    
    st.metric(f"{home_team} Finishing", 
              f"{home_finish:+.2f}", 
              home_finish_cat)
    st.metric(f"{home_team} Defense", 
              f"{home_def:+.2f}", 
              "Good" if home_def < -0.2 else "Poor" if home_def > 0.2 else "Average")

with col2:
    away_finish = prediction['totals']['away_finishing']
    away_finish_cat = prediction['totals']['away_finish_category']
    away_def = prediction['totals']['away_defense']
    
    st.metric(f"{away_team} Finishing", 
              f"{away_finish:+.2f}", 
              away_finish_cat)
    st.metric(f"{away_team} Defense", 
              f"{away_def:+.2f}", 
              "Good" if away_def < -0.2 else "Poor" if away_def > 0.2 else "Average")

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

# ========== EXPORT REPORT ==========
st.divider()
st.subheader("📤 Export Prediction Report")

report = f"""
⚽ FOOTBALL INTELLIGENCE ENGINE v4.0 - CORRECTED LOGIC
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 BETTING RECOMMENDATION
{recommendation['icon']} {recommendation['text']}
Type: {recommendation['subtext']}
Confidence: {recommendation['confidence']:.0f}/100
Reason: {recommendation['reason']}

📊 WINNER PREDICTION
Predicted Winner: {prediction['winner']['team']}
Probability: {prediction['winner']['probability']*100:.1f}%
Strength: {prediction['winner']['strength']}
Confidence: {prediction['winner']['confidence']} ({prediction['winner']['confidence_score']:.0f}/100)
Margin: {prediction['winner']['margin']*100:.1f}%
Most Likely Score: {prediction['winner']['most_likely_score']}

📊 TOTALS PREDICTION  
Direction: {prediction['totals']['direction']} 2.5
Probability: {prediction['totals']['probability']*100:.1f}%
Confidence: {prediction['totals']['confidence']} ({prediction['totals']['confidence_score']:.0f}/100)
Total Expected Goals: {prediction['totals']['total_xg']:.2f}

📊 TEAM TENDENCIES
{home_team} Finishing: {prediction['totals']['home_finishing']:+.2f} ({prediction['totals']['home_finish_category']})
{home_team} Defense: {prediction['totals']['home_defense']:+.2f} ({"Good" if prediction['totals']['home_defense'] < -0.2 else "Poor" if prediction['totals']['home_defense'] > 0.2 else "Average"})
{away_team} Finishing: {prediction['totals']['away_finishing']:+.2f} ({prediction['totals']['away_finish_category']})
{away_team} Defense: {prediction['totals']['away_defense']:+.2f} ({"Good" if prediction['totals']['away_defense'] < -0.2 else "Poor" if prediction['totals']['away_defense'] > 0.2 else "Average"})

📊 EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

📊 DETAILED PROBABILITIES
{home_team} Win: {prediction['probabilities']['home_win_probability']*100:.1f}%
Draw: {prediction['probabilities']['draw_probability']*100:.1f}%
{away_team} Win: {prediction['probabilities']['away_win_probability']*100:.1f}%
Both Teams Score: {prediction['probabilities']['btts_probability']*100:.1f}%
OVER 2.5: {prediction['probabilities']['over_2_5_probability']*100:.1f}%
UNDER 2.5: {prediction['probabilities']['under_2_5_probability']*100:.1f}%

⚠️ RISK FLAGS
{', '.join(prediction['totals']['risk_flags']) if prediction['totals']['risk_flags'] else 'None'}

🧠 INSIGHTS
{chr(10).join(prediction['insights']) if prediction['insights'] else 'None'}

---
CORRECTED LOGIC APPLIED:
1. Accounts for finishing tendencies (clinical vs wasteful)
2. Accounts for defensive performance (good vs poor defense)
3. Uses actual probabilities, not raw xG for decisions
4. Confidence correlates with probability
"""

st.code(report, language="text")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"corrected_{home_team}_vs_{away_team}.txt",
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
            'recommendation': recommendation
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
                    winner_prob = hist['prediction']['winner']['probability']*100
                    st.write(f"🏆 {winner}")
                    st.caption(f"{winner_prob:.1f}%")
                with col3:
                    unified = hist['recommendation']
                    st.write(f"🎯 {unified['subtext']}")
                    st.caption(f"{unified['icon']} {unified['text'][:20]}...")
                st.divider()
