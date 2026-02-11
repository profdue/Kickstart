import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
import json
import os
from supabase import create_client, Client
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine - LEAGUE-SPECIFIC MODELS",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Football Intelligence Engine - LEAGUE-SPECIFIC MODELS")
st.markdown("""
    **INDEPENDENT LEAGUE ENGINES** - Each league optimized based on 35-match analysis
""")

# ========== SUPABASE INITIALIZATION ==========
def init_supabase():
    """Initialize Supabase client"""
    try:
        supabase_url = st.secrets.get("SUPABASE_URL")
        supabase_key = st.secrets.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            st.warning("Supabase credentials not found in secrets. Using local storage only.")
            return None
            
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Error initializing Supabase: {str(e)}")
        return None

# Initialize
supabase = init_supabase()

# ========== LEAGUE-SPECIFIC ENGINES ==========

class PremierLeagueEngine:
    """COMPLETE REBUILD - Current model is 78% wrong (22% accuracy)"""
    
    def __init__(self):
        self.name = "Premier League"
        self.winner_accuracy = 0.222  # From 35-match analysis
        self.totals_accuracy = 0.444  # From 35-match analysis
        self.model_type = "REBUILD_FROM_SCRATCH"
        self.version = "v2.0_premier_fixed"
        
    def predict_winner(self, home_xg, away_xg, home_finish, away_finish):
        """SCRAP current model - use simple proven rules"""
        finish_diff = home_finish - away_finish
        xg_diff = home_xg - away_xg
        
        # Rule 1: Strong finishing advantage wins
        if finish_diff > 0.15 and xg_diff > 0:
            confidence = min(80, 60 + abs(finish_diff) * 100)
            return "HOME", confidence, "FINISHING_ADVANTAGE"
        elif finish_diff < -0.15 and xg_diff < 0:
            confidence = min(80, 60 + abs(finish_diff) * 100)
            return "AWAY", confidence, "FINISHING_ADVANTAGE"
        
        # Rule 2: Large xG difference wins (high threshold for low-accuracy league)
        elif xg_diff > 0.7:
            confidence = 55
            return "HOME", confidence, "XG_ADVANTAGE"
        elif xg_diff < -0.7:
            confidence = 55
            return "AWAY", confidence, "XG_ADVANTAGE"
        
        # Default: Draw (safest for unpredictable league)
        else:
            confidence = 50
            return "DRAW", confidence, "CAUTIOUS"
    
    def predict_totals(self, total_xg, home_finish, away_finish):
        """Premier League: xG correlation = -0.03 (USELESS) - Use finishing stats"""
        avg_finish = (home_finish + away_finish) / 2
        
        if avg_finish > 0.1:
            confidence = 60
            return "OVER", confidence, "GOOD_FINISHERS"
        elif avg_finish < -0.1:
            confidence = 60
            return "UNDER", confidence, "POOR_FINISHERS"
        else:
            # Neutral: slight edge to OVER for entertainment
            confidence = 52
            return "OVER", confidence, "NEUTRAL"
            
    def get_expected_improvement(self):
        return {
            "winner": {"current": 22.2, "expected": 45.0, "improvement": "+22.8%"},
            "totals": {"current": 44.4, "expected": 57.5, "improvement": "+13.1%"}
        }

class SerieAEngine:
    """LEVERAGE TOTALS STRENGTH - 75% totals accuracy"""
    
    def __init__(self):
        self.name = "Serie A"
        self.winner_accuracy = 0.500
        self.totals_accuracy = 0.750  # EXCELLENT from analysis
        self.model_type = "TOTALS_SPECIALIST"
        self.version = "v2.0_serie_totals"
        self.xg_calibration = 1.34  # Goals = xG × 1.34 (+34%)
        
    def predict_winner(self, home_xg, away_xg, home_finish, away_finish):
        """Standard approach - focus resources on totals"""
        finish_diff = home_finish - away_finish
        xg_diff = home_xg - away_xg
        
        if xg_diff > 0.5 and finish_diff > 0.1:
            confidence = 70
            return "HOME", confidence, "COMBINED_ADVANTAGE"
        elif xg_diff < -0.5 and finish_diff < -0.1:
            confidence = 70
            return "AWAY", confidence, "COMBINED_ADVANTAGE"
        elif xg_diff > 0.5:
            confidence = 60
            return "HOME", confidence, "XG_ADVANTAGE"
        elif xg_diff < -0.5:
            confidence = 60
            return "AWAY", confidence, "XG_ADVANTAGE"
        else:
            confidence = 50
            return "DRAW", confidence, "BALANCED"
    
    def predict_totals(self, total_xg, home_finish, away_finish):
        """Serie A: 75% accuracy for OVER predictions"""
        calibrated_xg = total_xg * self.xg_calibration
        
        if calibrated_xg > 2.2:  # Lower threshold for high-scoring league
            confidence = 80
            return "OVER", confidence, "HIGH_SCORING_LEAGUE"
        elif total_xg < 1.5:  # Only predict UNDER for very low xG
            confidence = 60
            return "UNDER", confidence, "VERY_LOW_XG"
        else:
            confidence = 75  # 75% accuracy from analysis
            return "OVER", confidence, "LEAGUE_TREND"
            
    def get_expected_improvement(self):
        return {
            "winner": {"current": 50.0, "expected": 57.5, "improvement": "+7.5%"},
            "totals": {"current": 75.0, "expected": 82.5, "improvement": "+7.5%"}
        }

class Ligue1Engine:
    """LEVERAGE WINNER STRENGTH - 75% winner accuracy"""
    
    def __init__(self):
        self.name = "Ligue 1"
        self.winner_accuracy = 0.750  # EXCELLENT from analysis
        self.totals_accuracy = 0.375  # TERRIBLE from analysis
        self.model_type = "WINNER_SPECIALIST"
        self.version = "v2.0_ligue_winners"
        self.xg_calibration = 0.61  # Goals = xG × 0.61 (-39%)
        
    def predict_winner(self, home_xg, away_xg, home_finish, away_finish):
        """Ligue 1: xG difference works well (75% accuracy)"""
        xg_diff = home_xg - away_xg
        
        # Simple threshold from analysis
        if xg_diff > 0.3:
            confidence = 80
            return "HOME", confidence, "XG_ADVANTAGE"
        elif xg_diff < -0.3:
            confidence = 80
            return "AWAY", confidence, "XG_ADVANTAGE"
        else:
            confidence = 50
            return "DRAW", confidence, "CLOSE_MATCH"
    
    def predict_totals(self, total_xg, home_finish, away_finish):
        """Ligue 1: Goals = 61% of xG (from analysis)"""
        calibrated_xg = total_xg * self.xg_calibration
        
        if calibrated_xg > 2.8:  # Higher threshold for low-scoring league
            confidence = 65
            return "OVER", confidence, "CALIBRATED_XG"
        else:
            confidence = 65
            return "UNDER", confidence, "CALIBRATED_XG"
            
    def get_expected_improvement(self):
        return {
            "winner": {"current": 75.0, "expected": 77.5, "improvement": "+2.5%"},
            "totals": {"current": 37.5, "expected": 62.5, "improvement": "+25.0%"}
        }

class BundesligaEngine:
    """MODERATE IMPROVEMENT - 50% accuracy across board"""
    
    def __init__(self):
        self.name = "Bundesliga"
        self.winner_accuracy = 0.500
        self.totals_accuracy = 0.500
        self.model_type = "BALANCED_IMPROVEMENT"
        self.version = "v2.0_bundesliga"
        self.xg_calibration = 1.15  # Goals slightly exceed xG
        
    def predict_winner(self, home_xg, away_xg, home_finish, away_finish):
        """Standard model with calibration"""
        finish_diff = home_finish - away_finish
        xg_diff = home_xg - away_xg
        
        if xg_diff > 0.5 and finish_diff > 0.1:
            confidence = 70
            return "HOME", confidence, "COMBINED_ADVANTAGE"
        elif xg_diff < -0.5 and finish_diff < -0.1:
            confidence = 70
            return "AWAY", confidence, "COMBINED_ADVANTAGE"
        elif xg_diff > 0.5:
            confidence = 60
            return "HOME", confidence, "XG_ADVANTAGE"
        elif xg_diff < -0.5:
            confidence = 60
            return "AWAY", confidence, "XG_ADVANTAGE"
        else:
            confidence = 50
            return "DRAW", confidence, "BALANCED"
    
    def predict_totals(self, total_xg, home_finish, away_finish):
        """Bundesliga: Goals slightly exceed xG"""
        calibrated_xg = total_xg * self.xg_calibration
        
        if calibrated_xg > 2.7:  # Slightly higher threshold
            confidence = 65
            return "OVER", confidence, "CALIBRATED_XG"
        else:
            confidence = 65
            return "UNDER", confidence, "CALIBRATED_XG"
            
    def get_expected_improvement(self):
        return {
            "winner": {"current": 50.0, "expected": 57.5, "improvement": "+7.5%"},
            "totals": {"current": 50.0, "expected": 62.5, "improvement": "+12.5%"}
        }

class LaLigaEngine:
    """MAINTAIN WINNER STRENGTH - 62.5% winner accuracy"""
    
    def __init__(self):
        self.name = "La Liga"
        self.winner_accuracy = 0.625  # GOOD from analysis
        self.totals_accuracy = 0.500
        self.model_type = "WINNER_FOCUSED"
        self.version = "v2.0_laliga"
        self.xg_calibration = 1.1  # Slight calibration
        
    def predict_winner(self, home_xg, away_xg, home_finish, away_finish):
        """La Liga: xG difference works (62.5% accuracy)"""
        xg_diff = home_xg - away_xg
        
        # Lower threshold from analysis (decisive matches)
        if xg_diff > 0.4:
            confidence = 75
            return "HOME", confidence, "XG_ADVANTAGE"
        elif xg_diff < -0.4:
            confidence = 75
            return "AWAY", confidence, "XG_ADVANTAGE"
        else:
            confidence = 50
            return "DRAW", confidence, "CLOSE_MATCH"
    
    def predict_totals(self, total_xg, home_finish, away_finish):
        """Standard with slight calibration"""
        calibrated_xg = total_xg * self.xg_calibration
        
        if calibrated_xg > 2.6:
            confidence = 60
            return "OVER", confidence, "CALIBRATED_XG"
        else:
            confidence = 60
            return "UNDER", confidence, "CALIBRATED_XG"
            
    def get_expected_improvement(self):
        return {
            "winner": {"current": 62.5, "expected": 67.5, "improvement": "+5.0%"},
            "totals": {"current": 50.0, "expected": 57.5, "improvement": "+7.5%"}
        }

class EredivisieEngine:
    """Similar to Bundesliga (estimated)"""
    
    def __init__(self):
        self.name = "Eredivisie"
        self.winner_accuracy = 0.500  # Estimated
        self.totals_accuracy = 0.500  # Estimated
        self.model_type = "ESTIMATED_MODEL"
        self.version = "v2.0_eredivisie"
        self.xg_calibration = 1.2  # High scoring league
        
    def predict_winner(self, home_xg, away_xg, home_finish, away_finish):
        """Use Bundesliga logic for now"""
        finish_diff = home_finish - away_finish
        xg_diff = home_xg - away_xg
        
        if xg_diff > 0.5 and finish_diff > 0.1:
            confidence = 65
            return "HOME", confidence, "COMBINED_ADVANTAGE"
        elif xg_diff < -0.5 and finish_diff < -0.1:
            confidence = 65
            return "AWAY", confidence, "COMBINED_ADVANTAGE"
        elif xg_diff > 0.5:
            confidence = 55
            return "HOME", confidence, "XG_ADVANTAGE"
        elif xg_diff < -0.5:
            confidence = 55
            return "AWAY", confidence, "XG_ADVANTAGE"
        else:
            confidence = 50
            return "DRAW", confidence, "BALANCED"
    
    def predict_totals(self, total_xg, home_finish, away_finish):
        """Eredivisie: High scoring league"""
        calibrated_xg = total_xg * self.xg_calibration
        
        if calibrated_xg > 2.9:  # Higher threshold for high scoring
            confidence = 60
            return "OVER", confidence, "CALIBRATED_XG"
        else:
            confidence = 60
            return "UNDER", confidence, "CALIBRATED_XG"
            
    def get_expected_improvement(self):
        return {
            "winner": {"current": 50.0, "expected": 55.0, "improvement": "+5.0%"},
            "totals": {"current": 50.0, "expected": 60.0, "improvement": "+10.0%"}
        }

class RFPLEngine:
    """Similar to Bundesliga (estimated)"""
    
    def __init__(self):
        self.name = "RFPL"
        self.winner_accuracy = 0.500  # Estimated
        self.totals_accuracy = 0.500  # Estimated
        self.model_type = "ESTIMATED_MODEL"
        self.version = "v2.0_rfpl"
        self.xg_calibration = 1.0  # No calibration data
        
    def predict_winner(self, home_xg, away_xg, home_finish, away_finish):
        """Use Bundesliga logic for now"""
        finish_diff = home_finish - away_finish
        xg_diff = home_xg - away_xg
        
        if xg_diff > 0.5 and finish_diff > 0.1:
            confidence = 65
            return "HOME", confidence, "COMBINED_ADVANTAGE"
        elif xg_diff < -0.5 and finish_diff < -0.1:
            confidence = 65
            return "AWAY", confidence, "COMBINED_ADVANTAGE"
        elif xg_diff > 0.5:
            confidence = 55
            return "HOME", confidence, "XG_ADVANTAGE"
        elif xg_diff < -0.5:
            confidence = 55
            return "AWAY", confidence, "XG_ADVANTAGE"
        else:
            confidence = 50
            return "DRAW", confidence, "BALANCED"
    
    def predict_totals(self, total_xg, home_finish, away_finish):
        """Standard approach"""
        calibrated_xg = total_xg * self.xg_calibration
        
        if calibrated_xg > 2.5:
            confidence = 60
            return "OVER", confidence, "STANDARD"
        else:
            confidence = 60
            return "UNDER", confidence, "STANDARD"
            
    def get_expected_improvement(self):
        return {
            "winner": {"current": 50.0, "expected": 55.0, "improvement": "+5.0%"},
            "totals": {"current": 50.0, "expected": 60.0, "improvement": "+10.0%"}
        }

# ========== LEAGUE ENGINE FACTORY ==========

class LeagueEngineFactory:
    """Factory to create the right engine for each league"""
    
    ENGINES = {
        "Premier League": PremierLeagueEngine,
        "Serie A": SerieAEngine,
        "Ligue 1": Ligue1Engine,
        "Bundesliga": BundesligaEngine,
        "La Liga": LaLigaEngine,
        "Eredivisie": EredivisieEngine,
        "RFPL": RFPLEngine,
    }
    
    @staticmethod
    def create_engine(league_name):
        """Create the appropriate engine for the league"""
        engine_class = LeagueEngineFactory.ENGINES.get(league_name)
        if engine_class:
            return engine_class()
        else:
            # Default to Bundesliga engine
            return BundesligaEngine()
    
    @staticmethod
    def get_league_stats():
        """Get accuracy stats from 35-match analysis"""
        return {
            "Premier League": {"winner": 22.2, "totals": 44.4},
            "Serie A": {"winner": 50.0, "totals": 75.0},
            "Ligue 1": {"winner": 75.0, "totals": 37.5},
            "Bundesliga": {"winner": 50.0, "totals": 50.0},
            "La Liga": {"winner": 62.5, "totals": 50.0},
            "Eredivisie": {"winner": 50.0, "totals": 50.0},
            "RFPL": {"winner": 50.0, "totals": 50.0},
        }

# ========== DATA COLLECTION FUNCTIONS ==========

def save_match_prediction(prediction_data, actual_score, league_name, engine):
    """Save match prediction data to Supabase"""
    try:
        # Parse actual score
        home_goals, away_goals = map(int, actual_score.split('-'))
        total_goals = home_goals + away_goals
        
        # Calculate actual results
        actual_winner = 'HOME' if home_goals > away_goals else 'AWAY' if away_goals > home_goals else 'DRAW'
        actual_over_under = 'OVER' if total_goals > 2.5 else 'UNDER'
        
        # Prepare match data
        match_data = {
            "match_date": datetime.now().date().isoformat(),
            "league": league_name,
            "home_team": prediction_data.get('home_team', 'Unknown'),
            "away_team": prediction_data.get('away_team', 'Unknown'),
            "home_xg": float(prediction_data.get('home_xg', 0)),
            "away_xg": float(prediction_data.get('away_xg', 0)),
            "home_finishing_vs_xg": float(prediction_data.get('home_finish', 0)),
            "away_finishing_vs_xg": float(prediction_data.get('away_finish', 0)),
            "home_defense_vs_xga": float(prediction_data.get('home_defense', 0)),
            "away_defense_vs_xga": float(prediction_data.get('away_defense', 0)),
            "home_adjusted_xg": float(prediction_data.get('home_adjusted_xg', 0)),
            "away_adjusted_xg": float(prediction_data.get('away_adjusted_xg', 0)),
            "delta_xg": float(prediction_data.get('delta_xg', 0)),
            "total_xg": float(prediction_data.get('total_xg', 0)),
            "finishing_sum": float(prediction_data.get('finishing_sum', 0)),
            "finishing_impact": float(prediction_data.get('finishing_impact', 0)),
            "adjusted_total_xg": float(prediction_data.get('adjusted_total_xg', 0)),
            "predicted_winner": prediction_data.get('predicted_winner', 'UNKNOWN'),
            "winner_confidence": float(prediction_data.get('winner_confidence', 50)),
            "predicted_totals_direction": prediction_data.get('predicted_totals', 'UNKNOWN'),
            "totals_confidence": float(prediction_data.get('totals_confidence', 50)),
            "finishing_alignment": prediction_data.get('finishing_alignment', 'NEUTRAL'),
            "total_xg_category": prediction_data.get('total_xg_category', 'UNKNOWN'),
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_total_goals": total_goals,
            "actual_winner": actual_winner,
            "actual_over_under": actual_over_under,
            "model_version": engine.version,
            "notes": f"League-specific engine: {engine.model_type}"
        }
        
        # Save to Supabase
        if supabase:
            response = supabase.table("match_predictions").insert(match_data).execute()
            if hasattr(response, 'data') and response.data:
                return True, "✅ Match data saved to Supabase"
            else:
                return False, "❌ Failed to save to Supabase"
        else:
            # Fallback: save locally
            with open("match_predictions_backup.json", "a") as f:
                f.write(json.dumps(match_data) + "\n")
            return True, "⚠️ Saved locally (no Supabase connection)"
        
    except Exception as e:
        return False, f"❌ Error saving match data: {str(e)}"

def get_match_stats():
    """Get statistics about collected matches"""
    try:
        if supabase:
            # Count total matches
            response = supabase.table("match_predictions").select("id", count="exact").execute()
            total_matches = response.count or 0
            
            # Get league distribution
            league_response = supabase.table("match_predictions").select("league").execute()
            leagues = [row['league'] for row in league_response.data] if league_response.data else []
            
            return {
                'total_matches': total_matches,
                'leagues': leagues,
                'league_counts': pd.Series(leagues).value_counts().to_dict() if leagues else {}
            }
        else:
            # Check local backup
            if os.path.exists("match_predictions_backup.json"):
                with open("match_predictions_backup.json", "r") as f:
                    lines = f.readlines()
                return {
                    'total_matches': len(lines),
                    'leagues': [],
                    'league_counts': {}
                }
            return {'total_matches': 0, 'leagues': [], 'league_counts': {}}
    except:
        return {'total_matches': 0, 'leagues': [], 'league_counts': {}}

# ========== EXPECTED GOALS CALCULATOR ==========

class ExpectedGoalsCalculator:
    """Calculate expected goals (keep your existing logic)"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_avg_goals = league_metrics.get('avg_goals_per_match', 2.5)
        self.league_name = league_name
    
    def predict_expected_goals(self, home_stats, away_stats):
        """Calculate expected goals - keep your existing logic"""
        home_adjGF = home_stats['goals_for_pm'] + 0.6 * home_stats['goals_vs_xg_pm']
        home_adjGA = home_stats['goals_against_pm'] + 0.6 * home_stats['goals_allowed_vs_xga_pm']
        
        away_adjGF = away_stats['goals_for_pm'] + 0.6 * away_stats['goals_vs_xg_pm']
        away_adjGA = away_stats['goals_against_pm'] + 0.6 * away_stats['goals_allowed_vs_xga_pm']
        
        venue_factor_home = 1 + 0.05 * (home_stats['points_pm'] - away_stats['points_pm']) / 3
        venue_factor_away = 1 + 0.05 * (away_stats['points_pm'] - home_stats['points_pm']) / 3
        
        venue_factor_home = max(0.8, min(1.2, venue_factor_home))
        venue_factor_away = max(0.8, min(1.2, venue_factor_away))
        
        home_xg = (home_adjGF + away_adjGA) / 2 * venue_factor_home
        away_xg = (away_adjGF + home_adjGA) / 2 * venue_factor_away
        
        normalization_factor = self.league_avg_goals / 2.5
        home_xg *= normalization_factor
        away_xg *= normalization_factor
        
        home_xg = max(0.2, min(5.0, home_xg))
        away_xg = max(0.2, min(5.0, away_xg))
        
        return home_xg, away_xg

def factorial_cache(n, cache={}):
    if n not in cache:
        cache[n] = math.factorial(n)
    return cache[n]

def poisson_pmf(k, lam):
    """Poisson probability mass function"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

class PoissonProbabilityEngine:
    """Probability engine for score probabilities"""
    
    @staticmethod
    def calculate_all_probabilities(home_xg, away_xg, max_goals=8):
        score_probabilities = []
        max_goals = min(max_goals, int(home_xg + away_xg) + 4)
        
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

# ========== DATA LOADING FUNCTIONS ==========

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
    """Prepare home and away data"""
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

# ========== STREAMLIT UI ==========

# Sidebar
with st.sidebar:
    st.header("⚙️ League-Specific Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie", "RFPL"]
    selected_league = st.selectbox("Select League", leagues)
    
    # Create league engine
    engine = LeagueEngineFactory.create_engine(selected_league)
    league_stats = LeagueEngineFactory.get_league_stats()
    
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
            
            if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
                calculate_btn = True
            else:
                calculate_btn = False
        else:
            st.error("Could not prepare team data")
            st.stop()
    
    # League Engine Info
    st.divider()
    st.header(f"🔧 {engine.name} Engine")
    st.write(f"**Type:** {engine.model_type}")
    st.write(f"**Version:** {engine.version}")
    
    # Current accuracy from analysis
    if selected_league in league_stats:
        stats = league_stats[selected_league]
        st.write(f"**Current Accuracy (from 35-match analysis):**")
        st.write(f"- Winner: {stats['winner']}%")
        st.write(f"- Totals: {stats['totals']}%")
    
    # Expected improvements
    improvements = engine.get_expected_improvement()
    st.write("**Expected Improvements:**")
    st.write(f"- Winner: {improvements['winner']['current']}% → {improvements['winner']['expected']}% ({improvements['winner']['improvement']})")
    st.write(f"- Totals: {improvements['totals']['current']}% → {improvements['totals']['expected']}% ({improvements['totals']['improvement']})")
    
    # Data Collection Stats
    st.divider()
    st.header("📊 Data Collection Stats")
    
    stats = get_match_stats()
    total_matches = stats['total_matches']
    
    st.metric("Total Matches Collected", total_matches)
    
    if total_matches > 0:
        st.progress(min(total_matches / 100, 1.0))
        st.caption(f"Target: 100 matches ({total_matches}/100)")
        
        if stats['league_counts']:
            st.write("**By League:**")
            for league, count in stats['league_counts'].items():
                st.write(f"- {league}: {count}")

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# Check if we should show prediction
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Calculate expected goals
        xg_calculator = ExpectedGoalsCalculator(league_metrics, selected_league)
        home_xg, away_xg = xg_calculator.predict_expected_goals(home_stats, away_stats)
        
        # Get finishing and defense stats
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        home_defense = home_stats['goals_allowed_vs_xga_pm']
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        # Calculate adjusted xG
        home_adjusted_xg = home_xg + home_finish - away_defense
        away_adjusted_xg = away_xg + away_finish - home_defense
        delta_xg = home_adjusted_xg - away_adjusted_xg
        
        # Get predictions from league-specific engine
        predicted_winner, winner_confidence, winner_logic = engine.predict_winner(
            home_xg, away_xg, home_finish, away_finish
        )
        
        predicted_totals, totals_confidence, totals_logic = engine.predict_totals(
            home_xg + away_xg, home_finish, away_finish
        )
        
        # Calculate probabilities
        prob_engine = PoissonProbabilityEngine()
        probabilities = prob_engine.calculate_all_probabilities(home_xg, away_xg)
        
        # Store prediction data
        prediction_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'total_xg': home_xg + away_xg,
            'home_finish': home_finish,
            'away_finish': away_finish,
            'home_defense': home_defense,
            'away_defense': away_defense,
            'home_adjusted_xg': home_adjusted_xg,
            'away_adjusted_xg': away_adjusted_xg,
            'delta_xg': delta_xg,
            'finishing_sum': home_finish + away_finish,
            'finishing_impact': (home_finish + away_finish) * 0.6,
            'adjusted_total_xg': (home_xg + away_xg) * (1 + (home_finish + away_finish) * 0.6),
            'predicted_winner': predicted_winner,
            'winner_confidence': winner_confidence,
            'predicted_totals': predicted_totals,
            'totals_confidence': totals_confidence,
            'finishing_alignment': 'NEUTRAL',
            'total_xg_category': 'MODERATE',
            'probabilities': probabilities,
            'engine_info': {
                'name': engine.name,
                'model_type': engine.model_type,
                'version': engine.version,
                'winner_logic': winner_logic,
                'totals_logic': totals_logic
            }
        }
        
        # Store for display
        st.session_state.prediction_data = prediction_data
        st.session_state.selected_teams = (home_team, away_team)
        st.session_state.engine = engine
        
    except KeyError as e:
        st.error(f"Team data error: {e}")
        st.stop()
elif 'prediction_data' in st.session_state:
    # Use stored prediction
    prediction_data = st.session_state.prediction_data
    home_team, away_team = st.session_state.selected_teams
    engine = st.session_state.engine
else:
    st.info("👈 Select teams and click 'Generate Prediction'")
    st.stop()

# ========== DISPLAY PREDICTION ==========
st.header(f"🎯 {home_team} vs {away_team}")
st.caption(f"League: {selected_league} | Engine: {engine.model_type} | Version: {engine.version}")

# Engine info
with st.expander("🔧 Engine Details"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Engine Configuration:**")
        st.write(f"- Name: {engine.name}")
        st.write(f"- Type: {engine.model_type}")
        st.write(f"- Version: {engine.version}")
        st.write(f"- Winner Logic: {prediction_data['engine_info']['winner_logic']}")
        st.write(f"- Totals Logic: {prediction_data['engine_info']['totals_logic']}")
    
    with col2:
        st.write("**Expected Performance:**")
        improvements = engine.get_expected_improvement()
        st.write(f"- Winner: {improvements['winner']['current']}% → {improvements['winner']['expected']}%")
        st.write(f"- Totals: {improvements['totals']['current']}% → {improvements['totals']['expected']}%")
        st.write(f"- Expected Overall: 65-70% accuracy")

# Prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction_data['predicted_winner']
    winner_conf = prediction_data['winner_confidence']
    winner_logic = prediction_data['engine_info']['winner_logic']
    
    prob = prediction_data['probabilities']
    winner_prob = prob['home_win_probability'] if winner_pred == 'HOME' else \
                  prob['away_win_probability'] if winner_pred == 'AWAY' else \
                  prob['draw_probability']
    
    # Confidence category
    if winner_conf >= 75:
        conf_category = "HIGH"
    elif winner_conf >= 60:
        conf_category = "MEDIUM"
    else:
        conf_category = "LOW"
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">WINNER ({engine.name})</h3>
        <div style="font-size: 36px; font-weight: bold; color: #60A5FA; margin: 10px 0;">
            {'🏠' if winner_pred == 'HOME' else '✈️' if winner_pred == 'AWAY' else '🤝'} {winner_pred}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_prob*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {conf_category} | Confidence: {winner_conf:.0f}/100
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            Logic: {winner_logic}
        </div>
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            ΔxG: {prediction_data['delta_xg']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    totals_pred = prediction_data['predicted_totals']
    totals_conf = prediction_data['totals_confidence']
    totals_logic = prediction_data['engine_info']['totals_logic']
    
    totals_prob = prob['over_2_5_probability'] if totals_pred == 'OVER' else \
                  prob['under_2_5_probability']
    
    # Confidence category
    if totals_conf >= 75:
        conf_category = "HIGH"
    elif totals_conf >= 60:
        conf_category = "MEDIUM"
    else:
        conf_category = "LOW"
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">TOTAL GOALS ({engine.name})</h3>
        <div style="font-size: 36px; font-weight: bold; color: #60A5FA; margin: 10px 0;">
            {'📈' if totals_pred == 'OVER' else '📉'} {totals_pred} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {totals_prob*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {conf_category} | Confidence: {totals_conf:.0f}/100
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            Logic: {totals_logic}
        </div>
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            xG: {prediction_data['total_xg']:.2f} → Adj: {prediction_data['adjusted_total_xg']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== DATA COLLECTION SECTION ==========
st.divider()
st.subheader("📝 COLLECT MATCH DATA")

col1, col2 = st.columns([2, 1])

with col1:
    score = st.text_input("Actual Final Score (e.g., 2-1)", key="score_input")
    
    with st.expander("📊 View Calculations"):
        st.write("**Expected Goals:**")
        st.write(f"- Home xG: {prediction_data['home_xg']:.2f}")
        st.write(f"- Away xG: {prediction_data['away_xg']:.2f}")
        st.write(f"- Total xG: {prediction_data['total_xg']:.2f}")
        st.write(f"- ΔxG: {prediction_data['delta_xg']:.2f}")
        
        st.write("**Finishing Stats:**")
        st.write(f"- Home finishing: {prediction_data['home_finish']:.3f}")
        st.write(f"- Away finishing: {prediction_data['away_finish']:.3f}")
        st.write(f"- Finishing impact: {prediction_data['finishing_impact']:.3f}")

with col2:
    if st.button("💾 Save Match Data", type="primary", use_container_width=True):
        if not score or '-' not in score:
            st.error("Enter valid score like '2-1'")
        else:
            try:
                with st.spinner("Saving match data..."):
                    success, message = save_match_prediction(
                        prediction_data, score, selected_league, engine
                    )
                    
                    if success:
                        st.success(f"""
                        {message}
                        
                        **Saved with:** {engine.model_type}
                        **Engine version:** {engine.version}
                        **Total matches:** {get_match_stats()['total_matches'] + 1}
                        """)
                        
                        st.balloons()
                        
                        # Reset for next match
                        if 'prediction_data' in st.session_state:
                            del st.session_state.prediction_data
                        if 'selected_teams' in st.session_state:
                            del st.session_state.selected_teams
                        if 'engine' in st.session_state:
                            del st.session_state.engine
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                        
            except ValueError:
                st.error("Enter numbers like '2-1'")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ========== ENGINE CALCULATIONS ==========
st.divider()
st.subheader("🔧 ENGINE CALCULATIONS")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**xG Calculations:**")
    st.write(f"- Home xG: {prediction_data['home_xg']:.2f}")
    st.write(f"- Away xG: {prediction_data['away_xg']:.2f}")
    st.write(f"- Total xG: {prediction_data['total_xg']:.2f}")
    st.write(f"- ΔxG: {prediction_data['delta_xg']:.2f}")

with col2:
    st.write("**Finishing Adjustments:**")
    st.write(f"- Home finishing: {prediction_data['home_finish']:.3f}")
    st.write(f"- Away finishing: {prediction_data['away_finish']:.3f}")
    st.write(f"- Sum: {prediction_data['finishing_sum']:.3f}")
    st.write(f"- Impact: {prediction_data['finishing_impact']:.3f}")

with col3:
    st.write("**Adjusted Values:**")
    st.write(f"- Home adjusted: {prediction_data['home_adjusted_xg']:.2f}")
    st.write(f"- Away adjusted: {prediction_data['away_adjusted_xg']:.2f}")
    st.write(f"- Total adjusted: {prediction_data['adjusted_total_xg']:.2f}")

# ========== PROBABILITIES ==========
st.divider()
st.subheader("🎲 PROBABILITIES")

prob = prediction_data['probabilities']

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Home Win", f"{prob['home_win_probability']*100:.1f}%")
    st.metric("Draw", f"{prob['draw_probability']*100:.1f}%")
    st.metric("Away Win", f"{prob['away_win_probability']*100:.1f}%")

with col2:
    st.metric("Over 2.5", f"{prob['over_2_5_probability']*100:.1f}%")
    st.metric("Under 2.5", f"{prob['under_2_5_probability']*100:.1f}%")
    st.metric("BTTS", f"{prob['btts_probability']*100:.1f}%")

with col3:
    st.write("**Most Likely Scores:**")
    for score, prob_val in prob['top_scores'][:3]:
        st.write(f"{score}: {prob_val*100:.1f}%")

# ========== LEAGUE PERFORMANCE SUMMARY ==========
st.divider()
st.subheader("📈 LEAGUE PERFORMANCE SUMMARY")

league_stats = LeagueEngineFactory.get_league_stats()

if selected_league in league_stats:
    stats = league_stats[selected_league]
    improvements = engine.get_expected_improvement()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{selected_league} Performance:**")
        st.write(f"- Current Winner Accuracy: {stats['winner']}%")
        st.write(f"- Current Totals Accuracy: {stats['totals']}%")
        st.write(f"- Matches Analyzed: 9" if selected_league == "Premier League" else 
                f"- Matches Analyzed: 8" if selected_league in ["Serie A", "Ligue 1"] else
                f"- Matches Analyzed: 5")
    
    with col2:
        st.write("**Expected with New Engine:**")
        st.write(f"- Winner: {improvements['winner']['expected']}% ({improvements['winner']['improvement']})")
        st.write(f"- Totals: {improvements['totals']['expected']}% ({improvements['totals']['improvement']})")
        st.write(f"- Overall Target: 65-70% accuracy")

# ========== FOOTER ==========
st.divider()
stats = get_match_stats()
st.caption(f"📊 League-Specific Engines | Total Matches: {stats['total_matches']} | Expected Accuracy: 65-70%")
