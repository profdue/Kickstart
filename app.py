import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
import json
import pickle
import hashlib
import os
from supabase import create_client, Client
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine - DATA COLLECTION MODE",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Football Intelligence Engine - DATA COLLECTION MODE")
st.markdown("""
    **FRESH START** - Collecting complete match data for analysis
    *40+ matches already recorded - building clean dataset*
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
        st.error(f"Error initializing Supabase: {e}")
        return None

# Initialize
supabase = init_supabase()

# ========== CONSTANTS ==========
MAX_GOALS_CALC = 8

# League-specific adjustments
LEAGUE_ADJUSTMENTS = {
    "Premier League": {"over_threshold": 2.5, "under_threshold": 2.5, "avg_goals": 2.79, "very_high_threshold": 3.3},
    "Bundesliga": {"over_threshold": 3.0, "under_threshold": 2.2, "avg_goals": 3.20, "very_high_threshold": 3.5},
    "Serie A": {"over_threshold": 2.7, "under_threshold": 2.3, "avg_goals": 2.40, "very_high_threshold": 3.0},
    "La Liga": {"over_threshold": 2.6, "under_threshold": 2.4, "avg_goals": 2.61, "very_high_threshold": 3.2},
    "Ligue 1": {"over_threshold": 2.8, "under_threshold": 2.2, "avg_goals": 2.85, "very_high_threshold": 3.3},
    "Eredivisie": {"over_threshold": 2.9, "under_threshold": 2.1, "avg_goals": 3.10, "very_high_threshold": 3.6},
    "RFPL": {"over_threshold": 2.5, "under_threshold": 2.2, "avg_goals": 2.53, "very_high_threshold": 3.1}
}

# ========== DATA COLLECTION FUNCTIONS ==========

def save_match_prediction(prediction, actual_score, league_name):
    """Save COMPLETE match prediction data to Supabase"""
    try:
        # Parse actual score
        home_goals, away_goals = map(int, actual_score.split('-'))
        total_goals = home_goals + away_goals
        
        # Get engine calculations
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Calculate all required values
        home_xg = prediction['expected_goals']['home']
        away_xg = prediction['expected_goals']['away']
        delta_xg = home_xg - away_xg
        
        home_finish = totals_pred.get('home_finishing', 0)
        away_finish = totals_pred.get('away_finishing', 0)
        finishing_sum = home_finish + away_finish
        finishing_impact = totals_pred.get('finishing_impact', finishing_sum * 0.6)
        
        # Get defense stats (from team data - need to pass these in)
        home_defense = prediction.get('home_defense', 0)
        away_defense = prediction.get('away_defense', 0)
        
        # Calculate adjusted xG values
        home_adjusted_xg = home_xg + home_finish - away_defense
        away_adjusted_xg = away_xg + away_finish - home_defense
        
        # Prepare complete data record
        match_data = {
            # Match info
            'league': league_name,
            'home_team': prediction['home_team'],
            'away_team': prediction['away_team'],
            'match_date': datetime.now().date().isoformat(),
            
            # Raw inputs
            'home_xg': float(home_xg),
            'away_xg': float(away_xg),
            'home_finishing_vs_xg': float(home_finish),
            'away_finishing_vs_xg': float(away_finish),
            'home_defense_vs_xga': float(home_defense),
            'away_defense_vs_xga': float(away_defense),
            
            # Engine calculations
            'home_adjusted_xg': float(home_adjusted_xg),
            'away_adjusted_xg': float(away_adjusted_xg),
            'delta_xg': float(delta_xg),
            'total_xg': float(totals_pred['total_xg']),
            'finishing_sum': float(finishing_sum),
            'finishing_impact': float(finishing_impact),
            'adjusted_total_xg': float(totals_pred.get('adjusted_xg', totals_pred['total_xg'])),
            
            # Predictions
            'predicted_winner': winner_pred['original_prediction'],
            'winner_confidence': float(winner_pred['original_confidence_score']),
            'predicted_totals_direction': totals_pred['original_direction'],
            'totals_confidence': float(totals_pred['original_confidence_score']),
            
            # Categories (for reference)
            'finishing_alignment': totals_pred.get('original_finishing_alignment', 'UNKNOWN'),
            'total_xg_category': totals_pred.get('original_total_category', 'UNKNOWN'),
            
            # Actual results
            'actual_home_goals': home_goals,
            'actual_away_goals': away_goals,
            'actual_total_goals': total_goals,
            'actual_winner': 'HOME' if home_goals > away_goals else 'AWAY' if away_goals > home_goals else 'DRAW',
            'actual_over_under': 'OVER' if total_goals > 2.5 else 'UNDER',
            
            # Model info
            'model_version': prediction.get('version', 'data_collection_v1'),
            'notes': f"Collection match"
        }
        
        # Save to Supabase
        if supabase:
            response = supabase.table("match_predictions").insert(match_data).execute()
            return True, "✅ Complete match data saved to database"
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

# ========== CORE PREDICTION ENGINE (ORIGINAL) ==========

class ExpectedGoalsPredictor:
    """Expected goals calculation"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_avg_goals = league_metrics.get('avg_goals_per_match', 2.5)
        self.league_name = league_name
    
    def predict_expected_goals(self, home_stats, away_stats):
        """Calculate expected goals"""
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

class WinnerPredictor:
    """Winner determination"""
    
    def predict_winner(self, home_xg, away_xg, home_stats, away_stats):
        """Predict winner"""
        home_finishing = home_stats['goals_vs_xg_pm']
        away_finishing = away_stats['goals_vs_xg_pm']
        home_defense = home_stats['goals_allowed_vs_xga_pm']
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        home_adjusted_xg = home_xg + home_finishing - away_defense
        away_adjusted_xg = away_xg + away_finishing - home_defense
        
        delta = home_adjusted_xg - away_adjusted_xg
        
        # Winner determination
        if delta > 1.2:
            predicted_winner = "HOME"
            strength = "STRONG"
        elif delta > 0.5:
            predicted_winner = "HOME"
            strength = "MODERATE"
        elif delta > 0.2:
            predicted_winner = "HOME"
            strength = "SLIGHT"
        elif delta < -1.2:
            predicted_winner = "AWAY"
            strength = "STRONG"
        elif delta < -0.5:
            predicted_winner = "AWAY"
            strength = "MODERATE"
        elif delta < -0.2:
            predicted_winner = "AWAY"
            strength = "SLIGHT"
        else:
            predicted_winner = "DRAW"
            strength = "CLOSE"
        
        # Confidence calculation
        base_confidence = min(100, abs(delta) / max(home_adjusted_xg, away_adjusted_xg, 0.5) * 150)
        win_rate_diff = home_stats['win_rate'] - away_stats['win_rate']
        form_bonus = min(20, max(0, win_rate_diff * 40))
        
        winner_confidence = min(100, max(30, base_confidence + form_bonus))
        
        # Confidence categorization
        if winner_confidence >= 90:
            confidence_category = "VERY HIGH"
        elif winner_confidence >= 75:
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
            'type': predicted_winner,
            'original_prediction': predicted_winner,
            'strength': strength,
            'confidence_score': winner_confidence,
            'confidence': confidence_category,
            'original_confidence': f"{winner_confidence:.1f}",
            'confidence_category': confidence_category,
            'delta': delta,
            'home_adjusted_xg': home_adjusted_xg,
            'away_adjusted_xg': away_adjusted_xg
        }

class TotalsPredictor:
    """Totals prediction with FINISHING FIX"""
    
    def __init__(self, league_name):
        self.league_name = league_name
        self.league_adjustments = LEAGUE_ADJUSTMENTS.get(league_name, LEAGUE_ADJUSTMENTS["Premier League"])
    
    def categorize_finishing(self, value):
        """Finishing categorization"""
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
        """Finishing alignment"""
        home_cat = self.categorize_finishing(home_finish)
        away_cat = self.categorize_finishing(away_finish)
        
        alignment_matrix = {
            "STRONG_OVERPERFORM": {
                "STRONG_OVERPERFORM": "HIGH_OVER",
                "MODERATE_OVERPERFORM": "MED_OVER",
                "NEUTRAL": "MED_OVER",
                "MODERATE_UNDERPERFORM": "RISKY",
                "STRONG_UNDERPERFORM": "HIGH_RISK"
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
        """Total xG categories"""
        very_high_thresh = self.league_adjustments.get('very_high_threshold', 3.3)
        
        if total_xg > very_high_thresh:
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
    
    def predict_totals(self, home_xg, away_xg, home_stats, away_stats):
        """Predict totals with FINISHING FIX"""
        total_xg = home_xg + away_xg
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        # THE FIX: Apply finishing impact
        finishing_impact = (home_finish + away_finish) * 0.6
        adjusted_xg = total_xg * (1 + finishing_impact)
        
        over_threshold = self.league_adjustments['over_threshold']
        
        # Use ADJUSTED xG for decision
        base_direction = "OVER" if adjusted_xg > over_threshold else "UNDER"
        
        # Finishing alignment
        finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
        total_category = self.categorize_total_xg(total_xg)
        
        # Base confidence
        base_confidence = 60
        
        # Risk assessment
        risk_flags = []
        if abs(home_finish) > 0.4 or abs(away_finish) > 0.4:
            risk_flags.append("HIGH_VARIANCE_TEAM")
        
        # CLOSE TO THRESHOLD with adjusted xG
        lower_thresh = self.league_adjustments['under_threshold'] - 0.1
        upper_thresh = self.league_adjustments['over_threshold'] + 0.1
        if lower_thresh < adjusted_xg < upper_thresh:
            risk_flags.append("CLOSE_TO_THRESHOLD")
            base_confidence -= 10
        
        base_confidence = max(5, min(95, base_confidence))
        
        # Confidence category
        if base_confidence >= 75:
            confidence_category = "VERY HIGH"
        elif base_confidence >= 65:
            confidence_category = "HIGH"
        elif base_confidence >= 55:
            confidence_category = "MEDIUM"
        elif base_confidence >= 45:
            confidence_category = "LOW"
        else:
            confidence_category = "VERY LOW"
        
        return {
            'direction': base_direction,
            'original_direction': base_direction,
            'total_xg': total_xg,
            'adjusted_xg': adjusted_xg,
            'finishing_impact': finishing_impact,
            'confidence': confidence_category,
            'confidence_score': base_confidence,
            'finishing_alignment': finishing_alignment,
            'original_finishing_alignment': finishing_alignment,
            'total_category': total_category,
            'original_total_category': total_category,
            'risk_flags': risk_flags,
            'home_finishing': home_finish,
            'away_finishing': away_finish,
            'league_threshold': over_threshold
        }

class PoissonProbabilityEngine:
    """Probability engine"""
    
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

def factorial_cache(n, cache={}):
    if n not in cache:
        cache[n] = math.factorial(n)
    return cache[n]

def poisson_pmf(k, lam):
    """Poisson probability mass function"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

class FootballEngine:
    """Main football engine"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_name = league_name
        
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate prediction"""
        
        # Get predictions
        home_xg, away_xg = self.xg_predictor.predict_expected_goals(home_stats, away_stats)
        
        probabilities = self.probability_engine.calculate_all_probabilities(home_xg, away_xg)
        
        winner_prediction = self.winner_predictor.predict_winner(
            home_xg, away_xg, home_stats, away_stats
        )
        
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Calculate additional engine values for data collection
        delta_xg = home_xg - away_xg
        finishing_sum = totals_prediction['home_finishing'] + totals_prediction['away_finishing']
        finishing_impact = totals_prediction['finishing_impact']
        
        home_defense = home_stats['goals_allowed_vs_xga_pm']
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        home_adjusted_xg = home_xg + totals_prediction['home_finishing'] - away_defense
        away_adjusted_xg = away_xg + totals_prediction['away_finishing'] - home_defense
        
        # Create prediction dictionary
        prediction_dict = {
            'home_team': home_team,
            'away_team': away_team,
            'winner': winner_prediction,
            'totals': totals_prediction,
            'probabilities': probabilities,
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': home_xg + away_xg},
            'engine_calculations': {
                'delta_xg': delta_xg,
                'home_adjusted_xg': home_adjusted_xg,
                'away_adjusted_xg': away_adjusted_xg,
                'finishing_sum': finishing_sum,
                'finishing_impact': finishing_impact,
                'home_defense': home_defense,
                'away_defense': away_defense
            },
            'version': 'data_collection_v1',
            'league': self.league_name
        }
        
        return prediction_dict

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

# ========== SESSION STATES ==========
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'last_teams' not in st.session_state:
    st.session_state.last_teams = None

# ========== STREAMLIT UI ==========

# Sidebar
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
            
            if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
                calculate_btn = True
            else:
                calculate_btn = False
        else:
            st.error("Could not prepare team data")
            st.stop()
    
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
    
    # Quick Analysis
    if total_matches >= 10:
        st.divider()
        st.header("🔍 Quick Insights")
        
        if st.button("Run Quick Analysis", use_container_width=True):
            # Simple analysis query
            try:
                if supabase:
                    response = supabase.table("match_predictions").select(
                        "predicted_winner", "actual_winner", "winner_confidence"
                    ).execute()
                    
                    if response.data:
                        df_analysis = pd.DataFrame(response.data)
                        df_analysis['correct'] = df_analysis['predicted_winner'] == df_analysis['actual_winner']
                        
                        # Group by confidence
                        if len(df_analysis) > 0:
                            df_analysis['conf_bucket'] = pd.cut(
                                df_analysis['winner_confidence'], 
                                bins=[0, 40, 60, 80, 100],
                                labels=['0-40%', '40-60%', '60-80%', '80-100%']
                            )
                            
                            accuracy = df_analysis.groupby('conf_bucket')['correct'].mean()
                            st.write("**Accuracy by Confidence:**")
                            for bucket, acc in accuracy.items():
                                st.write(f"{bucket}: {acc:.1%}")
            except:
                pass

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# Check if we should show prediction
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Generate prediction
        engine = FootballEngine(league_metrics, selected_league)
        prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)
        
        # Store for next time
        st.session_state.last_prediction = prediction
        st.session_state.last_teams = (home_team, away_team)
        
    except KeyError as e:
        st.error(f"Team data error: {e}")
        st.stop()
elif st.session_state.last_prediction and st.session_state.last_teams:
    # Use stored prediction
    prediction = st.session_state.last_prediction
    home_team, away_team = st.session_state.last_teams
else:
    st.info("👈 Select teams and click 'Generate Prediction'")
    st.stop()

# ========== DISPLAY PREDICTION ==========
st.header(f"🎯 {home_team} vs {away_team}")
st.caption(f"League: {selected_league} | Data Collection Mode | Matches: {get_match_stats()['total_matches']}")

# Prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction['winner']
    prob = prediction['probabilities']
    
    winner_prob = prob['home_win_probability'] if winner_pred['type'] == 'HOME' else \
                  prob['away_win_probability'] if winner_pred['type'] == 'AWAY' else \
                  prob['draw_probability']
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: #60A5FA; margin: 10px 0;">
            {'🏠' if winner_pred['type'] == 'HOME' else '✈️' if winner_pred['type'] == 'AWAY' else '🤝'} {winner_pred['type']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_prob*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {winner_pred['confidence']} | Confidence: {winner_pred['confidence_score']:.0f}/100
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            ΔxG: {winner_pred['delta']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    totals_pred = prediction['totals']
    prob = prediction['probabilities']
    
    totals_prob = prob['over_2_5_probability'] if totals_pred['direction'] == 'OVER' else \
                  prob['under_2_5_probability']
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">TOTAL GOALS</h3>
        <div style="font-size: 36px; font-weight: bold; color: #60A5FA; margin: 10px 0;">
            {'📈' if totals_pred['direction'] == 'OVER' else '📉'} {totals_pred['direction']} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {totals_prob*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {totals_pred['confidence']} | Confidence: {totals_pred['confidence_score']:.0f}/100
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            Raw xG: {totals_pred['total_xg']:.2f} | Adj. xG: {totals_pred['adjusted_xg']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== DATA COLLECTION SECTION ==========
st.divider()
st.subheader("📝 COLLECT MATCH DATA")

col1, col2 = st.columns([2, 1])

with col1:
    score = st.text_input("Actual Final Score (e.g., 2-1)", key="score_input")
    
    with st.expander("📊 View Engine Calculations"):
        st.write("**For Data Collection:**")
        st.write(f"- Home xG: {prediction['expected_goals']['home']:.2f}")
        st.write(f"- Away xG: {prediction['expected_goals']['away']:.2f}")
        st.write(f"- ΔxG: {prediction['engine_calculations']['delta_xg']:.2f}")
        st.write(f"- Home finishing: {prediction['totals']['home_finishing']:.3f}")
        st.write(f"- Away finishing: {prediction['totals']['away_finishing']:.3f}")
        st.write(f"- Finishing impact: {prediction['totals']['finishing_impact']:.3f}")
        st.write(f"- Raw → Adj. xG: {prediction['totals']['total_xg']:.2f} → {prediction['totals']['adjusted_xg']:.2f}")

with col2:
    if st.button("💾 Save Match Data", type="primary", use_container_width=True):
        if not score or '-' not in score:
            st.error("Enter valid score like '2-1'")
        else:
            try:
                with st.spinner("Saving complete match data..."):
                    # Add defense stats to prediction for data collection
                    prediction['home_defense'] = prediction['engine_calculations']['home_defense']
                    prediction['away_defense'] = prediction['engine_calculations']['away_defense']
                    
                    success, message = save_match_prediction(prediction, score, selected_league)
                    
                    if success:
                        st.success(f"""
                        {message}
                        
                        **Saved to database:**
                        - All xG values
                        - Finishing stats
                        - Engine calculations
                        - Predictions & confidence
                        - Actual results
                        
                        **Total matches:** {get_match_stats()['total_matches']}
                        """)
                        
                        # Show success animation
                        st.balloons()
                        
                        # Reset for next match
                        st.session_state.last_prediction = None
                        st.session_state.last_teams = None
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                        
            except ValueError:
                st.error("Enter numbers like '2-1'")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ========== ENGINE CALCULATIONS DISPLAY ==========
st.divider()
st.subheader("🔧 ENGINE CALCULATIONS")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**xG Calculations:**")
    st.write(f"- Home xG: {prediction['expected_goals']['home']:.2f}")
    st.write(f"- Away xG: {prediction['expected_goals']['away']:.2f}")
    st.write(f"- Total xG: {prediction['expected_goals']['total']:.2f}")
    st.write(f"- ΔxG: {prediction['engine_calculations']['delta_xg']:.2f}")

with col2:
    st.write("**Finishing Adjustments:**")
    st.write(f"- Home finishing: {prediction['totals']['home_finishing']:.3f}")
    st.write(f"- Away finishing: {prediction['totals']['away_finishing']:.3f}")
    st.write(f"- Sum: {prediction['engine_calculations']['finishing_sum']:.3f}")
    st.write(f"- Impact (×0.6): {prediction['totals']['finishing_impact']:.3f}")

with col3:
    st.write("**Adjusted Values:**")
    st.write(f"- Raw total: {prediction['totals']['total_xg']:.2f}")
    st.write(f"- Adjusted: {prediction['totals']['adjusted_xg']:.2f}")
    st.write(f"- Threshold: {prediction['totals']['league_threshold']}")
    st.write(f"- Decision: {prediction['totals']['adjusted_xg']:.2f} {'>' if prediction['totals']['adjusted_xg'] > prediction['totals']['league_threshold'] else '<'} {prediction['totals']['league_threshold']}")

# ========== PROBABILITIES ==========
st.divider()
st.subheader("🎲 PROBABILITIES")

prob = prediction['probabilities']

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

# ========== DATA ANALYSIS QUERIES ==========
st.divider()
st.subheader("📈 DATA ANALYSIS (After Collection)")

if get_match_stats()['total_matches'] >= 10:
    st.info(f"**{get_match_stats()['total_matches']} matches collected** - Ready for analysis")
    
    if st.button("Run Initial Analysis", use_container_width=True):
        # Show sample analysis
        st.write("**Sample Analysis Queries:**")
        st.code("""
        -- 1. Accuracy by confidence level
        SELECT 
            FLOOR(winner_confidence/10)*10 as confidence_decile,
            COUNT(*) as matches,
            AVG(CASE WHEN predicted_winner = actual_winner THEN 1.0 ELSE 0.0 END) as accuracy
        FROM match_predictions
        GROUP BY confidence_decile
        ORDER BY confidence_decile;
        
        -- 2. Finishing adjustment effectiveness
        SELECT 
            CASE 
                WHEN finishing_sum > 0.2 THEN 'OVERPERFORMERS'
                WHEN finishing_sum < -0.2 THEN 'UNDERPERFORMERS'
                ELSE 'NEUTRAL'
            END as group,
            COUNT(*) as matches,
            AVG(CASE WHEN predicted_totals_direction = actual_over_under THEN 1.0 ELSE 0.0 END) as accuracy
        FROM match_predictions
        GROUP BY group;
        """)
        
        # Try to run actual queries
        try:
            if supabase:
                # Simple accuracy query
                response = supabase.table("match_predictions").select(
                    "predicted_winner", "actual_winner", "predicted_totals_direction", "actual_over_under"
                ).execute()
                
                if response.data:
                    df_results = pd.DataFrame(response.data)
                    winner_accuracy = (df_results['predicted_winner'] == df_results['actual_winner']).mean()
                    totals_accuracy = (df_results['predicted_totals_direction'] == df_results['actual_over_under']).mean()
                    
                    st.metric("Winner Accuracy", f"{winner_accuracy*100:.1f}%")
                    st.metric("Totals Accuracy", f"{totals_accuracy*100:.1f}%")
        except:
            pass
else:
    st.warning(f"**{get_match_stats()['total_matches']} matches collected** - Need at least 10 for meaningful analysis")

# ========== FOOTER ==========
st.divider()
st.caption(f"📊 Data Collection Mode | Version: {prediction.get('version', '1.0')} | Matches in DB: {get_match_stats()['total_matches']}")
