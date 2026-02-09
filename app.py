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
    page_title="⚽ Football Intelligence Engine v4.0",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine v4.0")
st.markdown("""
    **ADAPTIVE BETTING SYSTEM** - Uses YOUR historical results to make profitable bets
    *Bet what actually wins, not what the algorithm predicts*
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

# ========== BETTING SYSTEM ==========

class AdaptiveBettingSystem:
    """Betting system that makes decisions based on YOUR actual results"""
    
    def __init__(self):
        self.pattern_memory = defaultdict(lambda: {'total': 0, 'success': 0})
        self.feature_weights = {
            'finishing_alignment': 1.0,
            'total_category': 1.0,
            'confidence_score': 1.0,
            'risk_flags': 1.0,
            'defense_quality': 1.0,
            'league': 0.8,
            'volatility': 1.2
        }
        self.outcomes = []
        self.supabase = init_supabase()
        
        # Load from Supabase
        self.load_learning()
    
    def _get_opposite_prediction(self, prediction):
        """Get opposite of prediction"""
        if prediction == "HOME":
            return "AWAY"
        elif prediction == "AWAY":
            return "HOME"
        elif prediction == "DRAW":
            return "HOME"  # Default opposite for draw
        elif prediction == "OVER":
            return "UNDER"
        elif prediction == "UNDER":
            return "OVER"
        elif isinstance(prediction, str) and "HOME" in prediction:
            return prediction.replace("HOME", "AWAY")
        elif isinstance(prediction, str) and "AWAY" in prediction:
            return prediction.replace("AWAY", "HOME")
        return prediction
    
    def get_betting_decision(self, algorithm_prediction, pattern_type, pattern_subtype, original_confidence, algorithm_details=None):
        """
        RETURNS WHAT TO ACTUALLY BET BASED ON REAL RESULTS
        
        pattern_type: "WINNER" or "TOTALS"
        pattern_subtype: e.g., "VERY_HIGH_90" or "MED_UNDER_VERY_HIGH"
        algorithm_prediction: What the algorithm predicts
        original_confidence: Algorithm's confidence score
        """
        
        # Get historical success for this EXACT pattern
        pattern_key = f"{pattern_type}_{pattern_subtype}"
        
        # Check if we have enough data
        if pattern_key in self.pattern_memory:
            stats = self.pattern_memory[pattern_key]
            
            if stats['total'] >= 3:  # Minimum reliable sample
                success_rate = stats['success'] / stats['total']
                
                # RULE 1: Pattern WINS consistently → BET IT
                if success_rate >= 0.7:
                    return {
                        'bet': algorithm_prediction,
                        'confidence': min(95, original_confidence + 20),
                        'reason': f"YOUR PROVEN PATTERN: {success_rate:.0%} SUCCESS ({stats['success']}/{stats['total']})",
                        'type': 'PROVEN_WINNER',
                        'success_rate': success_rate,
                        'sample_size': stats['total']
                    }
                
                # RULE 2: Pattern LOSES consistently → BET OPPOSITE
                elif success_rate <= 0.3:
                    opposite = self._get_opposite_prediction(algorithm_prediction)
                    return {
                        'bet': opposite,
                        'confidence': min(95, 100 - original_confidence + 20),
                        'reason': f"YOUR FAILING PATTERN: {success_rate:.0%} SUCCESS → BET OPPOSITE",
                        'type': 'PROVEN_LOSER',
                        'success_rate': success_rate,
                        'sample_size': stats['total']
                    }
                
                # RULE 3: Pattern is mediocre → Use algorithm but with caution
                else:
                    return {
                        'bet': algorithm_prediction,
                        'confidence': max(40, original_confidence - 10),
                        'reason': f"Mixed results: {success_rate:.0%} success ({stats['success']}/{stats['total']})",
                        'type': 'UNCLEAR_PATTERN',
                        'success_rate': success_rate,
                        'sample_size': stats['total']
                    }
        
        # RULE 4: Not enough data → Use algorithm
        return {
            'bet': algorithm_prediction,
            'confidence': original_confidence,
            'reason': "No historical data yet (using algorithm prediction)",
            'type': 'NEW_PATTERN',
            'success_rate': 0.5,
            'sample_size': 0
        }
    
    def save_learning(self):
        """Save ALL learning data to Supabase"""
        try:
            if not self.supabase:
                return self._save_learning_local()
            
            # Prepare all data for Supabase
            supabase_data = []
            
            # Save each pattern to Supabase
            for pattern_key, stats in self.pattern_memory.items():
                if stats['total'] == 0:
                    continue
                    
                success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                
                data = {
                    "pattern_key": pattern_key,
                    "total_matches": stats['total'],
                    "successful_matches": stats['success'],
                    "last_updated": datetime.now().isoformat(),
                    "metadata": json.dumps({
                        "last_updated": datetime.now().isoformat(),
                        "success_rate": success_rate,
                        "feature_weights": self.feature_weights
                    })
                }
                supabase_data.append(data)
            
            # Save outcomes as a separate record
            if self.outcomes:
                # Make sure all timestamps are strings
                serializable_outcomes = []
                for outcome in self.outcomes[-1000:]:
                    serialized_outcome = {}
                    for key, value in outcome.items():
                        if key == 'timestamp' and isinstance(value, datetime):
                            serialized_outcome[key] = value.isoformat()
                        elif isinstance(value, list):
                            serialized_outcome[key] = value
                        elif isinstance(value, dict):
                            serialized_outcome[key] = value
                        elif isinstance(value, (str, int, float, bool)):
                            serialized_outcome[key] = value
                        else:
                            serialized_outcome[key] = str(value)
                    serializable_outcomes.append(serialized_outcome)
                
                outcomes_data = {
                    "pattern_key": "ALL_OUTCOMES",
                    "total_matches": len(self.outcomes),
                    "successful_matches": sum(1 for o in self.outcomes if o.get('winner_correct') and o.get('totals_correct')),
                    "last_updated": datetime.now().isoformat(),
                    "metadata": json.dumps({
                        "outcomes": serializable_outcomes,
                        "outcome_count": len(self.outcomes),
                        "feature_weights": self.feature_weights,
                        "saved_at": datetime.now().isoformat()
                    })
                }
                supabase_data.append(outcomes_data)
            
            # Batch insert/update to Supabase
            if supabase_data:
                try:
                    # First, clear existing data
                    self.supabase.table("football_learning").delete().neq("pattern_key", "dummy").execute()
                    
                    # Insert all new data
                    response = self.supabase.table("football_learning").insert(supabase_data).execute()
                    
                    if hasattr(response, 'error') and response.error:
                        raise Exception(f"Supabase error: {response.error}")
                    
                    return True
                except Exception as e:
                    st.error(f"Supabase operation error: {e}")
                    return self._save_learning_local()
            
            return True
            
        except Exception as e:
            st.error(f"Error saving to Supabase: {e}")
            return self._save_learning_local()
    
    def _save_learning_local(self):
        """Fallback local storage"""
        try:
            with open("learning_data.pkl", "wb") as f:
                pickle.dump({
                    'pattern_memory': dict(self.pattern_memory),
                    'feature_weights': self.feature_weights,
                    'outcomes': self.outcomes
                }, f)
            return True
        except Exception as e:
            st.error(f"Local save failed: {e}")
            return False
    
    def load_learning(self):
        """Load learning data from Supabase"""
        try:
            if not self.supabase:
                return self._load_learning_local()
            
            # Load patterns from Supabase
            response = self.supabase.table("football_learning").select("*").execute()
            
            if not response.data:
                return True
            
            for row in response.data:
                pattern_key = row['pattern_key']
                
                if pattern_key == "ALL_OUTCOMES":
                    # Load outcomes
                    if 'metadata' in row and row['metadata']:
                        try:
                            metadata = row['metadata']
                            if isinstance(metadata, dict):
                                if 'outcomes' in metadata:
                                    self.outcomes = metadata['outcomes']
                                if 'feature_weights' in metadata:
                                    self.feature_weights.update(metadata['feature_weights'])
                        except Exception as e:
                            st.error(f"Error parsing metadata: {e}")
                else:
                    # Load pattern stats
                    self.pattern_memory[pattern_key] = {
                        'total': row['total_matches'] or 0,
                        'success': row['successful_matches'] or 0
                    }
            
            return True
            
        except Exception as e:
            return self._load_learning_local()
    
    def _load_learning_local(self):
        """Fallback local storage"""
        try:
            if os.path.exists("learning_data.pkl"):
                with open("learning_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.pattern_memory = defaultdict(lambda: {'total': 0, 'success': 0}, data['pattern_memory'])
                    self.feature_weights = data['feature_weights']
                    self.outcomes = data['outcomes']
                return True
        except:
            pass
        return False
    
    def record_outcome(self, prediction, pattern_indicators, actual_result, actual_score):
        """Record a match outcome for learning"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Determine actual outcomes
        home_goals, away_goals = map(int, actual_score.split('-'))
        
        if home_goals > away_goals:
            actual_winner = "HOME"
        elif away_goals > home_goals:
            actual_winner = "AWAY"
        else:
            actual_winner = "DRAW"
        
        total_goals = home_goals + away_goals
        actual_over = total_goals > 2.5
        
        # Store outcome
        outcome = {
            'timestamp': datetime.now().isoformat(),
            'home_team': prediction.get('home_team', 'Unknown'),
            'away_team': prediction.get('away_team', 'Unknown'),
            'winner_pattern': pattern_indicators['winner']['type'],
            'totals_pattern': pattern_indicators['totals']['type'],
            'winner_confidence': winner_pred['confidence_score'],
            'totals_confidence': totals_pred['confidence_score'],
            'winner_predicted': winner_pred['type'],
            'totals_predicted': totals_pred['direction'],
            'actual_winner': actual_winner,
            'actual_over': actual_over,
            'actual_score': actual_score,
            'finishing_alignment': totals_pred.get('finishing_alignment'),
            'total_category': totals_pred.get('total_category'),
            'risk_flags': totals_pred.get('risk_flags', []),
            'winner_correct': winner_pred['type'] == actual_winner,
            'totals_correct': (totals_pred['direction'] == "OVER") == actual_over
        }
        
        self.outcomes.append(outcome)
        
        # Create pattern keys
        winner_key = f"WINNER_{winner_pred['confidence']}_{winner_pred['confidence_score']//10*10}"
        totals_key = f"TOTALS_{totals_pred.get('finishing_alignment', 'N/A')}_{totals_pred.get('total_category', 'N/A')}"
        
        # Initialize if not exists
        if winner_key not in self.pattern_memory:
            self.pattern_memory[winner_key] = {'total': 0, 'success': 0}
        if totals_key not in self.pattern_memory:
            self.pattern_memory[totals_key] = {'total': 0, 'success': 0}
        
        # Update pattern memory
        self.pattern_memory[winner_key]['total'] += 1
        self.pattern_memory[winner_key]['success'] += 1 if outcome['winner_correct'] else 0
        
        self.pattern_memory[totals_key]['total'] += 1
        self.pattern_memory[totals_key]['success'] += 1 if outcome['totals_correct'] else 0
        
        # Adjust feature weights based on outcomes
        self._adjust_weights(outcome)
        
        # Save to Supabase
        save_success = self.save_learning()
        
        if save_success:
            return outcome, True, "✅ Outcome recorded and saved to Supabase!"
        else:
            return outcome, False, "⚠️ Outcome recorded locally but Supabase save failed"
    
    def _adjust_weights(self, outcome):
        """Adjust feature weights based on outcome success"""
        if not outcome['totals_correct']:
            if 'HIGH_OVER' in str(outcome.get('finishing_alignment', '')):
                self.feature_weights['finishing_alignment'] *= 0.9
            if 'HIGH_VARIANCE_TEAM' in outcome.get('risk_flags', []):
                self.feature_weights['risk_flags'] *= 0.95
            if outcome['totals_confidence'] < 50:
                self.feature_weights['confidence_score'] *= 1.1
        
        if outcome['totals_correct']:
            if 'MED_OVER' in str(outcome.get('finishing_alignment', '')):
                self.feature_weights['finishing_alignment'] *= 1.05
            if 'MED_UNDER' in str(outcome.get('finishing_alignment', '')):
                self.feature_weights['finishing_alignment'] *= 1.1
            if outcome['totals_confidence'] > 80:
                self.feature_weights['confidence_score'] *= 1.05
    
    def get_pattern_success_rate(self, pattern_type, pattern_subtype=None):
        """Get historical success rate for a pattern"""
        key = f"{pattern_type}_{pattern_subtype}" if pattern_subtype else pattern_type
        memory = self.pattern_memory
        
        exact_keys = [k for k in memory if key in k]
        if exact_keys:
            total = sum(memory[k]['total'] for k in exact_keys)
            success = sum(memory[k]['success'] for k in exact_keys)
            if total > 0:
                return success / total
        
        similar_keys = [k for k in memory if pattern_type in k]
        if similar_keys:
            total = sum(memory[k]['total'] for k in similar_keys)
            success = sum(memory[k]['success'] for k in similar_keys)
            if total > 0:
                return success / total
        
        return 0.5
    
    def generate_learned_insights(self):
        """Generate insights based on learned patterns"""
        insights = []
        
        if not self.outcomes:
            return ["🔄 **Betting System**: No historical data yet - record outcomes to start learning"]
        
        # Analyze last 20 outcomes
        recent = self.outcomes[-20:] if len(self.outcomes) > 20 else self.outcomes
        
        # Calculate success rates
        winner_success = sum(1 for o in recent if o['winner_correct']) / len(recent)
        totals_success = sum(1 for o in recent if o['totals_correct']) / len(recent)
        
        insights.append(f"📊 **Your Recent Accuracy**: Winners: {winner_success:.0%} | Totals: {totals_success:.0%}")
        
        # Identify strong patterns
        pattern_performance = defaultdict(lambda: {'total': 0, 'success': 0})
        for outcome in recent:
            key = f"{outcome.get('finishing_alignment', 'N/A')}+{outcome.get('total_category', 'N/A')}"
            pattern_performance[key]['total'] += 1
            pattern_performance[key]['success'] += 1 if outcome['totals_correct'] else 0
        
        # Find best and worst patterns
        for pattern, stats in list(pattern_performance.items()):
            if stats['total'] >= 3:
                success_rate = stats['success'] / stats['total']
                if success_rate >= 0.8:
                    insights.append(f"✅ **YOUR BETTING EDGE**: {pattern} - {stats['success']}/{stats['total']} correct")
                elif success_rate <= 0.2:
                    insights.append(f"💣 **YOUR REVERSE EDGE**: {pattern} - {stats['success']}/{stats['total']} correct → BET OPPOSITE")
        
        # Betting decision performance
        high_confidence_decisions = [o for o in recent if o['totals_confidence'] >= 70]
        if high_confidence_decisions:
            high_conf_success = sum(1 for o in high_confidence_decisions if o['totals_correct']) / len(high_confidence_decisions)
            insights.append(f"🎯 **Your High Confidence Bets**: {high_conf_success:.0%} success rate")
        
        return insights[:5]

# ========== INITIALIZE SESSION STATES ==========
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

if 'betting_system' not in st.session_state:
    st.session_state.betting_system = AdaptiveBettingSystem()

if 'match_history' not in st.session_state:
    st.session_state.match_history = []

if 'show_history' not in st.session_state:
    st.session_state.show_history = False

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'last_pattern_indicators' not in st.session_state:
    st.session_state.last_pattern_indicators = None

if 'last_teams' not in st.session_state:
    st.session_state.last_teams = None

if 'last_league' not in st.session_state:
    st.session_state.last_league = None

if 'last_engine' not in st.session_state:
    st.session_state.last_engine = None

if 'show_feedback_message' not in st.session_state:
    st.session_state.show_feedback_message = False

if 'save_status' not in st.session_state:
    st.session_state.save_status = None

if 'score_input' not in st.session_state:
    st.session_state.score_input = ""

def factorial_cache(n):
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    """Poisson probability mass function"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

# ========== CORE CLASSES ==========

class ExpectedGoalsPredictor:
    """Expected goals calculation"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_avg_goals = league_metrics.get('avg_goals_per_match', 2.5)
        self.league_name = league_name
    
    def predict_expected_goals(self, home_stats, away_stats):
        """Step 1 - Adjusted Team Strength"""
        home_adjGF = home_stats['goals_for_pm'] + 0.6 * home_stats['goals_vs_xg_pm']
        home_adjGA = home_stats['goals_against_pm'] + 0.6 * home_stats['goals_allowed_vs_xga_pm']
        
        away_adjGF = away_stats['goals_for_pm'] + 0.6 * away_stats['goals_vs_xg_pm']
        away_adjGA = away_stats['goals_against_pm'] + 0.6 * away_stats['goals_allowed_vs_xga_pm']
        
        # Dynamic Venue Factor
        venue_factor_home = 1 + 0.05 * (home_stats['points_pm'] - away_stats['points_pm']) / 3
        venue_factor_away = 1 + 0.05 * (away_stats['points_pm'] - home_stats['points_pm']) / 3
        
        venue_factor_home = max(0.8, min(1.2, venue_factor_home))
        venue_factor_away = max(0.8, min(1.2, venue_factor_away))
        
        # Expected Goals Calculation
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
    """Winner determination"""
    
    def predict_winner(self, home_xg, away_xg, home_stats, away_stats):
        """Winner determination with finishing adjustment"""
        
        # Get finishing trends
        home_finishing = home_stats['goals_vs_xg_pm']
        away_finishing = away_stats['goals_vs_xg_pm']
        
        # Get defensive performance
        home_defense = home_stats['goals_allowed_vs_xga_pm']
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        # Adjust xG for finishing ability
        home_adjusted_xg = home_xg + home_finishing - away_defense
        away_adjusted_xg = away_xg + away_finishing - home_defense
        
        # Calculate adjusted delta
        delta = home_adjusted_xg - away_adjusted_xg
        
        # Determine volatility flag
        volatility_high = False
        if abs(home_finishing) > 0.3 and abs(away_finishing) > 0.3:
            volatility_high = True
        elif home_finishing > 0.3 and away_finishing > 0.3:
            volatility_high = True
        elif home_finishing < -0.3 and away_finishing < -0.3:
            volatility_high = True
        
        # Winner determination
        if delta > 1.2:
            predicted_winner = "HOME"
            winner_strength = "STRONG"
            if volatility_high:
                winner_strength = "STRONG_HIGH_VOL"
                
        elif delta > 0.5:
            predicted_winner = "HOME"
            winner_strength = "MODERATE"
            if volatility_high:
                winner_strength = "MODERATE_HIGH_VOL"
                
        elif delta > 0.2:
            predicted_winner = "HOME"
            winner_strength = "SLIGHT"
            if volatility_high:
                winner_strength = "SLIGHT_HIGH_VOL"
                
        elif delta < -1.2:
            predicted_winner = "AWAY"
            winner_strength = "STRONG"
            if volatility_high:
                winner_strength = "STRONG_HIGH_VOL"
                
        elif delta < -0.5:
            predicted_winner = "AWAY"
            winner_strength = "MODERATE"
            if volatility_high:
                winner_strength = "MODERATE_HIGH_VOL"
                
        elif delta < -0.2:
            predicted_winner = "AWAY"
            winner_strength = "SLIGHT"
            if volatility_high:
                winner_strength = "SLIGHT_HIGH_VOL"
                
        else:
            predicted_winner = "DRAW"
            winner_strength = "CLOSE"
            if volatility_high:
                winner_strength = "CLOSE_HIGH_VOL"
        
        # Confidence calculation
        base_confidence = min(100, abs(delta) / max(home_adjusted_xg, away_adjusted_xg, 0.5) * 150)
        
        # Add bonuses
        venue_bonus = 0
        if home_stats['points_pm'] > 2.0:
            venue_bonus += 15
        if away_stats['points_pm'] < 1.0:
            venue_bonus += 15
        
        win_rate_diff = home_stats['win_rate'] - away_stats['win_rate']
        form_bonus = min(20, max(0, win_rate_diff * 40))
        
        winner_confidence = min(100, max(30, base_confidence + venue_bonus + form_bonus))
        
        # Penalize high volatility matches
        if volatility_high:
            winner_confidence = max(30, winner_confidence - 20)
        
        # Confidence categorization
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
            'confidence_score': winner_confidence,
            'winner_confidence_category': confidence_category,
            'delta': delta,
            'adjusted_delta': delta,
            'volatility_high': volatility_high,
            'home_adjusted_xg': home_adjusted_xg,
            'away_adjusted_xg': away_adjusted_xg,
            'home_finishing': home_finishing,
            'away_finishing': away_finishing,
            'home_defense_quality': home_defense,
            'away_defense_quality': away_defense
        }

class TotalsPredictor:
    """Totals prediction with defense quality rules"""
    
    def __init__(self, league_name):
        self.league_name = league_name
        self.league_adjustments = LEAGUE_ADJUSTMENTS.get(league_name, LEAGUE_ADJUSTMENTS["Premier League"])
    
    def categorize_finishing(self, value):
        """Categorize finishing strength"""
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
        """Finishing trend alignment matrix"""
        home_cat = self.categorize_finishing(home_finish)
        away_cat = self.categorize_finishing(away_finish)
        
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
                "MODERATE_UNDERFORM": "MED_UNDER",
                "STRONG_UNDERPERFORM": "HIGH_UNDER"
            }
        }
        
        return alignment_matrix[home_cat][away_cat]
    
    def categorize_total_xg(self, total_xg):
        """Total xG categories"""
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
    
    def check_defense_quality_rules(self, home_stats, away_stats):
        """Defense quality rules"""
        home_def = home_stats['goals_allowed_vs_xga_pm']
        away_def = away_stats['goals_allowed_vs_xga_pm']
        
        # RULE 1: Double bad defense = OVER 2.5
        if home_def >= 0.5 and away_def >= 0.5:
            min_def = min(home_def, away_def)
            if min_def >= 2.0:
                confidence = 80
                reason = f"DOUBLE VERY BAD DEFENSE: Home({home_def:.2f}) + Away({away_def:.2f}) = High scoring guaranteed"
            else:
                confidence = 70
                reason = f"DOUBLE BAD DEFENSE: Home({home_def:.2f}) + Away({away_def:.2f}) = High scoring likely"
            
            return {
                'direction': "OVER",
                'confidence': confidence,
                'reason': reason,
                'rule_triggered': 'DOUBLE_BAD_DEFENSE'
            }
        
        # RULE 2: Good defense present = UNDER 2.5
        if home_def <= -0.5 or away_def <= -0.5:
            if home_def <= -0.5 and away_def <= -0.5:
                confidence = 85
                reason = f"DOUBLE GOOD DEFENSE: Home({home_def:.2f}) + Away({away_def:.2f}) = Low scoring guaranteed"
            elif home_def <= -2.0 or away_def <= -2.0:
                confidence = 80
                reason = f"VERY GOOD DEFENSE PRESENT: Home({home_def:.2f}) Away({away_def:.2f}) = Low scoring likely"
            else:
                confidence = 70
                reason = f"GOOD DEFENSE PRESENT: Home({home_def:.2f}) Away({away_def:.2f}) = Low scoring likely"
            
            return {
                'direction': "UNDER",
                'confidence': confidence,
                'reason': reason,
                'rule_triggered': 'GOOD_DEFENSE_PRESENT'
            }
        
        return None
    
    def check_risk_flags(self, home_stats, away_stats, total_xg):
        """Risk flag system"""
        risk_flags = []
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
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
        
        # Bundesliga specific adjustment
        if self.league_name == "Bundesliga" and total_xg < 3.3:
            risk_flags.append("BUNDESLIGA_LOW_SCORING")
        
        return risk_flags
    
    def predict_totals(self, home_xg, away_xg, home_stats, away_stats):
        """Complete totals prediction with defense rules"""
        total_xg = home_xg + away_xg
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        # Check defense quality rules first
        defense_rule = self.check_defense_quality_rules(home_stats, away_stats)
        if defense_rule:
            direction = defense_rule['direction']
            base_confidence = defense_rule['confidence']
            rule_reason = defense_rule['reason']
            rule_triggered = defense_rule['rule_triggered']
            
            finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
            total_category = self.categorize_total_xg(total_xg)
            risk_flags = self.check_risk_flags(home_stats, away_stats, total_xg)
            
            # Adjust confidence based on risk flags
            final_confidence = base_confidence
            for flag in risk_flags:
                if flag == "VOLATILE_OVER_BOTH":
                    if direction == "OVER":
                        final_confidence -= 10
                    else:
                        final_confidence += 5
                elif flag == "CLOSE_TO_THRESHOLD":
                    final_confidence -= 10
            
            final_confidence = max(5, min(95, final_confidence))
            
            # Confidence category
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
                'away_finishing': away_finish,
                'defense_rule_triggered': rule_triggered,
                'defense_rule_reason': rule_reason
            }
        
        # Original logic
        over_threshold = self.league_adjustments['over_threshold']
        base_direction = "OVER" if total_xg > over_threshold else "UNDER"
        
        finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
        total_category = self.categorize_total_xg(total_xg)
        
        risk_flags = self.check_risk_flags(home_stats, away_stats, total_xg)
        
        # PROVEN PATTERN 1
        if finishing_alignment == "NEUTRAL" and total_xg > 3.0:
            return {
                'direction': "UNDER",
                'total_xg': total_xg,
                'confidence': "HIGH",
                'confidence_score': 80,
                'finishing_alignment': finishing_alignment,
                'total_category': total_category,
                'risk_flags': risk_flags,
                'home_finishing': home_finish,
                'away_finishing': away_finish,
                'defense_rule_triggered': 'NEUTRAL_HIGH_XG_UNDER'
            }
        
        # Decision matrix
        decision_matrix = {
            "VERY_HIGH": {
                "HIGH_OVER": ("OVER", "MEDIUM", 60),
                "MED_OVER": ("OVER", "HIGH", 75),
                "LOW_OVER": ("OVER", "MEDIUM", 65),
                "NEUTRAL": ("OVER", "MEDIUM", 60),
                "RISKY": ("OVER", "LOW", 45),
                "HIGH_RISK": (base_direction, "VERY LOW", 35),
                "MED_UNDER": ("OVER", "MEDIUM", 65),
                "LOW_UNDER": ("UNDER", "LOW", 45)
            },
            "HIGH": {
                "HIGH_OVER": ("OVER", "LOW", 50),
                "MED_OVER": ("OVER", "HIGH", 70),
                "LOW_OVER": ("OVER", "MEDIUM", 60),
                "NEUTRAL": (base_direction, "LOW", 50),
                "RISKY": (base_direction, "LOW", 45),
                "HIGH_RISK": (base_direction, "VERY LOW", 35),
                "MED_UNDER": ("OVER", "MEDIUM", 60),
                "LOW_UNDER": ("UNDER", "LOW", 45)
            },
            "MODERATE_HIGH": {
                "HIGH_OVER": ("OVER", "MEDIUM", 55),
                "MED_OVER": ("OVER", "MEDIUM", 65),
                "LOW_OVER": ("OVER", "MEDIUM", 60),
                "NEUTRAL": (base_direction, "LOW", 50),
                "LOW_UNDER": ("UNDER", "LOW", 45),
                "MED_UNDER": ("OVER", "MEDIUM", 55)
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
        
        # Apply risk flag penalties
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
        
        # Adjust confidence category
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
            'away_finishing': away_finish,
            'defense_rule_triggered': None
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
    """Generate enhanced insights with defense rules"""
    
    @staticmethod
    def generate_insights(winner_prediction, totals_prediction):
        insights = []
        
        # Winner insights
        if winner_prediction.get('winner_confidence_category') == "VERY HIGH":
            insights.append(f"🎯 **High Confidence Winner**: Model strongly favors {winner_prediction.get('predicted_winner', 'N/A')}")
        elif winner_prediction.get('winner_confidence_category') == "LOW":
            insights.append(f"⚠️ **Low Confidence Winner**: Exercise caution on {winner_prediction.get('predicted_winner', 'N/A')} prediction")
        
        # Defense rule insights
        defense_rule = totals_prediction.get('defense_rule_triggered')
        if defense_rule == 'DOUBLE_BAD_DEFENSE':
            insights.append(f"⚡ **DOUBLE BAD DEFENSE**: Both teams allow more goals than expected → HIGH SCORING likely")
        elif defense_rule == 'GOOD_DEFENSE_PRESENT':
            insights.append(f"🛡️ **GOOD DEFENSE PRESENT**: At least one team limits goals well → LOW SCORING likely")
        elif defense_rule == 'NEUTRAL_HIGH_XG_UNDER':
            insights.append(f"📉 **PROVEN PATTERN**: NEUTRAL finishing + HIGH xG = UNDER")
        
        # Volatility insight
        home_finish = totals_prediction.get('home_finishing', 0)
        away_finish = totals_prediction.get('away_finishing', 0)
        
        if home_finish > 0.35 and away_finish > 0.35:
            insights.append("⚠️ **Both teams strong overperformers** - High volatility expected")
        
        # PROVEN PATTERN INSIGHTS
        alignment = totals_prediction.get('finishing_alignment', 'NEUTRAL')
        total_category = totals_prediction.get('total_category', 'N/A')
        total_xg = totals_prediction.get('total_xg', 0)
        
        if alignment == "NEUTRAL" and total_xg > 3.0:
            insights.append("✅ **PROVEN PATTERN**: NEUTRAL + HIGH_xG (xG>3.0) = UNDER")
        elif alignment == "MED_UNDER" and total_xg > 3.0:
            insights.append("✅ **PROVEN PATTERN**: MED_UNDER + HIGH_xG (xG>3.0) = OVER")
        
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
            insights.append("⚠️ **HIGH_OVER pattern**: Be cautious with this pattern")
        elif alignment == "MED_OVER":
            insights.append("✅ **MED_OVER pattern**: Historically strong pattern")
        
        # Risk flag insights
        risk_flags = totals_prediction.get('risk_flags', [])
        if risk_flags:
            risk_count = len(risk_flags)
            flag_list = ", ".join(risk_flags)
            insights.append(f"⚠️ **{risk_count} risk flag(s) detected**: {flag_list}")
        
        # Defense quality insight
        home_def = winner_prediction.get('home_defense_quality', 0)
        away_def = winner_prediction.get('away_defense_quality', 0)
        
        if home_def >= 0.5 and away_def >= 0.5:
            insights.append(f"🚨 **Both teams have poor defense**: High scoring expected")
        elif home_def <= -0.5 or away_def <= -0.5:
            insights.append(f"✅ **Strong defense detected**: Could limit scoring")
        
        return insights[:8]

# ========== FOOTBALL ENGINE ==========

class FootballIntelligenceEngine:
    """Football prediction engine"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_name = league_name
        
        # Initialize predictors
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
        self.insights_generator = InsightsGenerator()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate prediction"""
        
        # Get base prediction
        home_xg, away_xg, calc_details = self.xg_predictor.predict_expected_goals(
            home_stats, away_stats
        )
        
        probabilities = self.probability_engine.calculate_all_probabilities(
            home_xg, away_xg
        )
        
        winner_prediction = self.winner_predictor.predict_winner(
            home_xg, away_xg, home_stats, away_stats
        )
        
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Generate insights
        insights = self.insights_generator.generate_insights(winner_prediction, totals_prediction)
        
        # Determine final probabilities
        if winner_prediction['predicted_winner'] == "HOME":
            winner_display = home_team
            winner_prob = probabilities['home_win_probability']
        elif winner_prediction['predicted_winner'] == "AWAY":
            winner_display = away_team
            winner_prob = probabilities['away_win_probability']
        else:
            winner_display = "DRAW"
            winner_prob = probabilities['draw_probability']
        
        if totals_prediction['direction'] == "OVER":
            total_prob = probabilities['over_2_5_probability']
        else:
            total_prob = probabilities['under_2_5_probability']
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'winner': {
                'team': winner_display,
                'type': winner_prediction['predicted_winner'],
                'probability': winner_prob,
                'confidence': winner_prediction['winner_confidence_category'],
                'confidence_score': winner_prediction['confidence_score'],
                'strength': winner_prediction['winner_strength'],
                'most_likely_score': probabilities['most_likely_score'],
                'adjusted_delta': winner_prediction['adjusted_delta'],
                'volatility_high': winner_prediction['volatility_high'],
                'home_finishing': winner_prediction['home_finishing'],
                'away_finishing': winner_prediction['away_finishing'],
            },
            
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
                'away_finishing': totals_prediction['away_finishing'],
                'defense_rule_triggered': totals_prediction.get('defense_rule_triggered'),
            },
            
            'probabilities': probabilities,
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': home_xg + away_xg},
            'insights': insights,
            'calculation_details': calc_details
        }

# ========== PATTERN INDICATORS ==========

class PatternIndicators:
    """Generate pattern indicators"""
    
    def __init__(self, betting_system):
        self.betting_system = betting_system
    
    def generate_indicators(self, prediction):
        """Generate pattern indicators"""
        indicators = {'winner': None, 'totals': None}
        
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # WINNER INDICATORS
        winner_success_rate = self.betting_system.get_pattern_success_rate(
            "WINNER", 
            f"{winner_pred['confidence']}_{winner_pred['confidence_score']//10*10}"
        )
        
        if winner_pred['confidence_score'] >= 90 and winner_success_rate > 0.7:
            indicators['winner'] = {
                'type': 'MET',
                'color': 'green',
                'text': 'YOUR PROVEN WINNER PATTERN',
                'explanation': f'Your historical success: {winner_success_rate:.0%} for this confidence level'
            }
        elif winner_pred['confidence_score'] < 45 and winner_success_rate < 0.4:
            indicators['winner'] = {
                'type': 'AVOID',
                'color': 'red',
                'text': 'YOUR WEAK WINNER PATTERN',
                'explanation': f'Your historical failure: {winner_success_rate:.0%} success rate'
            }
        elif winner_pred.get('volatility_high', False):
            vol_success = self.betting_system.get_pattern_success_rate("VOLATILE", "HIGH_VOLATILITY")
            indicators['winner'] = {
                'type': 'WARNING',
                'color': 'yellow',
                'text': 'HIGH VOLATILITY MATCH',
                'explanation': f'Your volatile matches: {vol_success:.0%} success rate historically'
            }
        else:
            indicators['winner'] = {
                'type': 'NO_PATTERN',
                'color': 'gray',
                'text': 'NO PATTERN YET',
                'explanation': f'Your historical success: {winner_success_rate:.0%}'
            }
        
        # TOTALS INDICATORS
        finishing_alignment = totals_pred.get('finishing_alignment', 'NEUTRAL')
        total_category = totals_pred.get('total_category', 'N/A')
        
        pattern_key = f"{finishing_alignment}_{total_category}"
        pattern_success = self.betting_system.get_pattern_success_rate("TOTALS", pattern_key)
        
        # Determine based on YOUR success rates
        if pattern_success > 0.7 and self.betting_system.pattern_memory.get(pattern_key, {}).get('total', 0) >= 3:
            indicators['totals'] = {
                'type': 'MET',
                'color': 'green',
                'text': f'YOUR STRONG PATTERN - {totals_pred["direction"]} 2.5',
                'explanation': f'Your historical success: {pattern_success:.0%} for this pattern'
            }
        elif pattern_success < 0.4 and self.betting_system.pattern_memory.get(pattern_key, {}).get('total', 0) >= 3:
            indicators['totals'] = {
                'type': 'AVOID',
                'color': 'red',
                'text': f'YOUR WEAK PATTERN - {totals_pred["direction"]} 2.5',
                'explanation': f'Your historical failure: {pattern_success:.0%} success rate'
            }
        elif pattern_success > 0.6:
            indicators['totals'] = {
                'type': 'PROMISING',
                'color': 'blue',
                'text': f'PROMISING PATTERN - {totals_pred["direction"]} 2.5',
                'explanation': f'Your historical success: {pattern_success:.0%}'
            }
        else:
            indicators['totals'] = {
                'type': 'NO_PATTERN',
                'color': 'gray',
                'text': 'NO PATTERN YET',
                'explanation': f'Your historical success: {pattern_success:.0%}'
            }
        
        return indicators

# ========== BETTING CARD ==========

class BettingCard:
    """Betting card with betting decisions"""
    
    def __init__(self, betting_system):
        self.betting_system = betting_system
    
    def get_recommendation(self, prediction, winner_decision, totals_decision):
        """Get betting recommendation based on decisions"""
        
        # Calculate combined confidence
        combined_confidence = (winner_decision['confidence'] + totals_decision['confidence']) / 2
        
        # Determine recommendation type
        if winner_decision['type'] in ['PROVEN_WINNER', 'PROVEN_LOSER'] and totals_decision['type'] in ['PROVEN_WINNER', 'PROVEN_LOSER']:
            return {
                'type': 'combo',
                'text': f"🎯 {winner_decision['bet']} + 📈 {totals_decision['bet']} 2.5",
                'confidence': combined_confidence,
                'color': '#10B981',
                'icon': '🎯',
                'subtext': 'DOUBLE BET (PROVEN PATTERNS)',
                'reason': f"Winner: {winner_decision['reason']} | Totals: {totals_decision['reason']}",
                'expected_value': combined_confidence / 100
            }
        elif winner_decision['type'] in ['PROVEN_WINNER', 'PROVEN_LOSER']:
            return {
                'type': 'single',
                'text': f"🏆 {winner_decision['bet']} to win",
                'confidence': winner_decision['confidence'],
                'color': '#3B82F6',
                'icon': '🏆',
                'subtext': 'WINNER BET',
                'reason': winner_decision['reason'],
                'expected_value': winner_decision['confidence'] / 100
            }
        elif totals_decision['type'] in ['PROVEN_WINNER', 'PROVEN_LOSER']:
            return {
                'type': 'single',
                'text': f"📈 {totals_decision['bet']} 2.5 Goals",
                'confidence': totals_decision['confidence'],
                'color': '#8B5CF6',
                'icon': '📈',
                'subtext': 'TOTALS BET',
                'reason': totals_decision['reason'],
                'expected_value': totals_decision['confidence'] / 100
            }
        else:
            return {
                'type': 'none',
                'text': "🚫 No Proven Pattern",
                'confidence': combined_confidence,
                'color': '#6B7280',
                'icon': '🤔',
                'subtext': 'NO BET',
                'reason': 'No proven patterns with enough data',
                'expected_value': 0
            }
    
    def display_card(self, recommendation):
        """Display the betting card"""
        ev = recommendation.get('expected_value', 0)
        
        # Color based on expected value
        if ev > 0.2:
            color = '#10B981'
        elif ev > 0.1:
            color = '#3B82F6'
        elif ev > 0:
            color = '#8B5CF6'
        else:
            color = '#6B7280'
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}20 0%, #1F2937 100%);
            padding: 25px;
            border-radius: 20px;
            border: 2px solid {color};
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
            <div style="font-size: 24px; color: {color}; margin-bottom: 10px; font-weight: bold;">
                {recommendation['subtext']}
            </div>
            <div style="font-size: 18px; color: #9CA3AF; margin-bottom: 15px;">
                Confidence: {recommendation['confidence']:.0f}/100 | Expected Value: {ev:.3f}
            </div>
            <div style="font-size: 16px; color: #D1D5DB; padding: 10px; background: rgba(59, 130, 246, 0.1); border-radius: 10px;">
                {recommendation['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

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

# ========== FIXED FEEDBACK SYSTEM ==========

def record_outcome_with_feedback(prediction, winner_decision, totals_decision, home_team, away_team):
    """Fixed feedback system"""
    
    st.divider()
    st.subheader("📝 Record Outcome for Learning")
    
    # Show previous feedback if exists
    if st.session_state.get('save_status'):
        status_type, status_message = st.session_state.save_status
        if status_type == "success":
            st.success(status_message)
            
            # Show what was learned in an expander
            if st.session_state.get('last_outcome'):
                with st.expander("📈 What was learned?", expanded=True):
                    outcome = st.session_state.last_outcome
                    st.write(f"**Match**: {outcome['home_team']} vs {outcome['away_team']}")
                    st.write(f"**Actual Score**: {outcome['actual_score']}")
                    st.write(f"**Betting Decision**: {winner_decision['bet']} & {totals_decision['bet']} 2.5")
                    st.write(f"**Winner Bet**: {'✅ Correct' if outcome['winner_correct'] else '❌ Wrong'}")
                    st.write(f"**Totals Bet**: {'✅ Correct' if outcome['totals_correct'] else '❌ Wrong'}")
                    
                    winner_pattern = f"WINNER_{prediction['winner']['confidence']}_{prediction['winner']['confidence_score']//10*10}"
                    totals_pattern = f"TOTALS_{prediction['totals'].get('finishing_alignment', 'N/A')}_{prediction['totals'].get('total_category', 'N/A')}"
                    
                    st.write(f"**Winner Pattern**: {winner_pattern}")
                    st.write(f"**Winner Decision Type**: {winner_decision['type']}")
                    st.write(f"**Totals Pattern**: {totals_pattern}")
                    st.write(f"**Totals Decision Type**: {totals_decision['type']}")
                    st.write(f"**Total Patterns Learned**: {len(st.session_state.betting_system.pattern_memory)}")
                    st.write(f"**Total Outcomes Recorded**: {len(st.session_state.betting_system.outcomes)}")
                    
                    # Show Supabase status
                    if st.session_state.betting_system.supabase:
                        st.success("✅ Saved to Supabase successfully!")
                    else:
                        st.warning("⚠️ Saved locally (Supabase not available)")
        else:
            st.error(status_message)
    
    # Simple input and button
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use a unique key for this input
        score_key = f"score_{home_team}_{away_team}"
        if score_key not in st.session_state:
            st.session_state[score_key] = ""
        
        score_input = st.text_input(
            "Actual Score (e.g., 2-1)", 
            value=st.session_state[score_key],
            help="Enter the actual match result.",
            key=f"input_{score_key}"
        )
        st.session_state[score_key] = score_input
    
    with col2:
        record_button = st.button(
            "✅ Record Outcome & Save to Supabase", 
            type="primary", 
            use_container_width=True,
            key=f"record_btn_{home_team}_{away_team}"
        )
    
    if record_button:
        if not score_input or score_input.strip() == "":
            st.error("❌ Please enter a score first")
            return
        
        # Clean and validate the score input
        score_input = score_input.strip()
        
        # Check if it contains a dash
        if '-' not in score_input:
            st.error("❌ Please enter score in format '2-1' (needs a dash)")
            return
        
        # Split and validate
        parts = score_input.split('-')
        if len(parts) != 2:
            st.error("❌ Please enter score in format '2-1' (exactly one dash)")
            return
        
        try:
            home_goals = int(parts[0].strip())
            away_goals = int(parts[1].strip())
            
            # Validate they're reasonable numbers
            if home_goals < 0 or away_goals < 0:
                st.error("❌ Goals cannot be negative")
                return
            if home_goals > 20 or away_goals > 20:
                st.error("❌ That's an unrealistic score!")
                return
            
            # Create pattern indicators for recording
            pattern_indicators = {
                'winner': {'type': winner_decision['type']},
                'totals': {'type': totals_decision['type']}
            }
            
            # Record outcome and SAVE TO SUPABASE
            with st.spinner("⏳ Saving to Supabase..."):
                try:
                    outcome, save_success, save_message = st.session_state.betting_system.record_outcome(
                        prediction, pattern_indicators, "", f"{home_goals}-{away_goals}"
                    )
                    
                    # Store results
                    st.session_state.last_outcome = outcome
                    st.session_state.save_status = ("success" if save_success else "error", save_message)
                    
                    # Clear the input
                    st.session_state[score_key] = ""
                    
                    # Add to history
                    history_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'home_team': home_team,
                        'away_team': away_team,
                        'betting_decision': f"{winner_decision['bet']} & {totals_decision['bet']} 2.5",
                        'actual_score': score_input,
                        'winner_correct': outcome['winner_correct'],
                        'totals_correct': outcome['totals_correct'],
                        'save_status': save_success
                    }
                    st.session_state.match_history.append(history_entry)
                    
                    # Show message
                    if save_success:
                        st.success(save_message)
                    else:
                        st.error(save_message)
                        
                    # Force a rerun to show the message
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error saving to Supabase: {str(e)}")
                    return
                
        except ValueError:
            st.error("❌ Please enter numbers only (e.g., '2-0' or '3-1')")
    
    # Clear messages button
    if st.session_state.get('save_status'):
        if st.button("🗑️ Clear Messages", type="secondary", use_container_width=True):
            st.session_state.save_status = None
            if score_key in st.session_state:
                st.session_state[score_key] = ""
            st.rerun()
    
    st.caption("💡 **Tip**: Enter the actual match result to improve the betting system.")

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
            
            st.markdown("### 🎯 Prediction Settings")
            show_details = st.checkbox("Show Detailed Analysis", value=True)
            
            if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
                calculate_btn = True
            else:
                calculate_btn = False
        else:
            st.error("Could not prepare team data")
            st.stop()

    # Betting System Section
    st.divider()
    st.header("💰 Betting System Status")
    
    # Supabase Status
    if st.session_state.betting_system.supabase:
        st.success("🔄 **Storage**: Connected to Supabase")
    else:
        st.warning("🔄 **Storage**: Local only (Supabase not available)")
    
    st.write(f"🎯 **Your Patterns**: {len(st.session_state.betting_system.pattern_memory)}")
    st.write(f"📈 **Your Outcomes**: {len(st.session_state.betting_system.outcomes)}")
    
    # Show proven patterns
    if st.session_state.betting_system.pattern_memory:
        proven_patterns = [
            (k, v) for k, v in st.session_state.betting_system.pattern_memory.items() 
            if v['total'] >= 3 and v['success']/v['total'] >= 0.7
        ]
        if proven_patterns:
            st.write(f"✅ **Proven Patterns**: {len(proven_patterns)}")
    
    # Refresh data button
    if st.button("🔄 Refresh Betting Data", use_container_width=True):
        success = st.session_state.betting_system.load_learning()
        if success:
            st.success("Betting data refreshed!")
        else:
            st.warning("Could not refresh from Supabase")
        st.rerun()
    
    st.divider()
    
    # Show betting statistics
    st.subheader("Your Betting Statistics")
    total_outcomes = len(st.session_state.betting_system.outcomes)
    if total_outcomes > 0:
        recent = st.session_state.betting_system.outcomes[-10:] if len(st.session_state.betting_system.outcomes) >= 10 else st.session_state.betting_system.outcomes
        if recent:
            winner_acc = sum(1 for o in recent if o['winner_correct']) / len(recent)
            totals_acc = sum(1 for o in recent if o['totals_correct']) / len(recent)
            
            st.metric("Your Total Matches", total_outcomes)
            st.metric("Your Recent Winner Acc", f"{winner_acc:.0%}")
            st.metric("Your Recent Totals Acc", f"{totals_acc:.0%}")
    else:
        st.info("No outcomes recorded yet. Record your first match outcome!")

if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# ========== CHECK IF WE SHOULD SHOW PREDICTION ==========
show_prediction = False
prediction = None
pattern_indicators = None
engine = None
winner_decision = None
totals_decision = None

# Option 1: User just clicked "Generate Prediction"
if 'calculate_btn' in locals() and calculate_btn:
    show_prediction = True
    
# Option 2: We have a stored prediction from last time
elif (st.session_state.last_prediction is not None and 
      st.session_state.last_teams is not None and
      st.session_state.last_league == selected_league):
    show_prediction = True
    # Use stored prediction
    prediction = st.session_state.last_prediction
    pattern_indicators = st.session_state.last_pattern_indicators
    engine = st.session_state.last_engine
    winner_decision = st.session_state.get('last_winner_decision')
    totals_decision = st.session_state.get('last_totals_decision')
    home_team, away_team = st.session_state.last_teams

# If no prediction to show
if not show_prediction:
    st.info("👈 Select teams and click 'Generate Prediction'")
    
    # Show betting insights
    with st.expander("💰 Betting System Insights", expanded=True):
        insights = st.session_state.betting_system.generate_learned_insights()
        for insight in insights:
            st.write(f"• {insight}")
    
    st.stop()

# ========== IF WE GET HERE, WE HAVE A PREDICTION TO SHOW ==========

# If this is a new prediction (user clicked "Generate Prediction")
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Generate prediction
        engine = FootballIntelligenceEngine(league_metrics, selected_league)
        prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)
        
        # Generate betting decisions
        winner_decision = st.session_state.betting_system.get_betting_decision(
            prediction['winner']['type'],
            "WINNER",
            f"{prediction['winner']['confidence']}_{prediction['winner']['confidence_score']//10*10}",
            prediction['winner']['confidence_score']
        )
        
        totals_decision = st.session_state.betting_system.get_betting_decision(
            prediction['totals']['direction'],
            "TOTALS", 
            f"{prediction['totals'].get('finishing_alignment', 'N/A')}_{prediction['totals'].get('total_category', 'N/A')}",
            prediction['totals']['confidence_score']
        )
        
        # Generate pattern indicators
        pattern_generator = PatternIndicators(st.session_state.betting_system)
        pattern_indicators = pattern_generator.generate_indicators(prediction)
        
        # Store in session state for next time
        st.session_state.last_prediction = prediction
        st.session_state.last_pattern_indicators = pattern_indicators
        st.session_state.last_winner_decision = winner_decision
        st.session_state.last_totals_decision = totals_decision
        st.session_state.last_teams = (home_team, away_team)
        st.session_state.last_league = selected_league
        st.session_state.last_engine = engine
        
    except KeyError as e:
        st.error(f"Team data error: {e}")
        st.stop()

# ========== DISPLAY THE BETTING DECISION ==========
st.header(f"💰 BETTING DECISION: {home_team} vs {away_team}")
st.caption(f"League: {selected_league} | Based on YOUR historical results")

# Display betting decisions
col1, col2 = st.columns(2)

with col1:
    # Winner betting decision
    decision_type = winner_decision['type']
    bet = winner_decision['bet']
    confidence = winner_decision['confidence']
    reason = winner_decision['reason']
    
    # Color coding for decision types
    if decision_type == 'PROVEN_WINNER':
        card_color = '#14532D'
        text_color = '#22C55E'
        icon = '✅'
        title = 'PROVEN WINNER PATTERN'
    elif decision_type == 'PROVEN_LOSER':
        card_color = '#7F1D1D'
        text_color = '#EF4444'
        icon = '🔄'
        title = 'REVERSE BET (Pattern fails)'
    elif decision_type == 'UNCLEAR_PATTERN':
        card_color = '#78350F'
        text_color = '#F59E0B'
        icon = '⚠️'
        title = 'MIXED RESULTS'
    else:
        card_color = '#1E293B'
        text_color = '#94A3B8'
        icon = '🆕'
        title = 'NEW PATTERN'
    
    # Map bet to display name
    if bet == "HOME":
        bet_display = home_team
        bet_icon = "🏠"
    elif bet == "AWAY":
        bet_display = away_team
        bet_icon = "✈️"
    else:
        bet_display = "DRAW"
        bet_icon = "🤝"
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0; border: 2px solid {text_color};">
        <div style="font-size: 14px; color: {text_color}; font-weight: bold; margin-bottom: 10px;">
            {icon} {title}
        </div>
        <div style="font-size: 24px; font-weight: bold; color: white; margin: 10px 0;">
            {bet_icon} {bet_display}
        </div>
        <div style="font-size: 36px; font-weight: bold; color: {text_color}; margin: 10px 0;">
            {confidence:.0f}%
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
            {reason}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Totals betting decision
    decision_type = totals_decision['type']
    bet = totals_decision['bet']
    confidence = totals_decision['confidence']
    reason = totals_decision['reason']
    
    # Color coding for decision types
    if decision_type == 'PROVEN_WINNER':
        card_color = '#14532D'
        text_color = '#22C55E'
        icon = '✅'
        title = 'PROVEN PATTERN'
    elif decision_type == 'PROVEN_LOSER':
        card_color = '#7F1D1D'
        text_color = '#EF4444'
        icon = '🔄'
        title = 'REVERSE BET (Pattern fails)'
    elif decision_type == 'UNCLEAR_PATTERN':
        card_color = '#78350F'
        text_color = '#F59E0B'
        icon = '⚠️'
        title = 'MIXED RESULTS'
    else:
        card_color = '#1E293B'
        text_color = '#94A3B8'
        icon = '🆕'
        title = 'NEW PATTERN'
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0; border: 2px solid {text_color};">
        <div style="font-size: 14px; color: {text_color}; font-weight: bold; margin-bottom: 10px;">
            {icon} {title}
        </div>
        <div style="font-size: 24px; font-weight: bold; color: white; margin: 10px 0;">
            {bet} 2.5
        </div>
        <div style="font-size: 36px; font-weight: bold; color: {text_color}; margin: 10px 0;">
            {confidence:.0f}%
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px; padding: 10px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;">
            {reason}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== BETTING CARD RECOMMENDATION ==========
st.divider()
st.subheader("🎯 BETTING CARD RECOMMENDATION")

# Generate betting card recommendation
betting_card = BettingCard(st.session_state.betting_system)
recommendation = betting_card.get_recommendation(prediction, winner_decision, totals_decision)

# Display the card
betting_card.display_card(recommendation)

# ========== ALGORITHM PREDICTION (FOR REFERENCE) ==========
with st.expander("🤖 Algorithm Prediction (For Reference)", expanded=False):
    st.write(f"**Algorithm originally predicted:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{prediction['winner']['team']} to win", 
                 f"{prediction['winner']['probability']*100:.1f}%",
                 f"Confidence: {prediction['winner']['confidence']} ({prediction['winner']['confidence_score']:.0f}/100)")
    with col2:
        st.metric(f"{prediction['totals']['direction']} 2.5",
                 f"{prediction['probabilities'][f'{prediction['totals']['direction'].lower()}_2_5_probability']*100:.1f}%",
                 f"Confidence: {prediction['totals']['confidence']} ({prediction['totals']['confidence_score']:.0f}/100)")
    
    st.caption("Note: Betting decisions above override algorithm predictions based on your historical results")

# ========== PATTERN INDICATORS ==========
st.divider()
st.subheader("🎯 Your Pattern Analysis")

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
    elif totals_indicator['type'] == 'PROMISING':
        st.markdown(f"""
        <div style="background-color: #1E3A8A; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #3B82F6;">
            <div style="font-size: 20px; font-weight: bold; color: #3B82F6; margin: 5px 0;">
                🔵 {totals_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #BFDBFE;">
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

st.caption("💡 **Pattern Analysis**: Shows how your historical results influence betting decisions")

# ========== INSIGHTS ==========
if prediction['insights']:
    st.subheader("🧠 Enhanced Insights")
    for insight in prediction['insights']:
        st.write(f"• {insight}")

# ========== RISK FLAGS ==========
if prediction['totals']['risk_flags']:
    st.warning(f"⚠️ **Risk Flags Detected**: {', '.join(prediction['totals']['risk_flags'])}")

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

# ========== FIXED FEEDBACK SECTION ==========
record_outcome_with_feedback(prediction, winner_decision, totals_decision, home_team, away_team)

# ========== BETTING SYSTEM INSIGHTS ==========
with st.expander("💰 Your Betting System Insights", expanded=True):
    insights = st.session_state.betting_system.generate_learned_insights()
    for insight in insights:
        st.write(f"• {insight}")
    
    # Show strongest learned patterns
    st.subheader("📊 Your Proven Patterns")
    patterns = dict(st.session_state.betting_system.pattern_memory)
    proven_patterns = sorted(
        [(k, v['success']/v['total']) for k, v in patterns.items() if v['total'] >= 3 and v['success']/v['total'] >= 0.7],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    if proven_patterns:
        for pattern, success in proven_patterns:
            st.success(f"**{pattern[:40]}...**: {success:.0%} success ({patterns[pattern]['success']}/{patterns[pattern]['total']})")
    
    # Show failing patterns
    failing_patterns = sorted(
        [(k, v['success']/v['total']) for k, v in patterns.items() if v['total'] >= 3 and v['success']/v['total'] <= 0.3],
        key=lambda x: x[1]
    )[:5]
    
    if failing_patterns:
        st.subheader("💣 Your Failing Patterns (Bet Opposite!)")
        for pattern, success in failing_patterns:
            st.error(f"**{pattern[:40]}...**: {success:.0%} success → BET OPPOSITE")

# ========== EXPORT REPORT ==========
st.divider()
st.subheader("📤 Export Betting Report")

report = f"""
💰 FOOTBALL BETTING SYSTEM - YOUR PERSONAL EDGE
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Storage: {'Supabase Connected' if st.session_state.betting_system.supabase else 'Local Storage Only'}

🎯 BETTING DECISIONS (Based on YOUR Results):
Winner: {winner_decision['bet']} ({winner_decision['confidence']:.0f}% confidence)
Reason: {winner_decision['reason']}
Decision Type: {winner_decision['type']}

Totals: {totals_decision['bet']} 2.5 ({totals_decision['confidence']:.0f}% confidence)
Reason: {totals_decision['reason']}
Decision Type: {totals_decision['type']}

🎯 BETTING CARD RECOMMENDATION:
{recommendation['icon']} {recommendation['text']}
Type: {recommendation['subtext']}
Confidence: {recommendation['confidence']:.0f}/100
Expected Value: {recommendation.get('expected_value', 0):.3f}
Reason: {recommendation['reason']}

🤖 ALGORITHM PREDICTION (For Reference):
Winner: {prediction['winner']['team']} ({prediction['winner']['probability']*100:.1f}%)
Confidence: {prediction['winner']['confidence']} ({prediction['winner']['confidence_score']:.0f}/100)

Totals: {prediction['totals']['direction']} 2.5 ({prediction['probabilities'][f'{prediction['totals']['direction'].lower()}_2_5_probability']*100:.1f}%)
Confidence: {prediction['totals']['confidence']} ({prediction['totals']['confidence_score']:.0f}/100)

📊 PATTERN ANALYSIS:
Winner Pattern: {pattern_indicators['winner']['text']}
Winner Explanation: {pattern_indicators['winner']['explanation']}

Totals Pattern: {pattern_indicators['totals']['text']}
Totals Explanation: {pattern_indicators['totals']['explanation']}

⚽ EXPECTED GOALS:
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

📊 FINISHING TRENDS:
{home_team}: {prediction['totals']['home_finishing']:+.2f} goals_vs_xg/game
{away_team}: {prediction['totals']['away_finishing']:+.2f} goals_vs_xg/game

⚠️ RISK FLAGS:
{', '.join(prediction['totals']['risk_flags']) if prediction['totals']['risk_flags'] else 'None'}

💰 YOUR BETTING SYSTEM STATS:
Your Outcomes Recorded: {len(st.session_state.betting_system.outcomes)}
Your Patterns Learned: {len(st.session_state.betting_system.pattern_memory)}
Storage Status: {'✅ Connected to Supabase' if st.session_state.betting_system.supabase else '⚠️ Local storage only'}

---
YOUR BETTING RULES:
1. Proven Patterns (≥70% success with ≥3 matches) → BET
2. Failing Patterns (≤30% success with ≥3 matches) → BET OPPOSITE
3. New/Mixed Patterns → Use algorithm with caution
4. All decisions based on YOUR actual historical results
"""

st.code(report, language="text")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"betting_{home_team}_vs_{away_team}.txt",
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
            'betting_decisions': {
                'winner': winner_decision,
                'totals': totals_decision
            },
            'algorithm_prediction': prediction,
            'betting_card': recommendation
        })
        st.success("Added to prediction history!")
