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

# ========== STATE MANAGEMENT ==========
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None

if 'current_teams' not in st.session_state:
    st.session_state.current_teams = None

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine v4.0",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine v4.0")
st.markdown("""
    **ADAPTIVE LEARNING SYSTEM** - Learns from historical outcomes to improve predictions
    *Pure Learning from Your Recorded Outcomes*
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

# ========== LEARNING SYSTEM ==========

class AdaptiveLearningSystem:
    """Machine Learning system that adapts based on YOUR historical results"""
    
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
        
        # Load ONLY from Supabase (NO pre-loaded test data)
        self.load_learning()
    
    def save_learning(self):
        """Save learning data to Supabase"""
        try:
            if not self.supabase:
                # Fallback to local storage
                self._save_learning_local()
                return
            
            # Save each pattern to Supabase
            for pattern_key, stats in self.pattern_memory.items():
                # Skip if no data
                if stats['total'] == 0:
                    continue
                    
                data = {
                    "pattern_key": pattern_key,
                    "total_matches": stats['total'],
                    "successful_matches": stats['success'],
                    "metadata": {
                        "feature_weights": self.feature_weights,
                        "last_updated": datetime.now().isoformat(),
                        "success_rate": stats['success'] / stats['total'] if stats['total'] > 0 else 0
                    }
                }
                
                # Upsert (insert or update) the pattern
                try:
                    self.supabase.table("football_learning").upsert(data, on_conflict="pattern_key").execute()
                except Exception as e:
                    st.error(f"Supabase upsert error: {e}")
                    self._save_learning_local()
                    return
            
            # Save outcomes as metadata
            if self.outcomes:
                outcomes_data = {
                    "pattern_key": "ALL_OUTCOMES",
                    "total_matches": len(self.outcomes),
                    "successful_matches": sum(1 for o in self.outcomes if o['winner_correct'] and o['totals_correct']),
                    "metadata": {
                        "outcomes": self.outcomes[-1000:],  # Keep last 1000 outcomes
                        "outcome_count": len(self.outcomes),
                        "feature_weights": self.feature_weights,
                        "saved_at": datetime.now().isoformat()
                    }
                }
                try:
                    self.supabase.table("football_learning").upsert(outcomes_data, on_conflict="pattern_key").execute()
                except Exception as e:
                    st.error(f"Supabase outcomes save error: {e}")
            
        except Exception as e:
            st.error(f"Error saving to Supabase: {e}")
            # Fallback to local storage
            self._save_learning_local()
    
    def _save_learning_local(self):
        """Fallback local storage"""
        try:
            with open("learning_data.pkl", "wb") as f:
                pickle.dump({
                    'pattern_memory': dict(self.pattern_memory),
                    'feature_weights': self.feature_weights,
                    'outcomes': self.outcomes
                }, f)
        except Exception as e:
            st.error(f"Local save failed: {e}")
    
    def load_learning(self):
        """Load learning data from Supabase"""
        try:
            if not self.supabase:
                # Fallback to local storage
                self._load_learning_local()
                return
            
            # Load patterns from Supabase
            response = self.supabase.table("football_learning").select("*").execute()
            
            if not response.data:
                # Fresh start - no previous data
                return
            
            patterns_loaded = 0
            outcomes_loaded = 0
            
            for row in response.data:
                pattern_key = row['pattern_key']
                
                if pattern_key == "ALL_OUTCOMES":
                    # Load outcomes
                    if 'metadata' in row and row['metadata']:
                        metadata = row['metadata']
                        if 'outcomes' in metadata:
                            self.outcomes = metadata['outcomes']
                            outcomes_loaded = len(self.outcomes)
                        if 'feature_weights' in metadata:
                            self.feature_weights.update(metadata['feature_weights'])
                else:
                    # Load pattern stats
                    self.pattern_memory[pattern_key] = {
                        'total': row['total_matches'],
                        'success': row['successful_matches']
                    }
                    patterns_loaded += 1
            
        except Exception as e:
            # Fallback to local storage
            self._load_learning_local()
    
    def _load_learning_local(self):
        """Fallback local storage"""
        try:
            if os.path.exists("learning_data.pkl"):
                with open("learning_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.pattern_memory = defaultdict(lambda: {'total': 0, 'success': 0}, data['pattern_memory'])
                    self.feature_weights = data['feature_weights']
                    self.outcomes = data['outcomes']
        except:
            pass
    
    def record_outcome(self, prediction, pattern_indicators, actual_result, actual_score):
        """Record a match outcome for learning"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Determine actual outcomes
        home_goals, away_goals = map(int, actual_score.split('-'))
        
        # Winner outcome
        if home_goals > away_goals:
            actual_winner = "HOME"
        elif away_goals > home_goals:
            actual_winner = "AWAY"
        else:
            actual_winner = "DRAW"
        
        # Totals outcome
        total_goals = home_goals + away_goals
        actual_over = total_goals > 2.5
        
        # Store outcome
        outcome = {
            'timestamp': datetime.now(),
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
        
        # Create pattern keys based on YOUR match
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
        
        # AUTO-SAVE TO SUPABASE
        self.save_learning()
        
        return outcome
    
    def _adjust_weights(self, outcome):
        """Adjust feature weights based on outcome success"""
        # If prediction was wrong, reduce weight of relevant features
        if not outcome['totals_correct']:
            if 'HIGH_OVER' in str(outcome.get('finishing_alignment', '')):
                self.feature_weights['finishing_alignment'] *= 0.9  # Reduce weight
            if 'HIGH_VARIANCE_TEAM' in outcome.get('risk_flags', []):
                self.feature_weights['risk_flags'] *= 0.95
            if outcome['totals_confidence'] < 50:
                self.feature_weights['confidence_score'] *= 1.1  # Increase weight for low confidence
        
        # If prediction was correct, increase weight of relevant features
        if outcome['totals_correct']:
            if 'MED_OVER' in str(outcome.get('finishing_alignment', '')):
                self.feature_weights['finishing_alignment'] *= 1.05
            if 'MED_UNDER' in str(outcome.get('finishing_alignment', '')):
                self.feature_weights['finishing_alignment'] *= 1.1  # Strong pattern
            if outcome['totals_confidence'] > 80:
                self.feature_weights['confidence_score'] *= 1.05
    
    def get_pattern_success_rate(self, pattern_type, pattern_subtype=None):
        """Get historical success rate for a pattern"""
        key = f"{pattern_type}_{pattern_subtype}" if pattern_subtype else pattern_type
        memory = self.pattern_memory
        
        # Look for exact matches first
        exact_keys = [k for k in memory if key in k]
        if exact_keys:
            total = sum(memory[k]['total'] for k in exact_keys)
            success = sum(memory[k]['success'] for k in exact_keys)
            if total > 0:
                return success / total
        
        # Look for similar patterns
        similar_keys = [k for k in memory if pattern_type in k]
        if similar_keys:
            total = sum(memory[k]['total'] for k in similar_keys)
            success = sum(memory[k]['success'] for k in similar_keys)
            if total > 0:
                return success / total
        
        return 0.5  # Default if no data
    
    def adjust_confidence(self, original_confidence, pattern_type, context):
        """Adjust confidence based on historical performance"""
        base_success = self.get_pattern_success_rate(pattern_type, context.get('subtype'))
        
        if base_success > 0.7:  # Strong pattern
            adjustment = min(20, (base_success - 0.7) * 100)
            return min(100, original_confidence + adjustment)
        elif base_success < 0.4:  # Weak pattern
            adjustment = min(30, (0.4 - base_success) * 100)
            return max(10, original_confidence - adjustment)
        else:
            return original_confidence
    
    def generate_learned_insights(self):
        """Generate insights based on learned patterns"""
        insights = []
        
        if not self.outcomes:
            return ["🔄 **Learning System**: No historical data yet - record outcomes to start learning"]
        
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
                    insights.append(f"✅ **YOUR STRONG PATTERN**: {pattern} - {stats['success']}/{stats['total']} correct")
                elif success_rate <= 0.3:
                    insights.append(f"❌ **YOUR WEAK PATTERN**: {pattern} - {stats['success']}/{stats['total']} correct")
        
        # Confidence level analysis
        high_conf = [o for o in recent if o['totals_confidence'] >= 70]
        if high_conf:
            high_conf_success = sum(1 for o in high_conf if o['totals_correct']) / len(high_conf)
            insights.append(f"🎯 **Your High Confidence (70+)**: {high_conf_success:.0%} success rate")
        
        return insights[:5]

# ========== ORIGINAL CORE FUNCTIONS ==========

# Initialize session states
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

if 'learning_system' not in st.session_state:
    st.session_state.learning_system = AdaptiveLearningSystem()

if 'match_history' not in st.session_state:
    st.session_state.match_history = []

if 'show_history' not in st.session_state:
    st.session_state.show_history = False

if 'last_outcome' not in st.session_state:
    st.session_state.last_outcome = None

if 'show_feedback_message' not in st.session_state:
    st.session_state.show_feedback_message = False

def factorial_cache(n):
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    """Poisson probability mass function"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

# ========== ORIGINAL CORE CLASSES ==========

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
    """FIXED LOGIC: Accounts for finishing ability in winner determination"""
    
    def predict_winner(self, home_xg, away_xg, home_stats, away_stats):
        """OUR IMPROVED LOGIC: Winner determination with finishing adjustment"""
        
        # Get finishing trends (goals_vs_xg per match)
        home_finishing = home_stats['goals_vs_xg_pm']  # e.g., +0.42, -0.10
        away_finishing = away_stats['goals_vs_xg_pm']
        
        # Get defensive performance (goals_allowed_vs_xga per match)
        home_defense = home_stats['goals_allowed_vs_xga_pm']  # negative = good defense
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        # ========== KEY FIX: ADJUST xG FOR FINISHING ABILITY ==========
        # Teams that finish well get boosted xG
        # Teams that waste chances get reduced xG
        
        home_adjusted_xg = home_xg + home_finishing - away_defense
        away_adjusted_xg = away_xg + away_finishing - home_defense
        
        # Calculate adjusted delta
        delta = home_adjusted_xg - away_adjusted_xg
        
        # ========== DETERMINE VOLATILITY FLAG ==========
        # Check for high-variance matchups
        volatility_high = False
        if abs(home_finishing) > 0.3 and abs(away_finishing) > 0.3:
            volatility_high = True  # Both extreme finishers
        elif home_finishing > 0.3 and away_finishing > 0.3:
            volatility_high = True  # Both clinical
        elif home_finishing < -0.3 and away_finishing < -0.3:
            volatility_high = True  # Both wasteful
        
        # ========== WINNER DETERMINATION WITH FINISHING AWARENESS ==========
        # Use adjusted delta, not raw delta
        
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
                # Close game with high volatility = very uncertain
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
        
        # ========== CONFIDENCE CALCULATION (ADJUSTED) ==========
        # Use adjusted values for confidence too
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
    """OUR IMPROVED LOGIC: Totals prediction with defense quality rules"""
    
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
    
    def check_defense_quality_rules(self, home_stats, away_stats):
        """NEW: Defense quality rules based on proven patterns"""
        home_def = home_stats['goals_allowed_vs_xga_pm']
        away_def = away_stats['goals_allowed_vs_xga_pm']
        
        # RULE 1: Double bad defense = OVER 2.5
        if home_def >= 0.5 and away_def >= 0.5:
            min_def = min(home_def, away_def)
            if min_def >= 2.0:
                confidence = 80  # VERY HIGH confidence for double VERY bad defense
                reason = f"DOUBLE VERY BAD DEFENSE: Home({home_def:.2f}) + Away({away_def:.2f}) = High scoring guaranteed"
            else:
                confidence = 70  # HIGH confidence for double bad defense
                reason = f"DOUBLE BAD DEFENSE: Home({home_def:.2f}) + Away({away_def:.2f}) = High scoring likely"
            
            return {
                'direction': "OVER",
                'confidence': confidence,
                'reason': reason,
                'rule_triggered': 'DOUBLE_BAD_DEFENSE'
            }
        
        # RULE 2: Good defense present = UNDER 2.5
        if home_def <= -0.5 or away_def <= -0.5:
            # Check if both have good defense
            if home_def <= -0.5 and away_def <= -0.5:
                confidence = 85  # VERY HIGH confidence for double good defense
                reason = f"DOUBLE GOOD DEFENSE: Home({home_def:.2f}) + Away({away_def:.2f}) = Low scoring guaranteed"
            elif home_def <= -2.0 or away_def <= -2.0:
                confidence = 80  # VERY HIGH confidence for very good defense
                reason = f"VERY GOOD DEFENSE PRESENT: Home({home_def:.2f}) Away({away_def:.2f}) = Low scoring likely"
            else:
                confidence = 70  # HIGH confidence for good defense present
                reason = f"GOOD DEFENSE PRESENT: Home({home_def:.2f}) Away({away_def:.2f}) = Low scoring likely"
            
            return {
                'direction': "UNDER",
                'confidence': confidence,
                'reason': reason,
                'rule_triggered': 'GOOD_DEFENSE_PRESENT'
            }
        
        return None
    
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
        """OUR IMPROVED LOGIC: Complete totals prediction with defense rules"""
        total_xg = home_xg + away_xg
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        # ========== NEW: CHECK DEFENSE QUALITY RULES FIRST ==========
        defense_rule = self.check_defense_quality_rules(home_stats, away_stats)
        if defense_rule:
            # Apply defense rule, then adjust with existing logic
            direction = defense_rule['direction']
            base_confidence = defense_rule['confidence']
            rule_reason = defense_rule['reason']
            rule_triggered = defense_rule['rule_triggered']
            
            # Still calculate finishing alignment for insights
            finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
            total_category = self.categorize_total_xg(total_xg)
            risk_flags = self.check_risk_flags(home_stats, away_stats, total_xg)
            
            # Adjust confidence based on risk flags
            final_confidence = base_confidence
            for flag in risk_flags:
                if flag == "VOLATILE_OVER_BOTH":
                    if direction == "OVER":
                        final_confidence -= 10  # Reduce confidence for OVER with volatile teams
                    else:
                        final_confidence += 5   # Increase confidence for UNDER with volatile teams
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
        
        # ========== ORIGINAL LOGIC (if no defense rule triggered) ==========
        over_threshold = self.league_adjustments['over_threshold']
        base_direction = "OVER" if total_xg > over_threshold else "UNDER"
        
        # OUR LOGIC: Finishing alignment
        finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
        total_category = self.categorize_total_xg(total_xg)
        
        # OUR LOGIC: Risk flags
        risk_flags = self.check_risk_flags(home_stats, away_stats, total_xg)
        
        # ========== NEW: PROVEN PATTERN 1 ==========
        # AUTO-UNDER RULE: NEUTRAL + HIGH_xG = UNDER (3/3 proven)
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
        
        # OUR IMPROVED LOGIC: Decision matrix with MED_UNDER fix
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
    """OUR IMPROVED LOGIC: Generate enhanced insights with defense rules"""
    
    @staticmethod
    def generate_insights(winner_prediction, totals_prediction):
        insights = []
        
        # Winner insights
        if winner_prediction.get('winner_confidence_category') == "VERY HIGH":
            insights.append(f"🎯 **High Confidence Winner**: Model strongly favors {winner_prediction.get('predicted_winner', 'N/A')}")
        elif winner_prediction.get('winner_confidence_category') == "LOW":
            insights.append(f"⚠️ **Low Confidence Winner**: Exercise caution on {winner_prediction.get('predicted_winner', 'N/A')} prediction (0/3 in backtests)")
        
        # NEW: Defense rule insights
        defense_rule = totals_prediction.get('defense_rule_triggered')
        if defense_rule == 'DOUBLE_BAD_DEFENSE':
            insights.append(f"⚡ **DOUBLE BAD DEFENSE**: Both teams allow more goals than expected → HIGH SCORING likely")
        elif defense_rule == 'GOOD_DEFENSE_PRESENT':
            insights.append(f"🛡️ **GOOD DEFENSE PRESENT**: At least one team limits goals well → LOW SCORING likely")
        elif defense_rule == 'NEUTRAL_HIGH_XG_UNDER':
            insights.append(f"📉 **PROVEN PATTERN**: NEUTRAL finishing + HIGH xG = UNDER (3/3 in tests)")
        
        # Volatility insight
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
        
        # Defense quality insight (from winner prediction)
        home_def = winner_prediction.get('home_defense_quality', 0)
        away_def = winner_prediction.get('away_defense_quality', 0)
        
        if home_def >= 0.5 and away_def >= 0.5:
            insights.append(f"🚨 **Both teams have poor defense**: High scoring expected")
        elif home_def <= -0.5 or away_def <= -0.5:
            insights.append(f"✅ **Strong defense detected**: Could limit scoring")
        
        return insights[:8]

# ========== ADAPTIVE FOOTBALL ENGINE ==========

class AdaptiveFootballIntelligenceEngineV4:
    """Version 4 with adaptive learning capabilities"""
    
    def __init__(self, league_metrics, league_name, learning_system=None):
        self.league_metrics = league_metrics
        self.league_name = league_name
        self.learning_system = learning_system or AdaptiveLearningSystem()
        
        # Initialize predictors
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
        self.insights_generator = InsightsGenerator()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate prediction with adaptive learning adjustments"""
        
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
        
        # Apply learning adjustments
        winner_prediction = self._adjust_with_learning(winner_prediction, 'winner', home_stats, away_stats)
        totals_prediction = self._adjust_with_learning(totals_prediction, 'totals', home_stats, away_stats)
        
        # Generate insights including learned patterns
        insights = self.insights_generator.generate_insights(winner_prediction, totals_prediction)
        learned_insights = self.learning_system.generate_learned_insights()
        insights.extend(learned_insights)
        
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
    
    def _adjust_with_learning(self, prediction, pred_type, home_stats, away_stats):
        """Apply learning-based adjustments to predictions"""
        if not self.learning_system:
            return prediction
        
        if pred_type == 'totals':
            # Adjust confidence based on historical performance of similar patterns
            context = {
                'finishing_alignment': prediction.get('finishing_alignment'),
                'total_category': prediction.get('total_category'),
                'subtype': f"{prediction.get('finishing_alignment', 'N/A')}_{prediction.get('total_category', 'N/A')}"
            }
            
            # Apply confidence adjustment
            original_conf = prediction.get('confidence_score', 50)
            adjusted_conf = self.learning_system.adjust_confidence(
                original_conf, 
                prediction.get('finishing_alignment', 'NEUTRAL'),
                context
            )
            
            # Update prediction with adjusted confidence
            prediction['confidence_score'] = adjusted_conf
            
            # Update confidence category
            if adjusted_conf >= 75:
                prediction['confidence'] = "VERY HIGH"
            elif adjusted_conf >= 65:
                prediction['confidence'] = "HIGH"
            elif adjusted_conf >= 55:
                prediction['confidence'] = "MEDIUM"
            elif adjusted_conf >= 45:
                prediction['confidence'] = "LOW"
            else:
                prediction['confidence'] = "VERY LOW"
        
        elif pred_type == 'winner':
            # Adjust winner confidence based on volatility patterns
            if prediction.get('volatility_high', False):
                # Historical data shows volatile matches are less predictable
                original_conf = prediction.get('confidence_score', 50)
                prediction['confidence_score'] = max(30, original_conf - 15)
                
                if prediction['confidence_score'] >= 75:
                    prediction['winner_confidence_category'] = "VERY HIGH"
                elif prediction['confidence_score'] >= 65:
                    prediction['winner_confidence_category'] = "HIGH"
                elif prediction['confidence_score'] >= 55:
                    prediction['winner_confidence_category'] = "MEDIUM"
                elif prediction['confidence_score'] >= 45:
                    prediction['winner_confidence_category'] = "LOW"
                else:
                    prediction['winner_confidence_category'] = "VERY LOW"
        
        return prediction

# ========== ADAPTIVE PATTERN INDICATORS ==========

class AdaptivePatternIndicators:
    """Generate pattern indicators with learned adjustments"""
    
    def __init__(self, learning_system):
        self.learning_system = learning_system
    
    def generate_indicators(self, prediction):
        """Generate pattern indicators with learned success rates"""
        indicators = {'winner': None, 'totals': None}
        
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # WINNER INDICATORS with learning
        winner_success_rate = self.learning_system.get_pattern_success_rate(
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
            vol_success = self.learning_system.get_pattern_success_rate("VOLATILE", "HIGH_VOLATILITY")
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
        
        # TOTALS INDICATORS with learning
        finishing_alignment = totals_pred.get('finishing_alignment', 'NEUTRAL')
        total_category = totals_pred.get('total_category', 'N/A')
        
        pattern_key = f"{finishing_alignment}_{total_category}"
        pattern_success = self.learning_system.get_pattern_success_rate("TOTALS", pattern_key)
        
        # Determine based on YOUR success rates
        if pattern_success > 0.7 and self.learning_system.pattern_memory.get(pattern_key, {}).get('total', 0) >= 3:
            indicators['totals'] = {
                'type': 'MET',
                'color': 'green',
                'text': f'YOUR STRONG PATTERN - {totals_pred["direction"]} 2.5',
                'explanation': f'Your historical success: {pattern_success:.0%} for this pattern'
            }
        elif pattern_success < 0.4 and self.learning_system.pattern_memory.get(pattern_key, {}).get('total', 0) >= 3:
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

# ========== ADAPTIVE BETTING CARD ==========

class AdaptiveBettingCard:
    """Betting card that adapts based on learned patterns"""
    
    def __init__(self, learning_system):
        self.learning_system = learning_system
    
    def get_recommendation(self, prediction, pattern_indicators):
        """Get betting recommendation with learned adjustments"""
        
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Calculate expected value based on learned success rates
        winner_ev = self._calculate_expected_value(winner_pred, pattern_indicators['winner'], 'winner')
        totals_ev = self._calculate_expected_value(totals_pred, pattern_indicators['totals'], 'totals')
        
        # Determine best bet based on expected value
        if winner_ev > 0.1 and totals_ev > 0.1:
            min_conf = min(winner_pred['confidence_score'], totals_pred['confidence_score'])
            return {
                'type': 'combo',
                'text': f"🎯 {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                'confidence': min_conf,
                'color': '#10B981',
                'icon': '🎯',
                'subtext': 'DOUBLE BET (HIGH EV)',
                'reason': f'Winner EV: {winner_ev:.2f} | Totals EV: {totals_ev:.2f}',
                'expected_value': (winner_ev + totals_ev) / 2
            }
        elif winner_ev > 0.15:
            return {
                'type': 'single',
                'text': f"🏆 {winner_pred['team']} to win",
                'confidence': winner_pred['confidence_score'],
                'color': '#3B82F6',
                'icon': '🏆',
                'subtext': 'WINNER BET',
                'reason': f'Expected Value: {winner_ev:.2f}',
                'expected_value': winner_ev
            }
        elif totals_ev > 0.15:
            return {
                'type': 'single',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'confidence': totals_pred['confidence_score'],
                'color': '#8B5CF6',
                'icon': '📈',
                'subtext': 'TOTALS BET',
                'reason': f'Expected Value: {totals_ev:.2f}',
                'expected_value': totals_ev
            }
        else:
            return {
                'type': 'none',
                'text': "🚫 No Value Bet",
                'confidence': max(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#6B7280',
                'icon': '🤔',
                'subtext': 'NO BET',
                'reason': f'Insufficient expected value (Winner: {winner_ev:.2f}, Totals: {totals_ev:.2f})',
                'expected_value': 0
            }
    
    def _calculate_expected_value(self, prediction, pattern_indicator, market_type):
        """Calculate expected value based on learned probabilities"""
        if pattern_indicator['type'] == 'AVOID':
            return -0.5  # Strong avoid
        
        # Get historical success rate
        if market_type == 'winner':
            success_rate = self.learning_system.get_pattern_success_rate(
                "WINNER", 
                f"{prediction['confidence']}_{prediction['confidence_score']//10*10}"
            )
            # Typical odds for winner bets
            implied_odds = 1 / prediction['probability'] if prediction['probability'] > 0 else 3.0
        else:
            finishing_alignment = prediction.get('finishing_alignment', 'NEUTRAL')
            total_category = prediction.get('total_category', 'N/A')
            success_rate = self.learning_system.get_pattern_success_rate(
                "TOTALS", 
                f"{finishing_alignment}_{total_category}"
            )
            # Typical odds for totals bets
            implied_odds = 1.9  # Average odds of ~1.9 for Over/Under
        
        # Calculate expected value
        ev = (success_rate * (implied_odds - 1)) - ((1 - success_rate) * 1)
        
        # Adjust for confidence
        confidence_factor = prediction['confidence_score'] / 100
        ev *= confidence_factor
        
        return ev
    
    def display_card(self, recommendation):
        """Display the adaptive betting card"""
        ev = recommendation.get('expected_value', 0)
        
        # Color based on expected value
        if ev > 0.2:
            color = '#10B981'  # Green for high EV
        elif ev > 0.1:
            color = '#3B82F6'  # Blue for medium EV
        elif ev > 0:
            color = '#8B5CF6'  # Purple for low EV
        else:
            color = '#6B7280'  # Gray for negative EV
        
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
                Confidence: {recommendation['confidence']:.0f}/100 | EV: {ev:.3f}
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

# ========== FEEDBACK SYSTEM ==========

def add_feedback_section(prediction, pattern_indicators, home_team, away_team):
    """Add section for recording actual outcomes WITHOUT page refresh"""
    st.divider()
    st.subheader("📝 Record Outcome for Learning")
    
    # Create a unique container for feedback
    feedback_container = st.container()
    
    # Use columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        actual_score = st.text_input("Actual Score (e.g., 2-1)", "", 
                                   help="Enter the actual match result",
                                   key=f"score_{home_team}_{away_team}")
    
    with col2:
        record_button = st.button("✅ Record Outcome & Learn", 
                                type="primary", 
                                use_container_width=True,
                                key=f"record_{home_team}_{away_team}")
    
    # Handle button click WITHOUT using forms
    if record_button:
        if actual_score and '-' in actual_score:
            try:
                home_goals, away_goals = map(int, actual_score.split('-'))
                
                # Record outcome
                outcome = st.session_state.learning_system.record_outcome(
                    prediction, pattern_indicators, "", actual_score
                )
                
                # Store in session state
                st.session_state.last_outcome = outcome
                st.session_state.show_feedback_message = True
                
                # Add to match history
                st.session_state.match_history.append({
                    'timestamp': datetime.now(),
                    'home_team': home_team,
                    'away_team': away_team,
                    'prediction': prediction,
                    'actual_score': actual_score,
                    'winner_correct': outcome['winner_correct'],
                    'totals_correct': outcome['totals_correct']
                })
                
                # Show success in the feedback container
                with feedback_container:
                    st.success(f"""
                    ✅ **Outcome Recorded!**  
                    **Match**: {home_team} vs {away_team}  
                    **Actual Score**: {actual_score}  
                    **Winner**: {'✅ Correct' if outcome['winner_correct'] else '❌ Wrong'}  
                    **Totals**: {'✅ Correct' if outcome['totals_correct'] else '❌ Wrong'}
                    """)
                    
                    # Show learning update
                    with st.expander("📈 What was learned?", expanded=True):
                        winner_success = st.session_state.learning_system.get_pattern_success_rate(
                            "WINNER", f"{prediction['winner']['confidence']}_{prediction['winner']['confidence_score']//10*10}"
                        )
                        totals_success = st.session_state.learning_system.get_pattern_success_rate(
                            "TOTALS", f"{prediction['totals'].get('finishing_alignment', 'N/A')}_{prediction['totals'].get('total_category', 'N/A')}"
                        )
                        
                        st.write(f"**Winner Pattern**: {prediction['winner']['confidence']} confidence")
                        st.write(f"**Winner Success Rate**: {winner_success:.0%}")
                        st.write(f"**Totals Pattern**: {prediction['totals'].get('finishing_alignment', 'N/A')} + {prediction['totals'].get('total_category', 'N/A')}")
                        st.write(f"**Totals Success Rate**: {totals_success:.0%}")
                        st.write(f"**Patterns in Database**: {len(st.session_state.learning_system.pattern_memory)}")
                        st.write(f"**Total Outcomes**: {len(st.session_state.learning_system.outcomes)}")
                
                # Force Streamlit to stop and show results
                st.stop()  # This prevents the page from going back to start
                
            except ValueError:
                st.error("❌ Please enter score in format '2-1' (numbers only)")
        else:
            st.error("❌ Please enter a valid score format '2-1'")
    
    # Show previous feedback if exists
    elif st.session_state.show_feedback_message and st.session_state.last_outcome:
        last_outcome = st.session_state.last_outcome
        with feedback_container:
            st.info(f"""
            📝 **Last Recorded Outcome**:  
            {last_outcome['home_team']} {last_outcome['actual_score']} {last_outcome['away_team']}  
            Winner: {'✅' if last_outcome['winner_correct'] else '❌'}  
            Totals: {'✅' if last_outcome['totals_correct'] else '❌'}
            """)
    
    st.caption("💡 **Tip**: Enter the actual match result to help the system learn.")

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

    # Learning System Section
    st.divider()
    st.header("📚 Learning System")
    
    # Supabase Status
    st.write("🔄 **Storage**: Supabase")
    st.write(f"📊 **Your Patterns**: {len(st.session_state.learning_system.pattern_memory)}")
    st.write(f"📈 **Your Outcomes**: {len(st.session_state.learning_system.outcomes)}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Learning", use_container_width=True):
            st.session_state.learning_system.save_learning()
            st.success("Learning data saved to Supabase!")
    
    with col2:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            # Reload learning data
            st.session_state.learning_system.load_learning()
            st.success("Learning stats refreshed!")
    
    # Clear feedback message button
    if st.session_state.show_feedback_message:
        if st.button("🗑️ Clear Feedback", type="secondary", use_container_width=True):
            st.session_state.show_feedback_message = False
            st.session_state.last_outcome = None
            st.rerun()
    
    st.divider()
    
    # Show learning statistics
    st.subheader("Your Learning Statistics")
    total_outcomes = len(st.session_state.learning_system.outcomes)
    if total_outcomes > 0:
        recent = st.session_state.learning_system.outcomes[-10:] if len(st.session_state.learning_system.outcomes) >= 10 else st.session_state.learning_system.outcomes
        if recent:
            winner_acc = sum(1 for o in recent if o['winner_correct']) / len(recent)
            totals_acc = sum(1 for o in recent if o['totals_correct']) / len(recent)
            
            st.metric("Your Total Matches", total_outcomes)
            st.metric("Your Recent Winner Acc", f"{winner_acc:.0%}")
            st.metric("Your Recent Totals Acc", f"{totals_acc:.0%}")
            
            # Show top patterns
            st.subheader("Your Top Patterns")
            patterns = dict(st.session_state.learning_system.pattern_memory)
            sorted_patterns = sorted(
                [(k, v['success']/v['total']) for k, v in patterns.items() if v['total'] >= 3],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for pattern, success in sorted_patterns:
                st.caption(f"`{pattern[:30]}...`: {success:.0%}")
    else:
        st.info("No outcomes recorded yet. Record your first match outcome!")

if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

if 'calculate_btn' in locals() and calculate_btn:
    show_prediction = True
    # Store the prediction in session state
    st.session_state.prediction_made = True 

# Check 2: Do we have a stored prediction from before?
elif st.session_state.prediction_made and st.session_state.current_prediction:
    show_prediction = True
    # Use stored prediction
    prediction = st.session_state.current_prediction
    home_team, away_team = st.session_state.current_teams
    # Skip generating new prediction, use stored one

# If no prediction to show, display start screen
if not show_prediction:
    st.info("👈 Select teams and click 'Generate Prediction'")
    
    # Show learning insights even when no prediction
    with st.expander("🧠 Learning System Insights", expanded=True):
        insights = st.session_state.learning_system.generate_learned_insights()
        for insight in insights:
            st.write(f"• {insight}")
    
    # Show history if requested
    if st.session_state.show_history and st.session_state.match_history:
        st.subheader("📊 Your Learning History")
        for hist in reversed(st.session_state.match_history[-10:]):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                with col1:
                    st.write(f"**{hist['home_team']} vs {hist['away_team']}**")
                    st.caption(f"{hist['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    st.write(f"Predicted: {hist['prediction']['winner']['team']}")
                    st.caption(f"Actual: {hist['actual_score']}")
                with col3:
                    st.write(f"Winner: {'✅' if hist['winner_correct'] else '❌'}")
                with col4:
                    st.write(f"Totals: {'✅' if hist['totals_correct'] else '❌'}")
                st.divider()
    
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

# Use adaptive engine
engine = AdaptiveFootballIntelligenceEngineV4(
    league_metrics, 
    selected_league, 
    st.session_state.learning_system
)

prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)

# Generate adaptive pattern indicators
pattern_generator = AdaptivePatternIndicators(st.session_state.learning_system)
pattern_indicators = pattern_generator.generate_indicators(prediction)

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
st.subheader("🎯 Your Pattern Indicators")

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

st.caption("💡 **Your Learning System**: Green = Your strong pattern | Red = Your weak pattern | Blue = Promising | Gray = No pattern yet")

# ========== ADAPTIVE BETTING CARD ==========
st.divider()
st.subheader("🎯 ADAPTIVE BETTING CARD (Based on YOUR Data)")

# Generate adaptive betting recommendation
betting_card = AdaptiveBettingCard(st.session_state.learning_system)
recommendation = betting_card.get_recommendation(prediction, pattern_indicators)

# Display the card
betting_card.display_card(recommendation)

# Show reasoning
with st.expander("🧠 Decision Logic (Based on YOUR Data)", expanded=False):
    st.write("**ADAPTIVE RULES (Learned from YOUR outcomes):**")
    st.write("1. 🟢 **YOUR STRONG PATTERNS**: >70% success with ≥3 matches → BET")
    st.write("2. 🔴 **YOUR WEAK PATTERNS**: <40% success with ≥3 matches → AVOID")
    st.write("3. 🔵 **PROMISING PATTERNS**: 60-70% success → Consider")
    st.write("4. 🤔 **NO PATTERN YET**: Check expected value (EV)")
    st.write("5. 📈 **DECISION RULE**: BET if EV > 0.15, DOUBLE BET if both markets EV > 0.10")
    
    # Show pattern details
    st.write("**For this match:**")
    st.write(f"**Winner Pattern**: {pattern_indicators['winner']['text']}")
    st.write(f"**Winner Explanation**: {pattern_indicators['winner']['explanation']}")
    st.write(f"**Totals Pattern**: {pattern_indicators['totals']['text']}")
    st.write(f"**Totals Explanation**: {pattern_indicators['totals']['explanation']}")
    
    if recommendation['type'] == 'combo':
        st.success("🎯 **DOUBLE BET** - Both markets show positive expected value based on YOUR data")
    elif recommendation['type'] == 'single':
        if 'winner' in recommendation['text']:
            st.success("🏆 **SINGLE WINNER BET** - Positive expected value based on YOUR data")
        else:
            st.success("📈 **SINGLE TOTALS BET** - Positive expected value based on YOUR data")
    else:
        st.warning("🚫 **NO BET** - Insufficient expected value based on YOUR data")

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

# ========== FEEDBACK SECTION ==========
add_feedback_section(prediction, pattern_indicators, home_team, away_team)

# ========== DETAILED ANALYSIS ==========
if show_details:
    with st.expander("🔍 Detailed Analysis", expanded=False):
        st.write("### Winner Prediction Analysis")
        st.write(f"- Expected Goals Difference: {prediction['winner'].get('strength', 'N/A')}")
        st.write(f"- Adjusted Delta: {prediction['winner'].get('adjusted_delta', 'N/A'):.2f}")
        st.write(f"- Confidence Level: {prediction['winner']['confidence']}")
        st.write(f"- Volatility High: {prediction['winner'].get('volatility_high', False)}")
        
        st.write("### Totals Prediction Analysis")
        st.write(f"- Total xG: {prediction['totals']['total_xg']:.2f}")
        st.write(f"- Finishing Alignment: {prediction['totals'].get('finishing_alignment', 'N/A')}")
        st.write(f"- Total Category: {prediction['totals'].get('total_category', 'N/A')}")
        st.write(f"- League-adjusted threshold: {LEAGUE_ADJUSTMENTS.get(selected_league, LEAGUE_ADJUSTMENTS['Premier League'])['over_threshold']}")
        
        if prediction['totals'].get('defense_rule_triggered'):
            st.write(f"- Defense Rule Triggered: {prediction['totals']['defense_rule_triggered']}")
        
        if prediction['totals']['risk_flags']:
            st.write("### Risk Analysis")
            for flag in prediction['totals']['risk_flags']:
                st.write(f"- {flag}")

# ========== LEARNING INSIGHTS PANEL ==========
with st.expander("🧠 Your Learning System Insights", expanded=True):
    insights = st.session_state.learning_system.generate_learned_insights()
    for insight in insights:
        st.write(f"• {insight}")
    
    # Show strongest learned patterns
    st.subheader("📊 Your Strongest Patterns")
    patterns = dict(st.session_state.learning_system.pattern_memory)
    strong_patterns = [(k, v) for k, v in patterns.items() if v['total'] >= 3 and v['success']/v['total'] >= 0.75]
    
    if strong_patterns:
        for pattern, stats in strong_patterns[:5]:
            success_rate = stats['success'] / stats['total']
            st.info(f"**{pattern[:40]}...**: {stats['success']}/{stats['total']} ({success_rate:.0%})")
    else:
        st.caption("Record more outcomes to identify your strong patterns")

# Show history if requested
if st.session_state.show_history and st.session_state.match_history:
    st.divider()
    st.subheader("📊 Your Learning History")
    for hist in reversed(st.session_state.match_history[-10:]):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            with col1:
                st.write(f"**{hist['home_team']} vs {hist['away_team']}**")
                st.caption(f"{hist['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            with col2:
                st.write(f"Predicted: {hist['prediction']['winner']['team']}")
                st.caption(f"Actual: {hist['actual_score']}")
            with col3:
                st.write(f"Winner: {'✅' if hist['winner_correct'] else '❌'}")
            with col4:
                st.write(f"Totals: {'✅' if hist['totals_correct'] else '❌'}")
            st.divider()

# ========== EXPORT REPORT ==========
st.divider()
st.subheader("📤 Export Prediction Report")

# Get original recommendations for report
report = f"""
⚽ FOOTBALL INTELLIGENCE ENGINE v4.0 - YOUR PERSONAL LEARNING SYSTEM
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Storage: Supabase Persistent

🎯 ADAPTIVE BETTING CARD (Based on YOUR Data)
{recommendation['icon']} {recommendation['text']}
Type: {recommendation['subtext']}
Confidence: {recommendation['confidence']:.0f}/100
Expected Value: {recommendation.get('expected_value', 0):.3f}
Reason: {recommendation['reason']}

📊 YOUR PATTERN ANALYSIS:
Winner Pattern: {pattern_indicators['winner']['text']}
Winner Explanation: {pattern_indicators['winner']['explanation']}
Winner Confidence: {prediction['winner']['confidence_score']:.0f}/100 ({prediction['winner']['confidence']})
Winner Volatility: {'HIGH' if prediction['winner'].get('volatility_high') else 'NORMAL'}

Totals Pattern: {pattern_indicators['totals']['text']}
Totals Explanation: {pattern_indicators['totals']['explanation']}
Totals Confidence: {prediction['totals']['confidence_score']:.0f}/100 ({prediction['totals']['confidence']})

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
Finishing Alignment: {prediction['totals'].get('finishing_alignment', 'N/A')}
Total xG Category: {prediction['totals'].get('total_category', 'N/A')}

📊 EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

📊 FINISHING TRENDS
{home_team}: {prediction['totals']['home_finishing']:+.2f} goals_vs_xg/game
{away_team}: {prediction['totals']['away_finishing']:+.2f} goals_vs_xg/game

⚠️ RISK FLAGS
{', '.join(prediction['totals']['risk_flags']) if prediction['totals']['risk_flags'] else 'None'}

🧠 YOUR LEARNING SYSTEM STATS
Your Outcomes Recorded: {len(st.session_state.learning_system.outcomes)}
Your Patterns Learned: {len(st.session_state.learning_system.pattern_memory)}
Storage: Supabase

---
YOUR ADAPTIVE LEARNING RULES:
1. Strong Patterns (>70% success with ≥3 matches) → BET
2. Weak Patterns (<40% success with ≥3 matches) → AVOID
3. Decision based on Expected Value (EV > 0.15 for single, > 0.10 for double)
"""

st.code(report, language="text")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"adaptive_{home_team}_vs_{away_team}.txt",
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
            'adaptive_recommendation': recommendation
        })
        st.success("Added to prediction history!")

# Show prediction history
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
                    if 'adaptive_recommendation' in hist:
                        unified = hist['adaptive_recommendation']
                        st.write(f"🎯 {unified['subtext']}")
                        st.caption(f"{unified['icon']} {unified['text'][:20]}...")
                st.divider()
