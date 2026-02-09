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
    page_title="⚽ Football Intelligence Engine v5.0",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine v5.0")
st.markdown("""
    **ALGORITHMIC BIAS EXPLOITER** - Learns where the algorithm is consistently wrong and bets opposite
    *Your Discovery: <40% success = BET OPPOSITE with 85% confidence*
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

# ========== IMPROVED LEARNING SYSTEM ==========

class AdaptiveLearningSystemV2:
    """YOUR IMPROVED VERSION: Bet opposite when algorithm is consistently wrong"""
    
    def __init__(self):
        self.pattern_memory = defaultdict(lambda: {'total': 0, 'success': 0})
        self.outcomes = []
        self.supabase = init_supabase()
        
        # YOUR BETTING RULES
        self.pattern_thresholds = {
            'min_matches': 3,           # Need at least 3 matches
            'strong_success': 0.7,      # 70%+ → BET THIS STRONGLY
            'weak_success': 0.4,        # <40% → BET OPPOSITE
            'neutral_range': (0.4, 0.7) # 40-70% → Use algorithm
        }
        
        # Load ONLY from Supabase
        self.load_learning()
    
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
                        "pattern_thresholds": self.pattern_thresholds
                    })
                }
                supabase_data.append(data)
            
            # Save outcomes as a separate record
            if self.outcomes:
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
                        "pattern_thresholds": self.pattern_thresholds,
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
                    'outcomes': self.outcomes,
                    'pattern_thresholds': self.pattern_thresholds
                }, f)
            return True
        except Exception as e:
            st.error(f"Local save failed: {e}")
            return False
    
    def load_learning(self):
        """Load learning data from Supabase"""
        try:
            if not self.supabase:
                # Fallback to local storage
                return self._load_learning_local()
            
            # Load patterns from Supabase
            response = self.supabase.table("football_learning").select("*").execute()
            
            if not response.data:
                # Fresh start - no previous data
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
                                if 'pattern_thresholds' in metadata:
                                    self.pattern_thresholds.update(metadata['pattern_thresholds'])
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
            # Fallback to local storage
            return self._load_learning_local()
    
    def _load_learning_local(self):
        """Fallback local storage"""
        try:
            if os.path.exists("learning_data.pkl"):
                with open("learning_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.pattern_memory = defaultdict(lambda: {'total': 0, 'success': 0}, data['pattern_memory'])
                    self.outcomes = data['outcomes']
                    self.pattern_thresholds = data.get('pattern_thresholds', self.pattern_thresholds)
                return True
        except:
            pass
        return False
    
    def record_outcome(self, prediction, pattern_indicators, actual_result, actual_score):
        """Record a match outcome for learning and SAVE TO SUPABASE"""
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
        
        # SAVE TO SUPABASE
        save_success = self.save_learning()
        
        if save_success:
            return outcome, True, "✅ Outcome recorded and saved to Supabase!"
        else:
            return outcome, False, "⚠️ Outcome recorded locally but Supabase save failed"
    
    def get_pattern_success_rate(self, pattern_type, pattern_subtype=None):
        """Get historical success rate for a pattern"""
        key = f"{pattern_type}_{pattern_subtype}" if pattern_subtype else pattern_type
        
        # Look for exact matches first
        exact_keys = [k for k in self.pattern_memory if key in k]
        if exact_keys:
            total = sum(self.pattern_memory[k]['total'] for k in exact_keys)
            success = sum(self.pattern_memory[k]['success'] for k in exact_keys)
            if total > 0:
                return success / total
        
        # Look for similar patterns
        similar_keys = [k for k in self.pattern_memory if pattern_type in k]
        if similar_keys:
            total = sum(self.pattern_memory[k]['total'] for k in similar_keys)
            success = sum(self.pattern_memory[k]['success'] for k in similar_keys)
            if total > 0:
                return success / total
        
        return 0.5  # Default to 50%
    
    def get_betting_decision(self, prediction_type, pattern_key, original_prediction):
        """YOUR KEY INSIGHT: Bet opposite when algorithm is consistently wrong"""
        
        # Get historical success rate for this pattern
        success_rate = self.get_pattern_success_rate(prediction_type, pattern_key)
        total_matches = self._get_pattern_matches(pattern_key)
        
        # Not enough data - use original algorithm
        if total_matches < self.pattern_thresholds['min_matches']:
            return {
                'action': 'FOLLOW_ALGORITHM',
                'original': original_prediction,
                'adjusted': original_prediction,
                'reason': f'Insufficient data ({total_matches} matches)',
                'confidence_multiplier': 1.0,
                'success_rate': success_rate
            }
        
        # YOUR GOLDEN RULES:
        
        # RULE 1: STRONG PATTERN (>70% success) → BET STRONGLY
        if success_rate >= self.pattern_thresholds['strong_success']:
            return {
                'action': 'BET_STRONGLY',
                'original': original_prediction,
                'adjusted': original_prediction,
                'reason': f'YOUR STRONG PATTERN: {success_rate:.0%} success ({total_matches} matches)',
                'confidence_multiplier': 1.5,
                'success_rate': success_rate,
                'expected_value': success_rate - 0.52
            }
        
        # RULE 2: WEAK PATTERN (<40% success) → BET OPPOSITE
        elif success_rate < self.pattern_thresholds['weak_success']:
            opposite_prediction = self._get_opposite_prediction(original_prediction, prediction_type)
            
            return {
                'action': 'BET_OPPOSITE',
                'original': original_prediction,
                'adjusted': opposite_prediction,
                'reason': f'YOUR WEAK PATTERN: {success_rate:.0%} success → BET OPPOSITE',
                'confidence_multiplier': 2.0,
                'success_rate': success_rate,
                'expected_value': (1 - success_rate) - 0.52  # EV of opposite bet
            }
        
        # RULE 3: NEUTRAL PATTERN (40-70%) → Use algorithm with adjusted confidence
        else:
            return {
                'action': 'ADJUST_CONFIDENCE',
                'original': original_prediction,
                'adjusted': self._adjust_confidence(original_prediction, success_rate),
                'reason': f'Neutral pattern: {success_rate:.0%} success',
                'confidence_multiplier': 1.0,
                'success_rate': success_rate,
                'expected_value': success_rate - 0.52
            }
    
    def _get_opposite_prediction(self, original_prediction, prediction_type):
        """Get the opposite of the original prediction"""
        if prediction_type == 'winner':
            opposite = original_prediction.copy()
            if original_prediction.get('type') == 'HOME':
                opposite['type'] = 'AWAY'
                opposite['confidence_score'] = 85  # High confidence in opposite!
                opposite['confidence'] = "HIGH"
            elif original_prediction.get('type') == 'AWAY':
                opposite['type'] = 'HOME'
                opposite['confidence_score'] = 85
                opposite['confidence'] = "HIGH"
            else:  # DRAW
                opposite['type'] = 'DRAW'
                opposite['confidence_score'] = original_prediction.get('confidence_score', 50)
            return opposite
        
        elif prediction_type == 'totals':
            opposite = original_prediction.copy()
            if original_prediction.get('direction') == 'OVER':
                opposite['direction'] = 'UNDER'
                opposite['confidence_score'] = 85  # High confidence in opposite!
                opposite['confidence'] = "HIGH"
            else:
                opposite['direction'] = 'OVER'
                opposite['confidence_score'] = 85
                opposite['confidence'] = "HIGH"
            return opposite
    
    def _adjust_confidence(self, prediction, success_rate):
        """Adjust confidence based on success rate"""
        adjusted = prediction.copy()
        
        # Scale confidence toward 50% (uncertainty increases)
        base_confidence = prediction.get('confidence_score', 50)
        adjustment = (0.5 - success_rate) * 50
        
        adjusted['confidence_score'] = max(20, min(80, base_confidence + adjustment))
        
        # Update confidence category
        if adjusted['confidence_score'] >= 75:
            adjusted['confidence'] = "VERY HIGH"
        elif adjusted['confidence_score'] >= 65:
            adjusted['confidence'] = "HIGH"
        elif adjusted['confidence_score'] >= 55:
            adjusted['confidence'] = "MEDIUM"
        elif adjusted['confidence_score'] >= 45:
            adjusted['confidence'] = "LOW"
        else:
            adjusted['confidence'] = "VERY LOW"
        
        return adjusted
    
    def _get_pattern_matches(self, pattern_key):
        """Get total matches for a pattern"""
        return self.pattern_memory.get(pattern_key, {}).get('total', 0)
    
    def generate_learned_insights(self):
        """Generate insights based on YOUR discovered patterns"""
        insights = []
        
        if not self.outcomes:
            return ["🔄 **Learning System**: No historical data yet - record outcomes to start learning"]
        
        # Analyze last 20 outcomes
        recent = self.outcomes[-20:] if len(self.outcomes) > 20 else self.outcomes
        
        # Calculate success rates
        winner_success = sum(1 for o in recent if o['winner_correct']) / len(recent)
        totals_success = sum(1 for o in recent if o['totals_correct']) / len(recent)
        
        insights.append(f"📊 **Your Recent Accuracy**: Winners: {winner_success:.0%} | Totals: {totals_success:.0%}")
        
        # Identify YOUR discovered patterns
        pattern_performance = defaultdict(lambda: {'total': 0, 'success': 0})
        for outcome in recent[-50:]:  # Last 50 for better patterns
            key = f"{outcome.get('finishing_alignment', 'N/A')}+{outcome.get('total_category', 'N/A')}"
            pattern_performance[key]['total'] += 1
            pattern_performance[key]['success'] += 1 if outcome['totals_correct'] else 0
        
        # YOUR KEY INSIGHTS: Strong and Weak patterns
        strong_patterns = []
        weak_patterns = []
        
        for pattern, stats in pattern_performance.items():
            if stats['total'] >= 3:
                success_rate = stats['success'] / stats['total']
                if success_rate >= 0.7:
                    strong_patterns.append((pattern, stats['success'], stats['total']))
                elif success_rate <= 0.4:
                    weak_patterns.append((pattern, stats['success'], stats['total']))
        
        if strong_patterns:
            pattern, success, total = strong_patterns[0]
            insights.append(f"✅ **YOUR DISCOVERED GOLD**: {pattern} - {success}/{total} ({success/total:.0%}) → BET STRONGLY")
        
        if weak_patterns:
            pattern, success, total = weak_patterns[0]
            insights.append(f"🎯 **YOUR PROFIT OPPORTUNITY**: {pattern} - {success}/{total} ({success/total:.0%}) → BET OPPOSITE!")
        
        # Algorithm bias detection
        high_conf_failures = [o for o in recent if o['totals_confidence'] >= 70 and not o['totals_correct']]
        if len(high_conf_failures) >= 3:
            failure_rate = len(high_conf_failures) / len([o for o in recent if o['totals_confidence'] >= 70])
            if failure_rate >= 0.5:
                insights.append(f"⚠️ **ALGORITHMIC BIAS DETECTED**: High confidence predictions failing {failure_rate:.0%} of time!")
        
        return insights[:5]

# ========== INITIALIZE SESSION STATES ==========
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

if 'learning_system' not in st.session_state:
    st.session_state.learning_system = AdaptiveLearningSystemV2()

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

if 'save_status' not in st.session_state:
    st.session_state.save_status = None

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
        
        # Get finishing trends
        home_finishing = home_stats['goals_vs_xg_pm']
        away_finishing = away_stats['goals_vs_xg_pm']
        
        # Get defensive performance
        home_defense = home_stats['goals_allowed_vs_xga_pm']
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        # ========== KEY FIX: ADJUST xG FOR FINISHING ABILITY ==========
        home_adjusted_xg = home_xg + home_finishing - away_defense
        away_adjusted_xg = away_xg + away_finishing - home_defense
        
        # Calculate adjusted delta
        delta = home_adjusted_xg - away_adjusted_xg
        
        # ========== DETERMINE VOLATILITY FLAG ==========
        volatility_high = False
        if abs(home_finishing) > 0.3 and abs(away_finishing) > 0.3:
            volatility_high = True
        elif home_finishing > 0.3 and away_finishing > 0.3:
            volatility_high = True
        elif home_finishing < -0.3 and away_finishing < -0.3:
            volatility_high = True
        
        # ========== WINNER DETERMINATION ==========
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
        
        # ========== CONFIDENCE CALCULATION ==========
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
            'type': predicted_winner,
            'winner_strength': winner_strength,
            'confidence_score': winner_confidence,
            'confidence': confidence_category,
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
            # Check if both have good defense
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
        """OUR IMPROVED LOGIC: Risk flag system"""
        risk_flags = []
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        # NEW RISK FLAG: Volatile overperformers
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
        
        # NEW: Bundesliga specific adjustment
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
        
        # ========== ORIGINAL LOGIC ==========
        over_threshold = self.league_adjustments['over_threshold']
        base_direction = "OVER" if total_xg > over_threshold else "UNDER"
        
        # OUR LOGIC: Finishing alignment
        finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
        total_category = self.categorize_total_xg(total_xg)
        
        # OUR LOGIC: Risk flags
        risk_flags = self.check_risk_flags(home_stats, away_stats, total_xg)
        
        # ========== PROVEN PATTERN 1 ==========
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
    """OUR IMPROVED LOGIC: Generate enhanced insights with YOUR betting rules"""
    
    @staticmethod
    def generate_insights(winner_prediction, totals_prediction, betting_decisions=None):
        insights = []
        
        # Add betting decision insights
        if betting_decisions:
            winner_decision = betting_decisions.get('winner')
            totals_decision = betting_decisions.get('totals')
            
            if winner_decision and winner_decision['action'] == 'BET_OPPOSITE':
                insights.append(f"🎯 **YOUR RULE**: BET OPPOSITE WINNER! Algorithm has {winner_decision['success_rate']:.0%} success rate for this pattern")
            
            if totals_decision and totals_decision['action'] == 'BET_OPPOSITE':
                insights.append(f"🎯 **YOUR RULE**: BET OPPOSITE TOTALS! Algorithm has {totals_decision['success_rate']:.0%} success rate for this pattern")
            
            if winner_decision and winner_decision['action'] == 'BET_STRONGLY':
                insights.append(f"✅ **YOUR STRONG PATTERN**: {winner_decision['reason']}")
            
            if totals_decision and totals_decision['action'] == 'BET_STRONGLY':
                insights.append(f"✅ **YOUR STRONG PATTERN**: {totals_decision['reason']}")
        
        # Winner insights
        if winner_prediction.get('confidence') == "VERY HIGH":
            insights.append(f"🎯 **High Confidence Winner**: Model strongly favors {winner_prediction.get('type', 'N/A')}")
        elif winner_prediction.get('confidence') == "LOW":
            insights.append(f"⚠️ **Low Confidence Winner**: Exercise caution on {winner_prediction.get('type', 'N/A')} prediction")
        
        # Defense rule insights
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
            insights.append("⚠️ **Both teams strong overperformers** - High volatility expected")
        
        # PROVEN PATTERN INSIGHTS
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
            flag_list = ", ".join(risk_flags[:3])
            insights.append(f"⚠️ **{risk_count} risk flag(s) detected**: {flag_list}")
        
        return insights[:8]

# ========== YOUR IMPROVED ADAPTIVE ENGINE ==========

class AdaptiveFootballIntelligenceEngineV5:
    """VERSION 5: Uses YOUR discovery to bet opposite when algorithm is wrong"""
    
    def __init__(self, league_metrics, league_name, learning_system=None):
        self.league_metrics = league_metrics
        self.league_name = league_name
        self.learning_system = learning_system or AdaptiveLearningSystemV2()
        
        # Initialize predictors
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
        self.insights_generator = InsightsGenerator()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate prediction with YOUR betting rules"""
        
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
        
        # Apply YOUR betting rules
        enhanced_prediction = self._apply_your_betting_rules(
            home_team, away_team, 
            winner_prediction, totals_prediction, 
            probabilities, home_xg, away_xg
        )
        
        # Store betting decisions for insights
        betting_decisions = self._get_betting_decisions(winner_prediction, totals_prediction)
        
        # Generate insights including YOUR rules
        insights = self.insights_generator.generate_insights(
            winner_prediction, totals_prediction, betting_decisions
        )
        learned_insights = self.learning_system.generate_learned_insights()
        insights.extend(learned_insights)
        
        # Update prediction with insights
        enhanced_prediction['insights'] = insights
        enhanced_prediction['calculation_details'] = calc_details
        
        return enhanced_prediction
    
    def _apply_your_betting_rules(self, home_team, away_team, winner_pred, totals_pred, probabilities, home_xg, away_xg):
        """Apply YOUR discovered betting rules"""
        
        # Create pattern keys
        winner_key = f"WINNER_{winner_pred['confidence']}_{winner_pred['confidence_score']//10*10}"
        totals_key = f"TOTALS_{totals_pred.get('finishing_alignment', 'N/A')}_{totals_pred.get('total_category', 'N/A')}"
        
        # Get betting decisions
        winner_decision = self.learning_system.get_betting_decision(
            'winner', winner_key, winner_pred
        )
        totals_decision = self.learning_system.get_betting_decision(
            'totals', totals_key, totals_pred
        )
        
        # Apply winner decision
        final_winner_pred = winner_decision['adjusted'].copy()
        
        # Determine team name and probability
        if final_winner_pred['type'] == 'HOME':
            winner_display = home_team
            winner_prob = probabilities['home_win_probability']
        elif final_winner_pred['type'] == 'AWAY':
            winner_display = away_team
            winner_prob = probabilities['away_win_probability']
        else:
            winner_display = "DRAW"
            winner_prob = probabilities['draw_probability']
        
        # Apply totals decision
        final_totals_pred = totals_decision['adjusted'].copy()
        
        # Determine probability for totals
        if final_totals_pred['direction'] == "OVER":
            total_prob = probabilities['over_2_5_probability']
        else:
            total_prob = probabilities['under_2_5_probability']
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'winner': {
                'team': winner_display,
                'type': final_winner_pred['type'],
                'probability': winner_prob,
                'confidence': final_winner_pred['confidence'],
                'confidence_score': final_winner_pred['confidence_score'],
                'strength': final_winner_pred.get('winner_strength', 'N/A'),
                'most_likely_score': probabilities['most_likely_score'],
                'betting_decision': winner_decision['action'],
                'original_prediction': winner_decision['original'].get('type'),
                'success_rate': winner_decision['success_rate'],
                'opposite_bet': winner_decision['action'] == 'BET_OPPOSITE'
            },
            
            'totals': {
                'direction': final_totals_pred['direction'],
                'probability': total_prob,
                'confidence': final_totals_pred['confidence'],
                'confidence_score': final_totals_pred['confidence_score'],
                'total_xg': totals_pred['total_xg'],
                'finishing_alignment': totals_pred.get('finishing_alignment'),
                'total_category': totals_pred.get('total_category'),
                'risk_flags': totals_pred.get('risk_flags', []),
                'betting_decision': totals_decision['action'],
                'original_direction': totals_decision['original'].get('direction'),
                'success_rate': totals_decision['success_rate'],
                'opposite_bet': totals_decision['action'] == 'BET_OPPOSITE'
            },
            
            'probabilities': probabilities,
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': home_xg + away_xg},
            'betting_decisions': {
                'winner': winner_decision,
                'totals': totals_decision
            }
        }
    
    def _get_betting_decisions(self, winner_pred, totals_pred):
        """Get betting decisions for insights"""
        winner_key = f"WINNER_{winner_pred['confidence']}_{winner_pred['confidence_score']//10*10}"
        totals_key = f"TOTALS_{totals_pred.get('finishing_alignment', 'N/A')}_{totals_pred.get('total_category', 'N/A')}"
        
        return {
            'winner': self.learning_system.get_betting_decision('winner', winner_key, winner_pred),
            'totals': self.learning_system.get_betting_decision('totals', totals_key, totals_pred)
        }

# ========== ADAPTIVE PATTERN INDICATORS ==========

class AdaptivePatternIndicators:
    """Generate pattern indicators with YOUR betting rules"""
    
    def __init__(self, learning_system):
        self.learning_system = learning_system
    
    def generate_indicators(self, prediction):
        """Generate pattern indicators with YOUR betting rules"""
        indicators = {'winner': None, 'totals': None}
        
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # WINNER INDICATORS with YOUR rules
        winner_key = f"WINNER_{winner_pred['confidence']}_{winner_pred['confidence_score']//10*10}"
        winner_success_rate = self.learning_system.get_pattern_success_rate('WINNER', winner_key)
        winner_matches = self.learning_system._get_pattern_matches(winner_key)
        
        if winner_pred.get('opposite_bet', False):
            indicators['winner'] = {
                'type': 'OPPOSITE_BET',
                'color': 'red',
                'text': '🎯 BET OPPOSITE!',
                'explanation': f'Algorithm has only {winner_success_rate:.0%} success rate for this pattern ({winner_matches} matches)'
            }
        elif winner_success_rate >= 0.7 and winner_matches >= 3:
            indicators['winner'] = {
                'type': 'STRONG_PATTERN',
                'color': 'green',
                'text': '✅ YOUR STRONG PATTERN',
                'explanation': f'Your historical success: {winner_success_rate:.0%} for this pattern'
            }
        elif winner_success_rate < 0.4 and winner_matches >= 3:
            indicators['winner'] = {
                'type': 'WEAK_PATTERN',
                'color': 'orange',
                'text': '⚠️ YOUR WEAK PATTERN',
                'explanation': f'Your historical failure: {winner_success_rate:.0%} success rate'
            }
        else:
            indicators['winner'] = {
                'type': 'NO_PATTERN',
                'color': 'gray',
                'text': 'NO PATTERN YET',
                'explanation': f'Your historical success: {winner_success_rate:.0%} ({winner_matches} matches)'
            }
        
        # TOTALS INDICATORS with YOUR rules
        finishing_alignment = totals_pred.get('finishing_alignment', 'N/A')
        total_category = totals_pred.get('total_category', 'N/A')
        
        totals_key = f"TOTALS_{finishing_alignment}_{total_category}"
        totals_success_rate = self.learning_system.get_pattern_success_rate('TOTALS', totals_key)
        totals_matches = self.learning_system._get_pattern_matches(totals_key)
        
        if totals_pred.get('opposite_bet', False):
            indicators['totals'] = {
                'type': 'OPPOSITE_BET',
                'color': 'red',
                'text': '🎯 BET OPPOSITE!',
                'explanation': f'Algorithm has only {totals_success_rate:.0%} success rate for this pattern ({totals_matches} matches)'
            }
        elif totals_success_rate >= 0.7 and totals_matches >= 3:
            indicators['totals'] = {
                'type': 'STRONG_PATTERN',
                'color': 'green',
                'text': '✅ YOUR STRONG PATTERN',
                'explanation': f'Your historical success: {totals_success_rate:.0%} for this pattern'
            }
        elif totals_success_rate < 0.4 and totals_matches >= 3:
            indicators['totals'] = {
                'type': 'WEAK_PATTERN',
                'color': 'orange',
                'text': '⚠️ YOUR WEAK PATTERN',
                'explanation': f'Your historical failure: {totals_success_rate:.0%} success rate'
            }
        else:
            indicators['totals'] = {
                'type': 'NO_PATTERN',
                'color': 'gray',
                'text': 'NO PATTERN YET',
                'explanation': f'Your historical success: {totals_success_rate:.0%} ({totals_matches} matches)'
            }
        
        return indicators

# ========== YOUR INTELLIGENT BETTING CARD ==========

class IntelligentBettingCard:
    """Betting card that EXPLOITS algorithmic biases"""
    
    def __init__(self, learning_system):
        self.learning_system = learning_system
    
    def get_recommendation(self, prediction):
        """Get recommendation based on YOUR discovered patterns"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Check if we should bet opposite
        should_bet_opposite = winner_pred.get('opposite_bet', False) or totals_pred.get('opposite_bet', False)
        
        if should_bet_opposite:
            return self._get_opposite_recommendation(prediction)
        elif winner_pred.get('betting_decision') == 'BET_STRONGLY' or totals_pred.get('betting_decision') == 'BET_STRONGLY':
            return self._get_strong_pattern_recommendation(prediction)
        else:
            return self._get_standard_recommendation(prediction)
    
    def _get_opposite_recommendation(self, prediction):
        """Generate recommendation to bet OPPOSITE"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        actions = []
        
        if winner_pred.get('opposite_bet'):
            original = winner_pred.get('original_prediction', 'N/A')
            actions.append(f"🎯 BET OPPOSITE WINNER (Algorithm predicted: {original})")
        
        if totals_pred.get('opposite_bet'):
            original = totals_pred.get('original_direction', 'N/A')
            actions.append(f"📈 BET OPPOSITE TOTALS (Algorithm predicted: {original} 2.5)")
        
        return {
            'type': 'OPPOSITE_BET',
            'text': "🎯 EXPLOIT ALGORITHMIC BIAS",
            'actions': actions,
            'confidence': 85,
            'color': '#DC2626',
            'icon': '🎯',
            'subtext': 'BET OPPOSITE (Algorithm Wrong!)',
            'reason': f"Algorithm consistently wrong on these patterns",
            'expected_value': 0.35,
            'risk_level': 'MEDIUM',
            'stake_multiplier': 1.0
        }
    
    def _get_strong_pattern_recommendation(self, prediction):
        """Generate recommendation for strong patterns"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        if winner_pred.get('betting_decision') == 'BET_STRONGLY' and totals_pred.get('betting_decision') == 'BET_STRONGLY':
            return {
                'type': 'DOUBLE_STRONG',
                'text': f"✅ {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']) * 1.2,
                'color': '#10B981',
                'icon': '🎯',
                'subtext': 'DOUBLE STRONG PATTERN',
                'reason': f'Both patterns have >70% historical success',
                'expected_value': 0.25,
                'risk_level': 'LOW',
                'stake_multiplier': 1.5
            }
        elif winner_pred.get('betting_decision') == 'BET_STRONGLY':
            return {
                'type': 'WINNER_STRONG',
                'text': f"✅ {winner_pred['team']} to win",
                'confidence': winner_pred['confidence_score'] * 1.2,
                'color': '#3B82F6',
                'icon': '🏆',
                'subtext': 'STRONG WINNER PATTERN',
                'reason': f'Winner pattern has {winner_pred.get("success_rate", 0):.0%} historical success',
                'expected_value': winner_pred.get("success_rate", 0.5) - 0.52,
                'risk_level': 'LOW',
                'stake_multiplier': 1.2
            }
        else:
            return {
                'type': 'TOTALS_STRONG',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'confidence': totals_pred['confidence_score'] * 1.2,
                'color': '#8B5CF6',
                'icon': '📈',
                'subtext': 'STRONG TOTALS PATTERN',
                'reason': f'Totals pattern has {totals_pred.get("success_rate", 0):.0%} historical success',
                'expected_value': totals_pred.get("success_rate", 0.5) - 0.52,
                'risk_level': 'LOW',
                'stake_multiplier': 1.2
            }
    
    def _get_standard_recommendation(self, prediction):
        """Standard recommendation when no strong patterns"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Calculate expected values
        winner_ev = winner_pred.get("success_rate", 0.5) - 0.52
        totals_ev = totals_pred.get("success_rate", 0.5) - 0.52
        
        if winner_ev > 0.1 and totals_ev > 0.1:
            return {
                'type': 'COMBO',
                'text': f"🎯 {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#10B981',
                'icon': '🎯',
                'subtext': 'DOUBLE BET (POSITIVE EV)',
                'reason': f'Winner EV: {winner_ev:.2f} | Totals EV: {totals_ev:.2f}',
                'expected_value': (winner_ev + totals_ev) / 2,
                'risk_level': 'MEDIUM',
                'stake_multiplier': 1.0
            }
        elif winner_ev > 0.15:
            return {
                'type': 'SINGLE_WINNER',
                'text': f"🏆 {winner_pred['team']} to win",
                'confidence': winner_pred['confidence_score'],
                'color': '#3B82F6',
                'icon': '🏆',
                'subtext': 'WINNER BET',
                'reason': f'Expected Value: {winner_ev:.2f}',
                'expected_value': winner_ev,
                'risk_level': 'MEDIUM',
                'stake_multiplier': 1.0
            }
        elif totals_ev > 0.15:
            return {
                'type': 'SINGLE_TOTALS',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'confidence': totals_pred['confidence_score'],
                'color': '#8B5CF6',
                'icon': '📈',
                'subtext': 'TOTALS BET',
                'reason': f'Expected Value: {totals_ev:.2f}',
                'expected_value': totals_ev,
                'risk_level': 'MEDIUM',
                'stake_multiplier': 1.0
            }
        else:
            return {
                'type': 'NO_BET',
                'text': "🚫 No Value Bet",
                'confidence': max(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#6B7280',
                'icon': '🤔',
                'subtext': 'NO BET',
                'reason': f'Insufficient expected value (Winner: {winner_ev:.2f}, Totals: {totals_ev:.2f})',
                'expected_value': 0,
                'risk_level': 'LOW',
                'stake_multiplier': 0
            }
    
    def display_card(self, recommendation):
        """Display the intelligent betting card"""
        ev = recommendation.get('expected_value', 0)
        risk_level = recommendation.get('risk_level', 'MEDIUM')
        
        # Color based on recommendation type
        if recommendation['type'] == 'OPPOSITE_BET':
            color = '#DC2626'
            border_color = '#EF4444'
        elif recommendation['type'] in ['DOUBLE_STRONG', 'WINNER_STRONG', 'TOTALS_STRONG']:
            color = '#10B981'
            border_color = '#34D399'
        elif recommendation['type'] == 'NO_BET':
            color = '#6B7280'
            border_color = '#9CA3AF'
        else:
            color = '#3B82F6'
            border_color = '#60A5FA'
        
        # Risk level indicator
        risk_colors = {
            'LOW': '#10B981',
            'MEDIUM': '#F59E0B',
            'HIGH': '#DC2626'
        }
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}20 0%, #1F2937 100%);
            padding: 25px;
            border-radius: 20px;
            border: 3px solid {border_color};
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        ">
            <div style="font-size: 48px; margin-bottom: 15px;">
                {recommendation['icon']}
            </div>
            <div style="font-size: 36px; font-weight: bold; color: white; margin-bottom: 10px;">
                {recommendation['text']}
            </div>
            <div style="font-size: 24px; color: {border_color}; margin-bottom: 10px; font-weight: bold;">
                {recommendation['subtext']}
            </div>
            
            <div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0;">
                <div style="background: rgba(59, 130, 246, 0.2); padding: 10px 20px; border-radius: 10px;">
                    <div style="font-size: 14px; color: #9CA3AF;">Confidence</div>
                    <div style="font-size: 24px; font-weight: bold; color: white;">{recommendation['confidence']:.0f}/100</div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.2); padding: 10px 20px; border-radius: 10px;">
                    <div style="font-size: 14px; color: #9CA3AF;">Expected Value</div>
                    <div style="font-size: 24px; font-weight: bold; color: white;">{ev:.3f}</div>
                </div>
                <div style="background: rgba(245, 158, 11, 0.2); padding: 10px 20px; border-radius: 10px;">
                    <div style="font-size: 14px; color: #9CA3AF;">Risk Level</div>
                    <div style="font-size: 24px; font-weight: bold; color: {risk_colors[risk_level]};">{risk_level}</div>
                </div>
            </div>
            
            <div style="font-size: 16px; color: #D1D5DB; padding: 15px; background: rgba(59, 130, 246, 0.1); border-radius: 10px; margin-top: 15px;">
                {recommendation['reason']}
            </div>
            
            {f'''
            <div style="margin-top: 15px; padding: 10px; background: rgba(220, 38, 38, 0.1); border-radius: 10px; border: 1px solid rgba(220, 38, 38, 0.3);">
                <div style="font-size: 16px; color: #FCA5A5; font-weight: bold;">🎯 ALGORITHMIC BIAS DETECTED</div>
                <div style="font-size: 14px; color: #FECACA;">Betting opposite of algorithm's prediction</div>
            </div>
            ''' if recommendation['type'] == 'OPPOSITE_BET' else ''}
            
            {f'''
            <div style="margin-top: 15px; padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 10px; border: 1px solid rgba(16, 185, 129, 0.3);">
                <div style="font-size: 16px; color: #6EE7B7; font-weight: bold;">✅ YOUR PROVEN PATTERN</div>
                <div style="font-size: 14px; color: #A7F3D0;">Based on your historical success >70%</div>
            </div>
            ''' if 'STRONG' in recommendation['type'] else ''}
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

def record_outcome_with_feedback(prediction, pattern_indicators, home_team, away_team):
    """Fixed feedback system"""
    
    st.divider()
    st.subheader("📝 Record Outcome for Learning")
    
    # Show previous feedback if exists
    if st.session_state.get('save_status'):
        status_type, status_message = st.session_state.save_status
        if status_type == "success":
            st.success(status_message)
            
            # Show what was learned
            if st.session_state.get('last_outcome'):
                with st.expander("📈 What was learned?", expanded=True):
                    outcome = st.session_state.last_outcome
                    st.write(f"**Match**: {outcome['home_team']} vs {outcome['away_team']}")
                    st.write(f"**Actual Score**: {outcome['actual_score']}")
                    st.write(f"**Winner Prediction**: {'✅ Correct' if outcome['winner_correct'] else '❌ Wrong'}")
                    st.write(f"**Totals Prediction**: {'✅ Correct' if outcome['totals_correct'] else '❌ Wrong'}")
                    
                    # Show pattern updates
                    winner_key = f"WINNER_{prediction['winner']['confidence']}_{prediction['winner']['confidence_score']//10*10}"
                    totals_key = f"TOTALS_{prediction['totals'].get('finishing_alignment', 'N/A')}_{prediction['totals'].get('total_category', 'N/A')}"
                    
                    winner_stats = st.session_state.learning_system.pattern_memory.get(winner_key, {'total': 0, 'success': 0})
                    totals_stats = st.session_state.learning_system.pattern_memory.get(totals_key, {'total': 0, 'success': 0})
                    
                    st.write(f"**Winner Pattern**: {winner_key}")
                    st.write(f"**Winner Success Rate**: {winner_stats['success']}/{winner_stats['total']} ({winner_stats['success']/winner_stats['total']:.0% if winner_stats['total'] > 0 else 'N/A'})")
                    st.write(f"**Totals Pattern**: {totals_key}")
                    st.write(f"**Totals Success Rate**: {totals_stats['success']}/{totals_stats['total']} ({totals_stats['success']/totals_stats['total']:.0% if totals_stats['total'] > 0 else 'N/A'})")
                    
                    # Show if pattern now qualifies for betting rules
                    if winner_stats['total'] >= 3:
                        success_rate = winner_stats['success'] / winner_stats['total']
                        if success_rate >= 0.7:
                            st.success(f"🎯 **NEW STRONG PATTERN**: Winner pattern now qualifies for BET STRONGLY!")
                        elif success_rate <= 0.4:
                            st.warning(f"🎯 **NEW WEAK PATTERN**: Winner pattern now qualifies for BET OPPOSITE!")
                    
                    if totals_stats['total'] >= 3:
                        success_rate = totals_stats['success'] / totals_stats['total']
                        if success_rate >= 0.7:
                            st.success(f"🎯 **NEW STRONG PATTERN**: Totals pattern now qualifies for BET STRONGLY!")
                        elif success_rate <= 0.4:
                            st.warning(f"🎯 **NEW WEAK PATTERN**: Totals pattern now qualifies for BET OPPOSITE!")
        
        else:
            st.error(status_message)
    
    # Simple input and button
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
        
        if '-' not in score_input:
            st.error("❌ Please enter score in format '2-1' (needs a dash)")
            return
        
        parts = score_input.split('-')
        if len(parts) != 2:
            st.error("❌ Please enter score in format '2-1' (exactly one dash)")
            return
        
        try:
            home_goals = int(parts[0].strip())
            away_goals = int(parts[1].strip())
            
            if home_goals < 0 or away_goals < 0:
                st.error("❌ Goals cannot be negative")
                return
            if home_goals > 20 or away_goals > 20:
                st.error("❌ That's an unrealistic score!")
                return
            
            # Record outcome and SAVE TO SUPABASE
            with st.spinner("⏳ Saving to Supabase..."):
                try:
                    outcome, save_success, save_message = st.session_state.learning_system.record_outcome(
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
                        'prediction': prediction,
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
    st.header("🎯 YOUR BETTING RULES")
    
    st.info("""
    **YOUR DISCOVERED RULES:**
    
    ✅ **STRONG PATTERN** (>70% success, ≥3 matches)
    → BET STRONGLY with boosted confidence
    
    🎯 **WEAK PATTERN** (<40% success, ≥3 matches)  
    → BET OPPOSITE with 85% confidence!
    
    ⚪ **NEUTRAL PATTERN** (40-70% success)
    → Use algorithm with adjusted confidence
    """)
    
    # Supabase Status
    if st.session_state.learning_system.supabase:
        st.success("🔄 **Storage**: Connected to Supabase")
    else:
        st.warning("🔄 **Storage**: Local only (Supabase not available)")
    
    st.write(f"📊 **Your Patterns**: {len(st.session_state.learning_system.pattern_memory)}")
    st.write(f"📈 **Your Outcomes**: {len(st.session_state.learning_system.outcomes)}")
    
    # Show YOUR discovered patterns
    st.subheader("Your Discovered Patterns")
    patterns = dict(st.session_state.learning_system.pattern_memory)
    qualifying_patterns = [(k, v) for k, v in patterns.items() if v['total'] >= 3]
    
    if qualifying_patterns:
        for pattern, stats in qualifying_patterns[-5:]:  # Show last 5
            success_rate = stats['success'] / stats['total']
            if success_rate >= 0.7:
                st.success(f"✅ {pattern[:30]}...: {success_rate:.0%} ({stats['success']}/{stats['total']})")
            elif success_rate <= 0.4:
                st.error(f"🎯 {pattern[:30]}...: {success_rate:.0%} ({stats['success']}/{stats['total']}) → BET OPPOSITE!")
            else:
                st.info(f"⚪ {pattern[:30]}...: {success_rate:.0%} ({stats['success']}/{stats['total']})")
    else:
        st.caption("Record 3+ matches with same pattern to see your rules in action")
    
    # Refresh data button
    if st.button("🔄 Refresh Learning Data", use_container_width=True):
        success = st.session_state.learning_system.load_learning()
        if success:
            st.success("Learning data refreshed!")
        else:
            st.warning("Could not refresh from Supabase")
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
    home_team, away_team = st.session_state.last_teams

# If no prediction to show
if not show_prediction:
    st.info("👈 Select teams and click 'Generate Prediction'")
    
    # Show learning insights
    with st.expander("🧠 Learning System Insights", expanded=True):
        insights = st.session_state.learning_system.generate_learned_insights()
        for insight in insights:
            st.write(f"• {insight}")
    
    st.stop()

# ========== IF WE GET HERE, WE HAVE A PREDICTION TO SHOW ==========

# If this is a new prediction (user clicked "Generate Prediction")
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Generate prediction with YOUR improved engine
        engine = AdaptiveFootballIntelligenceEngineV5(
            league_metrics, 
            selected_league, 
            st.session_state.learning_system
        )
        
        prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)
        pattern_generator = AdaptivePatternIndicators(st.session_state.learning_system)
        pattern_indicators = pattern_generator.generate_indicators(prediction)
        
        # Store in session state for next time
        st.session_state.last_prediction = prediction
        st.session_state.last_pattern_indicators = pattern_indicators
        st.session_state.last_teams = (home_team, away_team)
        st.session_state.last_league = selected_league
        st.session_state.last_engine = engine
        
    except KeyError as e:
        st.error(f"Team data error: {e}")
        st.stop()

# ========== DISPLAY THE PREDICTION ==========
st.header(f"🎯 {home_team} vs {away_team}")
st.caption(f"League: {selected_league} | League Avg Goals: {league_metrics['avg_goals_per_match']:.2f}")

# Main prediction cards
col1, col2 = st.columns(2)

with col1:
    # Winner prediction
    winner_pred = prediction['winner']
    
    # Color based on betting decision
    if winner_pred.get('opposite_bet', False):
        card_color = "#7F1D1D"
        text_color = "#EF4444"
        icon = "🎯"
        subtitle = "BET OPPOSITE!"
    elif winner_pred.get('betting_decision') == 'BET_STRONGLY':
        card_color = "#14532D"
        text_color = "#22C55E"
        icon = "✅"
        subtitle = "YOUR STRONG PATTERN"
    else:
        if winner_pred['type'] == "HOME":
            text_color = "#22C55E" if winner_pred['confidence'] in ["VERY HIGH", "HIGH"] else "#4ADE80" if winner_pred['confidence'] == "MEDIUM" else "#84CC16"
            icon = "🏠"
        elif winner_pred['type'] == "AWAY":
            text_color = "#22C55E" if winner_pred['confidence'] in ["VERY HIGH", "HIGH"] else "#4ADE80" if winner_pred['confidence'] == "MEDIUM" else "#84CC16"
            icon = "✈️"
        else:
            text_color = "#F59E0B"
            icon = "🤝"
        
        if winner_pred['confidence'] == "VERY HIGH":
            card_color = "#14532D"
        elif winner_pred['confidence'] == "HIGH":
            card_color = "#166534"
        elif winner_pred['confidence'] == "MEDIUM":
            card_color = "#365314"
        elif winner_pred['confidence'] == "LOW":
            card_color = "#3F6212"
        else:
            card_color = "#1E293B"
        subtitle = winner_pred['confidence']
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">PREDICTED WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: {text_color}; margin: 10px 0;">
            {icon} {winner_pred['team']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {subtitle} | Confidence: {winner_pred['confidence_score']:.0f}/100
        </div>
        {f'<div style="font-size: 14px; color: #FCA5A5; margin-top: 10px; font-weight: bold;">🎯 BET OPPOSITE! (Algorithm wrong {winner_pred.get("success_rate", 0):.0%} of time)</div>' if winner_pred.get('opposite_bet') else ''}
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Totals prediction
    totals_pred = prediction['totals']
    direction = totals_pred['direction']
    confidence = totals_pred['confidence']
    conf_score = totals_pred['confidence_score']
    
    # Color based on betting decision
    if totals_pred.get('opposite_bet', False):
        card_color = "#7F1D1D"
        text_color = "#EF4444"
        subtitle = "BET OPPOSITE!"
    elif totals_pred.get('betting_decision') == 'BET_STRONGLY':
        card_color = "#14532D"
        text_color = "#22C55E"
        subtitle = "YOUR STRONG PATTERN"
    else:
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
        subtitle = confidence
    
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
            {subtitle} | Confidence: {conf_score:.0f}/100
        </div>
        {f'<div style="font-size: 14px; color: #FCA5A5; margin-top: 10px; font-weight: bold;">🎯 BET OPPOSITE! (Algorithm wrong {totals_pred.get("success_rate", 0):.0%} of time)</div>' if totals_pred.get('opposite_bet') else ''}
    </div>
    """, unsafe_allow_html=True)

# ========== YOUR INTELLIGENT BETTING CARD ==========
st.divider()
st.subheader("🎯 YOUR INTELLIGENT BETTING CARD")

# Generate adaptive betting recommendation
betting_card = IntelligentBettingCard(st.session_state.learning_system)
recommendation = betting_card.get_recommendation(prediction)

# Display the card
betting_card.display_card(recommendation)

# Show stake suggestion if not "NO BET"
if recommendation['type'] != 'NO_BET':
    st.info(f"💡 **Stake Suggestion**: Use {recommendation.get('stake_multiplier', 1.0):.1f}x your normal stake for this bet")

# ========== PATTERN INDICATORS ==========
st.divider()
st.subheader("🔍 Your Pattern Analysis")

col1, col2 = st.columns(2)

with col1:
    winner_indicator = pattern_indicators['winner']
    if winner_indicator['type'] == 'OPPOSITE_BET':
        st.markdown(f"""
        <div style="background-color: #7F1D1D; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #EF4444;">
            <div style="font-size: 20px; font-weight: bold; color: #EF4444; margin: 5px 0;">
                🎯 {winner_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #FECACA;">
                {winner_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif winner_indicator['type'] == 'STRONG_PATTERN':
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
    elif winner_indicator['type'] == 'WEAK_PATTERN':
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
    if totals_indicator['type'] == 'OPPOSITE_BET':
        st.markdown(f"""
        <div style="background-color: #7F1D1D; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0; border: 2px solid #EF4444;">
            <div style="font-size: 20px; font-weight: bold; color: #EF4444; margin: 5px 0;">
                🎯 {totals_indicator['text']}
            </div>
            <div style="font-size: 14px; color: #FECACA;">
                {totals_indicator['explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif totals_indicator['type'] == 'STRONG_PATTERN':
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
    elif totals_indicator['type'] == 'WEAK_PATTERN':
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

st.caption("💡 **YOUR RULES**: Green = Bet strongly | Red = Bet opposite! | Orange = Weak pattern | Gray = No pattern yet")

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
    home_finish = prediction['totals'].get('home_finishing', 0)
    if home_finish > 0:
        st.metric(f"{home_team} Finishing", f"{home_finish:+.2f}", "Overperforms xG")
    else:
        st.metric(f"{home_team} Finishing", f"{home_finish:+.2f}", "Underperforms xG")

with col2:
    away_finish = prediction['totals'].get('away_finishing', 0)
    if away_finish > 0:
        st.metric(f"{away_team} Finishing", f"{away_finish:+.2f}", "Overperforms xG")
    else:
        st.metric(f"{away_team} Finishing", f"{away_finish:+.2f}", "Underperforms xG")

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

# ========== FIXED FEEDBACK SECTION ==========
record_outcome_with_feedback(prediction, pattern_indicators, home_team, away_team)

# ========== DETAILED ANALYSIS ==========
if show_details:
    with st.expander("🔍 Detailed Analysis", expanded=False):
        st.write("### Winner Prediction Analysis")
        st.write(f"- Expected Goals Difference: {prediction['winner'].get('strength', 'N/A')}")
        st.write(f"- Confidence Level: {prediction['winner']['confidence']}")
        st.write(f"- Betting Decision: {prediction['winner'].get('betting_decision', 'N/A')}")
        st.write(f"- Historical Success Rate: {prediction['winner'].get('success_rate', 0):.0%}")
        
        if prediction['winner'].get('opposite_bet'):
            st.write(f"- 🎯 **ACTION**: BET OPPOSITE! Original prediction was: {prediction['winner'].get('original_prediction', 'N/A')}")
        
        st.write("### Totals Prediction Analysis")
        st.write(f"- Total xG: {prediction['totals']['total_xg']:.2f}")
        st.write(f"- Finishing Alignment: {prediction['totals'].get('finishing_alignment', 'N/A')}")
        st.write(f"- Total Category: {prediction['totals'].get('total_category', 'N/A')}")
        st.write(f"- Betting Decision: {prediction['totals'].get('betting_decision', 'N/A')}")
        st.write(f"- Historical Success Rate: {prediction['totals'].get('success_rate', 0):.0%}")
        
        if prediction['totals'].get('opposite_bet'):
            st.write(f"- 🎯 **ACTION**: BET OPPOSITE! Original prediction was: {prediction['totals'].get('original_direction', 'N/A')} 2.5")
        
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
    st.subheader("📊 Your Strongest & Weakest Patterns")
    patterns = dict(st.session_state.learning_system.pattern_memory)
    strong_patterns = [(k, v) for k, v in patterns.items() if v['total'] >= 3 and v['success']/v['total'] >= 0.7]
    weak_patterns = [(k, v) for k, v in patterns.items() if v['total'] >= 3 and v['success']/v['total'] <= 0.4]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if strong_patterns:
            st.subheader("✅ STRONG PATTERNS (Bet This)")
            for pattern, stats in strong_patterns[:3]:
                success_rate = stats['success'] / stats['total']
                st.success(f"**{pattern[:25]}...**: {stats['success']}/{stats['total']} ({success_rate:.0%})")
        else:
            st.info("No strong patterns yet (need ≥3 matches with >70% success)")
    
    with col2:
        if weak_patterns:
            st.subheader("🎯 WEAK PATTERNS (Bet Opposite!)")
            for pattern, stats in weak_patterns[:3]:
                success_rate = stats['success'] / stats['total']
                st.error(f"**{pattern[:25]}...**: {stats['success']}/{stats['total']} ({success_rate:.0%})")
                st.caption(f"→ BET OPPOSITE for {(1-success_rate):.0%} expected success!")
        else:
            st.info("No weak patterns yet (need ≥3 matches with <40% success)")
    
    st.caption("💡 **Your patterns need at least 3 matches to qualify for betting rules**")

# ========== EXPORT REPORT ==========
st.divider()
st.subheader("📤 Export Prediction Report")

report = f"""
⚽ FOOTBALL INTELLIGENCE ENGINE v5.0 - ALGORITHMIC BIAS EXPLOITER
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Storage: {'Supabase Connected' if st.session_state.learning_system.supabase else 'Local Storage Only'}

🎯 YOUR BETTING RULES IN ACTION:
STRONG PATTERN (>70% success, ≥3 matches) → BET STRONGLY
WEAK PATTERN (<40% success, ≥3 matches) → BET OPPOSITE with 85% confidence!
NEUTRAL PATTERN (40-70% success) → Use algorithm with adjusted confidence

🎯 INTELLIGENT BETTING CARD:
{recommendation['icon']} {recommendation['text']}
Type: {recommendation['subtext']}
Confidence: {recommendation['confidence']:.0f}/100
Expected Value: {recommendation.get('expected_value', 0):.3f}
Risk Level: {recommendation.get('risk_level', 'MEDIUM')}
Stake Multiplier: {recommendation.get('stake_multiplier', 1.0):.1f}x
Reason: {recommendation['reason']}

📊 YOUR PATTERN ANALYSIS:
Winner Pattern: {pattern_indicators['winner']['text']}
Winner Explanation: {pattern_indicators['winner']['explanation']}
Winner Confidence: {prediction['winner']['confidence_score']:.0f}/100 ({prediction['winner']['confidence']})
Winner Betting Decision: {prediction['winner'].get('betting_decision', 'N/A')}
Winner Historical Success: {prediction['winner'].get('success_rate', 0):.0%}

Totals Pattern: {pattern_indicators['totals']['text']}
Totals Explanation: {pattern_indicators['totals']['explanation']}
Totals Confidence: {prediction['totals']['confidence_score']:.0f}/100 ({prediction['totals']['confidence']})
Totals Betting Decision: {prediction['totals'].get('betting_decision', 'N/A')}
Totals Historical Success: {prediction['totals'].get('success_rate', 0):.0%}

🎯 WINNER PREDICTION
Predicted Winner: {prediction['winner']['team']}
Probability: {prediction['winner']['probability']*100:.1f}%
Strength: {prediction['winner']['strength']}
Confidence: {prediction['winner']['confidence']} ({prediction['winner']['confidence_score']:.0f}/100)
Most Likely Score: {prediction['winner']['most_likely_score']}
{'🎯 BET OPPOSITE! Algorithm consistently wrong on this pattern' if prediction['winner'].get('opposite_bet') else ''}

🎯 TOTALS PREDICTION  
Direction: {prediction['totals']['direction']} 2.5
Probability: {prediction['probabilities'][f'{prediction["totals"]["direction"].lower()}_2_5_probability']*100:.1f}%
Confidence: {prediction['totals']['confidence']} ({prediction['totals']['confidence_score']:.0f}/100)
Total Expected Goals: {prediction['expected_goals']['total']:.2f}
Finishing Alignment: {prediction['totals'].get('finishing_alignment', 'N/A')}
Total xG Category: {prediction['totals'].get('total_category', 'N/A')}
{'🎯 BET OPPOSITE! Algorithm consistently wrong on this pattern' if prediction['totals'].get('opposite_bet') else ''}

📊 EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

📊 FINISHING TRENDS
{home_team}: {prediction['totals'].get('home_finishing', 0):+.2f} goals_vs_xg/game
{away_team}: {prediction['totals'].get('away_finishing', 0):+.2f} goals_vs_xg/game

⚠️ RISK FLAGS
{', '.join(prediction['totals']['risk_flags']) if prediction['totals']['risk_flags'] else 'None'}

🧠 YOUR LEARNING SYSTEM STATS
Your Outcomes Recorded: {len(st.session_state.learning_system.outcomes)}
Your Patterns Learned: {len(st.session_state.learning_system.pattern_memory)}
Qualifying Patterns (≥3 matches): {len([k for k, v in st.session_state.learning_system.pattern_memory.items() if v['total'] >= 3])}
Strong Patterns (>70%): {len([k for k, v in st.session_state.learning_system.pattern_memory.items() if v['total'] >= 3 and v['success']/v['total'] >= 0.7])}
Weak Patterns (<40%): {len([k for k, v in st.session_state.learning_system.pattern_memory.items() if v['total'] >= 3 and v['success']/v['total'] <= 0.4])}

---
YOUR PROVEN STRATEGY:
1. Identify patterns where algorithm is consistently wrong (<40% success)
2. BET OPPOSITE with 85% confidence
3. Identify patterns where algorithm is consistently right (>70% success)
4. BET STRONGLY with boosted confidence
5. All other patterns → Use standard algorithm
"""

st.code(report, language="text")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"algorithmic_bias_{home_team}_vs_{away_team}.txt",
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
