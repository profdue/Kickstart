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
    page_title="⚽ Football Intelligence Engine v6.0",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine v6.0")
st.markdown("""
    **YOUR DISCOVERED RULES IN ACTION** - Bets opposite when algorithm is consistently wrong
    *Your Data: 4 weak patterns (<40%) vs 2 strong patterns (>70%)*
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

class ImprovedLearningSystem:
    """YOUR IMPROVED RULES with proper thresholds"""
    
    def __init__(self):
        self.pattern_memory = {}
        self.supabase = init_supabase()
        
        # YOUR IMPROVED RULES based on actual data
        self.thresholds = {
            'min_matches': 3,
            'strong_success': 0.70,   # >70% = STRONG (your proven winners)
            'weak_success': 0.40,     # <40% = WEAK (bet opposite!)
            'promising_min': 0.60,    # 60-70% = PROMISING (small boost)
            'promising_max': 0.70
        }
        
        self.load_learning()
    
    def save_learning(self):
        """Save learning data to Supabase"""
        try:
            if not self.supabase:
                return self._save_learning_local()
            
            # Prepare data
            supabase_data = []
            for pattern_key, stats in self.pattern_memory.items():
                if stats['total'] == 0:
                    continue
                    
                data = {
                    "pattern_key": pattern_key,
                    "total_matches": stats['total'],
                    "successful_matches": stats['success'],
                    "last_updated": datetime.now().isoformat(),
                    "metadata": json.dumps({
                        "thresholds": self.thresholds,
                        "success_rate": stats['success'] / stats['total'] if stats['total'] > 0 else 0
                    })
                }
                supabase_data.append(data)
            
            # Save to Supabase
            if supabase_data:
                self.supabase.table("football_learning").delete().neq("pattern_key", "dummy").execute()
                response = self.supabase.table("football_learning").insert(supabase_data).execute()
                return True
                
            return True
            
        except Exception as e:
            st.error(f"Supabase save error: {e}")
            return self._save_learning_local()
    
    def _save_learning_local(self):
        """Fallback local storage"""
        try:
            with open("learning_data.pkl", "wb") as f:
                pickle.dump({
                    'pattern_memory': self.pattern_memory,
                    'thresholds': self.thresholds
                }, f)
            return True
        except Exception as e:
            st.error(f"Local save error: {e}")
            return False
    
    def load_learning(self):
        """Load learning data from Supabase"""
        try:
            if not self.supabase:
                return self._load_learning_local()
            
            response = self.supabase.table("football_learning").select("*").execute()
            
            if response.data:
                for row in response.data:
                    self.pattern_memory[row['pattern_key']] = {
                        'total': row['total_matches'] or 0,
                        'success': row['successful_matches'] or 0
                    }
                
                # Load thresholds from metadata if available
                for row in response.data:
                    if row.get('metadata'):
                        try:
                            metadata = json.loads(row['metadata'])
                            if 'thresholds' in metadata:
                                self.thresholds.update(metadata['thresholds'])
                                break
                        except:
                            pass
                
                return True
            
            return True
            
        except Exception as e:
            return self._load_learning_local()
    
    def _load_learning_local(self):
        """Fallback local storage"""
        try:
            if os.path.exists("learning_data.pkl"):
                with open("learning_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.pattern_memory = data.get('pattern_memory', {})
                    self.thresholds = data.get('thresholds', self.thresholds)
                return True
        except:
            pass
        return False
    
    def record_outcome(self, prediction, actual_score):
        """Record match outcome - SIMPLE and CLEAR"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Parse score
        try:
            home_goals, away_goals = map(int, actual_score.split('-'))
        except:
            return None, "Invalid score format"
        
        # Determine actual outcomes
        if home_goals > away_goals:
            actual_winner = "HOME"
        elif away_goals > home_goals:
            actual_winner = "AWAY"
        else:
            actual_winner = "DRAW"
        
        total_goals = home_goals + away_goals
        actual_over = total_goals > 2.5
        
        # Create pattern keys (SIMPLE and CONSISTENT)
        winner_key = f"WINNER_{winner_pred['original_prediction']}_{winner_pred['original_confidence']}"
        totals_key = f"TOTALS_{totals_pred['original_finishing_alignment']}_{totals_pred['original_total_category']}"
        
        # Initialize if not exists
        if winner_key not in self.pattern_memory:
            self.pattern_memory[winner_key] = {'total': 0, 'success': 0}
        if totals_key not in self.pattern_memory:
            self.pattern_memory[totals_key] = {'total': 0, 'success': 0}
        
        # Check predictions
        winner_correct = winner_pred['original_prediction'] == actual_winner
        totals_correct = (totals_pred['original_direction'] == "OVER") == actual_over
        
        # Update patterns
        self.pattern_memory[winner_key]['total'] += 1
        self.pattern_memory[winner_key]['success'] += 1 if winner_correct else 0
        
        self.pattern_memory[totals_key]['total'] += 1
        self.pattern_memory[totals_key]['success'] += 1 if totals_correct else 0
        
        # Save to Supabase
        save_success = self.save_learning()
        
        return {
            'winner_correct': winner_correct,
            'totals_correct': totals_correct,
            'winner_key': winner_key,
            'totals_key': totals_key,
            'save_success': save_success
        }, "Outcome recorded successfully!"
    
    def get_betting_advice(self, winner_pred, totals_pred):
        """APPLY YOUR IMPROVED RULES"""
        
        # Create pattern keys
        winner_key = f"WINNER_{winner_pred['type']}_{winner_pred['confidence']}"
        totals_key = f"TOTALS_{totals_pred.get('finishing_alignment', 'N/A')}_{totals_pred.get('total_category', 'N/A')}"
        
        advice = {
            'winner': {'action': 'FOLLOW', 'bet_on': winner_pred['type'], 'confidence': winner_pred['confidence_score']},
            'totals': {'action': 'FOLLOW', 'bet_on': totals_pred['direction'], 'confidence': totals_pred['confidence_score']}
        }
        
        # Apply YOUR RULES to each market
        for market_type, pattern_key, original in [
            ('winner', winner_key, winner_pred),
            ('totals', totals_key, totals_pred)
        ]:
            if pattern_key in self.pattern_memory:
                stats = self.pattern_memory[pattern_key]
                
                if stats['total'] >= self.thresholds['min_matches']:
                    success_rate = stats['success'] / stats['total']
                    
                    # YOUR RULE 1: STRONG PATTERN (>70% success)
                    if success_rate > self.thresholds['strong_success']:
                        advice[market_type]['action'] = 'BET_STRONGLY'
                        # Boost confidence for strong patterns
                        advice[market_type]['confidence'] = min(95, original['confidence_score'] * 1.3)
                        advice[market_type]['reason'] = f"STRONG PATTERN: {stats['success']}/{stats['total']} ({success_rate:.0%}) wins"
                        advice[market_type]['color'] = 'green'
                    
                    # YOUR RULE 2: PROMISING PATTERN (60-70% success)
                    elif (success_rate >= self.thresholds['promising_min'] and 
                          success_rate <= self.thresholds['promising_max']):
                        advice[market_type]['action'] = 'PROMISING'
                        # Small confidence boost for promising patterns
                        advice[market_type]['confidence'] = min(85, original['confidence_score'] * 1.15)
                        advice[market_type]['reason'] = f"PROMISING: {stats['success']}/{stats['total']} ({success_rate:.0%}) wins"
                        advice[market_type]['color'] = 'blue'
                    
                    # YOUR RULE 3: WEAK PATTERN (<40% success) → BET OPPOSITE!
                    elif success_rate < self.thresholds['weak_success']:
                        advice[market_type]['action'] = 'BET_OPPOSITE'
                        advice[market_type]['confidence'] = 85  # Your rule: 85% confidence in opposite!
                        
                        # Determine opposite bet
                        if market_type == 'winner':
                            if original['type'] == 'HOME':
                                advice[market_type]['bet_on'] = 'AWAY'
                            elif original['type'] == 'AWAY':
                                advice[market_type]['bet_on'] = 'HOME'
                            else:
                                advice[market_type]['bet_on'] = 'DRAW'
                        else:
                            advice[market_type]['bet_on'] = 'UNDER' if original['direction'] == 'OVER' else 'OVER'
                        
                        advice[market_type]['reason'] = f"WEAK PATTERN: Only {stats['success']}/{stats['total']} ({success_rate:.0%}) wins → BET OPPOSITE!"
                        advice[market_type]['color'] = 'red'
                    
                    # RULE 4: NEUTRAL PATTERN (40-60% success)
                    else:
                        advice[market_type]['action'] = 'NEUTRAL'
                        # Small adjustment toward 50%
                        adjustment = (0.5 - success_rate) * 20
                        advice[market_type]['confidence'] = max(20, min(80, original['confidence_score'] + adjustment))
                        advice[market_type]['reason'] = f"NEUTRAL: {stats['success']}/{stats['total']} ({success_rate:.0%}) wins"
                        advice[market_type]['color'] = 'gray'
        
        return advice
    
    def get_pattern_stats(self, pattern_key):
        """Get stats for a pattern"""
        return self.pattern_memory.get(pattern_key, {'total': 0, 'success': 0})
    
    def get_pattern_success_rate(self, pattern_key):
        """Get success rate for a pattern"""
        stats = self.get_pattern_stats(pattern_key)
        if stats['total'] > 0:
            return stats['success'] / stats['total']
        return 0.5  # Default to 50%

# ========== INITIALIZE SESSION STATES ==========
if 'learning_system' not in st.session_state:
    st.session_state.learning_system = ImprovedLearningSystem()

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'last_teams' not in st.session_state:
    st.session_state.last_teams = None

if 'save_status' not in st.session_state:
    st.session_state.save_status = None

def factorial_cache(n, cache={}):
    if n not in cache:
        cache[n] = math.factorial(n)
    return cache[n]

def poisson_pmf(k, lam):
    """Poisson probability mass function"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

# ========== CORE PREDICTION CLASSES ==========

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
            'strength': strength,
            'confidence_score': winner_confidence,
            'confidence': confidence_category,
            'delta': delta
        }

class TotalsPredictor:
    """Totals prediction"""
    
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
        """Finishing trend alignment"""
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
    
    def predict_totals(self, home_xg, away_xg, home_stats, away_stats):
        """Predict totals"""
        total_xg = home_xg + away_xg
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        over_threshold = self.league_adjustments['over_threshold']
        base_direction = "OVER" if total_xg > over_threshold else "UNDER"
        
        # Finishing alignment
        finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
        total_category = self.categorize_total_xg(total_xg)
        
        # Risk flags
        risk_flags = []
        if abs(home_finish) > 0.4 or abs(away_finish) > 0.4:
            risk_flags.append("HIGH_VARIANCE_TEAM")
        
        lower_thresh = self.league_adjustments['under_threshold'] - 0.1
        upper_thresh = self.league_adjustments['over_threshold'] + 0.1
        if lower_thresh < total_xg < upper_thresh:
            risk_flags.append("CLOSE_TO_THRESHOLD")
        
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
        
        # Get decision
        direction = base_direction
        confidence_category = "LOW"
        base_confidence = 40
        
        if total_category in decision_matrix and finishing_alignment in decision_matrix[total_category]:
            direction, confidence_category, base_confidence = decision_matrix[total_category][finishing_alignment]
        
        # Apply risk flag penalties
        final_confidence = base_confidence
        for flag in risk_flags:
            if flag == "HIGH_VARIANCE_TEAM":
                final_confidence -= 15
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
            'away_finishing': away_finish
        }

class PoissonProbabilityEngine:
    """Calculate probabilities"""
    
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

# ========== IMPROVED FOOTBALL ENGINE ==========

class ImprovedFootballEngine:
    """Engine that applies YOUR IMPROVED RULES"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_name = league_name
        
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate prediction with YOUR RULES"""
        
        # Get base prediction
        home_xg, away_xg = self.xg_predictor.predict_expected_goals(home_stats, away_stats)
        
        probabilities = self.probability_engine.calculate_all_probabilities(home_xg, away_xg)
        
        winner_prediction = self.winner_predictor.predict_winner(
            home_xg, away_xg, home_stats, away_stats
        )
        
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Get betting advice from YOUR IMPROVED RULES
        betting_advice = st.session_state.learning_system.get_betting_advice(
            winner_prediction, totals_prediction
        )
        
        # Apply betting advice
        final_winner = self._apply_advice_to_winner(
            winner_prediction, betting_advice['winner'], home_team, away_team
        )
        
        final_totals = self._apply_advice_to_totals(
            totals_prediction, betting_advice['totals']
        )
        
        # Get probabilities
        winner_prob = self._get_probability_for_winner(final_winner, probabilities)
        totals_prob = self._get_probability_for_totals(final_totals, probabilities)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'winner': {
                'team': final_winner['team'],
                'type': final_winner['type'],
                'probability': winner_prob,
                'confidence': final_winner['confidence'],
                'confidence_score': final_winner['confidence_score'],
                'strength': final_winner.get('strength', 'N/A'),
                'most_likely_score': probabilities['most_likely_score'],
                'betting_action': betting_advice['winner']['action'],
                'original_prediction': winner_prediction['type'],
                'original_confidence': winner_prediction['confidence'],
                'reason': betting_advice['winner'].get('reason', 'Algorithm prediction'),
                'color': betting_advice['winner'].get('color', 'gray')
            },
            
            'totals': {
                'direction': final_totals['direction'],
                'probability': totals_prob,
                'confidence': final_totals['confidence'],
                'confidence_score': final_totals['confidence_score'],
                'total_xg': totals_prediction['total_xg'],
                'finishing_alignment': totals_prediction.get('finishing_alignment'),
                'total_category': totals_prediction.get('total_category'),
                'risk_flags': totals_prediction.get('risk_flags', []),
                'betting_action': betting_advice['totals']['action'],
                'original_direction': totals_prediction['direction'],
                'original_finishing_alignment': totals_prediction.get('finishing_alignment'),
                'original_total_category': totals_prediction.get('total_category'),
                'reason': betting_advice['totals'].get('reason', 'Algorithm prediction'),
                'color': betting_advice['totals'].get('color', 'gray')
            },
            
            'probabilities': probabilities,
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': home_xg + away_xg},
            'betting_advice': betting_advice
        }
    
    def _apply_advice_to_winner(self, original, advice, home_team, away_team):
        """Apply betting advice to winner"""
        final = original.copy()
        
        if advice['action'] == 'BET_OPPOSITE':
            # Bet opposite!
            if original['type'] == 'HOME':
                final['type'] = 'AWAY'
                final['team'] = away_team
            elif original['type'] == 'AWAY':
                final['type'] = 'HOME'
                final['team'] = home_team
            else:
                final['type'] = 'DRAW'
                final['team'] = 'DRAW'
            
            final['confidence_score'] = advice['confidence']
            final['confidence'] = 'HIGH' if advice['confidence'] >= 65 else 'MEDIUM'
        
        elif advice['action'] in ['BET_STRONGLY', 'PROMISING']:
            # Boost confidence
            final['confidence_score'] = advice['confidence']
            if advice['confidence'] >= 75:
                final['confidence'] = 'VERY HIGH'
            elif advice['confidence'] >= 65:
                final['confidence'] = 'HIGH'
            else:
                final['confidence'] = 'MEDIUM'
            
            # Set team name
            final['team'] = home_team if original['type'] == 'HOME' else away_team if original['type'] == 'AWAY' else 'DRAW'
        
        else:
            # Use algorithm
            final['team'] = home_team if original['type'] == 'HOME' else away_team if original['type'] == 'AWAY' else 'DRAW'
        
        return final
    
    def _apply_advice_to_totals(self, original, advice):
        """Apply betting advice to totals"""
        final = original.copy()
        
        if advice['action'] == 'BET_OPPOSITE':
            # Bet opposite!
            final['direction'] = 'UNDER' if original['direction'] == 'OVER' else 'OVER'
            final['confidence_score'] = advice['confidence']
            final['confidence'] = 'HIGH' if advice['confidence'] >= 65 else 'MEDIUM'
        
        elif advice['action'] in ['BET_STRONGLY', 'PROMISING']:
            # Boost confidence
            final['confidence_score'] = advice['confidence']
            if advice['confidence'] >= 75:
                final['confidence'] = 'VERY HIGH'
            elif advice['confidence'] >= 65:
                final['confidence'] = 'HIGH'
            else:
                final['confidence'] = 'MEDIUM'
        
        return final
    
    def _get_probability_for_winner(self, winner_pred, probabilities):
        """Get probability for winner"""
        if winner_pred['type'] == 'HOME':
            return probabilities['home_win_probability']
        elif winner_pred['type'] == 'AWAY':
            return probabilities['away_win_probability']
        else:
            return probabilities['draw_probability']
    
    def _get_probability_for_totals(self, totals_pred, probabilities):
        """Get probability for totals"""
        if totals_pred['direction'] == 'OVER':
            return probabilities['over_2_5_probability']
        else:
            return probabilities['under_2_5_probability']

# ========== IMPROVED BETTING CARD ==========

class ImprovedBettingCard:
    """Betting card that shows YOUR IMPROVED RULES"""
    
    @staticmethod
    def get_recommendation(prediction):
        """Get betting recommendation"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Check for opposite bets
        if winner_pred['betting_action'] == 'BET_OPPOSITE' and totals_pred['betting_action'] == 'BET_OPPOSITE':
            return {
                'type': 'DOUBLE_OPPOSITE',
                'text': f"🎯 {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                'subtext': 'DOUBLE BET OPPOSITE!',
                'reason': f"Algorithm consistently wrong on both patterns",
                'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#DC2626',
                'icon': '🎯'
            }
        
        elif winner_pred['betting_action'] == 'BET_OPPOSITE':
            return {
                'type': 'WINNER_OPPOSITE',
                'text': f"🎯 {winner_pred['team']} to win",
                'subtext': 'BET OPPOSITE WINNER!',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯'
            }
        
        elif totals_pred['betting_action'] == 'BET_OPPOSITE':
            return {
                'type': 'TOTALS_OPPOSITE',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'BET OPPOSITE TOTALS!',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯'
            }
        
        elif winner_pred['betting_action'] == 'BET_STRONGLY' and totals_pred['betting_action'] == 'BET_STRONGLY':
            return {
                'type': 'DOUBLE_STRONG',
                'text': f"✅ {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                'subtext': 'DOUBLE STRONG PATTERN',
                'reason': f"Both patterns have >70% historical success",
                'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#10B981',
                'icon': '✅'
            }
        
        elif winner_pred['betting_action'] == 'BET_STRONGLY':
            return {
                'type': 'WINNER_STRONG',
                'text': f"✅ {winner_pred['team']} to win",
                'subtext': 'STRONG PATTERN (>70%)',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅'
            }
        
        elif totals_pred['betting_action'] == 'BET_STRONGLY':
            return {
                'type': 'TOTALS_STRONG',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'STRONG PATTERN (>70%)',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅'
            }
        
        elif winner_pred['betting_action'] == 'PROMISING' and totals_pred['betting_action'] == 'PROMISING':
            return {
                'type': 'DOUBLE_PROMISING',
                'text': f"🔵 {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                'subtext': 'DOUBLE PROMISING (60-70%)',
                'reason': f"Both patterns have 60-70% historical success",
                'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#3B82F6',
                'icon': '🔵'
            }
        
        elif winner_pred['betting_action'] == 'PROMISING':
            return {
                'type': 'WINNER_PROMISING',
                'text': f"🔵 {winner_pred['team']} to win",
                'subtext': 'PROMISING PATTERN (60-70%)',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#3B82F6',
                'icon': '🔵'
            }
        
        elif totals_pred['betting_action'] == 'PROMISING':
            return {
                'type': 'TOTALS_PROMISING',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'PROMISING PATTERN (60-70%)',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#3B82F6',
                'icon': '🔵'
            }
        
        else:
            # Calculate expected value
            winner_ev = winner_pred['probability'] - 0.5
            totals_ev = totals_pred['probability'] - 0.5
            
            if winner_ev > 0.1 and totals_ev > 0.1:
                return {
                    'type': 'COMBO',
                    'text': f"🎯 {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                    'subtext': 'DOUBLE BET',
                    'reason': f'Both positive expected value',
                    'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']),
                    'color': '#8B5CF6',
                    'icon': '🎯'
                }
            elif winner_ev > 0.15:
                return {
                    'type': 'SINGLE_WINNER',
                    'text': f"🏆 {winner_pred['team']} to win",
                    'subtext': 'WINNER BET',
                    'reason': f'Positive expected value',
                    'confidence': winner_pred['confidence_score'],
                    'color': '#8B5CF6',
                    'icon': '🏆'
                }
            elif totals_ev > 0.15:
                return {
                    'type': 'SINGLE_TOTALS',
                    'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                    'subtext': 'TOTALS BET',
                    'reason': f'Positive expected value',
                    'confidence': totals_pred['confidence_score'],
                    'color': '#8B5CF6',
                    'icon': '📈'
                }
            else:
                return {
                    'type': 'NO_BET',
                    'text': "🤔 No Clear Bet",
                    'subtext': 'NO BET',
                    'reason': f'Insufficient expected value or neutral patterns',
                    'confidence': max(winner_pred['confidence_score'], totals_pred['confidence_score']),
                    'color': '#6B7280',
                    'icon': '🤔'
                }
    
    @staticmethod
    def display_card(recommendation):
        """Display the betting card"""
        color = recommendation['color']
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}20 0%, #1F2937 100%);
            padding: 25px;
            border-radius: 20px;
            border: 3px solid {color};
            text-align: center;
            margin: 20px 0;
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
                Confidence: {recommendation['confidence']:.0f}/100
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

# ========== FEEDBACK SYSTEM ==========

def record_outcome_improved(prediction):
    """Improved feedback system with Supabase confirmation"""
    
    st.divider()
    st.subheader("📝 Record Outcome for Learning")
    
    # Show current patterns
    winner_key = f"WINNER_{prediction['winner']['original_prediction']}_{prediction['winner']['original_confidence']}"
    totals_key = f"TOTALS_{prediction['totals']['original_finishing_alignment']}_{prediction['totals']['original_total_category']}"
    
    winner_stats = st.session_state.learning_system.get_pattern_stats(winner_key)
    totals_stats = st.session_state.learning_system.get_pattern_stats(totals_key)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Winner Pattern:**")
        st.code(winner_key)
        if winner_stats['total'] > 0:
            success = winner_stats['success'] / winner_stats['total']
            st.write(f"Current: {winner_stats['success']}/{winner_stats['total']} ({success:.0%})")
            if winner_stats['total'] >= 3:
                if success > 0.7:
                    st.success("✅ STRONG PATTERN (>70%)")
                elif success >= 0.6:
                    st.info("🔵 PROMISING (60-70%)")
                elif success < 0.4:
                    st.error("🎯 WEAK PATTERN (<40%) - BET OPPOSITE!")
                else:
                    st.info("⚪ NEUTRAL (40-60%)")
    
    with col2:
        st.write("**Totals Pattern:**")
        st.code(totals_key)
        if totals_stats['total'] > 0:
            success = totals_stats['success'] / totals_stats['total']
            st.write(f"Current: {totals_stats['success']}/{totals_stats['total']} ({success:.0%})")
            if totals_stats['total'] >= 3:
                if success > 0.7:
                    st.success("✅ STRONG PATTERN (>70%)")
                elif success >= 0.6:
                    st.info("🔵 PROMISING (60-70%)")
                elif success < 0.4:
                    st.error("🎯 WEAK PATTERN (<40%) - BET OPPOSITE!")
                else:
                    st.info("⚪ NEUTRAL (40-60%)")
    
    # Score input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        score = st.text_input("Actual Score (e.g., 2-1)", key="score_input")
    
    with col2:
        if st.button("✅ Record Outcome", type="primary", use_container_width=True):
            if not score or '-' not in score:
                st.error("Enter valid score like '2-1'")
            else:
                try:
                    with st.spinner("Saving to Supabase..."):
                        result, message = st.session_state.learning_system.record_outcome(prediction, score)
                        
                        if result:
                            if result['save_success']:
                                st.success("✅ Saved to Supabase successfully!")
                            else:
                                st.warning("⚠️ Saved locally (Supabase failed)")
                            
                            # Show what happened
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if result['winner_correct']:
                                    st.success(f"✅ Winner correct! Pattern updated")
                                else:
                                    st.error(f"❌ Winner wrong! Pattern updated")
                            
                            with col2:
                                if result['totals_correct']:
                                    st.success(f"✅ Totals correct! Pattern updated")
                                else:
                                    st.error(f"❌ Totals wrong! Pattern updated")
                            
                            st.rerun()
                        else:
                            st.error(message)
                            
                except ValueError:
                    st.error("Enter numbers like '2-1'")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

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

    # Learning System Section
    st.divider()
    st.header("🎯 YOUR IMPROVED RULES")
    
    st.info("""
    **BASED ON YOUR DATA:**
    
    ✅ **STRONG** (>70% success, ≥3 matches)
    → BET STRONGLY with boosted confidence
    
    🔵 **PROMISING** (60-70% success, ≥3 matches)
    → Small confidence boost
    
    ⚪ **NEUTRAL** (40-60% success, ≥3 matches)
    → Use algorithm's prediction
    
    🎯 **WEAK** (<40% success, ≥3 matches)
    → BET OPPOSITE with 85% confidence!
    """)
    
    # Show current match patterns if available
    if st.session_state.last_prediction:
        winner_pred = st.session_state.last_prediction['winner']
        totals_pred = st.session_state.last_prediction['totals']
        
        winner_key = f"WINNER_{winner_pred['original_prediction']}_{winner_pred['original_confidence']}"
        totals_key = f"TOTALS_{totals_pred['original_finishing_alignment']}_{totals_pred['original_total_category']}"
        
        winner_stats = st.session_state.learning_system.get_pattern_stats(winner_key)
        totals_stats = st.session_state.learning_system.get_pattern_stats(totals_key)
        
        st.subheader("Current Match:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if winner_stats['total'] >= 3:
                success = winner_stats['success'] / winner_stats['total']
                if success > 0.7:
                    st.success(f"✅ Winner: {success:.0%}")
                elif success >= 0.6:
                    st.info(f"🔵 Winner: {success:.0%}")
                elif success < 0.4:
                    st.error(f"🎯 Winner: {success:.0%}")
                else:
                    st.info(f"⚪ Winner: {success:.0%}")
            else:
                st.caption(f"Winner: {winner_stats['total']}/3 matches")
        
        with col2:
            if totals_stats['total'] >= 3:
                success = totals_stats['success'] / totals_stats['total']
                if success > 0.7:
                    st.success(f"✅ Totals: {success:.0%}")
                elif success >= 0.6:
                    st.info(f"🔵 Totals: {success:.0%}")
                elif success < 0.4:
                    st.error(f"🎯 Totals: {success:.0%}")
                else:
                    st.info(f"⚪ Totals: {success:.0%}")
            else:
                st.caption(f"Totals: {totals_stats['total']}/3 matches")
    
    st.divider()
    
    # Show your best patterns
    st.subheader("Your Best Patterns:")
    
    patterns = list(st.session_state.learning_system.pattern_memory.items())
    qualifying_patterns = [(k, v) for k, v in patterns if v['total'] >= 3]
    
    if qualifying_patterns:
        # Sort by success rate
        qualifying_patterns.sort(key=lambda x: x[1]['success'] / x[1]['total'], reverse=True)
        
        for i, (pattern, stats) in enumerate(qualifying_patterns[:6]):
            success = stats['success'] / stats['total']
            if success > 0.7:
                st.success(f"✅ {pattern[:25]}...: {success:.0%} ({stats['success']}/{stats['total']})")
            elif success >= 0.6:
                st.info(f"🔵 {pattern[:25]}...: {success:.0%} ({stats['success']}/{stats['total']})")
            elif success < 0.4:
                st.error(f"🎯 {pattern[:25]}...: {success:.0%} ({stats['success']}/{stats['total']})")
            else:
                st.info(f"⚪ {pattern[:25]}...: {success:.0%} ({stats['success']}/{stats['total']})")
    else:
        st.caption("No patterns with 3+ matches yet")
    
    # Statistics
    st.divider()
    st.subheader("Statistics:")
    
    total_patterns = len(st.session_state.learning_system.pattern_memory)
    qualifying = len([v for v in st.session_state.learning_system.pattern_memory.values() if v['total'] >= 3])
    strong = len([v for v in st.session_state.learning_system.pattern_memory.values() 
                 if v['total'] >= 3 and v['success']/v['total'] > 0.7])
    promising = len([v for v in st.session_state.learning_system.pattern_memory.values() 
                    if v['total'] >= 3 and 0.6 <= v['success']/v['total'] <= 0.7])
    weak = len([v for v in st.session_state.learning_system.pattern_memory.values() 
               if v['total'] >= 3 and v['success']/v['total'] < 0.4])
    
    st.write(f"Total patterns: {total_patterns}")
    st.write(f"Qualifying (≥3 matches): {qualifying}")
    st.write(f"Strong (>70%): {strong}")
    st.write(f"Promising (60-70%): {promising}")
    st.write(f"Weak (<40%): {weak}")
    
    # Supabase status
    if st.session_state.learning_system.supabase:
        st.success("✅ Connected to Supabase")
    else:
        st.warning("⚠️ Local storage only")

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
        engine = ImprovedFootballEngine(league_metrics, selected_league)
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
st.caption(f"League: {selected_league}")

# Main prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction['winner']
    color = winner_pred.get('color', '#6B7280')
    
    # Determine icon and subtitle based on betting action
    if winner_pred['betting_action'] == 'BET_OPPOSITE':
        icon = "🎯"
        subtitle = "BET OPPOSITE!"
        card_color = "#7F1D1D"
    elif winner_pred['betting_action'] == 'BET_STRONGLY':
        icon = "✅"
        subtitle = "STRONG PATTERN"
        card_color = "#14532D"
    elif winner_pred['betting_action'] == 'PROMISING':
        icon = "🔵"
        subtitle = "PROMISING"
        card_color = "#1E3A8A"
    else:
        if winner_pred['type'] == "HOME":
            icon = "🏠"
        elif winner_pred['type'] == "AWAY":
            icon = "✈️"
        else:
            icon = "🤝"
        
        if winner_pred['confidence'] == "VERY HIGH":
            card_color = "#14532D"
        elif winner_pred['confidence'] == "HIGH":
            card_color = "#166534"
        elif winner_pred['confidence'] == "MEDIUM":
            card_color = "#365314"
        else:
            card_color = "#1E293B"
        
        subtitle = winner_pred['confidence']
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">PREDICTED WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {icon} {winner_pred['team']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {subtitle} | Confidence: {winner_pred['confidence_score']:.0f}/100
        </div>
        {f'<div style="font-size: 14px; color: #FCA5A5; margin-top: 10px;">🎯 {winner_pred["reason"]}</div>' if winner_pred['betting_action'] == 'BET_OPPOSITE' else ''}
        {f'<div style="font-size: 14px; color: #BBF7D0; margin-top: 10px;">✅ {winner_pred["reason"]}</div>' if winner_pred['betting_action'] == 'BET_STRONGLY' else ''}
        {f'<div style="font-size: 14px; color: #BFDBFE; margin-top: 10px;">🔵 {winner_pred["reason"]}</div>' if winner_pred['betting_action'] == 'PROMISING' else ''}
    </div>
    """, unsafe_allow_html=True)

with col2:
    totals_pred = prediction['totals']
    color = totals_pred.get('color', '#6B7280')
    
    # Determine icon and subtitle
    if totals_pred['betting_action'] == 'BET_OPPOSITE':
        icon = "🎯"
        subtitle = "BET OPPOSITE!"
        card_color = "#7F1D1D"
    elif totals_pred['betting_action'] == 'BET_STRONGLY':
        icon = "✅"
        subtitle = "STRONG PATTERN"
        card_color = "#14532D"
    elif totals_pred['betting_action'] == 'PROMISING':
        icon = "🔵"
        subtitle = "PROMISING"
        card_color = "#1E3A8A"
    else:
        icon = "📈"
        if totals_pred['confidence'] == "VERY HIGH":
            card_color = "#14532D" if totals_pred['direction'] == "OVER" else "#7F1D1D"
        elif totals_pred['confidence'] == "HIGH":
            card_color = "#166534" if totals_pred['direction'] == "OVER" else "#991B1B"
        elif totals_pred['confidence'] == "MEDIUM":
            card_color = "#365314" if totals_pred['direction'] == "OVER" else "#78350F"
        else:
            card_color = "#1E293B"
        subtitle = totals_pred['confidence']
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">TOTAL GOALS</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {icon} {totals_pred['direction']} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {totals_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {subtitle} | Confidence: {totals_pred['confidence_score']:.0f}/100
        </div>
        {f'<div style="font-size: 14px; color: #FCA5A5; margin-top: 10px;">🎯 {totals_pred["reason"]}</div>' if totals_pred['betting_action'] == 'BET_OPPOSITE' else ''}
        {f'<div style="font-size: 14px; color: #BBF7D0; margin-top: 10px;">✅ {totals_pred["reason"]}</div>' if totals_pred['betting_action'] == 'BET_STRONGLY' else ''}
        {f'<div style="font-size: 14px; color: #BFDBFE; margin-top: 10px;">🔵 {totals_pred["reason"]}</div>' if totals_pred['betting_action'] == 'PROMISING' else ''}
    </div>
    """, unsafe_allow_html=True)

# ========== BETTING CARD ==========
st.divider()
st.subheader("🎯 YOUR BETTING ADVICE")

recommendation = ImprovedBettingCard.get_recommendation(prediction)
ImprovedBettingCard.display_card(recommendation)

# ========== PATTERN EXPLANATION ==========
st.divider()
st.subheader("🔍 Pattern Analysis")

col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction['winner']
    winner_key = f"WINNER_{winner_pred['original_prediction']}_{winner_pred['original_confidence']}"
    winner_stats = st.session_state.learning_system.get_pattern_stats(winner_key)
    
    st.write("**Winner Pattern:**")
    st.code(winner_key)
    
    if winner_stats['total'] > 0:
        success_rate = winner_stats['success'] / winner_stats['total']
        
        if winner_stats['total'] >= 3:
            if success_rate > 0.7:
                st.success(f"✅ STRONG PATTERN: {winner_stats['success']}/{winner_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **BET STRONGLY** with boosted confidence")
            elif success_rate >= 0.6:
                st.info(f"🔵 PROMISING: {winner_stats['success']}/{winner_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **Small confidence boost** for promising pattern")
            elif success_rate < 0.4:
                st.error(f"🎯 WEAK PATTERN: {winner_stats['success']}/{winner_stats['total']} ({success_rate:.0%}) wins")
                st.write(f"→ **BET OPPOSITE!** Algorithm only wins {success_rate:.0%} of the time")
            else:
                st.info(f"⚪ NEUTRAL: {winner_stats['success']}/{winner_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **Use algorithm's prediction**")
        else:
            st.warning(f"⚠️ NEED MORE DATA: {winner_stats['total']}/3 matches")
            st.write(f"→ Need {3 - winner_stats['total']} more match(es) to apply rules")
    else:
        st.info("🆕 NEW PATTERN")
        st.write("→ Record outcome to start tracking")

with col2:
    totals_pred = prediction['totals']
    totals_key = f"TOTALS_{totals_pred['original_finishing_alignment']}_{totals_pred['original_total_category']}"
    totals_stats = st.session_state.learning_system.get_pattern_stats(totals_key)
    
    st.write("**Totals Pattern:**")
    st.code(totals_key)
    
    if totals_stats['total'] > 0:
        success_rate = totals_stats['success'] / totals_stats['total']
        
        if totals_stats['total'] >= 3:
            if success_rate > 0.7:
                st.success(f"✅ STRONG PATTERN: {totals_stats['success']}/{totals_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **BET STRONGLY** with boosted confidence")
            elif success_rate >= 0.6:
                st.info(f"🔵 PROMISING: {totals_stats['success']}/{totals_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **Small confidence boost** for promising pattern")
            elif success_rate < 0.4:
                st.error(f"🎯 WEAK PATTERN: {totals_stats['success']}/{totals_stats['total']} ({success_rate:.0%}) wins")
                st.write(f"→ **BET OPPOSITE!** Algorithm only wins {success_rate:.0%} of the time")
            else:
                st.info(f"⚪ NEUTRAL: {totals_stats['success']}/{totals_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **Use algorithm's prediction**")
        else:
            st.warning(f"⚠️ NEED MORE DATA: {totals_stats['total']}/3 matches")
            st.write(f"→ Need {3 - totals_stats['total']} more match(es) to apply rules")
    else:
        st.info("🆕 NEW PATTERN")
        st.write("→ Record outcome to start tracking")

# ========== INSIGHTS ==========
st.divider()
st.subheader("🧠 Insights")

if winner_pred['betting_action'] == 'BET_OPPOSITE':
    st.error(f"🎯 **WINNER**: {winner_pred['reason']}")

if totals_pred['betting_action'] == 'BET_OPPOSITE':
    st.error(f"🎯 **TOTALS**: {totals_pred['reason']}")

if winner_pred['betting_action'] == 'BET_STRONGLY':
    st.success(f"✅ **WINNER**: {winner_pred['reason']}")

if totals_pred['betting_action'] == 'BET_STRONGLY':
    st.success(f"✅ **TOTALS**: {totals_pred['reason']}")

if winner_pred['betting_action'] == 'PROMISING':
    st.info(f"🔵 **WINNER**: {winner_pred['reason']}")

if totals_pred['betting_action'] == 'PROMISING':
    st.info(f"🔵 **TOTALS**: {totals_pred['reason']}")

# Risk flags
if prediction['totals']['risk_flags']:
    st.warning(f"⚠️ **Risk Flags**: {', '.join(prediction['totals']['risk_flags'])}")

# ========== PROBABILITIES ==========
st.divider()
st.subheader("🎲 Probabilities")

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

# Most likely scores
st.subheader("🎯 Most Likely Scores")
scores_cols = st.columns(5)
for idx, (score, prob) in enumerate(prediction['probabilities']['top_scores'][:5]):
    with scores_cols[idx]:
        st.metric(f"{score}", f"{prob*100:.1f}%")

# Expected goals
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

# ========== RECORD OUTCOME ==========
record_outcome_improved(prediction)

# ========== EXPORT ==========
st.divider()
st.subheader("📤 Export")

report = f"""⚽ FOOTBALL INTELLIGENCE ENGINE v6.0 - YOUR IMPROVED RULES
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 YOUR BETTING ADVICE:
{recommendation['icon']} {recommendation['text']}
{recommendation['subtext']}
Reason: {recommendation['reason']}
Confidence: {recommendation['confidence']:.0f}/100

🎯 WINNER:
Bet on: {winner_pred['team']}
Original algorithm: {winner_pred['original_prediction']}
Betting action: {winner_pred['betting_action']}
Reason: {winner_pred['reason']}
Probability: {winner_pred['probability']*100:.1f}%
Confidence: {winner_pred['confidence']} ({winner_pred['confidence_score']:.0f}/100)
Pattern: WINNER_{winner_pred['original_prediction']}_{winner_pred['original_confidence']}
Pattern stats: {winner_stats['success']}/{winner_stats['total']} wins ({winner_stats['success']/winner_stats['total']:.0% if winner_stats['total'] > 0 else 'N/A'})

🎯 TOTALS:
Bet on: {totals_pred['direction']} 2.5
Original algorithm: {totals_pred['original_direction']} 2.5
Betting action: {totals_pred['betting_action']}
Reason: {totals_pred['reason']}
Probability: {totals_pred['probability']*100:.1f}%
Confidence: {totals_pred['confidence']} ({totals_pred['confidence_score']:.0f}/100)
Pattern: TOTALS_{totals_pred['original_finishing_alignment']}_{totals_pred['original_total_category']}
Pattern stats: {totals_stats['success']}/{totals_stats['total']} wins ({totals_stats['success']/totals_stats['total']:.0% if totals_stats['total'] > 0 else 'N/A'})

📊 EXPECTED GOALS:
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

⚠️ RISK FLAGS: {', '.join(prediction['totals']['risk_flags']) if prediction['totals']['risk_flags'] else 'None'}

---
YOUR IMPROVED RULES (based on your data):
1. STRONG (>70% success, ≥3 matches) → ✅ BET STRONGLY with boosted confidence
2. PROMISING (60-70% success, ≥3 matches) → 🔵 Small confidence boost
3. NEUTRAL (40-60% success, ≥3 matches) → ⚪ Use algorithm's prediction
4. WEAK (<40% success, ≥3 matches) → 🎯 BET OPPOSITE with 85% confidence!

YOUR DATA SHOWS:
- {len([v for v in st.session_state.learning_system.pattern_memory.values() if v['total'] >= 3 and v['success']/v['total'] > 0.7])} strong patterns (>70%)
- {len([v for v in st.session_state.learning_system.pattern_memory.values() if v['total'] >= 3 and 0.6 <= v['success']/v['total'] <= 0.7])} promising patterns (60-70%)
- {len([v for v in st.session_state.learning_system.pattern_memory.values() if v['total'] >= 3 and v['success']/v['total'] < 0.4])} weak patterns (<40%) → BET OPPOSITE!
"""

st.code(report, language="text")

st.download_button(
    label="📥 Download Report",
    data=report,
    file_name=f"improved_rules_{home_team}_vs_{away_team}.txt",
    mime="text/plain",
    use_container_width=True
)
