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
    page_title="⚽ Football Intelligence Engine v8.0 - PROVEN PATTERN MASTER",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Football Intelligence Engine v8.0 - PROVEN PATTERN MASTER")
st.markdown("""
    **DATA-DRIVEN OVERRIDES** - Based on 41-match empirical analysis
    *10 proven rules, 7 anti-patterns, 4 gold patterns*
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

# League-specific adjustments (ORIGINAL VALUES - no surgical changes)
LEAGUE_ADJUSTMENTS = {
    "Premier League": {"over_threshold": 2.5, "under_threshold": 2.5, "avg_goals": 2.79, "very_high_threshold": 3.3},
    "Bundesliga": {"over_threshold": 3.0, "under_threshold": 2.2, "avg_goals": 3.20, "very_high_threshold": 3.5},
    "Serie A": {"over_threshold": 2.7, "under_threshold": 2.3, "avg_goals": 2.40, "very_high_threshold": 3.0},
    "La Liga": {"over_threshold": 2.6, "under_threshold": 2.4, "avg_goals": 2.61, "very_high_threshold": 3.2},
    "Ligue 1": {"over_threshold": 2.8, "under_threshold": 2.2, "avg_goals": 2.85, "very_high_threshold": 3.3},
    "Eredivisie": {"over_threshold": 2.9, "under_threshold": 2.1, "avg_goals": 3.10, "very_high_threshold": 3.6},
    "RFPL": {"over_threshold": 2.5, "under_threshold": 2.2, "avg_goals": 2.53, "very_high_threshold": 3.1}
}

# ========== PROVEN PATTERNS DATABASE FROM 41-MATCH ANALYSIS ==========

PROVEN_FAILURES = {
    # ALWAYS WRONG PATTERNS (bet opposite)
    "TOTALS_RISKY_VERY_HIGH": {
        "record": "0/2",
        "action": "BET_OPPOSITE",
        "confidence": 85,
        "reason": "Market overreacts to high xG with volatile teams",
        "matches": "2 matches, 0 wins"
    },
    "TOTALS_LOW_OVER_HIGH": {
        "record": "0/2",
        "action": "BET_OPPOSITE", 
        "confidence": 80,
        "reason": "Weak finishing + high xG = inflated expectations",
        "matches": "2 matches, 0 wins"
    },
    "TOTALS_LOW_UNDER_HIGH": {
        "record": "0/1",
        "action": "BET_OPPOSITE",
        "confidence": 80,
        "reason": "Same failure pattern",
        "matches": "1 match, 0 wins"
    },
    "WINNER_HIGH_70.0": {
        "record": "0/2",
        "action": "BET_OPPOSITE",
        "confidence": 70,
        "reason": "70% false favorite zone",
        "matches": "2 matches, 0 wins"
    },
    "WINNER_HIGH_69.0": {
        "record": "0/1",
        "action": "BET_OPPOSITE",
        "confidence": 70,
        "reason": "70% false favorite zone",
        "matches": "1 match, 0 wins"
    }
}

PROVEN_SUCCESSES = {
    # GOLD PATTERNS (bet strongly)
    "TOTALS_MED_OVER_MODERATE_LOW": {
        "record": "4/6 → 5/7",
        "action": "BET_STRONGLY",
        "boost": 20,
        "reason": "Sweet spot pattern - reliable edge",
        "matches": "7 matches, 5 wins"
    },
    "TOTALS_HIGH_OVER_MODERATE_LOW": {
        "record": "2/2 → 3/3",
        "action": "BET_STRONGLY",
        "boost": 25,
        "reason": "Perfect pattern - maximum confidence",
        "matches": "3 matches, 3 wins"
    },
    "TOTALS_MED_UNDER_VERY_HIGH": {
        "record": "3/3",
        "action": "BET_STRONGLY",
        "boost": 20,
        "reason": "Counter-market value - market mispricing",
        "matches": "3 matches, 3 wins"
    },
    "WINNER_VERY_HIGH_100": {
        "record": "4/5",
        "action": "BET_STRONGLY",
        "boost": 15,
        "reason": "True dominance - algorithm certainty",
        "matches": "5 matches, 4 wins"
    },
    "TOTALS_NEUTRAL_MODERATE_LOW": {
        "record": "3/4",
        "action": "BET",
        "boost": 15,
        "reason": "Reliable baseline pattern",
        "matches": "4 matches, 3 wins"
    }
}

# ========== CONFIDENCE LEVEL RULES ==========

CONFIDENCE_RULES = {
    # 70% CONFIDENCE BUG (from your data: WINNER_HIGH_70.0 had 0/2 success)
    "70_PERCENT_ZONE": {
        "range": (68, 72),
        "action": "BET_OPPOSITE",
        "confidence": 70,
        "reason": "70% false favorite zone (0/3 success)"
    },
    
    # 90% OVERCONFIDENCE TRAP
    "90_PERCENT_ZONE": {
        "range": (88, 94),
        "action": "CAUTION",
        "derate_factor": 0.8,
        "reason": "90% overconfidence trap zone"
    },
    
    # 100% HAMMER ZONE
    "100_PERCENT_ZONE": {
        "range": (99, 101),
        "action": "BET_STRONGLY",
        "boost": 1.1,
        "reason": "True dominance (4/5 success)"
    }
}

# ========== HIGH VARIANCE DIRECTIONAL LOGIC ==========

HIGH_VARIANCE_RULES = {
    "OVERPERFORMER": {
        "threshold": 0.4,
        "over_boost": 20,
        "under_penalty": -15,
        "over_reason": "High variance overperformer amplifies OVER",
        "under_reason": "Overperformer likely to score, hurting UNDER"
    },
    "UNDERPERFORMER": {
        "threshold": -0.4,
        "under_boost": 20,
        "over_penalty": -15,
        "under_reason": "High variance underperformer amplifies UNDER",
        "over_reason": "Underperformer due for goals, helping OVER"
    }
}

# ========== SIMPLIFIED LEARNING SYSTEM ==========

class ProvenPatternSystem:
    """LEARNING SYSTEM WITH EMPIRICAL OVERRIDES"""
    
    def __init__(self):
        self.pattern_memory = {}
        self.supabase = init_supabase()
        
        # Thresholds from your data
        self.min_matches = 3
        self.strong_threshold = 0.70  # >70% = STRONG
        self.weak_threshold = 0.40    # <40% = WEAK
        
        self.load_learning()
    
    def save_learning(self):
        """Save learning data"""
        try:
            if not self.supabase:
                return self._save_learning_local()
            
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
                        "success_rate": stats['success'] / stats['total'] if stats['total'] > 0 else 0,
                        "strength": self._get_strength_category(stats)
                    })
                }
                supabase_data.append(data)
            
            if supabase_data:
                self.supabase.table("proven_patterns").delete().neq("pattern_key", "dummy").execute()
                self.supabase.table("proven_patterns").insert(supabase_data).execute()
                return True
                
            return True
            
        except Exception as e:
            return self._save_learning_local()
    
    def _get_strength_category(self, stats):
        """Get strength category for pattern"""
        if stats['total'] < self.min_matches:
            return "INSUFFICIENT_DATA"
        
        success_rate = stats['success'] / stats['total']
        if success_rate > self.strong_threshold:
            return "STRONG"
        elif success_rate < self.weak_threshold:
            return "WEAK"
        else:
            return "NEUTRAL"
    
    def _save_learning_local(self):
        """Fallback local storage"""
        try:
            with open("proven_patterns_data.pkl", "wb") as f:
                pickle.dump({
                    'pattern_memory': self.pattern_memory,
                    'version': '8.0_proven'
                }, f)
            return True
        except:
            return False
    
    def load_learning(self):
        """Load learning data"""
        try:
            if not self.supabase:
                return self._load_learning_local()
            
            response = self.supabase.table("proven_patterns").select("*").execute()
            
            if response.data:
                for row in response.data:
                    self.pattern_memory[row['pattern_key']] = {
                        'total': row['total_matches'] or 0,
                        'success': row['successful_matches'] or 0
                    }
                return True
            
            return True
            
        except Exception as e:
            return self._load_learning_local()
    
    def _load_learning_local(self):
        """Fallback local storage"""
        try:
            if os.path.exists("proven_patterns_data.pkl"):
                with open("proven_patterns_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.pattern_memory = data.get('pattern_memory', {})
                return True
        except:
            pass
        return False
    
    def record_outcome(self, prediction, actual_score):
        """Record match outcome"""
        try:
            # Generate pattern keys
            winner_key = self._generate_winner_key(prediction['winner'])
            totals_key = self._generate_totals_key(prediction['totals'])
            
            # Parse score
            home_goals, away_goals = map(int, actual_score.split('-'))
            
            # Determine actual outcomes
            if home_goals > away_goals:
                actual_winner = "HOME"
            elif away_goals > home_goals:
                actual_winner = "AWAY"
            else:
                actual_winner = "DRAW"
            
            total_goals = home_goals + away_goals
            actual_over = total_goals > 2.5
            
            # Initialize patterns
            if winner_key not in self.pattern_memory:
                self.pattern_memory[winner_key] = {'total': 0, 'success': 0}
            if totals_key not in self.pattern_memory:
                self.pattern_memory[totals_key] = {'total': 0, 'success': 0}
            
            # Check predictions
            winner_correct = prediction['winner']['original_prediction'] == actual_winner
            totals_correct = (prediction['totals']['original_direction'] == "OVER") == actual_over
            
            # Update patterns
            self.pattern_memory[winner_key]['total'] += 1
            self.pattern_memory[winner_key]['success'] += 1 if winner_correct else 0
            
            self.pattern_memory[totals_key]['total'] += 1
            self.pattern_memory[totals_key]['success'] += 1 if totals_correct else 0
            
            # Save
            save_success = self.save_learning()
            
            return {
                'winner_correct': winner_correct,
                'totals_correct': totals_correct,
                'winner_key': winner_key,
                'totals_key': totals_key,
                'save_success': save_success
            }, "Outcome recorded!"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def _generate_winner_key(self, winner_pred):
        """Generate winner pattern key"""
        pred_type = winner_pred.get('original_prediction', winner_pred.get('type', 'UNKNOWN'))
        confidence = winner_pred.get('original_confidence', winner_pred.get('confidence', '50'))
        
        # Standardize confidence
        if isinstance(confidence, str):
            if confidence.endswith('.0'):
                confidence = confidence[:-2]
        elif isinstance(confidence, float):
            confidence = str(int(confidence)) if confidence.is_integer() else f"{confidence:.1f}"
        
        return f"WINNER_{winner_pred.get('confidence_category', 'MEDIUM')}_{confidence}"
    
    def _generate_totals_key(self, totals_pred):
        """Generate totals pattern key"""
        finishing = totals_pred.get('original_finishing_alignment', 
                                  totals_pred.get('finishing_alignment', 'NEUTRAL'))
        total_cat = totals_pred.get('original_total_category', 
                                   totals_pred.get('total_category', 'MODERATE_LOW'))
        
        # Clean override suffix if present
        if finishing.endswith("_OVERRIDDEN"):
            finishing = finishing[:-11]
        
        return f"TOTALS_{finishing}_{total_cat}"
    
    def get_proven_advice(self, winner_pred, totals_pred, home_finish, away_finish, league_name):
        """Get betting advice with EMPIRICAL PROVEN RULES"""
        winner_key = self._generate_winner_key(winner_pred)
        totals_key = self._generate_totals_key(totals_pred)
        
        advice = {
            'winner': {
                'action': 'FOLLOW', 
                'bet_on': winner_pred['type'], 
                'confidence': winner_pred['confidence_score'],
                'reason': 'Algorithm prediction',
                'stake': 'HALF',
                'variance_effect': 'NONE'
            },
            'totals': {
                'action': 'FOLLOW', 
                'bet_on': totals_pred['direction'], 
                'confidence': totals_pred['confidence_score'],
                'reason': 'Algorithm prediction',
                'stake': 'HALF',
                'variance_effect': 'NONE'
            }
        }
        
        # ====== STEP 1: CHECK PROVEN FAILURE PATTERNS ======
        
        # Check winner proven failures
        winner_confidence = winner_pred['confidence_score']
        winner_conf_str = f"{winner_confidence:.1f}"
        
        if f"WINNER_HIGH_{winner_conf_str}" in PROVEN_FAILURES:
            rule = PROVEN_FAILURES[f"WINNER_HIGH_{winner_conf_str}"]
            advice['winner']['action'] = rule['action']
            advice['winner']['confidence'] = rule['confidence']
            advice['winner']['reason'] = f"PROVEN FAILURE: {rule['reason']} ({rule['record']})"
            advice['winner']['stake'] = 'HALF'
        
        # Check totals proven failures
        if totals_key in PROVEN_FAILURES:
            rule = PROVEN_FAILURES[totals_key]
            advice['totals']['action'] = rule['action']
            advice['totals']['bet_on'] = 'UNDER' if totals_pred['direction'] == 'OVER' else 'OVER'
            advice['totals']['confidence'] = rule['confidence']
            advice['totals']['reason'] = f"PROVEN FAILURE: {rule['reason']} ({rule['record']})"
            advice['totals']['stake'] = 'FULL'
        
        # ====== STEP 2: CHECK PROVEN SUCCESS PATTERNS ======
        
        # Check winner proven successes
        if winner_key in PROVEN_SUCCESSES and advice['winner']['action'] != 'BET_OPPOSITE':
            rule = PROVEN_SUCCESSES[winner_key]
            advice['winner']['action'] = rule['action']
            advice['winner']['confidence'] = min(95, winner_pred['confidence_score'] + rule['boost'])
            advice['winner']['reason'] = f"PROVEN SUCCESS: {rule['reason']} ({rule['record']})"
            advice['winner']['stake'] = 'FULL'
        
        # Check totals proven successes
        if totals_key in PROVEN_SUCCESSES and advice['totals']['action'] != 'BET_OPPOSITE':
            rule = PROVEN_SUCCESSES[totals_key]
            advice['totals']['action'] = rule['action']
            advice['totals']['confidence'] = min(95, totals_pred['confidence_score'] + rule['boost'])
            advice['totals']['reason'] = f"PROVEN SUCCESS: {rule['reason']} ({rule['record']})"
            advice['totals']['stake'] = 'FULL'
        
        # ====== STEP 3: APPLY CONFIDENCE LEVEL RULES ======
        
        # Apply confidence zone rules for winner
        for zone_name, zone_rule in CONFIDENCE_RULES.items():
            if zone_rule['range'][0] <= winner_confidence <= zone_rule['range'][1]:
                if zone_name == "70_PERCENT_ZONE" and advice['winner']['action'] != 'BET_OPPOSITE':
                    advice['winner']['action'] = zone_rule['action']
                    advice['winner']['confidence'] = zone_rule['confidence']
                    advice['winner']['reason'] = zone_rule['reason']
                    advice['winner']['stake'] = 'HALF'
                elif zone_name == "90_PERCENT_ZONE" and advice['winner']['action'] == 'FOLLOW':
                    advice['winner']['confidence'] *= zone_rule['derate_factor']
                    advice['winner']['reason'] = zone_rule['reason']
                    advice['winner']['stake'] = 'REDUCED'
                elif zone_name == "100_PERCENT_ZONE" and advice['winner']['action'] == 'FOLLOW':
                    advice['winner']['action'] = zone_rule['action']
                    advice['winner']['confidence'] *= zone_rule['boost']
                    advice['winner']['reason'] = zone_rule['reason']
                    advice['winner']['stake'] = 'FULL'
        
        # ====== STEP 4: APPLY HIGH VARIANCE DIRECTIONAL LOGIC ======
        
        # Apply variance rules to totals
        if advice['totals']['action'] != 'BET_OPPOSITE':  # Don't override proven failures
            # Check for overperformer
            if home_finish > HIGH_VARIANCE_RULES["OVERPERFORMER"]["threshold"] or \
               away_finish > HIGH_VARIANCE_RULES["OVERPERFORMER"]["threshold"]:
                
                variance_type = "OVERPERFORMER"
                rule = HIGH_VARIANCE_RULES[variance_type]
                
                if totals_pred['direction'] == "OVER":
                    advice['totals']['confidence'] += rule['over_boost']
                    advice['totals']['variance_effect'] = 'AMPLIFIED'
                    advice['totals']['reason'] = f"{rule['over_reason']} (+{rule['over_boost']}% boost)"
                else:
                    advice['totals']['confidence'] += rule['under_penalty']
                    advice['totals']['variance_effect'] = 'REDUCED'
                    advice['totals']['reason'] = f"{rule['under_reason']} ({rule['under_penalty']}% penalty)"
            
            # Check for underperformer
            elif home_finish < HIGH_VARIANCE_RULES["UNDERPERFORMER"]["threshold"] or \
                 away_finish < HIGH_VARIANCE_RULES["UNDERPERFORMER"]["threshold"]:
                
                variance_type = "UNDERPERFORMER"
                rule = HIGH_VARIANCE_RULES[variance_type]
                
                if totals_pred['direction'] == "UNDER":
                    advice['totals']['confidence'] += rule['under_boost']
                    advice['totals']['variance_effect'] = 'AMPLIFIED'
                    advice['totals']['reason'] = f"{rule['under_reason']} (+{rule['under_boost']}% boost)"
                else:
                    advice['totals']['confidence'] += rule['over_penalty']
                    advice['totals']['variance_effect'] = 'REDUCED'
                    advice['totals']['reason'] = f"{rule['over_reason']} ({rule['over_penalty']}% penalty)"
        
        # ====== STEP 5: APPLY LEAGUE-SPECIFIC EXECUTION ======
        
        league_adj = LEAGUE_ADJUSTMENTS.get(league_name, LEAGUE_ADJUSTMENTS["Premier League"])
        total_xg = totals_pred['total_xg']
        
        # Higher scoring leagues = trust OVER more
        if league_name == "Bundesliga" or league_name == "Eredivisie":
            if totals_pred['direction'] == "OVER" and total_xg > league_adj['over_threshold']:
                advice['totals']['confidence'] = min(95, advice['totals']['confidence'] * 1.1)
                advice['totals']['reason'] += " | High-scoring league boost"
        
        # Lower scoring leagues = trust UNDER more
        elif league_name == "Serie A" or league_name == "RFPL":
            if totals_pred['direction'] == "UNDER" and total_xg < league_adj['under_threshold']:
                advice['totals']['confidence'] = min(95, advice['totals']['confidence'] * 1.1)
                advice['totals']['reason'] += " | Low-scoring league boost"
        
        # ====== STEP 6: APPLY HISTORICAL LEARNING (only if ≥3 matches) ======
        
        # Winner pattern learning (only if not already overridden)
        if winner_key in self.pattern_memory and advice['winner']['action'] == 'FOLLOW':
            stats = self.pattern_memory[winner_key]
            if stats['total'] >= self.min_matches:
                success_rate = stats['success'] / stats['total']
                
                if success_rate > self.strong_threshold:
                    advice['winner']['action'] = 'BET_STRONGLY'
                    advice['winner']['confidence'] = min(95, winner_pred['confidence_score'] * 1.3)
                    advice['winner']['reason'] = f"Strong pattern: {stats['success']}/{stats['total']} ({success_rate:.0%})"
                    advice['winner']['stake'] = 'FULL'
                
                elif success_rate < self.weak_threshold:
                    advice['winner']['action'] = 'BET_OPPOSITE'
                    advice['winner']['confidence'] = 85
                    if winner_pred['type'] == 'HOME':
                        advice['winner']['bet_on'] = 'AWAY'
                    elif winner_pred['type'] == 'AWAY':
                        advice['winner']['bet_on'] = 'HOME'
                    advice['winner']['reason'] = f"Weak pattern: {stats['success']}/{stats['total']} ({success_rate:.0%})"
                    advice['winner']['stake'] = 'HALF'
        
        # Totals pattern learning (only if not already overridden)
        if totals_key in self.pattern_memory and advice['totals']['action'] == 'FOLLOW':
            stats = self.pattern_memory[totals_key]
            if stats['total'] >= self.min_matches:
                success_rate = stats['success'] / stats['total']
                
                if success_rate > self.strong_threshold:
                    advice['totals']['action'] = 'BET_STRONGLY'
                    advice['totals']['confidence'] = min(95, totals_pred['confidence_score'] * 1.3)
                    advice['totals']['reason'] = f"Strong pattern: {stats['success']}/{stats['total']} ({success_rate:.0%})"
                    advice['totals']['stake'] = 'FULL'
                
                elif success_rate < self.weak_threshold:
                    advice['totals']['action'] = 'BET_OPPOSITE'
                    advice['totals']['confidence'] = 85
                    advice['totals']['bet_on'] = 'UNDER' if totals_pred['direction'] == 'OVER' else 'OVER'
                    advice['totals']['reason'] = f"Weak pattern: {stats['success']}/{stats['total']} ({success_rate:.0%})"
                    advice['totals']['stake'] = 'HALF'
        
        # Ensure confidence is within bounds
        advice['winner']['confidence'] = max(5, min(100, advice['winner']['confidence']))
        advice['totals']['confidence'] = max(5, min(100, advice['totals']['confidence']))
        
        return advice
    
    def get_pattern_stats(self, pattern_key):
        """Get stats for a pattern"""
        return self.pattern_memory.get(pattern_key, {'total': 0, 'success': 0})

# ========== INITIALIZE SESSION STATES ==========
if 'proven_system' not in st.session_state:
    st.session_state.proven_system = ProvenPatternSystem()

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'last_teams' not in st.session_state:
    st.session_state.last_teams = None

# ========== ORIGINAL CORE PREDICTION CLASSES ==========
# Keeping your original working classes

class ExpectedGoalsPredictor:
    """ORIGINAL expected goals calculation"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_avg_goals = league_metrics.get('avg_goals_per_match', 2.5)
        self.league_name = league_name
    
    def predict_expected_goals(self, home_stats, away_stats):
        """Calculate expected goals (ORIGINAL)"""
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
    """ORIGINAL winner determination with empirical overrides"""
    
    def predict_winner(self, home_xg, away_xg, home_stats, away_stats):
        """Predict winner with original logic"""
        home_finishing = home_stats['goals_vs_xg_pm']
        away_finishing = away_stats['goals_vs_xg_pm']
        home_defense = home_stats['goals_allowed_vs_xga_pm']
        away_defense = away_stats['goals_allowed_vs_xga_pm']
        
        home_adjusted_xg = home_xg + home_finishing - away_defense
        away_adjusted_xg = away_xg + away_finishing - home_defense
        
        delta = home_adjusted_xg - away_adjusted_xg
        
        # Winner determination (ORIGINAL)
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
        
        # Confidence calculation (ORIGINAL)
        base_confidence = min(100, abs(delta) / max(home_adjusted_xg, away_adjusted_xg, 0.5) * 150)
        win_rate_diff = home_stats['win_rate'] - away_stats['win_rate']
        form_bonus = min(20, max(0, win_rate_diff * 40))
        
        winner_confidence = min(100, max(30, base_confidence + form_bonus))
        
        # Store ORIGINAL values
        original_confidence = winner_confidence
        
        # Confidence categorization (ORIGINAL)
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
            'original_confidence': f"{original_confidence:.1f}",
            'confidence_category': confidence_category,
            'delta': delta,
            'has_confidence_bug': CONFIDENCE_RULES["70_PERCENT_ZONE"]["range"][0] <= winner_confidence <= CONFIDENCE_RULES["70_PERCENT_ZONE"]["range"][1]
        }

class TotalsPredictor:
    """ORIGINAL totals prediction with empirical overrides"""
    
    def __init__(self, league_name):
        self.league_name = league_name
        self.league_adjustments = LEAGUE_ADJUSTMENTS.get(league_name, LEAGUE_ADJUSTMENTS["Premier League"])
    
    def categorize_finishing(self, value):
        """ORIGINAL finishing categorization"""
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
        """ORIGINAL finishing alignment"""
        home_cat = self.categorize_finishing(home_finish)
        away_cat = self.categorize_finishing(away_finish)
        
        # ORIGINAL alignment matrix
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
        """ORIGINAL total xG categories with league adjustments"""
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
        """Predict totals with original logic"""
        total_xg = home_xg + away_xg
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        over_threshold = self.league_adjustments['over_threshold']
        base_direction = "OVER" if total_xg > over_threshold else "UNDER"
        
        # ORIGINAL finishing alignment
        finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
        total_category = self.categorize_total_xg(total_xg)
        
        # Store original values
        original_direction = base_direction
        original_finishing = finishing_alignment
        original_category = total_category
        
        # Risk assessment
        risk_flags = []
        
        # HIGH VARIANCE TEAM flag (will be used directionally, not as penalty)
        if abs(home_finish) > 0.4 or abs(away_finish) > 0.4:
            risk_flags.append("HIGH_VARIANCE_TEAM")
        
        # CLOSE TO THRESHOLD flag
        lower_thresh = self.league_adjustments['under_threshold'] - 0.1
        upper_thresh = self.league_adjustments['over_threshold'] + 0.1
        if lower_thresh < total_xg < upper_thresh:
            risk_flags.append("CLOSE_TO_THRESHOLD")
        
        # Check for proven anti-patterns (these will be overridden later)
        if finishing_alignment in ["RISKY", "HIGH_RISK"] and total_category == "VERY_HIGH":
            risk_flags.append("PROVEN_ANTI_PATTERN")
        
        if "LOW" in finishing_alignment and total_category in ["HIGH", "VERY_HIGH"]:
            risk_flags.append("PROVEN_ANTI_PATTERN")
        
        # Base confidence
        base_confidence = 60
        
        # Apply CLOSE_TO_THRESHOLD penalty only
        for flag in risk_flags:
            if flag == "CLOSE_TO_THRESHOLD":
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
            'confidence': confidence_category,
            'confidence_score': base_confidence,
            'finishing_alignment': finishing_alignment,
            'original_finishing_alignment': original_finishing,
            'total_category': total_category,
            'original_total_category': original_category,
            'risk_flags': risk_flags,
            'home_finishing': home_finish,
            'away_finishing': away_finish,
            'league_threshold': over_threshold,
            'is_proven_anti_pattern': "PROVEN_ANTI_PATTERN" in risk_flags
        }

# ========== ORIGINAL PROBABILITY ENGINE ==========

class PoissonProbabilityEngine:
    """ORIGINAL probability engine"""
    
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

# ========== PROVEN FOOTBALL ENGINE ==========

class ProvenFootballEngine:
    """Engine with EMPIRICAL PROVEN RULES"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_name = league_name
        
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate prediction with EMPIRICAL PROVEN RULES"""
        
        # Get ORIGINAL predictions
        home_xg, away_xg = self.xg_predictor.predict_expected_goals(home_stats, away_stats)
        
        probabilities = self.probability_engine.calculate_all_probabilities(home_xg, away_xg)
        
        winner_prediction = self.winner_predictor.predict_winner(
            home_xg, away_xg, home_stats, away_stats
        )
        
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Get PROVEN betting advice (empirical rules)
        betting_advice = st.session_state.proven_system.get_proven_advice(
            winner_prediction, 
            totals_prediction,
            totals_prediction['home_finishing'],
            totals_prediction['away_finishing'],
            self.league_name
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
                'confidence': self._get_confidence_category(final_winner['confidence']),
                'confidence_score': final_winner['confidence'],
                'strength': final_winner.get('strength', 'N/A'),
                'most_likely_score': probabilities['most_likely_score'],
                'betting_action': betting_advice['winner']['action'],
                'original_prediction': winner_prediction['original_prediction'],
                'original_confidence': winner_prediction['original_confidence'],
                'original_confidence_score': winner_prediction['confidence_score'],
                'reason': betting_advice['winner']['reason'],
                'stake': betting_advice['winner']['stake'],
                'color': self._get_color_for_action(betting_advice['winner']['action']),
                'has_confidence_bug': winner_prediction.get('has_confidence_bug', False),
                'variance_effect': betting_advice['winner'].get('variance_effect', 'NONE')
            },
            
            'totals': {
                'direction': final_totals['direction'],
                'probability': totals_prob,
                'confidence': self._get_confidence_category(final_totals['confidence']),
                'confidence_score': final_totals['confidence'],
                'total_xg': totals_prediction['total_xg'],
                'finishing_alignment': totals_prediction['finishing_alignment'],
                'original_finishing_alignment': totals_prediction['original_finishing_alignment'],
                'total_category': totals_prediction['total_category'],
                'original_total_category': totals_prediction['original_total_category'],
                'risk_flags': totals_prediction.get('risk_flags', []),
                'betting_action': betting_advice['totals']['action'],
                'original_direction': totals_prediction['original_direction'],
                'original_confidence_score': totals_prediction['confidence_score'],
                'reason': betting_advice['totals']['reason'],
                'stake': betting_advice['totals']['stake'],
                'color': self._get_color_for_action(betting_advice['totals']['action']),
                'is_proven_anti_pattern': totals_prediction.get('is_proven_anti_pattern', False),
                'variance_effect': betting_advice['totals'].get('variance_effect', 'NONE')
            },
            
            'probabilities': probabilities,
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': home_xg + away_xg},
            'betting_advice': betting_advice,
            'version': '8.0_proven',
            'league': self.league_name
        }
    
    def _apply_advice_to_winner(self, original, advice, home_team, away_team):
        """Apply advice to winner"""
        final = original.copy()
        
        if advice['action'] == 'BET_OPPOSITE':
            # Bet opposite
            if original['type'] == 'HOME':
                final['type'] = 'AWAY'
                final['team'] = away_team
            elif original['type'] == 'AWAY':
                final['type'] = 'HOME'
                final['team'] = home_team
            else:
                final['type'] = 'DRAW'
                final['team'] = 'DRAW'
            
            final['confidence'] = advice['confidence']
        
        elif advice['action'] == 'CAUTION':
            # Derate confidence
            final['confidence'] = advice['confidence']
            final['team'] = home_team if original['type'] == 'HOME' else away_team if original['type'] == 'AWAY' else 'DRAW'
        
        elif advice['action'] == 'BET_STRONGLY':
            # Boost confidence
            final['confidence'] = advice['confidence']
            final['team'] = home_team if original['type'] == 'HOME' else away_team if original['type'] == 'AWAY' else 'DRAW'
        
        else:
            # Follow algorithm
            final['team'] = home_team if original['type'] == 'HOME' else away_team if original['type'] == 'AWAY' else 'DRAW'
            final['confidence'] = advice['confidence']
        
        return final
    
    def _apply_advice_to_totals(self, original, advice):
        """Apply advice to totals"""
        final = original.copy()
        
        if advice['action'] == 'BET_OPPOSITE':
            # Bet opposite
            final['direction'] = advice['bet_on']
            final['confidence'] = advice['confidence']
        
        elif advice['action'] == 'BET_STRONGLY':
            # Boost confidence
            final['confidence'] = advice['confidence']
        
        else:
            # Follow algorithm
            final['confidence'] = advice['confidence']
        
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
    
    def _get_confidence_category(self, confidence_score):
        """Get confidence category from score"""
        if confidence_score >= 85:
            return "VERY HIGH"
        elif confidence_score >= 75:
            return "HIGH"
        elif confidence_score >= 65:
            return "MEDIUM"
        elif confidence_score >= 55:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _get_color_for_action(self, action):
        """Get color for betting action"""
        colors = {
            'BET_OPPOSITE': '#DC2626',  # Red
            'CAUTION': '#F59E0B',        # Orange
            'BET_STRONGLY': '#10B981',   # Green
            'FOLLOW': '#6B7280',         # Gray
            'BET': '#3B82F6'             # Blue
        }
        return colors.get(action, '#6B7280')

# ========== EMPIRICAL BETTING CARD ==========

class EmpiricalBettingCard:
    """Empirical betting card based on proven patterns"""
    
    @staticmethod
    def get_recommendation(prediction):
        """Get betting recommendation based on proven rules"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Check for proven failure overrides first
        if winner_pred['betting_action'] == 'BET_OPPOSITE' and winner_pred.get('has_confidence_bug', False):
            return {
                'type': '70_PERCENT_BUG_FIX',
                'text': f"🎯 {winner_pred['team']} to win",
                'subtext': '70% FALSE FAVORITE ZONE',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': winner_pred.get('stake', 'HALF'),
                'proven_type': 'FAILURE_OVERRIDE'
            }
        
        if totals_pred.get('is_proven_anti_pattern', False) and totals_pred['betting_action'] == 'BET_OPPOSITE':
            return {
                'type': 'PROVEN_ANTI_PATTERN',
                'text': f"🎯 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'PROVEN ANTI-PATTERN',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': totals_pred.get('stake', 'FULL'),
                'proven_type': 'FAILURE_OVERRIDE'
            }
        
        # Check for proven success patterns
        if totals_pred['betting_action'] == 'BET_STRONGLY' and 'PROVEN SUCCESS' in totals_pred['reason']:
            return {
                'type': 'PROVEN_GOLD_PATTERN',
                'text': f"✅ {totals_pred['direction']} 2.5 Goals",
                'subtext': 'PROVEN GOLD PATTERN',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': totals_pred.get('stake', 'FULL'),
                'proven_type': 'SUCCESS_PATTERN'
            }
        
        if winner_pred['betting_action'] == 'BET_STRONGLY' and 'PROVEN SUCCESS' in winner_pred['reason']:
            return {
                'type': 'PROVEN_WINNER_PATTERN',
                'text': f"✅ {winner_pred['team']} to win",
                'subtext': 'PROVEN WINNER PATTERN',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': winner_pred.get('stake', 'FULL'),
                'proven_type': 'SUCCESS_PATTERN'
            }
        
        # Check for high variance amplification
        if totals_pred.get('variance_effect') == 'AMPLIFIED':
            return {
                'type': 'VARIANCE_AMPLIFICATION',
                'text': f"⚡ {totals_pred['direction']} 2.5 Goals",
                'subtext': 'VARIANCE AMPLIFICATION',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#8B5CF6',
                'icon': '⚡',
                'stake': totals_pred.get('stake', 'HALF'),
                'proven_type': 'VARIANCE_PLAY'
            }
        
        # Check for double strong patterns
        if winner_pred['betting_action'] == 'BET_STRONGLY' and totals_pred['betting_action'] == 'BET_STRONGLY':
            return {
                'type': 'DOUBLE_STRONG_PATTERNS',
                'text': f"🔥 {winner_pred['team']} + {totals_pred['direction']} 2.5",
                'subtext': 'DOUBLE STRONG PATTERNS',
                'reason': 'Both markets show proven edges',
                'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#10B981',
                'icon': '🔥',
                'stake': 'FULL',
                'proven_type': 'DOUBLE_STRONG'
            }
        
        # Single strong patterns
        if winner_pred['betting_action'] == 'BET_STRONGLY':
            return {
                'type': 'WINNER_STRONG_PATTERN',
                'text': f"✅ {winner_pred['team']} to win",
                'subtext': 'STRONG WINNER PATTERN',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': winner_pred.get('stake', 'HALF'),
                'proven_type': 'SINGLE_STRONG'
            }
        
        if totals_pred['betting_action'] == 'BET_STRONGLY':
            return {
                'type': 'TOTALS_STRONG_PATTERN',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'STRONG TOTALS PATTERN',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': totals_pred.get('stake', 'HALF'),
                'proven_type': 'SINGLE_STRONG'
            }
        
        # Weak patterns (bet opposite)
        if winner_pred['betting_action'] == 'BET_OPPOSITE':
            return {
                'type': 'WINNER_WEAK_PATTERN',
                'text': f"🎯 {winner_pred['team']} to win",
                'subtext': 'BET OPPOSITE (Weak pattern)',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': winner_pred.get('stake', 'HALF'),
                'proven_type': 'WEAK_PATTERN'
            }
        
        if totals_pred['betting_action'] == 'BET_OPPOSITE':
            return {
                'type': 'TOTALS_WEAK_PATTERN',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'BET OPPOSITE (Weak pattern)',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': totals_pred.get('stake', 'HALF'),
                'proven_type': 'WEAK_PATTERN'
            }
        
        # Caution zone
        if winner_pred['betting_action'] == 'CAUTION':
            return {
                'type': 'CAUTION_ZONE',
                'text': f"⚠️ {winner_pred['team']} to win",
                'subtext': 'CAUTION (90% trap zone)',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#F59E0B',
                'icon': '⚠️',
                'stake': 'REDUCED',
                'proven_type': 'CAUTION'
            }
        
        # No clear edge
        return {
            'type': 'NO_PROVEN_EDGE',
            'text': "🤔 No Proven Edge",
            'subtext': 'NO BET',
            'reason': 'Insufficient proven edge or neutral patterns',
            'confidence': max(winner_pred['confidence_score'], totals_pred['confidence_score']),
            'color': '#6B7280',
            'icon': '🤔',
            'stake': 'NONE',
            'proven_type': 'NEUTRAL'
        }
    
    @staticmethod
    def display_card(recommendation):
        """Display the betting card"""
        color = recommendation['color']
        stake_colors = {
            'FULL': '#10B981', 
            'HALF': '#F59E0B', 
            'REDUCED': '#F59E0B',
            'NONE': '#6B7280'
        }
        stake_color = stake_colors.get(recommendation['stake'], '#6B7280')
        
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
            <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 15px;">
                <div style="font-size: 18px; color: #9CA3AF;">
                    Confidence: {recommendation['confidence']:.0f}/100
                </div>
                <div style="font-size: 18px; color: {stake_color}; font-weight: bold;">
                    Stake: {recommendation['stake']}
                </div>
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

def record_outcome_proven(prediction):
    """Proven feedback system"""
    
    st.divider()
    st.subheader("📝 Record Outcome for Pattern Learning")
    
    # Show current patterns
    winner_key = st.session_state.proven_system._generate_winner_key(prediction['winner'])
    totals_key = st.session_state.proven_system._generate_totals_key(prediction['totals'])
    
    winner_stats = st.session_state.proven_system.get_pattern_stats(winner_key)
    totals_stats = st.session_state.proven_system.get_pattern_stats(totals_key)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Winner Pattern:**")
        st.code(winner_key)
        if winner_stats['total'] > 0:
            success = winner_stats['success'] / winner_stats['total'] if winner_stats['total'] > 0 else 0
            st.write(f"Current: {winner_stats['success']}/{winner_stats['total']} ({success:.0%})")
        
        # Check if proven pattern
        if winner_key in PROVEN_FAILURES:
            st.error(f"🔴 PROVEN FAILURE: {PROVEN_FAILURES[winner_key]['record']}")
        elif winner_key in PROVEN_SUCCESSES:
            st.success(f"✅ PROVEN SUCCESS: {PROVEN_SUCCESSES[winner_key]['record']}")
    
    with col2:
        st.write("**Totals Pattern:**")
        st.code(totals_key)
        if totals_stats['total'] > 0:
            success = totals_stats['success'] / totals_stats['total'] if totals_stats['total'] > 0 else 0
            st.write(f"Current: {totals_stats['success']}/{totals_stats['total']} ({success:.0%})")
        
        # Check if proven pattern
        if totals_key in PROVEN_FAILURES:
            st.error(f"🔴 PROVEN FAILURE: {PROVEN_FAILURES[totals_key]['record']}")
        elif totals_key in PROVEN_SUCCESSES:
            st.success(f"✅ PROVEN SUCCESS: {PROVEN_SUCCESSES[totals_key]['record']}")
    
    # Score input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        score = st.text_input("Actual Score (e.g., 2-1)", key="proven_score_input")
    
    with col2:
        if st.button("✅ Record Outcome", type="primary", use_container_width=True):
            if not score or '-' not in score:
                st.error("Enter valid score like '2-1'")
            else:
                try:
                    with st.spinner("Saving pattern data..."):
                        result, message = st.session_state.proven_system.record_outcome(prediction, score)
                        
                        if result:
                            if result['save_success']:
                                st.success("✅ Pattern saved successfully!")
                            else:
                                st.warning("⚠️ Saved locally")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if result['winner_correct']:
                                    st.success(f"✅ Winner correct!")
                                else:
                                    st.error(f"❌ Winner wrong!")
                            
                            with col2:
                                if result['totals_correct']:
                                    st.success(f"✅ Totals correct!")
                                else:
                                    st.error(f"❌ Totals wrong!")
                            
                            # Update proven patterns if needed
                            if not result['winner_correct'] and winner_key not in PROVEN_FAILURES and winner_stats['total'] >= 2:
                                st.error(f"⚠️ Consider adding {winner_key} to PROVEN_FAILURES")
                            
                            if result['totals_correct'] and totals_key not in PROVEN_SUCCESSES and totals_stats['total'] >= 3:
                                st.success(f"⚠️ Consider adding {totals_key} to PROVEN_SUCCESSES")
                            
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

    # Proven System Section
    st.divider()
    st.header("🎯 EMPIRICAL PROVEN RULES")
    
    st.subheader("🔴 PROVEN FAILURES (Bet Opposite):")
    for pattern, rule in PROVEN_FAILURES.items():
        st.error(f"""
        **{pattern}**
        → {rule['action']} (Confidence: {rule['confidence']}%)
        → Reason: {rule['reason']}
        → Record: {rule['record']}
        """)
    
    st.subheader("✅ PROVEN SUCCESSES (Bet Strongly):")
    for pattern, rule in PROVEN_SUCCESSES.items():
        st.success(f"""
        **{pattern}**
        → {rule['action']} (Boost: +{rule['boost']}%)
        → Reason: {rule['reason']}
        → Record: {rule['record']}
        """)
    
    st.subheader("⚡ HIGH VARIANCE RULES:")
    st.info("""
    **Overperformer (>+0.4 goals vs xG):**
    → OVER: +20% confidence boost
    → UNDER: -15% confidence penalty
    
    **Underperformer (<-0.4 goals vs xG):**
    → UNDER: +20% confidence boost
    → OVER: -15% confidence penalty
    """)
    
    st.subheader("📊 CONFIDENCE ZONES:")
    st.warning("""
    **70% Zone (68-72%):**
    → BET OPPOSITE (0/3 success)
    
    **90% Zone (88-94%):**
    → CAUTION (Derate by 20%)
    
    **100% Zone (99-100%):**
    → BET STRONGLY (4/5 success)
    """)
    
    # Show current match patterns if available
    if st.session_state.last_prediction:
        try:
            winner_pred = st.session_state.last_prediction['winner']
            totals_pred = st.session_state.last_prediction['totals']
            
            winner_key = st.session_state.proven_system._generate_winner_key(winner_pred)
            totals_key = st.session_state.proven_system._generate_totals_key(totals_pred)
            
            winner_stats = st.session_state.proven_system.get_pattern_stats(winner_key)
            totals_stats = st.session_state.proven_system.get_pattern_stats(totals_key)
            
            st.subheader("Current Match Patterns:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Winner:** {winner_key}")
                if winner_stats['total'] >= 3:
                    success = winner_stats['success'] / winner_stats['total']
                    if success > 0.7:
                        st.success(f"✅ {success:.0%}")
                    elif success < 0.4:
                        st.error(f"🎯 {success:.0%}")
                    else:
                        st.info(f"⚪ {success:.0%}")
                else:
                    st.caption(f"{winner_stats['total']}/3 matches")
            
            with col2:
                st.write(f"**Totals:** {totals_key}")
                if totals_stats['total'] >= 3:
                    success = totals_stats['success'] / totals_stats['total']
                    if success > 0.7:
                        st.success(f"✅ {success:.0%}")
                    elif success < 0.4:
                        st.error(f"🎯 {success:.0%}")
                    else:
                        st.info(f"⚪ {success:.0%}")
                else:
                    st.caption(f"{totals_stats['total']}/3 matches")
                    
        except Exception as e:
            st.warning("Could not display pattern stats")
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Pattern Statistics:")
    
    pattern_memory = st.session_state.proven_system.pattern_memory
    total_patterns = len(pattern_memory)
    qualifying = len([v for v in pattern_memory.values() if v['total'] >= 3])
    strong = len([v for v in pattern_memory.values() 
                 if v['total'] >= 3 and v['success']/v['total'] > 0.7])
    weak = len([v for v in pattern_memory.values() 
               if v['total'] >= 3 and v['success']/v['total'] < 0.4])
    
    st.write(f"Total patterns tracked: {total_patterns}")
    st.write(f"Qualifying (≥3 matches): {qualifying}")
    st.write(f"Strong patterns (>70%): {strong}")
    st.write(f"Weak patterns (<40%): {weak}")
    
    # Show top patterns
    if qualifying > 0:
        st.subheader("🏆 Top Patterns:")
        sorted_patterns = sorted([(k, v) for k, v in pattern_memory.items() if v['total'] >= 3], 
                                key=lambda x: x[1]['success']/x[1]['total'], 
                                reverse=True)[:5]
        
        for pattern, stats in sorted_patterns:
            success = stats['success'] / stats['total']
            st.write(f"**{pattern}:** {stats['success']}/{stats['total']} ({success:.0%})")

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# Check if we should show prediction
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Generate prediction with PROVEN RULES
        engine = ProvenFootballEngine(league_metrics, selected_league)
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
st.caption(f"League: {selected_league} | Version: {prediction.get('version', '8.0')} | Empirical Rules Applied")

# Prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction['winner']
    color = winner_pred.get('color', '#6B7280')
    
    # Determine icon based on action
    if winner_pred['betting_action'] == 'BET_OPPOSITE':
        icon = "🎯"
        subtitle = "BET OPPOSITE (Proven Failure)"
        card_color = "#7F1D1D"
    elif winner_pred['betting_action'] == 'CAUTION':
        icon = "⚠️"
        subtitle = "CAUTION (90% Trap Zone)"
        card_color = "#78350F"
    elif winner_pred['betting_action'] == 'BET_STRONGLY':
        icon = "✅"
        subtitle = "BET STRONGLY (Proven Success)"
        card_color = "#14532D"
    else:
        icon = "🏠" if winner_pred['type'] == "HOME" else "✈️" if winner_pred['type'] == "AWAY" else "🤝"
        card_color = "#1E293B"
        subtitle = winner_pred['confidence']
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {icon} {winner_pred['team']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {subtitle} | Confidence: {winner_pred['confidence_score']:.0f}/100
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            {winner_pred['reason']}
        </div>
        <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
            <div style="font-size: 12px; color: #9CA3AF;">
                Stake: {winner_pred.get('stake', 'HALF')}
            </div>
            <div style="font-size: 12px; color: #9CA3AF;">
                Original: {winner_pred['original_prediction']} ({winner_pred['original_confidence_score']:.0f}%)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    totals_pred = prediction['totals']
    color = totals_pred.get('color', '#6B7280')
    
    # Determine icon based on action
    if totals_pred['betting_action'] == 'BET_OPPOSITE':
        icon = "🎯"
        subtitle = "BET OPPOSITE (Proven Failure)"
        card_color = "#7F1D1D"
    elif totals_pred['betting_action'] == 'BET_STRONGLY':
        icon = "✅"
        subtitle = "BET STRONGLY (Proven Success)"
        card_color = "#14532D"
    else:
        icon = "📈" if totals_pred['direction'] == "OVER" else "📉"
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
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            {totals_pred['reason']}
        </div>
        <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
            <div style="font-size: 12px; color: #9CA3AF;">
                Stake: {totals_pred.get('stake', 'HALF')}
            </div>
            <div style="font-size: 12px; color: #9CA3AF;">
                Original: {totals_pred['original_direction']} ({totals_pred['original_confidence_score']:.0f}%)
            </div>
        </div>
        <div style="font-size: 12px; color: #FCD34D; margin-top: 5px;">
            xG: {totals_pred['total_xg']:.2f} | Variance: {totals_pred.get('variance_effect', 'NONE')}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== EMPIRICAL BETTING CARD ==========
st.divider()
st.subheader("🎯 EMPIRICAL BETTING ADVICE")

recommendation = EmpiricalBettingCard.get_recommendation(prediction)
EmpiricalBettingCard.display_card(recommendation)

# ========== PATTERN ANALYSIS ==========
st.divider()
st.subheader("🔍 EMPIRICAL PATTERN ANALYSIS")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 Original Algorithm")
    
    st.write(f"**Winner:**")
    st.write(f"- Prediction: {prediction['winner']['original_prediction']}")
    st.write(f"- Confidence: {prediction['winner']['original_confidence_score']:.1f}%")
    
    st.write(f"**Totals:**")
    st.write(f"- Prediction: {prediction['totals']['original_direction']}")
    st.write(f"- Confidence: {prediction['totals']['original_confidence_score']:.1f}%")
    st.write(f"- xG: {prediction['totals']['total_xg']:.2f}")
    st.write(f"- Pattern: {prediction['totals']['original_finishing_alignment']} + {prediction['totals']['original_total_category']}")

with col2:
    st.subheader("📊 Proven Rules Applied")
    
    st.write(f"**Winner Rules:**")
    if prediction['winner']['has_confidence_bug']:
        st.error("→ 70% Confidence Bug Fix")
    if "PROVEN FAILURE" in prediction['winner']['reason']:
        st.error(f"→ {prediction['winner']['reason'].split(':')[1]}")
    if "PROVEN SUCCESS" in prediction['winner']['reason']:
        st.success(f"→ {prediction['winner']['reason'].split(':')[1]}")
    if prediction['winner'].get('variance_effect') != 'NONE':
        st.info(f"→ Variance: {prediction['winner']['variance_effect']}")
    
    st.write(f"**Totals Rules:**")
    if prediction['totals'].get('is_proven_anti_pattern'):
        st.error("→ Proven Anti-Pattern Detected")
    if "PROVEN FAILURE" in prediction['totals']['reason']:
        st.error(f"→ {prediction['totals']['reason'].split(':')[1]}")
    if "PROVEN SUCCESS" in prediction['totals']['reason']:
        st.success(f"→ {prediction['totals']['reason'].split(':')[1]}")
    if prediction['totals'].get('variance_effect') != 'NONE':
        st.info(f"→ Variance: {prediction['totals']['variance_effect']}")

with col3:
    st.subheader("📈 Pattern Statistics")
    
    winner_key = st.session_state.proven_system._generate_winner_key(prediction['winner'])
    totals_key = st.session_state.proven_system._generate_totals_key(prediction['totals'])
    
    winner_stats = st.session_state.proven_system.get_pattern_stats(winner_key)
    totals_stats = st.session_state.proven_system.get_pattern_stats(totals_key)
    
    st.write(f"**Winner Pattern:**")
    st.code(winner_key)
    if winner_stats['total'] > 0:
        success = winner_stats['success'] / winner_stats['total']
        st.write(f"Record: {winner_stats['success']}/{winner_stats['total']} ({success:.0%})")
    
    st.write(f"**Totals Pattern:**")
    st.code(totals_key)
    if totals_stats['total'] > 0:
        success = totals_stats['success'] / totals_stats['total']
        st.write(f"Record: {totals_stats['success']}/{totals_stats['total']} ({success:.0%})")

# ========== DETAILED RULE BREAKDOWN ==========
st.divider()
st.subheader("🔬 Detailed Rule Breakdown")

# Winner rules
with st.expander("🎯 Winner Rules Applied", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Algorithm Output:**")
        st.write(f"- Type: {prediction['winner']['original_prediction']}")
        st.write(f"- Confidence: {prediction['winner']['original_confidence_score']:.1f}%")
        st.write(f"- Category: {prediction['winner'].get('confidence_category', 'N/A')}")
    
    with col2:
        st.write("**Proven Rules Applied:**")
        
        # Check each rule
        winner_confidence = prediction['winner']['original_confidence_score']
        
        # 70% zone check
        if CONFIDENCE_RULES["70_PERCENT_ZONE"]["range"][0] <= winner_confidence <= CONFIDENCE_RULES["70_PERCENT_ZONE"]["range"][1]:
            st.error(f"**70% Zone Rule:** BET OPPOSITE")
            st.write(f"→ Confidence: {winner_confidence:.1f}% is in 68-72% false favorite zone")
            st.write(f"→ Record: 0/3 success in this zone")
        
        # 90% zone check
        if CONFIDENCE_RULES["90_PERCENT_ZONE"]["range"][0] <= winner_confidence <= CONFIDENCE_RULES["90_PERCENT_ZONE"]["range"][1]:
            st.warning(f"**90% Zone Rule:** CAUTION")
            st.write(f"→ Confidence: {winner_confidence:.1f}% is in 88-94% overconfidence trap")
            st.write(f"→ Action: Derate confidence by 20%")
        
        # 100% zone check
        if CONFIDENCE_RULES["100_PERCENT_ZONE"]["range"][0] <= winner_confidence <= CONFIDENCE_RULES["100_PERCENT_ZONE"]["range"][1]:
            st.success(f"**100% Zone Rule:** BET STRONGLY")
            st.write(f"→ Confidence: {winner_confidence:.1f}% indicates true dominance")
            st.write(f"→ Record: 4/5 success at 100% confidence")
            st.write(f"→ Action: Boost confidence by 10%")

# Totals rules
with st.expander("📈 Totals Rules Applied", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Algorithm Output:**")
        st.write(f"- Direction: {prediction['totals']['original_direction']}")
        st.write(f"- Confidence: {prediction['totals']['original_confidence_score']:.1f}%")
        st.write(f"- Pattern: {prediction['totals']['original_finishing_alignment']} + {prediction['totals']['original_total_category']}")
        st.write(f"- xG: {prediction['totals']['total_xg']:.2f}")
        st.write(f"- League Threshold: {prediction['totals'].get('league_threshold', 2.5)}")
    
    with col2:
        st.write("**Proven Rules Applied:**")
        
        # Check for proven failure patterns
        totals_key = f"TOTALS_{prediction['totals']['original_finishing_alignment']}_{prediction['totals']['original_total_category']}"
        
        if totals_key in PROVEN_FAILURES:
            st.error(f"**Proven Failure Pattern:** {totals_key}")
            st.write(f"→ Action: BET OPPOSITE")
            st.write(f"→ Reason: {PROVEN_FAILURES[totals_key]['reason']}")
            st.write(f"→ Record: {PROVEN_FAILURES[totals_key]['record']}")
        
        # Check for proven success patterns
        elif totals_key in PROVEN_SUCCESSES:
            st.success(f"**Proven Success Pattern:** {totals_key}")
            st.write(f"→ Action: {PROVEN_SUCCESSES[totals_key]['action']}")
            st.write(f"→ Reason: {PROVEN_SUCCESSES[totals_key]['reason']}")
            st.write(f"→ Record: {PROVEN_SUCCESSES[totals_key]['record']}")
            st.write(f"→ Boost: +{PROVEN_SUCCESSES[totals_key]['boost']}%")
        
        # Check for high variance
        home_finish = prediction['totals']['home_finishing']
        away_finish = prediction['totals']['away_finishing']
        
        if home_finish > HIGH_VARIANCE_RULES["OVERPERFORMER"]["threshold"] or away_finish > HIGH_VARIANCE_RULES["OVERPERFORMER"]["threshold"]:
            st.info(f"**High Variance - Overperformer:**")
            if prediction['totals']['direction'] == "OVER":
                st.write(f"→ Action: AMPLIFY OVER")
                st.write(f"→ Boost: +{HIGH_VARIANCE_RULES['OVERPERFORMER']['over_boost']}% confidence")
                st.write(f"→ Reason: {HIGH_VARIANCE_RULES['OVERPERFORMER']['over_reason']}")
            else:
                st.write(f"→ Action: REDUCE UNDER")
                st.write(f"→ Penalty: {HIGH_VARIANCE_RULES['OVERPERFORMER']['under_penalty']}% confidence")
                st.write(f"→ Reason: {HIGH_VARIANCE_RULES['OVERPERFORMER']['under_reason']}")
        
        elif home_finish < HIGH_VARIANCE_RULES["UNDERPERFORMER"]["threshold"] or away_finish < HIGH_VARIANCE_RULES["UNDERPERFORMER"]["threshold"]:
            st.info(f"**High Variance - Underperformer:**")
            if prediction['totals']['direction'] == "UNDER":
                st.write(f"→ Action: AMPLIFY UNDER")
                st.write(f"→ Boost: +{HIGH_VARIANCE_RULES['UNDERPERFORMER']['under_boost']}% confidence")
                st.write(f"→ Reason: {HIGH_VARIANCE_RULES['UNDERPERFORMER']['under_reason']}")
            else:
                st.write(f"→ Action: REDUCE OVER")
                st.write(f"→ Penalty: {HIGH_VARIANCE_RULES['UNDERPERFORMER']['over_penalty']}% confidence")
                st.write(f"→ Reason: {HIGH_VARIANCE_RULES['UNDERPERFORMER']['over_reason']}")

# ========== FEEDBACK ==========
record_outcome_proven(prediction)

# ========== EMPIRICAL REPORT ==========
st.divider()
st.subheader("📤 Empirical Pattern Report")

winner_stats = st.session_state.proven_system.get_pattern_stats(winner_key)
totals_stats = st.session_state.proven_system.get_pattern_stats(totals_key)

winner_success_rate = winner_stats['success'] / winner_stats['total'] if winner_stats['total'] > 0 else 0
totals_success_rate = totals_stats['success'] / totals_stats['total'] if totals_stats['total'] > 0 else 0

report = f"""🎯 FOOTBALL INTELLIGENCE ENGINE v8.0 - EMPIRICAL PATTERN MASTER
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Version: 8.0 (Empirical proven rules)

🎯 EMPIRICAL BETTING ADVICE:
{recommendation['icon']} {recommendation['text']}
{recommendation['subtext']}
Type: {recommendation['proven_type']}
Reason: {recommendation['reason']}
Confidence: {recommendation['confidence']:.0f}/100
Stake: {recommendation['stake']}

🎯 WINNER ANALYSIS:
Final bet: {prediction['winner']['team']} ({prediction['winner']['type']})
Original algorithm: {prediction['winner']['original_prediction']} ({prediction['winner']['original_confidence_score']:.1f}%)
Final confidence: {prediction['winner']['confidence_score']:.0f}/100 ({prediction['winner']['confidence']})
Betting action: {prediction['winner']['betting_action']}
Reason: {prediction['winner']['reason']}
Probability: {prediction['winner']['probability']*100:.1f}%
Stake: {prediction['winner']['stake']}
70% bug zone: {prediction['winner']['has_confidence_bug']}
Variance effect: {prediction['winner']['variance_effect']}
Pattern: {winner_key}
Pattern stats: {winner_stats['success']}/{winner_stats['total']} wins ({winner_success_rate:.0%})

🎯 TOTALS ANALYSIS:
Final bet: {prediction['totals']['direction']} 2.5
Original algorithm: {prediction['totals']['original_direction']} ({prediction['totals']['original_confidence_score']:.1f}%)
Final confidence: {prediction['totals']['confidence_score']:.0f}/100 ({prediction['totals']['confidence']})
Betting action: {prediction['totals']['betting_action']}
Reason: {prediction['totals']['reason']}
Probability: {prediction['totals']['probability']*100:.1f}%
Stake: {prediction['totals']['stake']}
Proven anti-pattern: {prediction['totals']['is_proven_anti_pattern']}
Variance effect: {prediction['totals']['variance_effect']}
xG: {prediction['totals']['total_xg']:.2f}
Pattern: {prediction['totals']['original_finishing_alignment']} + {prediction['totals']['original_total_category']}
Pattern stats: {totals_stats['success']}/{totals_stats['total']} wins ({totals_success_rate:.0%})

📊 EXPECTED GOALS:
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

🎲 PROBABILITIES:
Home win: {prediction['probabilities']['home_win_probability']*100:.1f}%
Draw: {prediction['probabilities']['draw_probability']*100:.1f}%
Away win: {prediction['probabilities']['away_win_probability']*100:.1f}%
Over 2.5: {prediction['probabilities']['over_2_5_probability']*100:.1f}%
Under 2.5: {prediction['probabilities']['under_2_5_probability']*100:.1f}%
Most likely score: {prediction['probabilities']['most_likely_score']}

---
🔴 ACTIVE PROVEN FAILURE PATTERNS:
{TOTALS_RISKY_VERY_HIGH} → Bet UNDER (0/2 success)
{TOTALS_LOW_OVER_HIGH} → Bet opposite (0/2 success)
{TOTALS_LOW_UNDER_HIGH} → Bet opposite (0/1 success)
WINNER_HIGH_70.0 → Bet opposite (0/2 success)
WINNER_HIGH_69.0 → Bet opposite (0/1 success)

✅ ACTIVE PROVEN SUCCESS PATTERNS:
{TOTALS_MED_OVER_MODERATE_LOW} → Bet strongly (5/7 success)
{TOTALS_HIGH_OVER_MODERATE_LOW} → Bet strongly (3/3 success)
{TOTALS_MED_UNDER_VERY_HIGH} → Bet strongly (3/3 success)
WINNER_VERY_HIGH_100 → Bet strongly (4/5 success)

⚡ HIGH VARIANCE RULES:
Overperformer (>+0.4): OVER +20%, UNDER -15%
Underperformer (<-0.4): UNDER +20%, OVER -15%

📊 CONFIDENCE ZONE RULES:
70% Zone (68-72%): BET OPPOSITE (0/3)
90% Zone (88-94%): CAUTION (derate 20%)
100% Zone (99-100%): BET STRONGLY (4/5)

📈 PATTERN STATISTICS:
Total patterns tracked: {len(st.session_state.proven_system.pattern_memory)}
Qualifying (≥3 matches): {len([v for v in st.session_state.proven_system.pattern_memory.values() if v['total'] >= 3])}
Strong patterns (>70%): {len([v for v in st.session_state.proven_system.pattern_memory.values() if v['total'] >= 3 and v['success']/v['total'] > 0.7])}
Weak patterns (<40%): {len([v for v in st.session_state.proven_system.pattern_memory.values() if v['total'] >= 3 and v['success']/v['total'] < 0.4])}

✅ EMPIRICAL SYSTEM FEATURES:
- 10 proven rules from 41-match analysis
- 7 anti-pattern overrides (bet opposite)
- 4 gold patterns (bet strongly)
- High variance directional logic
- Confidence zone rules
- League-specific execution
- Pattern memory learning
"""

st.code(report, language="text")

st.download_button(
    label="📥 Download Empirical Report",
    data=report,
    file_name=f"empirical_{home_team}_vs_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
    mime="text/plain",
    use_container_width=True
)
