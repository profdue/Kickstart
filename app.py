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
    page_title="⚽ Football Intelligence Engine v8.2 - LOGIC FIXED",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Football Intelligence Engine v8.2 - LOGIC FIXED")
st.markdown("""
    **LOGIC BUG FIXED** - Now uses finishing data in totals decisions
    *Testing without empirical crutches*
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

# ========== PROVEN PATTERNS DATABASE - MINIMAL CRUTCHES ==========
# KEEPING ONLY THE CONFIRMED BUGS, REMOVING TOTALS CRUTCHES

PROVEN_FAILURES = {
    # ONLY KEEPING WINNER CONFIDENCE BUGS (confirmed algorithm bug)
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
    
    # REMOVED TOTALS ANTI-PATTERNS - Let fixed algorithm handle these
    # "TOTALS_RISKY_VERY_HIGH": {...},
    # "TOTALS_LOW_OVER_HIGH": {...},
    # "TOTALS_LOW_UNDER_HIGH": {...},
}

PROVEN_SUCCESSES = {
    # ONLY KEEPING 100% CONFIDENCE WINNERS (confirmed algorithm strength)
    "WINNER_VERY_HIGH_100": {
        "record": "4/5",
        "action": "BET_STRONGLY",
        "boost": 15,
        "reason": "True dominance - algorithm certainty",
        "matches": "5 matches, 4 wins"
    }
    
    # REMOVED TOTALS SUCCESS PATTERNS - Let fixed algorithm handle these
    # "TOTALS_MED_OVER_MODERATE_LOW": {...},
    # "TOTALS_HIGH_OVER_MODERATE_LOW": {...},
    # "TOTALS_MED_UNDER_VERY_HIGH": {...},
    # "TOTALS_NEUTRAL_MODERATE_LOW": {...},
}

# ========== CONFIDENCE LEVEL RULES ==========
# KEEPING ONLY CONFIRMED BUGS

CONFIDENCE_RULES = {
    # 70% CONFIDENCE BUG (confirmed algorithm bug)
    "70_PERCENT_ZONE": {
        "range": (68, 72),
        "action": "BET_OPPOSITE",
        "confidence": 70,
        "reason": "70% false favorite zone (0/3 success)"
    },
    
    # 90% OVERCONFIDENCE TRAP (keep as caution)
    "90_PERCENT_ZONE": {
        "range": (88, 94),
        "action": "CAUTION",
        "derate_factor": 0.8,
        "reason": "90% overconfidence trap zone"
    },
    
    # 100% HAMMER ZONE (confirmed strength)
    "100_PERCENT_ZONE": {
        "range": (99, 101),
        "action": "BET_STRONGLY",
        "boost": 1.1,
        "reason": "True dominance (4/5 success)"
    }
}

# ========== HIGH VARIANCE DIRECTIONAL LOGIC ==========
# KEEPING - This is strategic insight, not a crutch

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
    """LEARNING SYSTEM WITH MINIMAL CRUTCHES"""
    
    def __init__(self):
        self.pattern_memory = {}
        self.supabase = init_supabase()
        
        # Thresholds from your data
        self.min_matches = 3
        self.strong_threshold = 0.70  # >70% = STRONG
        self.weak_threshold = 0.40    # <40% = WEAK
        
        self.load_learning()
    
    def save_learning(self):
        """Save learning data to Supabase (football_learning table)"""
        try:
            if not self.supabase:
                st.warning("No Supabase connection. Saving locally only.")
                return self._save_learning_local()
            
            # Prepare data for Supabase
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
                        "strength": self._get_strength_category(stats),
                        "last_updated": datetime.now().isoformat()
                    })
                }
                supabase_data.append(data)
            
            if supabase_data:
                # Try to update existing records first, then insert new ones
                for data in supabase_data:
                    try:
                        # Check if pattern exists
                        response = self.supabase.table("football_learning").select("*").eq("pattern_key", data['pattern_key']).execute()
                        
                        if response.data and len(response.data) > 0:
                            # Update existing record
                            self.supabase.table("football_learning").update({
                                "total_matches": data['total_matches'],
                                "successful_matches": data['successful_matches'],
                                "last_updated": data['last_updated'],
                                "metadata": data['metadata']
                            }).eq("pattern_key", data['pattern_key']).execute()
                            st.success(f"✅ Updated pattern: {data['pattern_key']}")
                        else:
                            # Insert new record
                            self.supabase.table("football_learning").insert(data).execute()
                            st.success(f"✅ Inserted pattern: {data['pattern_key']}")
                            
                    except Exception as e:
                        st.error(f"❌ Error saving pattern {data['pattern_key']}: {str(e)}")
                        # Fallback to local storage
                        return self._save_learning_local()
                
                return True
            else:
                st.warning("No pattern data to save")
                return True
            
        except Exception as e:
            st.error(f"❌ Supabase save error: {str(e)}")
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
                    'version': '8.2_no_crutches',
                    'last_saved': datetime.now().isoformat()
                }, f)
            return True
        except Exception as e:
            st.error(f"❌ Local save error: {str(e)}")
            return False
    
    def load_learning(self):
        """Load learning data from Supabase (football_learning table)"""
        try:
            if not self.supabase:
                st.warning("No Supabase connection. Loading local data only.")
                return self._load_learning_local()
            
            # First try to load from Supabase
            response = self.supabase.table("football_learning").select("*").execute()
            
            if response.data:
                self.pattern_memory = {}
                for row in response.data:
                    self.pattern_memory[row['pattern_key']] = {
                        'total': row['total_matches'] or 0,
                        'success': row['successful_matches'] or 0
                    }
                st.success(f"✅ Loaded {len(response.data)} patterns from Supabase")
                return True
            else:
                st.warning("No data in Supabase, trying local storage")
                return self._load_learning_local()
            
        except Exception as e:
            st.error(f"❌ Supabase load error: {str(e)}")
            return self._load_learning_local()
    
    def _load_learning_local(self):
        """Fallback local storage"""
        try:
            if os.path.exists("proven_patterns_data.pkl"):
                with open("proven_patterns_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.pattern_memory = data.get('pattern_memory', {})
                st.success(f"✅ Loaded {len(self.pattern_memory)} patterns from local storage")
                return True
            else:
                st.warning("No local pattern data found. Starting fresh.")
                return True
        except Exception as e:
            st.error(f"❌ Local load error: {str(e)}")
            return False
    
    def record_outcome(self, prediction, actual_score):
        """Record match outcome - FIXED VERSION"""
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
            
            # Initialize patterns if they don't exist
            if winner_key not in self.pattern_memory:
                self.pattern_memory[winner_key] = {'total': 0, 'success': 0}
            if totals_key not in self.pattern_memory:
                self.pattern_memory[totals_key] = {'total': 0, 'success': 0}
            
            # Check predictions against ORIGINAL predictions, not final ones
            winner_correct = prediction['winner']['original_prediction'] == actual_winner
            totals_correct = (prediction['totals']['original_direction'] == "OVER") == actual_over
            
            # Update patterns
            self.pattern_memory[winner_key]['total'] += 1
            self.pattern_memory[winner_key]['success'] += 1 if winner_correct else 0
            
            self.pattern_memory[totals_key]['total'] += 1
            self.pattern_memory[totals_key]['success'] += 1 if totals_correct else 0
            
            # Save to Supabase and local
            save_success = self.save_learning()
            
            return {
                'winner_correct': winner_correct,
                'totals_correct': totals_correct,
                'winner_key': winner_key,
                'totals_key': totals_key,
                'save_success': save_success,
                'winner_stats': self.pattern_memory[winner_key],
                'totals_stats': self.pattern_memory[totals_key]
            }, "Outcome recorded!"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def _generate_winner_key(self, winner_pred):
        """Generate winner pattern key"""
        # Use the original prediction for pattern key
        original_pred = winner_pred.get('original_prediction', winner_pred.get('type', 'UNKNOWN'))
        confidence = winner_pred.get('original_confidence', winner_pred.get('confidence', '50'))
        
        # Standardize confidence
        if isinstance(confidence, str):
            if confidence.endswith('.0'):
                confidence = confidence[:-2]
        elif isinstance(confidence, float):
            confidence = str(int(confidence)) if confidence.is_integer() else f"{confidence:.1f}"
        
        # Use confidence category from prediction
        confidence_category = winner_pred.get('confidence_category', 'MEDIUM')
        
        return f"WINNER_{confidence_category}_{confidence}"
    
    def _generate_totals_key(self, totals_pred):
        """Generate totals pattern key"""
        # Use ORIGINAL finishing alignment and total category for pattern key
        finishing = totals_pred.get('original_finishing_alignment', 
                                  totals_pred.get('finishing_alignment', 'NEUTRAL'))
        total_cat = totals_pred.get('original_total_category', 
                                   totals_pred.get('total_category', 'MODERATE_LOW'))
        
        # Clean override suffix if present
        if finishing.endswith("_OVERRIDDEN"):
            finishing = finishing[:-11]
        
        return f"TOTALS_{finishing}_{total_cat}"
    
    def get_proven_advice(self, winner_pred, totals_pred, home_finish, away_finish, league_name):
        """Get betting advice with MINIMAL CRUTCHES"""
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
        
        # Check winner proven failures (KEEPING ONLY CONFIRMED BUGS)
        winner_confidence = winner_pred['confidence_score']
        winner_conf_str = f"{winner_confidence:.1f}"
        
        # Check for specific confidence values in proven failures
        if f"WINNER_HIGH_{winner_conf_str}" in PROVEN_FAILURES:
            rule = PROVEN_FAILURES[f"WINNER_HIGH_{winner_conf_str}"]
            advice['winner']['action'] = rule['action']
            advice['winner']['confidence'] = rule['confidence']
            advice['winner']['reason'] = f"PROVEN BUG: {rule['reason']} ({rule['record']})"
            advice['winner']['stake'] = 'HALF'
        
        # Check totals proven failures (REMOVED - let fixed algorithm handle)
        # if totals_key in PROVEN_FAILURES:
        #     rule = PROVEN_FAILURES[totals_key]
        #     advice['totals']['action'] = rule['action']
        #     advice['totals']['bet_on'] = 'UNDER' if totals_pred['direction'] == 'OVER' else 'OVER'
        #     advice['totals']['confidence'] = rule['confidence']
        #     advice['totals']['reason'] = f"PROVEN FAILURE: {rule['reason']} ({rule['record']})"
        #     advice['totals']['stake'] = 'FULL'
        
        # ====== STEP 2: CHECK PROVEN SUCCESS PATTERNS ======
        
        # Check winner proven successes (KEEPING ONLY 100% CONFIDENCE)
        if winner_key in PROVEN_SUCCESSES and advice['winner']['action'] != 'BET_OPPOSITE':
            rule = PROVEN_SUCCESSES[winner_key]
            advice['winner']['action'] = rule['action']
            advice['winner']['confidence'] = min(95, winner_pred['confidence_score'] + rule['boost'])
            advice['winner']['reason'] = f"PROVEN STRENGTH: {rule['reason']} ({rule['record']})"
            advice['winner']['stake'] = 'FULL'
        
        # Check totals proven successes (REMOVED - let fixed algorithm handle)
        # if totals_key in PROVEN_SUCCESSES and advice['totals']['action'] != 'BET_OPPOSITE':
        #     rule = PROVEN_SUCCESSES[totals_key]
        #     advice['totals']['action'] = rule['action']
        #     advice['totals']['confidence'] = min(95, totals_pred['confidence_score'] + rule['boost'])
        #     advice['totals']['reason'] = f"PROVEN SUCCESS: {rule['reason']} ({rule['record']})"
        #     advice['totals']['stake'] = 'FULL'
        
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
        # KEEPING - This is strategic insight, not a crutch
        
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
        # KEEPING - This is strategic insight
        
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
        # KEEPING - This is true learning, not a crutch
        
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
# Keeping your original working classes WITH THE LOGIC FIX

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
    """ORIGINAL totals prediction - NOW WITH LOGIC FIX"""
    
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
        """Predict totals with original logic - FIXED VERSION"""
        total_xg = home_xg + away_xg
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        # ========== THE LOGIC FIX ==========
        # Apply finishing impact to adjust xG BEFORE making decision
        finishing_impact = (home_finish + away_finish) * 0.6
        adjusted_xg = total_xg * (1 + finishing_impact)
        
        over_threshold = self.league_adjustments['over_threshold']
        
        # Use ADJUSTED xG for the decision (not raw xG)
        base_direction = "OVER" if adjusted_xg > over_threshold else "UNDER"
        # ========== END FIX ==========
        
        # ORIGINAL finishing alignment (unchanged)
        finishing_alignment = self.get_finishing_alignment(home_finish, away_finish)
        total_category = self.categorize_total_xg(total_xg)  # Note: Still use raw xG for category
        
        # Store original values
        original_direction = base_direction
        original_finishing = finishing_alignment
        original_category = total_category
        
        # Risk assessment
        risk_flags = []
        
        # HIGH VARIANCE TEAM flag (will be used directionally, not as penalty)
        if abs(home_finish) > 0.4 or abs(away_finish) > 0.4:
            risk_flags.append("HIGH_VARIANCE_TEAM")
        
        # CLOSE TO THRESHOLD flag - Now using ADJUSTED xG
        lower_thresh = self.league_adjustments['under_threshold'] - 0.1
        upper_thresh = self.league_adjustments['over_threshold'] + 0.1
        if lower_thresh < adjusted_xg < upper_thresh:
            risk_flags.append("CLOSE_TO_THRESHOLD")
        
        # REMOVED PROVEN ANTI-PATTERN CHECKS - let fixed algorithm handle
        
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
        
        # Include adjusted_xg in the output for debugging
        return {
            'direction': base_direction,
            'original_direction': base_direction,
            'total_xg': total_xg,
            'adjusted_xg': adjusted_xg,  # NEW: for debugging
            'finishing_impact': finishing_impact,  # NEW: for debugging
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
            'home_finish_value': home_finish,
            'away_finish_value': away_finish
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

# ========== CLEAN FOOTBALL ENGINE ==========

class CleanFootballEngine:
    """Engine with LOGIC FIX and MINIMAL CRUTCHES"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_name = league_name
        
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = WinnerPredictor()
        self.totals_predictor = TotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate prediction with LOGIC FIX and minimal crutches"""
        
        # Get ORIGINAL predictions
        home_xg, away_xg = self.xg_predictor.predict_expected_goals(home_stats, away_stats)
        
        probabilities = self.probability_engine.calculate_all_probabilities(home_xg, away_xg)
        
        winner_prediction = self.winner_predictor.predict_winner(
            home_xg, away_xg, home_stats, away_stats
        )
        
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Get MINIMAL betting advice (only confirmed bugs)
        betting_advice = st.session_state.proven_system.get_proven_advice(
            winner_prediction, 
            totals_prediction,
            totals_prediction['home_finish_value'],
            totals_prediction['away_finish_value'],
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
        
        # Create the final prediction dictionary
        prediction_dict = {
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
                'variance_effect': betting_advice['winner'].get('variance_effect', 'NONE'),
                'confidence_category': winner_prediction.get('confidence_category', 'MEDIUM')
            },
            
            'totals': {
                'direction': final_totals['direction'],
                'probability': totals_prob,
                'confidence': self._get_confidence_category(final_totals['confidence']),
                'confidence_score': final_totals['confidence'],
                'total_xg': totals_prediction['total_xg'],
                'adjusted_xg': totals_prediction.get('adjusted_xg', totals_prediction['total_xg']),
                'finishing_impact': totals_prediction.get('finishing_impact', 0),
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
                'variance_effect': betting_advice['totals'].get('variance_effect', 'NONE'),
                'home_finishing': totals_prediction['home_finish_value'],
                'away_finishing': totals_prediction['away_finish_value']
            },
            
            'probabilities': probabilities,
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': home_xg + away_xg},
            'betting_advice': betting_advice,
            'version': '8.2_no_crutches',
            'league': self.league_name
        }
        
        return prediction_dict
    
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

# ========== CLEAN BETTING CARD ==========

class CleanBettingCard:
    """Clean betting card based on algorithm output"""
    
    @staticmethod
    def get_recommendation(prediction):
        """Get betting recommendation based on algorithm output"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Check for 70% confidence bug
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
                'proven_type': 'BUG_FIX'
            }
        
        # Check for 100% confidence strength
        if winner_pred['betting_action'] == 'BET_STRONGLY' and '100%' in winner_pred.get('reason', ''):
            return {
                'type': '100_PERCENT_STRENGTH',
                'text': f"✅ {winner_pred['team']} to win",
                'subtext': '100% CONFIDENCE ZONE',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': winner_pred.get('stake', 'FULL'),
                'proven_type': 'STRENGTH'
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
        
        # Check for strong patterns from learning
        if winner_pred['betting_action'] == 'BET_STRONGLY' and 'Strong pattern' in winner_pred.get('reason', ''):
            return {
                'type': 'LEARNED_STRONG_PATTERN',
                'text': f"✅ {winner_pred['team']} to win",
                'subtext': 'LEARNED STRONG PATTERN',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': winner_pred.get('stake', 'HALF'),
                'proven_type': 'LEARNED'
            }
        
        if totals_pred['betting_action'] == 'BET_STRONGLY' and 'Strong pattern' in totals_pred.get('reason', ''):
            return {
                'type': 'LEARNED_STRONG_PATTERN',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'LEARNED STRONG PATTERN',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': totals_pred.get('stake', 'HALF'),
                'proven_type': 'LEARNED'
            }
        
        # Weak patterns (bet opposite) from learning
        if winner_pred['betting_action'] == 'BET_OPPOSITE' and 'Weak pattern' in winner_pred.get('reason', ''):
            return {
                'type': 'LEARNED_WEAK_PATTERN',
                'text': f"🎯 {winner_pred['team']} to win",
                'subtext': 'LEARNED WEAK PATTERN',
                'reason': winner_pred['reason'],
                'confidence': winner_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': winner_pred.get('stake', 'HALF'),
                'proven_type': 'LEARNED'
            }
        
        if totals_pred['betting_action'] == 'BET_OPPOSITE' and 'Weak pattern' in totals_pred.get('reason', ''):
            return {
                'type': 'LEARNED_WEAK_PATTERN',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'LEARNED WEAK PATTERN',
                'reason': totals_pred['reason'],
                'confidence': totals_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': totals_pred.get('stake', 'HALF'),
                'proven_type': 'LEARNED'
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
        
        # Default: Follow algorithm
        # Pick the stronger prediction
        if winner_pred['confidence_score'] > totals_pred['confidence_score']:
            return {
                'type': 'ALGORITHM_WINNER',
                'text': f"🏠 {winner_pred['team']} to win",
                'subtext': 'ALGORITHM PREDICTION',
                'reason': 'Following algorithm prediction',
                'confidence': winner_pred['confidence_score'],
                'color': '#3B82F6',
                'icon': '🏠' if winner_pred['type'] == 'HOME' else '✈️' if winner_pred['type'] == 'AWAY' else '🤝',
                'stake': 'HALF',
                'proven_type': 'ALGORITHM'
            }
        else:
            return {
                'type': 'ALGORITHM_TOTALS',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'ALGORITHM PREDICTION',
                'reason': 'Following algorithm prediction',
                'confidence': totals_pred['confidence_score'],
                'color': '#3B82F6',
                'icon': '📈' if totals_pred['direction'] == 'OVER' else '📉',
                'stake': 'HALF',
                'proven_type': 'ALGORITHM'
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

def record_outcome_clean(prediction):
    """Clean feedback system with Supabase integration"""
    
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
        
        # Check if it's a known bug
        if winner_key in PROVEN_FAILURES:
            st.error(f"🔴 CONFIRMED BUG: {PROVEN_FAILURES[winner_key]['record']}")
    
    with col2:
        st.write("**Totals Pattern:**")
        st.code(totals_key)
        if totals_stats['total'] > 0:
            success = totals_stats['success'] / totals_stats['total'] if totals_stats['total'] > 0 else 0
            st.write(f"Current: {totals_stats['success']}/{totals_stats['total']} ({success:.0%})")
        
        # No more proven patterns for totals - algorithm is fixed
    
    # Score input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        score = st.text_input("Actual Score (e.g., 2-1)", key="clean_score_input")
    
    with col2:
        if st.button("✅ Record Outcome", type="primary", use_container_width=True):
            if not score or '-' not in score:
                st.error("Enter valid score like '2-1'")
            else:
                try:
                    with st.spinner("Saving pattern data to Supabase..."):
                        result, message = st.session_state.proven_system.record_outcome(prediction, score)
                        
                        if result:
                            col_success, col_stats = st.columns(2)
                            
                            with col_success:
                                if result['save_success']:
                                    st.success("✅ Pattern saved to Supabase!")
                                else:
                                    st.warning("⚠️ Saved locally only (check Supabase connection)")
                            
                            with col_stats:
                                st.write(f"**Winner:** {'✅ Correct' if result['winner_correct'] else '❌ Wrong'}")
                                st.write(f"**Totals:** {'✅ Correct' if result['totals_correct'] else '❌ Wrong'}")
                            
                            # Show updated stats
                            st.write("### 📊 Updated Pattern Stats:")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                new_winner_stats = result['winner_stats']
                                success_rate = new_winner_stats['success'] / new_winner_stats['total'] if new_winner_stats['total'] > 0 else 0
                                st.write(f"**{winner_key}**")
                                st.write(f"Record: {new_winner_stats['success']}/{new_winner_stats['total']} ({success_rate:.0%})")
                            
                            with col2:
                                new_totals_stats = result['totals_stats']
                                success_rate = new_totals_stats['success'] / new_totals_stats['total'] if new_totals_stats['total'] > 0 else 0
                                st.write(f"**{totals_key}**")
                                st.write(f"Record: {new_totals_stats['success']}/{new_totals_stats['total']} ({success_rate:.0%})")
                            
                            # Check for pattern significance
                            if new_winner_stats['total'] >= 3:
                                success_rate = new_winner_stats['success'] / new_winner_stats['total']
                                if success_rate < 0.4 and winner_key not in PROVEN_FAILURES:
                                    st.error(f"⚠️ **Potential new bug:** {winner_key} ({new_winner_stats['success']}/{new_winner_stats['total']})")
                                elif success_rate > 0.7 and winner_key not in PROVEN_SUCCESSES:
                                    st.success(f"⚠️ **Potential new strength:** {winner_key} ({new_winner_stats['success']}/{new_winner_stats['total']})")
                            
                            if new_totals_stats['total'] >= 3:
                                success_rate = new_totals_stats['success'] / new_totals_stats['total']
                                if success_rate < 0.4:
                                    st.error(f"⚠️ **Weak totals pattern:** {totals_key} ({new_totals_stats['success']}/{new_totals_stats['total']})")
                                elif success_rate > 0.7:
                                    st.success(f"⚠️ **Strong totals pattern:** {totals_key} ({new_totals_stats['success']}/{new_totals_stats['total']})")
                            
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                            
                except ValueError:
                    st.error("Enter numbers like '2-1'")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

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

    # System Section
    st.divider()
    st.header("🎯 SYSTEM STATUS")
    
    st.success("**✅ LOGIC BUG FIXED**")
    st.write("Totals predictions now correctly incorporate finishing data")
    
    st.subheader("🔧 Active Adjustments:")
    
    st.error("**70% Confidence Zone Bug:**")
    st.write("→ BET OPPOSITE when winner confidence is 68-72%")
    st.write("→ Confirmed bug from data (0/3 success)")
    
    st.warning("**90% Confidence Zone:**")
    st.write("→ CAUTION when winner confidence is 88-94%")
    st.write("→ Overconfidence trap zone")
    
    st.success("**100% Confidence Zone:**")
    st.write("→ BET STRONGLY when winner confidence is 99-100%")
    st.write("→ Confirmed strength (4/5 success)")
    
    st.info("**High Variance Logic:**")
    st.write("→ Overperformers: OVER +20%, UNDER -15%")
    st.write("→ Underperformers: UNDER +20%, OVER -15%")
    
    st.subheader("🔄 Removed Crutches:")
    st.write("""
    ✅ **TOTALS anti-patterns REMOVED:**
    - TOTALS_RISKY_VERY_HIGH
    - TOTALS_LOW_OVER_HIGH  
    - TOTALS_LOW_UNDER_HIGH
    
    ✅ **TOTALS success patterns REMOVED:**
    - TOTALS_MED_OVER_MODERATE_LOW
    - TOTALS_HIGH_OVER_MODERATE_LOW
    - TOTALS_MED_UNDER_VERY_HIGH
    - TOTALS_NEUTRAL_MODERATE_LOW
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
        
        # Generate prediction with LOGIC FIX
        engine = CleanFootballEngine(league_metrics, selected_league)
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
st.caption(f"League: {selected_league} | Version: {prediction.get('version', '8.2')} | Logic Bug Fixed")

# Prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction['winner']
    color = winner_pred.get('color', '#6B7280')
    
    # Determine icon based on action
    if winner_pred['betting_action'] == 'BET_OPPOSITE':
        icon = "🎯"
        subtitle = "BET OPPOSITE (70% Bug)"
        card_color = "#7F1D1D"
    elif winner_pred['betting_action'] == 'CAUTION':
        icon = "⚠️"
        subtitle = "CAUTION (90% Trap)"
        card_color = "#78350F"
    elif winner_pred['betting_action'] == 'BET_STRONGLY':
        icon = "✅"
        subtitle = "BET STRONGLY (100% Strength)"
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
        subtitle = "BET OPPOSITE (Learned)"
        card_color = "#7F1D1D"
    elif totals_pred['betting_action'] == 'BET_STRONGLY':
        icon = "✅"
        subtitle = "BET STRONGLY (Learned)"
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
            <div style="font-size: 12px; color: #60A5FA;">
                Adj. xG: {totals_pred.get('adjusted_xg', totals_pred['total_xg']):.2f}
            </div>
        </div>
        <div style="font-size: 12px; color: #FCD34D; margin-top: 5px;">
            Raw xG: {totals_pred['total_xg']:.2f} | Impact: {totals_pred.get('finishing_impact', 0):.1%} | Variance: {totals_pred.get('variance_effect', 'NONE')}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== CLEAN BETTING CARD ==========
st.divider()
st.subheader("🎯 BETTING ADVICE")

recommendation = CleanBettingCard.get_recommendation(prediction)
CleanBettingCard.display_card(recommendation)

# ========== LOGIC ANALYSIS ==========
st.divider()
st.subheader("🔍 LOGIC ANALYSIS")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 Algorithm Output")
    
    st.write(f"**Winner:**")
    st.write(f"- Prediction: {prediction['winner']['original_prediction']}")
    st.write(f"- Confidence: {prediction['winner']['original_confidence_score']:.1f}%")
    
    st.write(f"**Totals:**")
    st.write(f"- Prediction: {prediction['totals']['original_direction']}")
    st.write(f"- Confidence: {prediction['totals']['original_confidence_score']:.1f}%")
    st.write(f"- Raw xG: {prediction['totals']['total_xg']:.2f}")
    st.write(f"- Adj. xG: {prediction['totals'].get('adjusted_xg', prediction['totals']['total_xg']):.2f}")
    st.write(f"- Finishing Impact: {prediction['totals'].get('finishing_impact', 0):.1%}")
    st.write(f"- Pattern: {prediction['totals']['original_finishing_alignment']} + {prediction['totals']['original_total_category']}")

with col2:
    st.subheader("⚡ Active Adjustments")
    
    st.write(f"**Winner Adjustments:**")
    if prediction['winner']['has_confidence_bug']:
        st.error("→ 70% Confidence Bug Fix")
    if "100%" in prediction['winner'].get('reason', ''):
        st.success("→ 100% Confidence Strength")
    if prediction['winner']['betting_action'] == 'CAUTION':
        st.warning("→ 90% Overconfidence Caution")
    if "Strong pattern" in prediction['winner'].get('reason', ''):
        st.success("→ Learned Strong Pattern")
    if "Weak pattern" in prediction['winner'].get('reason', ''):
        st.error("→ Learned Weak Pattern")
    
    st.write(f"**Totals Adjustments:**")
    if prediction['totals'].get('variance_effect') == 'AMPLIFIED':
        st.info(f"→ Variance Amplification: {prediction['totals']['variance_effect']}")
    elif prediction['totals'].get('variance_effect') == 'REDUCED':
        st.warning(f"→ Variance Reduction: {prediction['totals']['variance_effect']}")
    if "Strong pattern" in prediction['totals'].get('reason', ''):
        st.success("→ Learned Strong Pattern")
    if "Weak pattern" in prediction['totals'].get('reason', ''):
        st.error("→ Learned Weak Pattern")

with col3:
    st.subheader("📊 Pattern Statistics")
    
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

# ========== FINISHING IMPACT BREAKDOWN ==========
st.divider()
st.subheader("🎯 FINISHING IMPACT BREAKDOWN")

with st.expander("📈 Totals Decision Logic", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Finishing Data:**")
        home_finish = prediction['totals'].get('home_finishing', 0)
        away_finish = prediction['totals'].get('away_finishing', 0)
        
        st.write(f"- Home finishing vs xG: {home_finish:.3f}")
        st.write(f"- Away finishing vs xG: {away_finish:.3f}")
        st.write(f"- Total finishing: {home_finish + away_finish:.3f}")
        
        # Show finishing categories
        totals_predictor = TotalsPredictor(prediction['league'])
        home_cat = totals_predictor.categorize_finishing(home_finish)
        away_cat = totals_predictor.categorize_finishing(away_finish)
        st.write(f"- Home category: {home_cat}")
        st.write(f"- Away category: {away_cat}")
        
    with col2:
        st.write("**Adjustment Calculation:**")
        total_xg = prediction['totals']['total_xg']
        finishing_impact = prediction['totals'].get('finishing_impact', 0)
        adjusted_xg = prediction['totals'].get('adjusted_xg', total_xg)
        threshold = prediction['totals'].get('league_threshold', 2.5)
        
        st.write(f"1. Raw xG: {total_xg:.3f}")
        st.write(f"2. Finishing impact: ({home_finish:.3f} + {away_finish:.3f}) × 0.6 = {finishing_impact:.3f}")
        st.write(f"3. Adjusted xG: {total_xg:.3f} × (1 + {finishing_impact:.3f}) = {adjusted_xg:.3f}")
        st.write(f"4. Threshold: {threshold:.1f}")
        st.write(f"5. Decision: {adjusted_xg:.3f} {'>' if adjusted_xg > threshold else '<'} {threshold:.1f} → **{prediction['totals']['original_direction']}**")
        
        # Show the fix in action
        st.write("**Logic Fix Comparison:**")
        st.write(f"- Old logic: {total_xg:.3f} > {threshold:.1f}? = {'OVER' if total_xg > threshold else 'UNDER'}")
        st.write(f"- New logic: {adjusted_xg:.3f} > {threshold:.1f}? = {prediction['totals']['original_direction']}")

# ========== FEEDBACK ==========
record_outcome_clean(prediction)

# ========== SYSTEM REPORT ==========
st.divider()
st.subheader("📤 System Report")

winner_stats = st.session_state.proven_system.get_pattern_stats(winner_key)
totals_stats = st.session_state.proven_system.get_pattern_stats(totals_key)

winner_success_rate = winner_stats['success'] / winner_stats['total'] if winner_stats['total'] > 0 else 0
totals_success_rate = totals_stats['success'] / totals_stats['total'] if totals_stats['total'] > 0 else 0

report = f"""🎯 FOOTBALL INTELLIGENCE ENGINE v8.2 - LOGIC BUG FIXED
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Version: 8.2 (Logic bug fixed, no crutches)

🎯 BETTING ADVICE:
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
Pattern: {winner_key}
Pattern stats: {winner_stats['success']}/{winner_stats['total']} wins ({winner_success_rate:.0%})

🎯 TOTALS ANALYSIS (LOGIC FIXED):
Final bet: {prediction['totals']['direction']} 2.5
Original algorithm: {prediction['totals']['original_direction']} ({prediction['totals']['original_confidence_score']:.1f}%)
Final confidence: {prediction['totals']['confidence_score']:.0f}/100 ({prediction['totals']['confidence']})
Betting action: {prediction['totals']['betting_action']}
Reason: {prediction['totals']['reason']}
Probability: {prediction['totals']['probability']*100:.1f}%
Stake: {prediction['totals']['stake']}
Variance effect: {prediction['totals']['variance_effect']}
Raw xG: {prediction['totals']['total_xg']:.2f}
Adj. xG: {prediction['totals'].get('adjusted_xg', prediction['totals']['total_xg']):.2f}
Finishing Impact: {prediction['totals'].get('finishing_impact', 0):.1%}
Pattern: {prediction['totals']['original_finishing_alignment']} + {prediction['totals']['original_total_category']}
Home finishing: {prediction['totals'].get('home_finishing', 0):.2f}
Away finishing: {prediction['totals'].get('away_finishing', 0):.2f}
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
✅ LOGIC BUG FIXED:
- Totals predictions now correctly incorporate finishing data
- Old bug: Used raw xG ignoring finishing tendencies
- New logic: adjusted_xg = total_xg × (1 + finishing_impact)
- finishing_impact = (home_finish + away_finish) × 0.6

🔧 ACTIVE ADJUSTMENTS:
70% Zone (68-72%): BET OPPOSITE (confirmed bug)
90% Zone (88-94%): CAUTION (overconfidence trap)
100% Zone (99-100%): BET STRONGLY (confirmed strength)
High Variance: Directional amplification/penalty

🔄 REMOVED CRUTCHES:
- TOTALS anti-patterns (algorithm can handle these now)
- TOTALS success patterns (algorithm should get these right)

📊 PATTERN STATISTICS:
Total patterns tracked: {len(st.session_state.proven_system.pattern_memory)}
Qualifying (≥3 matches): {len([v for v in st.session_state.proven_system.pattern_memory.values() if v['total'] >= 3])}
Strong patterns (>70%): {len([v for v in st.session_state.proven_system.pattern_memory.values() if v['total'] >= 3 and v['success']/v['total'] > 0.7])}
Weak patterns (<40%): {len([v for v in st.session_state.proven_system.pattern_memory.values() if v['total'] >= 3 and v['success']/v['total'] < 0.4])}

📈 TESTING PHASE:
- Running without empirical crutches
- Letting fixed algorithm do its job
- Monitoring for true algorithm weaknesses
"""

st.code(report, language="text")

st.download_button(
    label="📥 Download System Report",
    data=report,
    file_name=f"system_{home_team}_vs_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
    mime="text/plain",
    use_container_width=True
)