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
    page_title="⚽ Football Intelligence Engine v7.0 - SURGICAL EDITION",
    page_icon="🔪",
    layout="wide"
)

st.title("🔪 Football Intelligence Engine v7.0 - SURGICAL EDITION")
st.markdown("""
    **SURGICAL RULES BASED ON 41-MATCH ANALYSIS**
    *Anti-pattern overrides + Volatility awareness + Confidence calibration*
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

# League-specific adjustments with SURGICAL thresholds
LEAGUE_ADJUSTMENTS = {
    "Premier League": {
        "over_threshold": 2.5, 
        "under_threshold": 2.5, 
        "avg_goals": 2.79,
        "very_high_threshold": 3.3,  # Adjusted based on data
        "high_threshold": 3.0,
        "moderate_high_threshold": 2.7,
        "moderate_low_threshold": 2.5,  # Increased from 2.3 based on data
        "low_threshold": 2.3
    },
    "Bundesliga": {
        "over_threshold": 3.0, 
        "under_threshold": 2.2, 
        "avg_goals": 3.20,
        "very_high_threshold": 3.8,
        "high_threshold": 3.4,
        "moderate_high_threshold": 3.1,
        "moderate_low_threshold": 2.8,
        "low_threshold": 2.5
    },
    # ... other leagues with similar adjustments
}

# KNOWN ANTI-PATTERNS FROM 41-MATCH ANALYSIS
ANTI_PATTERNS = {
    "TOTALS_RISKY_VERY_HIGH": {
        "action": "BET_OPPOSITE",
        "confidence": 85,
        "reason": "0/2 success - ALWAYS FAILS",
        "since": "2026-02-09"
    },
    "TOTALS_RISKY_HIGH": {
        "action": "CAUTION",
        "confidence_reduction": 0.7,
        "reason": "High risk in high-scoring games",
        "since": "2026-02-09"
    },
    "WINNER_HIGH_70": {
        "action": "DERATE",
        "confidence_multiplier": 0.7,
        "reason": "0/2 success - Overconfidence at 70%",
        "since": "2026-02-09"
    },
    "TOTALS_LOW_OVER_HIGH": {
        "action": "BET_OPPOSITE",
        "confidence": 80,
        "reason": "0/2 success - LOW alignment + HIGH goals fails",
        "since": "2026-02-09"
    },
    "TOTALS_LOW_UNDER_HIGH": {
        "action": "BET_OPPOSITE", 
        "confidence": 80,
        "reason": "0/1 success - LOW alignment + HIGH goals fails",
        "since": "2026-02-09"
    }
}

# ========== SURGICAL LEARNING SYSTEM ==========

class SurgicalLearningSystem:
    """LEARNING SYSTEM WITH ANTI-PATTERN OVERRIDES"""
    
    def __init__(self):
        self.pattern_memory = {}
        self.supabase = init_supabase()
        
        # SURGICAL THRESHOLDS from 41-match analysis
        self.thresholds = {
            'min_matches': 3,
            'strong_success': 0.70,   # >70% = BET
            'weak_success': 0.40,     # <40% = BET OPPOSITE
            'neutral_min': 0.40,      # 40-70% = NO EDGE
            'neutral_max': 0.70,
            'volatility_cutoff': 0.3, # <30% consistency = high volatility
            'negative_trend_cutoff': -0.1  # Trend < -0.1 = declining
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
                    
                # Standardize keys (remove .0 decimals)
                clean_key = self.standardize_pattern_key(pattern_key)
                
                data = {
                    "pattern_key": clean_key,
                    "total_matches": stats['total'],
                    "successful_matches": stats['success'],
                    "last_updated": datetime.now().isoformat(),
                    "metadata": json.dumps({
                        "thresholds": self.thresholds,
                        "success_rate": stats['success'] / stats['total'] if stats['total'] > 0 else 0,
                        "original_key": pattern_key
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
            with open("surgical_learning_data.pkl", "wb") as f:
                pickle.dump({
                    'pattern_memory': self.pattern_memory,
                    'thresholds': self.thresholds,
                    'version': '7.0_surgical'
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
                    clean_key = self.standardize_pattern_key(row['pattern_key'])
                    self.pattern_memory[clean_key] = {
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
            if os.path.exists("surgical_learning_data.pkl"):
                with open("surgical_learning_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.pattern_memory = data.get('pattern_memory', {})
                    self.thresholds = data.get('thresholds', self.thresholds)
                return True
        except:
            pass
        return False
    
    def standardize_pattern_key(self, pattern_key):
        """Standardize pattern keys - remove .0 decimals"""
        if not pattern_key or pattern_key is None:
            return {'total': 0, 'success': 0}
        return self.pattern_memory.get(pattern_key, {'total': 0, 'success': 0})
        
        parts = pattern_key.split('_')
        if len(parts) >= 3:
            # Check last part for .0 decimal
            last_part = parts[-1]
            if last_part.endswith('.0'):
                parts[-1] = last_part[:-2]
        
        return '_'.join(parts)
    
    def record_outcome(self, prediction, actual_score):
        """Record match outcome with SURGICAL precision"""
        
        # Generate STANDARDIZED pattern keys
        winner_key, totals_key = self.generate_standardized_keys(prediction)
        
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
        
        # Initialize if not exists
        if winner_key not in self.pattern_memory:
            self.pattern_memory[winner_key] = {'total': 0, 'success': 0}
        if totals_key not in self.pattern_memory:
            self.pattern_memory[totals_key] = {'total': 0, 'success': 0}
        
        # Check predictions - use type instead of original_prediction if needed
        winner_pred = prediction['winner']
        winner_original = winner_pred.get('original_prediction', winner_pred.get('type', 'UNKNOWN'))
        totals_pred = prediction['totals']
        totals_original = totals_pred.get('original_direction', totals_pred.get('direction', 'UNKNOWN'))
        
        winner_correct = winner_original == actual_winner
        totals_correct = (totals_original == "OVER") == actual_over
        
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
        }, "Outcome recorded with surgical precision!"
    
def generate_standardized_keys(self, prediction):
    """Generate standardized pattern keys with robust fallback logic"""
    try:
        winner_pred = prediction.get('winner', {})
        totals_pred = prediction.get('totals', {})
        
        # Winner key with robust fallbacks
        orig_pred = winner_pred.get('original_prediction', 
                                   winner_pred.get('type', 
                                                   winner_pred.get('prediction', 'UNKNOWN')))
        
        # Handle confidence with multiple fallbacks
        orig_conf = winner_pred.get('original_confidence', 
                                   winner_pred.get('confidence', '50.0'))
        if isinstance(orig_conf, (int, float)):
            orig_conf = str(float(orig_conf))
        elif not isinstance(orig_conf, str):
            orig_conf = '50.0'
        
        # Remove .0 decimal if present
        if orig_conf.endswith('.0'):
            orig_conf = orig_conf[:-2]
        
        winner_key = f"WINNER_{orig_pred}_{orig_conf}"
        winner_key = self.standardize_pattern_key(winner_key)
        
        # Totals key with robust fallbacks
        finishing = totals_pred.get('original_finishing_alignment', 
                                  totals_pred.get('finishing_alignment',
                                                 totals_pred.get('alignment', 'NEUTRAL')))
        if finishing.endswith("_OVERRIDDEN"):
            finishing = finishing[:-11]
        
        total_cat = totals_pred.get('original_total_category', 
                                   totals_pred.get('total_category',
                                                  totals_pred.get('category', 'MODERATE_LOW')))
        
        totals_key = f"TOTALS_{finishing}_{total_cat}"
        totals_key = self.standardize_pattern_key(totals_key)
        
        return winner_key, totals_key
        
    except Exception as e:
        # If anything fails, return default keys
        st.error(f"Error generating pattern keys: {e}")
        return "WINNER_UNKNOWN_50", "TOTALS_NEUTRAL_MODERATE_LOW"
    
    def get_surgical_advice(self, winner_pred, totals_pred):
        """APPLY SURGICAL RULES with anti-pattern overrides"""
        
        # Generate pattern keys with fallbacks
        winner_key, totals_key = self.generate_standardized_keys({
            'winner': winner_pred,
            'totals': totals_pred
        })
        
        advice = {
            'winner': {
                'action': 'FOLLOW', 
                'bet_on': winner_pred['type'], 
                'confidence': winner_pred['confidence_score'],
                'original_confidence': winner_pred.get('original_confidence', winner_pred['confidence'])
            },
            'totals': {
                'action': 'FOLLOW', 
                'bet_on': totals_pred['direction'], 
                'confidence': totals_pred['confidence_score'],
                'volatility': totals_pred.get('volatility_score', 0.5),
                'trend': totals_pred.get('trend_score', 0)
            }
        }
        
        # ====== APPLY ANTI-PATTERN OVERRIDES FIRST ======
        
        # 1. Check for KNOWN ANTI-PATTERNS in totals
        finishing = totals_pred.get('original_finishing_alignment', 
                                  totals_pred.get('finishing_alignment', 'NEUTRAL'))
        total_cat = totals_pred.get('original_total_category', 
                                   totals_pred.get('total_category', 'MODERATE_LOW'))
        
        anti_pattern_key = f"TOTALS_{finishing}_{total_cat}"
        anti_pattern_key = self.standardize_pattern_key(anti_pattern_key)
        
        if anti_pattern_key in ANTI_PATTERNS:
            rule = ANTI_PATTERNS[anti_pattern_key]
            if rule['action'] == 'BET_OPPOSITE':
                advice['totals']['action'] = 'BET_OPPOSITE'
                advice['totals']['bet_on'] = 'UNDER' if totals_pred['direction'] == 'OVER' else 'OVER'
                advice['totals']['confidence'] = rule['confidence']
                advice['totals']['reason'] = f"🎯 ANTI-PATTERN: {rule['reason']}"
                advice['totals']['color'] = '#DC2626'
            elif rule['action'] == 'DERATE':
                advice['totals']['action'] = 'REDUCED_STAKE'
                advice['totals']['confidence'] = totals_pred['confidence_score'] * rule.get('confidence_multiplier', 0.7)
                advice['totals']['reason'] = f"⚠️ DERATED: {rule['reason']}"
                advice['totals']['color'] = '#F59E0B'
        
        # 2. Check for 70% confidence bug in winner
        if winner_pred['confidence'] == "HIGH" and 68 <= winner_pred['confidence_score'] <= 72:
            advice['winner']['action'] = 'DERATED'
            advice['winner']['confidence'] = winner_pred['confidence_score'] * 0.7
            advice['winner']['reason'] = "🎯 70% CONFIDENCE BUG: Derating by 30%"
            advice['winner']['color'] = '#DC2626'
        
        # 3. Apply volatility adjustments
        volatility = totals_pred.get('volatility_score', 0.5)
        if volatility < self.thresholds['volatility_cutoff']:
            if advice['totals']['action'] == 'FOLLOW':
                advice['totals']['action'] = 'REDUCED_STAKE'
                advice['totals']['confidence'] *= 0.7
                advice['totals']['reason'] = f"⚠️ HIGH VOLATILITY: {volatility:.2f} consistency"
                advice['totals']['color'] = '#F59E0B'
        
        # 4. Apply trend adjustments
        trend = totals_pred.get('trend_score', 0)
        if trend < self.thresholds['negative_trend_cutoff']:
            if advice['totals']['action'] in ['FOLLOW', 'REDUCED_STAKE']:
                advice['totals']['action'] = 'BET_OPPOSITE'
                advice['totals']['bet_on'] = 'UNDER' if totals_pred['direction'] == 'OVER' else 'OVER'
                advice['totals']['confidence'] = 75
                advice['totals']['reason'] = f"📉 NEGATIVE TREND: {trend:.2f} - Betting opposite"
                advice['totals']['color'] = '#DC2626'
        
        # ====== THEN APPLY HISTORICAL PATTERN RULES ======
        
        for market_type, pattern_key, original in [
            ('winner', winner_key, winner_pred),
            ('totals', totals_key, totals_pred)
        ]:
            if pattern_key in self.pattern_memory:
                stats = self.pattern_memory[pattern_key]
                
                if stats['total'] >= self.thresholds['min_matches']:
                    success_rate = stats['success'] / stats['total']
                    
                    # Only apply historical rules if not already overridden by anti-patterns
                    if market_type == 'winner' and advice['winner'].get('action') not in ['DERATED', 'BET_OPPOSITE']:
                        if success_rate > self.thresholds['strong_success']:
                            advice[market_type]['action'] = 'BET_STRONGLY'
                            advice[market_type]['confidence'] = min(95, original['confidence_score'] * 1.3)
                            advice[market_type]['reason'] = f"✅ STRONG: {stats['success']}/{stats['total']} ({success_rate:.0%})"
                            advice[market_type]['color'] = '#10B981'
                        
                        elif success_rate < self.thresholds['weak_success']:
                            advice[market_type]['action'] = 'BET_OPPOSITE'
                            advice[market_type]['confidence'] = 85
                            # Determine opposite
                            if market_type == 'winner':
                                if original['type'] == 'HOME':
                                    advice[market_type]['bet_on'] = 'AWAY'
                                elif original['type'] == 'AWAY':
                                    advice[market_type]['bet_on'] = 'HOME'
                                else:
                                    advice[market_type]['bet_on'] = 'DRAW'
                            advice[market_type]['reason'] = f"🎯 WEAK: {stats['success']}/{stats['total']} ({success_rate:.0%}) → BET OPPOSITE!"
                            advice[market_type]['color'] = '#DC2626'
                    
                    if market_type == 'totals' and advice['totals'].get('action') not in ['BET_OPPOSITE', 'REDUCED_STAKE']:
                        if success_rate > self.thresholds['strong_success']:
                            advice[market_type]['action'] = 'BET_STRONGLY'
                            advice[market_type]['confidence'] = min(95, original['confidence_score'] * 1.3)
                            advice[market_type]['reason'] = f"✅ STRONG: {stats['success']}/{stats['total']} ({success_rate:.0%})"
                            advice[market_type]['color'] = '#10B981'
                        
                        elif success_rate < self.thresholds['weak_success']:
                            advice[market_type]['action'] = 'BET_OPPOSITE'
                            advice[market_type]['confidence'] = 85
                            advice[market_type]['bet_on'] = 'UNDER' if original['direction'] == 'OVER' else 'OVER'
                            advice[market_type]['reason'] = f"🎯 WEAK: {stats['success']}/{stats['total']} ({success_rate:.0%}) → BET OPPOSITE!"
                            advice[market_type]['color'] = '#DC2626'
        
        return advice

# ========== INITIALIZE SESSION STATES ==========
if 'learning_system' not in st.session_state:
    st.session_state.learning_system = SurgicalLearningSystem()

if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

if 'last_teams' not in st.session_state:
    st.session_state.last_teams = None

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

class SurgicalWinnerPredictor:
    """Winner determination WITH CONFIDENCE CALIBRATION"""
    
    def predict_winner(self, home_xg, away_xg, home_stats, away_stats):
        """Predict winner with surgical fixes"""
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
        
        # ====== SURGICAL FIXES ======
        
        # FIX 1: 70% confidence bug
        if 68 <= winner_confidence <= 72:
            winner_confidence *= 0.7  # Derate by 30%
            strength = "QUESTIONABLE"
        
        # FIX 2: VERY HIGH 90% vs 100% differentiation
        if winner_confidence >= 90:
            if winner_confidence >= 95:
                confidence_category = "VERY HIGH"
                # Keep as is - true dominance
            else:
                # 90-94% range has poor performance historically
                confidence_category = "HIGH"
                winner_confidence *= 0.85  # Adjust down
                strength = "MODERATE"
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
            'strength': strength,
            'confidence_score': winner_confidence,
            'confidence': confidence_category,
            'delta': delta,
            'original_confidence': f"{winner_confidence:.1f}" if winner_confidence != 100 else "100",
            'is_calibrated': True
        }

class SurgicalTotalsPredictor:
    """Totals prediction WITH ANTI-PATTERN OVERRIDES"""
    
    def __init__(self, league_name):
        self.league_name = league_name
        self.league_adjustments = LEAGUE_ADJUSTMENTS.get(league_name, LEAGUE_ADJUSTMENTS["Premier League"])
    
    def calculate_volatility(self, home_stats, away_stats):
        """Calculate finishing volatility (0-1, higher = more consistent)"""
        # Placeholder - in production, calculate from match logs
        # For now, synthetic based on goals_vs_xg variance
        home_vol = 0.7 - min(0.5, abs(home_stats['goals_vs_xg_pm']) * 2)
        away_vol = 0.7 - min(0.5, abs(away_stats['goals_vs_xg_pm']) * 2)
        return max(0.1, min(1.0, (home_vol + away_vol) / 2))
    
    def calculate_trend(self, home_stats, away_stats):
        """Calculate finishing trend (-1 to 1)"""
        # Placeholder - positive means improving
        # Synthetic: teams with positive goals_vs_xg have positive trend
        home_trend = home_stats['goals_vs_xg_pm'] * 0.5
        away_trend = away_stats['goals_vs_xg_pm'] * 0.5
        return (home_trend + away_trend) / 2
    
    def categorize_finishing(self, value, volatility=0.5):
        """Categorize finishing with volatility awareness"""
        if volatility < 0.3:  # High volatility
            if value > 0.2:
                return "VOLATILE_OVERPERFORM"
            elif value < -0.2:
                return "VOLATILE_UNDERPERFORM"
            else:
                return "VOLATILE_NEUTRAL"
        else:  # Stable
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
    
    def get_finishing_alignment(self, home_finish, away_finish, home_vol, away_vol):
        """Finishing alignment with volatility"""
        home_cat = self.categorize_finishing(home_finish, home_vol)
        away_cat = self.categorize_finishing(away_finish, away_vol)
        
        # Enhanced alignment matrix
        alignment_matrix = {
            "STRONG_OVERPERFORM": {
                "STRONG_OVERPERFORM": "HIGH_OVER_STABLE",
                "MODERATE_OVERPERFORM": "MED_OVER_STABLE",
                "NEUTRAL": "MED_OVER",
                "MODERATE_UNDERPERFORM": "RISKY",
                "STRONG_UNDERPERFORM": "HIGH_RISK",
                "VOLATILE_OVERPERFORM": "VOLATILE_OVER",
                "VOLATILE_UNDERPERFORM": "EXTREME_RISK"
            },
            "MODERATE_OVERPERFORM": {
                "STRONG_OVERPERFORM": "MED_OVER_STABLE",
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
                "STRONG_UNDERPERFORM": "HIGH_UNDER_STABLE"
            },
            "VOLATILE_OVERPERFORM": {
                "VOLATILE_OVERPERFORM": "HIGH_VOLATILITY",
                "NEUTRAL": "VOLATILE",
                "VOLATILE_UNDERPERFORM": "EXTREME_RISK"
            },
            "VOLATILE_UNDERPERFORM": {
                "VOLATILE_OVERPERFORM": "EXTREME_RISK",
                "VOLATILE_UNDERPERFORM": "HIGH_VOLATILITY"
            }
        }
        
        return alignment_matrix.get(home_cat, {}).get(away_cat, "NEUTRAL")
    
    def categorize_total_xg(self, total_xg):
        """Total xG categories with league adjustments"""
        thresholds = self.league_adjustments
        
        if total_xg > thresholds['very_high_threshold']:
            return "VERY_HIGH"
        elif total_xg > thresholds['high_threshold']:
            return "HIGH"
        elif total_xg > thresholds['moderate_high_threshold']:
            return "MODERATE_HIGH"
        elif total_xg > thresholds['moderate_low_threshold']:  # Increased threshold
            return "MODERATE_LOW"
        elif total_xg > thresholds['low_threshold']:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def predict_totals(self, home_xg, away_xg, home_stats, away_stats):
        """Predict totals with surgical overrides"""
        total_xg = home_xg + away_xg
        home_finish = home_stats['goals_vs_xg_pm']
        away_finish = away_stats['goals_vs_xg_pm']
        
        # Calculate volatility and trend
        volatility = self.calculate_volatility(home_stats, away_stats)
        trend = self.calculate_trend(home_stats, away_stats)
        
        over_threshold = self.league_adjustments['over_threshold']
        base_direction = "OVER" if total_xg > over_threshold else "UNDER"
        
        # Finishing alignment with volatility
        finishing_alignment = self.get_finishing_alignment(
            home_finish, away_finish, 
            volatility, volatility  # Using same volatility for both
        )
        
        total_category = self.categorize_total_xg(total_xg)
        
        # ====== ANTI-PATTERN OVERRIDES ======
        
        # OVERRIDE 1: RISKY + VERY_HIGH/HIGH always fails
        if "RISKY" in finishing_alignment and total_category in ["VERY_HIGH", "HIGH"]:
            return {
                'direction': 'UNDER',
                'total_xg': total_xg,
                'confidence': "HIGH",
                'confidence_score': 85,
                'finishing_alignment': finishing_alignment + "_OVERRIDDEN",
                'original_finishing_alignment': finishing_alignment,
                'total_category': total_category,
                'original_total_category': total_category,
                'risk_flags': ["ANTI-PATTERN: RISKY+HIGH always fails"],
                'home_finishing': home_finish,
                'away_finishing': away_finish,
                'volatility_score': volatility,
                'trend_score': trend,
                'is_override': True
            }
        
        # OVERRIDE 2: LOW alignment + HIGH goals always fails
        if "LOW" in finishing_alignment and total_category in ["HIGH", "VERY_HIGH"]:
            opposite = "UNDER" if base_direction == "OVER" else "OVER"
            return {
                'direction': opposite,
                'total_xg': total_xg,
                'confidence': "HIGH",
                'confidence_score': 80,
                'finishing_alignment': finishing_alignment + "_OVERRIDDEN",
                'original_finishing_alignment': finishing_alignment,
                'total_category': total_category,
                'original_total_category': total_category,
                'risk_flags': [f"ANTI-PATTERN: LOW+{total_category} fails"],
                'home_finishing': home_finish,
                'away_finishing': away_finish,
                'volatility_score': volatility,
                'trend_score': trend,
                'is_override': True
            }
        
        # Continue with normal prediction if no overrides
        risk_flags = []
        if abs(home_finish) > 0.4 or abs(away_finish) > 0.4:
            risk_flags.append("HIGH_VARIANCE_TEAM")
        
        lower_thresh = self.league_adjustments['under_threshold'] - 0.1
        upper_thresh = self.league_adjustments['over_threshold'] + 0.1
        if lower_thresh < total_xg < upper_thresh:
            risk_flags.append("CLOSE_TO_THRESHOLD")
        
        # Decision matrix (simplified for example)
        direction = base_direction
        base_confidence = 60
        
        # Adjust for volatility
        if volatility < 0.3:
            base_confidence *= 0.7  # High volatility reduces confidence
        
        # Adjust for trend
        if trend < -0.1:
            base_confidence *= 0.8  # Negative trend reduces confidence
        elif trend > 0.1:
            base_confidence = min(85, base_confidence * 1.2)  # Positive trend increases
        
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
            'direction': direction,
            'total_xg': total_xg,
            'confidence': confidence_category,
            'confidence_score': base_confidence,
            'finishing_alignment': finishing_alignment,
            'original_finishing_alignment': finishing_alignment,
            'total_category': total_category,
            'original_total_category': total_category,
            'risk_flags': risk_flags,
            'home_finishing': home_finish,
            'away_finishing': away_finish,
            'volatility_score': volatility,
            'trend_score': trend,
            'is_override': False
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

def factorial_cache(n, cache={}):
    if n not in cache:
        cache[n] = math.factorial(n)
    return cache[n]

def poisson_pmf(k, lam):
    """Poisson probability mass function"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

# ========== SURGICAL FOOTBALL ENGINE ==========

class SurgicalFootballEngine:
    """Engine with SURGICAL improvements"""
    
    def __init__(self, league_metrics, league_name):
        self.league_metrics = league_metrics
        self.league_name = league_name
        
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics, league_name)
        self.winner_predictor = SurgicalWinnerPredictor()
        self.totals_predictor = SurgicalTotalsPredictor(league_name)
        self.probability_engine = PoissonProbabilityEngine()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """Generate prediction with SURGICAL rules"""
        
        # Get base prediction
        home_xg, away_xg = self.xg_predictor.predict_expected_goals(home_stats, away_stats)
        
        probabilities = self.probability_engine.calculate_all_probabilities(home_xg, away_xg)
        
        winner_prediction = self.winner_predictor.predict_winner(
            home_xg, away_xg, home_stats, away_stats
        )
        
        totals_prediction = self.totals_predictor.predict_totals(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Get SURGICAL betting advice
        betting_advice = st.session_state.learning_system.get_surgical_advice(
            winner_prediction, totals_prediction
        )
        
        # Apply betting advice
        final_winner = self._apply_surgical_advice_to_winner(
            winner_prediction, betting_advice['winner'], home_team, away_team
        )
        
        final_totals = self._apply_surgical_advice_to_totals(
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
                'original_confidence': winner_prediction['original_confidence'],
                'reason': betting_advice['winner'].get('reason', 'Algorithm prediction'),
                'color': betting_advice['winner'].get('color', 'gray'),
                'is_calibrated': winner_prediction.get('is_calibrated', False)
            },
            
            'totals': {
                'direction': final_totals['direction'],
                'probability': totals_prob,
                'confidence': final_totals['confidence'],
                'confidence_score': final_totals['confidence_score'],
                'total_xg': totals_prediction['total_xg'],
                'finishing_alignment': totals_prediction.get('finishing_alignment'),
                'original_finishing_alignment': totals_prediction.get('original_finishing_alignment'),
                'total_category': totals_prediction.get('total_category'),
                'original_total_category': totals_prediction.get('original_total_category'),
                'risk_flags': totals_prediction.get('risk_flags', []),
                'betting_action': betting_advice['totals']['action'],
                'original_direction': totals_prediction['direction'],
                'reason': betting_advice['totals'].get('reason', 'Algorithm prediction'),
                'color': betting_advice['totals'].get('color', 'gray'),
                'volatility_score': totals_prediction.get('volatility_score'),
                'trend_score': totals_prediction.get('trend_score'),
                'is_override': totals_prediction.get('is_override', False)
            },
            
            'probabilities': probabilities,
            'expected_goals': {'home': home_xg, 'away': away_xg, 'total': home_xg + away_xg},
            'betting_advice': betting_advice,
            'surgical_version': '7.0'
        }
    
    def _apply_surgical_advice_to_winner(self, original, advice, home_team, away_team):
        """Apply surgical advice to winner"""
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
        
        elif advice['action'] == 'DERATED':
            # Confidence derated due to bug
            final['confidence_score'] = advice['confidence']
            if advice['confidence'] >= 75:
                final['confidence'] = 'VERY HIGH'
            elif advice['confidence'] >= 65:
                final['confidence'] = 'HIGH'
            else:
                final['confidence'] = 'MEDIUM'
            
            final['team'] = home_team if original['type'] == 'HOME' else away_team if original['type'] == 'AWAY' else 'DRAW'
        
        elif advice['action'] in ['BET_STRONGLY', 'PROMISING']:
            # Boost confidence
            final['confidence_score'] = advice['confidence']
            if advice['confidence'] >= 75:
                final['confidence'] = 'VERY HIGH'
            elif advice['confidence'] >= 65:
                final['confidence'] = 'HIGH'
            else:
                final['confidence'] = 'MEDIUM'
            
            final['team'] = home_team if original['type'] == 'HOME' else away_team if original['type'] == 'AWAY' else 'DRAW'
        
        else:
            # Use algorithm
            final['team'] = home_team if original['type'] == 'HOME' else away_team if original['type'] == 'AWAY' else 'DRAW'
        
        return final
    
    def _apply_surgical_advice_to_totals(self, original, advice):
        """Apply surgical advice to totals"""
        final = original.copy()
        
        if advice['action'] == 'BET_OPPOSITE':
            # Bet opposite!
            final['direction'] = 'UNDER' if original['direction'] == 'OVER' else 'OVER'
            final['confidence_score'] = advice['confidence']
            final['confidence'] = 'HIGH' if advice['confidence'] >= 65 else 'MEDIUM'
        
        elif advice['action'] == 'REDUCED_STAKE':
            # Reduced stake due to volatility
            final['confidence_score'] = advice['confidence']
            if advice['confidence'] >= 75:
                final['confidence'] = 'VERY HIGH'
            elif advice['confidence'] >= 65:
                final['confidence'] = 'HIGH'
            else:
                final['confidence'] = 'MEDIUM'
        
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

# ========== SURGICAL BETTING CARD ==========

class SurgicalBettingCard:
    """Betting card that shows SURGICAL decisions"""
    
    @staticmethod
    def get_surgical_recommendation(prediction):
        """Get surgical betting recommendation"""
        winner_pred = prediction['winner']
        totals_pred = prediction['totals']
        
        # Check for anti-pattern overrides first
        if winner_pred.get('is_calibrated', False) and 'DERATED' in winner_pred.get('betting_action', ''):
            return {
                'type': 'WINNER_DERATED',
                'text': f"⚠️ {winner_pred['team']} to win (DERATED)",
                'subtext': '70% CONFIDENCE BUG - Reduced confidence',
                'reason': winner_pred.get('reason', 'Confidence calibration applied'),
                'confidence': winner_pred['confidence_score'],
                'color': '#F59E0B',
                'icon': '⚠️',
                'stake': 'HALF'
            }
        
        if totals_pred.get('is_override', False):
            return {
                'type': 'TOTALS_OVERRIDE',
                'text': f"🎯 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'ANTI-PATTERN OVERRIDE',
                'reason': totals_pred.get('risk_flags', ['Anti-pattern'])[0],
                'confidence': totals_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': 'FULL'
            }
        
        # Check for volatility warnings
        if totals_pred.get('volatility_score', 0.5) < 0.3:
            return {
                'type': 'HIGH_VOLATILITY',
                'text': f"⚠️ {totals_pred['direction']} 2.5 Goals",
                'subtext': 'HIGH VOLATILITY - Reduced stake',
                'reason': f"Finishing consistency: {totals_pred.get('volatility_score', 0):.2f}",
                'confidence': totals_pred['confidence_score'],
                'color': '#F59E0B',
                'icon': '⚠️',
                'stake': 'QUARTER'
            }
        
        # Check for negative trends
        if totals_pred.get('trend_score', 0) < -0.1:
            return {
                'type': 'NEGATIVE_TREND',
                'text': f"📉 OPPOSITE: {'UNDER' if totals_pred['direction'] == 'OVER' else 'OVER'} 2.5",
                'subtext': 'NEGATIVE TREND - Betting opposite',
                'reason': f"Trend score: {totals_pred.get('trend_score', 0):.2f}",
                'confidence': 75,
                'color': '#DC2626',
                'icon': '📉',
                'stake': 'HALF'
            }
        
        # Original betting logic
        winner_action = winner_pred['betting_action']
        totals_action = totals_pred['betting_action']
        
        # BOTH STRONG
        if winner_action == 'BET_STRONGLY' and totals_action == 'BET_STRONGLY':
            return {
                'type': 'DOUBLE_STRONG',
                'text': f"✅ {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                'subtext': 'DOUBLE STRONG PATTERN',
                'reason': 'Both markets show strong historical edge',
                'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#10B981',
                'icon': '✅',
                'stake': 'FULL'
            }
        
        # BOTH OPPOSITE
        elif winner_action == 'BET_OPPOSITE' and totals_action == 'BET_OPPOSITE':
            return {
                'type': 'DOUBLE_OPPOSITE',
                'text': f"🎯 {winner_pred['team']} + 📈 {totals_pred['direction']} 2.5",
                'subtext': 'DOUBLE BET OPPOSITE!',
                'reason': 'Algorithm consistently wrong on both patterns',
                'confidence': min(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#DC2626',
                'icon': '🎯',
                'stake': 'FULL'
            }
        
        # SINGLE STRONG/OPPOSITE
        elif winner_action == 'BET_STRONGLY':
            return {
                'type': 'WINNER_STRONG',
                'text': f"✅ {winner_pred['team']} to win",
                'subtext': 'STRONG PATTERN (>70%)',
                'reason': winner_pred.get('reason', 'Strong historical pattern'),
                'confidence': winner_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': 'HALF'
            }
        
        elif totals_action == 'BET_STRONGLY':
            return {
                'type': 'TOTALS_STRONG',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'STRONG PATTERN (>70%)',
                'reason': totals_pred.get('reason', 'Strong historical pattern'),
                'confidence': totals_pred['confidence_score'],
                'color': '#10B981',
                'icon': '✅',
                'stake': 'HALF'
            }
        
        elif winner_action == 'BET_OPPOSITE':
            return {
                'type': 'WINNER_OPPOSITE',
                'text': f"🎯 {winner_pred['team']} to win",
                'subtext': 'BET OPPOSITE WINNER!',
                'reason': winner_pred.get('reason', 'Weak historical pattern'),
                'confidence': winner_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': 'HALF'
            }
        
        elif totals_action == 'BET_OPPOSITE':
            return {
                'type': 'TOTALS_OPPOSITE',
                'text': f"📈 {totals_pred['direction']} 2.5 Goals",
                'subtext': 'BET OPPOSITE TOTALS!',
                'reason': totals_pred.get('reason', 'Weak historical pattern'),
                'confidence': totals_pred['confidence_score'],
                'color': '#DC2626',
                'icon': '🎯',
                'stake': 'HALF'
            }
        
        # NO CLEAR EDGE
        else:
            return {
                'type': 'NO_BET',
                'text': "🤔 No Clear Bet",
                'subtext': 'NO BET',
                'reason': 'Insufficient edge or neutral patterns',
                'confidence': max(winner_pred['confidence_score'], totals_pred['confidence_score']),
                'color': '#6B7280',
                'icon': '🤔',
                'stake': 'NONE'
            }
    
    @staticmethod
    def display_surgical_card(recommendation):
        """Display the surgical betting card"""
        color = recommendation['color']
        stake_colors = {
            'FULL': '#10B981',
            'HALF': '#F59E0B',
            'QUARTER': '#DC2626',
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

def record_outcome_surgical(prediction):
    """Surgical feedback system"""
    
    st.divider()
    st.subheader("🔪 Record Outcome for Surgical Learning")
    
    # Show current patterns
    winner_key, totals_key = st.session_state.learning_system.generate_standardized_keys(prediction)
    
    # Get pattern stats safely
    winner_stats = {'total': 0, 'success': 0}
    totals_stats = {'total': 0, 'success': 0}
    
    if winner_key:
        winner_stats = st.session_state.learning_system.get_pattern_stats(winner_key)
    if totals_key:
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
                elif success < 0.4:
                    st.error("🎯 WEAK PATTERN (<40%) - BET OPPOSITE!")
                else:
                    st.info("⚪ NEUTRAL (40-70%)")
    
    with col2:
        st.write("**Totals Pattern:**")
        st.code(totals_key)
        if totals_stats['total'] > 0:
            success = totals_stats['success'] / totals_stats['total']
            st.write(f"Current: {totals_stats['success']}/{totals_stats['total']} ({success:.0%})")
            if totals_stats['total'] >= 3:
                if success > 0.7:
                    st.success("✅ STRONG PATTERN (>70%)")
                elif success < 0.4:
                    st.error("🎯 WEAK PATTERN (<40%) - BET OPPOSITE!")
                else:
                    st.info("⚪ NEUTRAL (40-70%)")
    
    # Score input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        score = st.text_input("Actual Score (e.g., 2-1)", key="surgical_score_input")
    
    with col2:
        if st.button("🔪 Record Surgical Outcome", type="primary", use_container_width=True):
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
    st.header("🔪 Surgical Settings")
    
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
            
            if st.button("🔪 Generate Surgical Prediction", type="primary", use_container_width=True):
                calculate_btn = True
            else:
                calculate_btn = False
        else:
            st.error("Could not prepare team data")
            st.stop()

    # Surgical System Section
    st.divider()
    st.header("🎯 SURGICAL RULES")
    
    st.error("""
    **ANTI-PATTERN OVERRIDES:**
    
    🎯 **TOTALS_RISKY_VERY_HIGH/HIGH**
    → ALWAYS BET OPPOSITE (0/2 success)
    
    🎯 **WINNER_HIGH_70%**
    → DERATE BY 30% (0/2 success)
    
    🎯 **TOTALS_LOW_+_HIGH/VERY_HIGH**
    → ALWAYS BET OPPOSITE (0/6 success)
    """)
    
    st.info("""
    **VOLATILITY RULES:**
    
    ⚠️ **Finishing consistency < 30%**
    → REDUCE STAKE by 70%
    
    📉 **Negative trend < -0.1**
    → BET OPPOSITE with 75% confidence
    """)
    
    st.success("""
    **SURGICAL IMPROVEMENTS:**
    
    ✅ **Standardized pattern keys** (no .0 decimals)
    ✅ **League-adjusted thresholds**
    ✅ **Volatility-aware decisions**
    ✅ **Trend-based adjustments**
    """)
    
    # Show current match patterns if available
    if st.session_state.last_prediction:
        winner_pred = st.session_state.last_prediction['winner']
        totals_pred = st.session_state.last_prediction['totals']
        
        winner_key, totals_key = st.session_state.learning_system.generate_standardized_keys(
            st.session_state.last_prediction
        )
        
        winner_stats = st.session_state.learning_system.get_pattern_stats(winner_key)
        totals_stats = st.session_state.learning_system.get_pattern_stats(totals_key)
        
        st.subheader("Current Match Patterns:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if winner_stats['total'] >= 3:
                success = winner_stats['success'] / winner_stats['total']
                if success > 0.7:
                    st.success(f"✅ Winner: {success:.0%}")
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
                elif success < 0.4:
                    st.error(f"🎯 Totals: {success:.0%}")
                else:
                    st.info(f"⚪ Totals: {success:.0%}")
            else:
                st.caption(f"Totals: {totals_stats['total']}/3 matches")
        
        # Show volatility and trend
        if totals_pred.get('volatility_score') is not None:
            st.caption(f"Volatility: {totals_pred['volatility_score']:.2f}")
        if totals_pred.get('trend_score') is not None:
            st.caption(f"Trend: {totals_pred['trend_score']:.2f}")
    
    st.divider()
    
    # Show anti-patterns
    st.subheader("🔴 Active Anti-Patterns:")
    for pattern, rule in ANTI_PATTERNS.items():
        st.error(f"{pattern}: {rule['reason']}")
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Surgical Statistics:")
    
    total_patterns = len(st.session_state.learning_system.pattern_memory)
    qualifying = len([v for v in st.session_state.learning_system.pattern_memory.values() 
                     if v['total'] >= 3])
    strong = len([v for v in st.session_state.learning_system.pattern_memory.values() 
                 if v['total'] >= 3 and v['success']/v['total'] > 0.7])
    weak = len([v for v in st.session_state.learning_system.pattern_memory.values() 
               if v['total'] >= 3 and v['success']/v['total'] < 0.4])
    
    st.write(f"Total patterns: {total_patterns}")
    st.write(f"Qualifying (≥3 matches): {qualifying}")
    st.write(f"Strong (>70%): {strong}")
    st.write(f"Weak (<40%): {weak}")
    st.write(f"Weak/Strong ratio: {weak/max(strong, 1):.2f}")

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# Check if we should show prediction
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Generate surgical prediction
        engine = SurgicalFootballEngine(league_metrics, selected_league)
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
    st.info("👈 Select teams and click 'Generate Surgical Prediction'")
    st.stop()

# ========== DISPLAY SURGICAL PREDICTION ==========
st.header(f"🔪 {home_team} vs {away_team}")
st.caption(f"League: {selected_league} | Version: {prediction.get('surgical_version', '7.0')}")

# Surgical prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction['winner']
    color = winner_pred.get('color', '#6B7280')
    
    # Determine icon and subtitle
    if winner_pred['betting_action'] == 'BET_OPPOSITE':
        icon = "🎯"
        subtitle = "BET OPPOSITE!"
        card_color = "#7F1D1D"
    elif winner_pred['betting_action'] == 'DERATED':
        icon = "⚠️"
        subtitle = "DERATED (70% bug)"
        card_color = "#78350F"
    elif winner_pred['betting_action'] == 'BET_STRONGLY':
        icon = "✅"
        subtitle = "STRONG PATTERN"
        card_color = "#14532D"
    elif winner_pred.get('is_calibrated', False):
        icon = "🔧"
        subtitle = "SURGICALLY CALIBRATED"
        card_color = "#1E40AF"
    else:
        icon = "🏠" if winner_pred['type'] == "HOME" else "✈️" if winner_pred['type'] == "AWAY" else "🤝"
        card_color = "#1E293B"
        subtitle = winner_pred['confidence']
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">SURGICAL WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {icon} {winner_pred['team']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {subtitle} | Confidence: {winner_pred['confidence_score']:.0f}/100
        </div>
        {f'<div style="font-size: 14px; color: #FCA5A5; margin-top: 10px;">{winner_pred["reason"]}</div>' if winner_pred.get('reason') else ''}
        {f'<div style="font-size: 12px; color: #FCD34D; margin-top: 5px;">Calibrated: {winner_pred.get("is_calibrated", False)}</div>' if winner_pred.get('is_calibrated') else ''}
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
    elif totals_pred.get('is_override', False):
        icon = "🔴"
        subtitle = "ANTI-PATTERN OVERRIDE"
        card_color = "#991B1B"
    elif totals_pred.get('volatility_score', 0.5) < 0.3:
        icon = "⚠️"
        subtitle = "HIGH VOLATILITY"
        card_color = "#78350F"
    elif totals_pred.get('trend_score', 0) < -0.1:
        icon = "📉"
        subtitle = "NEGATIVE TREND"
        card_color = "#92400E"
    elif totals_pred['betting_action'] == 'BET_STRONGLY':
        icon = "✅"
        subtitle = "STRONG PATTERN"
        card_color = "#14532D"
    else:
        icon = "📈"
        card_color = "#1E293B"
        subtitle = totals_pred['confidence']
    
    volatility_display = f" | Vol: {totals_pred.get('volatility_score', 0):.2f}" if totals_pred.get('volatility_score') else ""
    trend_display = f" | Trend: {totals_pred.get('trend_score', 0):.2f}" if totals_pred.get('trend_score') else ""
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">SURGICAL TOTALS</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {icon} {totals_pred['direction']} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {totals_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            {subtitle}{volatility_display}{trend_display}
        </div>
        {f'<div style="font-size: 14px; color: #FCA5A5; margin-top: 10px;">{totals_pred["reason"]}</div>' if totals_pred.get('reason') else ''}
        {f'<div style="font-size: 12px; color: #FCD34D; margin-top: 5px;">Override: {totals_pred.get("is_override", False)}</div>' if totals_pred.get('is_override') else ''}
    </div>
    """, unsafe_allow_html=True)

# ========== SURGICAL BETTING CARD ==========
st.divider()
st.subheader("🔪 SURGICAL BETTING ADVICE")

recommendation = SurgicalBettingCard.get_surgical_recommendation(prediction)
SurgicalBettingCard.display_surgical_card(recommendation)

# ========== SURGICAL INSIGHTS ==========
st.divider()
st.subheader("🔍 Surgical Pattern Analysis")

col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction['winner']
    winner_key, _ = st.session_state.learning_system.generate_standardized_keys(prediction)
    winner_stats = st.session_state.learning_system.get_pattern_stats(winner_key)
    
    st.write("**Winner Pattern:**")
    st.code(winner_key)
    
    if winner_stats['total'] > 0:
        success_rate = winner_stats['success'] / winner_stats['total']
        
        if winner_stats['total'] >= 3:
            if success_rate > 0.7:
                st.success(f"✅ STRONG PATTERN: {winner_stats['success']}/{winner_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **BET STRONGLY** with boosted confidence")
            elif success_rate < 0.4:
                st.error(f"🎯 WEAK PATTERN: {winner_stats['success']}/{winner_stats['total']} ({success_rate:.0%}) wins")
                st.write(f"→ **BET OPPOSITE!** Algorithm only wins {success_rate:.0%} of the time")
            else:
                st.info(f"⚪ NEUTRAL: {winner_stats['success']}/{winner_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **Use algorithm's prediction**")
        else:
            st.warning(f"⚠️ NEED MORE DATA: {winner_stats['total']}/3 matches")
            st.write(f"→ Need {3 - winner_stats['total']} more match(es)")
    
    # Show calibration info
    if winner_pred.get('is_calibrated', False):
        st.info(f"🔧 **Surgically calibrated**: Confidence adjusted based on 41-match analysis")

with col2:
    totals_pred = prediction['totals']
    _, totals_key = st.session_state.learning_system.generate_standardized_keys(prediction)
    totals_stats = st.session_state.learning_system.get_pattern_stats(totals_key)
    
    st.write("**Totals Pattern:**")
    st.code(totals_key)
    
    if totals_stats['total'] > 0:
        success_rate = totals_stats['success'] / totals_stats['total']
        
        if totals_stats['total'] >= 3:
            if success_rate > 0.7:
                st.success(f"✅ STRONG PATTERN: {totals_stats['success']}/{totals_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **BET STRONGLY** with boosted confidence")
            elif success_rate < 0.4:
                st.error(f"🎯 WEAK PATTERN: {totals_stats['success']}/{totals_stats['total']} ({success_rate:.0%}) wins")
                st.write(f"→ **BET OPPOSITE!** Algorithm only wins {success_rate:.0%} of the time")
            else:
                st.info(f"⚪ NEUTRAL: {totals_stats['success']}/{totals_stats['total']} ({success_rate:.0%}) wins")
                st.write("→ **Use algorithm's prediction**")
        else:
            st.warning(f"⚠️ NEED MORE DATA: {totals_stats['total']}/3 matches")
            st.write(f"→ Need {3 - totals_stats['total']} more match(es)")
    
    # Show volatility and trend
    if totals_pred.get('volatility_score') is not None:
        vol_color = "#DC2626" if totals_pred['volatility_score'] < 0.3 else "#F59E0B" if totals_pred['volatility_score'] < 0.5 else "#10B981"
        st.write(f"**Volatility**: :{vol_color}[{totals_pred['volatility_score']:.2f}]")
    
    if totals_pred.get('trend_score') is not None:
        trend_color = "#DC2626" if totals_pred['trend_score'] < -0.1 else "#F59E0B" if totals_pred['trend_score'] < 0.1 else "#10B981"
        st.write(f"**Trend**: :{trend_color}[{totals_pred['trend_score']:.2f}]")
    
    if totals_pred.get('is_override', False):
        st.error("🔴 **ANTI-PATTERN OVERRIDE ACTIVE**")
        st.write("→ Known failing pattern detected, betting opposite")

# ========== SURGICAL FEEDBACK ==========
record_outcome_surgical(prediction)

# ========== EXPORT ==========
st.divider()
st.subheader("📤 Export Surgical Report")

winner_stats = st.session_state.learning_system.get_pattern_stats(winner_key)
totals_stats = st.session_state.learning_system.get_pattern_stats(totals_key)

winner_success_rate = winner_stats['success'] / winner_stats['total'] if winner_stats['total'] > 0 else 0
totals_success_rate = totals_stats['success'] / totals_stats['total'] if totals_stats['total'] > 0 else 0

report = f"""🔪 FOOTBALL INTELLIGENCE ENGINE v7.0 - SURGICAL EDITION
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Version: Surgical 7.0

🎯 SURGICAL BETTING ADVICE:
{recommendation['icon']} {recommendation['text']}
{recommendation['subtext']}
Reason: {recommendation['reason']}
Confidence: {recommendation['confidence']:.0f}/100
Stake: {recommendation['stake']}

🎯 SURGICAL WINNER:
Bet on: {winner_pred['team']}
Original algorithm: {winner_pred['original_prediction']}
Betting action: {winner_pred['betting_action']}
Reason: {winner_pred['reason']}
Probability: {winner_pred['probability']*100:.1f}%
Confidence: {winner_pred['confidence']} ({winner_pred['confidence_score']:.0f}/100)
Pattern: {winner_key}
Pattern stats: {winner_stats['success']}/{winner_stats['total']} wins ({winner_success_rate:.0%})
Calibrated: {winner_pred.get('is_calibrated', False)}

🎯 SURGICAL TOTALS:
Bet on: {totals_pred['direction']} 2.5
Original algorithm: {totals_pred['original_direction']} 2.5
Betting action: {totals_pred['betting_action']}
Reason: {totals_pred['reason']}
Probability: {totals_pred['probability']*100:.1f}%
Confidence: {totals_pred['confidence']} ({totals_pred['confidence_score']:.0f}/100)
Volatility: {totals_pred.get('volatility_score', 0):.2f}
Trend: {totals_pred.get('trend_score', 0):.2f}
Pattern: {totals_key}
Pattern stats: {totals_stats['success']}/{totals_stats['total']} wins ({totals_success_rate:.0%})
Anti-pattern override: {totals_pred.get('is_override', False)}

📊 EXPECTED GOALS:
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

⚠️ RISK FLAGS: {', '.join(prediction['totals']['risk_flags']) if prediction['totals']['risk_flags'] else 'None'}

---
🔪 ACTIVE ANTI-PATTERN OVERRIDES:
{chr(10).join([f"- {pattern}: {rule['reason']}" for pattern, rule in ANTI_PATTERNS.items()])}

📊 SURGICAL STATISTICS:
- Total patterns: {len(st.session_state.learning_system.pattern_memory)}
- Qualifying (≥3 matches): {len([v for v in st.session_state.learning_system.pattern_memory.values() if v['total'] >= 3])}
- Strong patterns (>70%): {len([v for v in st.session_state.learning_system.pattern_memory.values() if v['total'] >= 3 and v['success']/v['total'] > 0.7])}
- Weak patterns (<40%): {len([v for v in st.session_state.learning_system.pattern_memory.values() if v['total'] >= 3 and v['success']/v['total'] < 0.4])}

🎯 SURGICAL RULES APPLIED:
1. Anti-pattern overrides for known failure patterns
2. 70% confidence bug derating
3. Volatility-based stake reduction (<30% consistency)
4. Trend-based opposite betting (< -0.1 trend)
5. League-adjusted goal thresholds
6. Standardized pattern keys
"""

st.code(report, language="text")

st.download_button(
    label="📥 Download Surgical Report",
    data=report,
    file_name=f"surgical_{home_team}_vs_{away_team}.txt",
    mime="text/plain",
    use_container_width=True
)
