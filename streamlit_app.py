import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Enhanced Hybrid Precision Prediction Engine",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .expected-score {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        color: white;
    }
    .probability-badge {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        font-weight: bold;
        margin: 0.3rem;
        display: inline-block;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    .team-analysis-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .confidence-high {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: black;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .confidence-low {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        margin-bottom: 2rem;
    }
    .prediction-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedPredictionEngine:
    def __init__(self):
        self.league_avg_xg = 1.35
        self.league_avg_xga = 1.35
        
        # Comprehensive team database with enhanced metrics
        self.team_database = {
            # Premier League Teams with enhanced data
            "Arsenal": {
                "league": "EPL", "xg": 2.10, "xga": 0.95, "possession": 58, 
                "tactical_style": "HIGH_PRESS", "formation": "4-3-3",
                "conversion_rate": 0.12, "fatigue_days": 7,
                "recent_form": [2.1, 1.8, 2.3, 1.9, 2.0]  # Last 5 matches xG
            },
            "Manchester City": {
                "league": "EPL", "xg": 2.35, "xga": 0.85, "possession": 65, 
                "tactical_style": "POSSESSION", "formation": "4-3-3",
                "conversion_rate": 0.14, "fatigue_days": 6,
                "recent_form": [2.4, 2.1, 2.6, 2.3, 2.2]
            },
            "Liverpool": {
                "league": "EPL", "xg": 2.25, "xga": 1.05, "possession": 62, 
                "tactical_style": "GEGENPRESS", "formation": "4-3-3",
                "conversion_rate": 0.13, "fatigue_days": 8,
                "recent_form": [2.2, 2.4, 1.9, 2.1, 2.3]
            },
            "Aston Villa": {
                "league": "EPL", "xg": 1.85, "xga": 1.25, "possession": 55, 
                "tactical_style": "HIGH_LINE", "formation": "4-4-2",
                "conversion_rate": 0.11, "fatigue_days": 7,
                "recent_form": [1.8, 1.9, 1.7, 2.0, 1.6]
            },
            "Tottenham": {
                "league": "EPL", "xg": 1.95, "xga": 1.45, "possession": 58, 
                "tactical_style": "HIGH_PRESS", "formation": "4-2-3-1",
                "conversion_rate": 0.10, "fatigue_days": 9,
                "recent_form": [2.0, 1.8, 2.1, 1.7, 1.9]
            },
            "Newcastle": {
                "league": "EPL", "xg": 1.75, "xga": 1.35, "possession": 52, 
                "tactical_style": "COUNTER_ATTACK", "formation": "4-3-3",
                "conversion_rate": 0.09, "fatigue_days": 6,
                "recent_form": [1.7, 1.6, 1.9, 1.8, 1.5]
            },
            "Brighton": {
                "league": "EPL", "xg": 1.90, "xga": 1.55, "possession": 60, 
                "tactical_style": "POSSESSION", "formation": "4-2-3-1",
                "conversion_rate": 0.08, "fatigue_days": 8,
                "recent_form": [1.8, 2.1, 1.7, 2.0, 1.9]
            },
            "Manchester United": {
                "league": "EPL", "xg": 1.65, "xga": 1.40, "possession": 54, 
                "tactical_style": "TRANSITION", "formation": "4-2-3-1",
                "conversion_rate": 0.11, "fatigue_days": 7,
                "recent_form": [1.6, 1.4, 1.8, 1.5, 1.7]
            },
            "West Ham": {
                "league": "EPL", "xg": 1.55, "xga": 1.65, "possession": 48, 
                "tactical_style": "COUNTER_ATTACK", "formation": "4-2-3-1",
                "conversion_rate": 0.12, "fatigue_days": 10,
                "recent_form": [1.5, 1.7, 1.4, 1.6, 1.3]
            },
            "Chelsea": {
                "league": "EPL", "xg": 1.70, "xga": 1.50, "possession": 59, 
                "tactical_style": "POSSESSION", "formation": "4-2-3-1",
                "conversion_rate": 0.10, "fatigue_days": 6,
                "recent_form": [1.6, 1.8, 1.5, 1.9, 1.7]
            },
            # Additional teams with basic enhanced data
            "Bournemouth": {"league": "EPL", "xg": 1.45, "xga": 1.70, "possession": 46, "tactical_style": "COUNTER_ATTACK", "formation": "4-4-2", "conversion_rate": 0.09, "fatigue_days": 7, "recent_form": [1.4, 1.3, 1.6, 1.2, 1.5]},
            "Crystal Palace": {"league": "EPL", "xg": 1.35, "xga": 1.60, "possession": 49, "tactical_style": "DEFENSIVE", "formation": "4-3-3", "conversion_rate": 0.08, "fatigue_days": 8, "recent_form": [1.3, 1.2, 1.5, 1.1, 1.4]},
            "Real Madrid": {"league": "La Liga", "xg": 2.20, "xga": 0.80, "possession": 60, "tactical_style": "COUNTER_ATTACK", "formation": "4-3-3", "conversion_rate": 0.15, "fatigue_days": 6, "recent_form": [2.3, 2.1, 2.4, 2.0, 2.2]},
            "Barcelona": {"league": "La Liga", "xg": 2.15, "xga": 0.90, "possession": 68, "tactical_style": "POSSESSION", "formation": "4-3-3", "conversion_rate": 0.13, "fatigue_days": 7, "recent_form": [2.2, 2.0, 2.3, 1.9, 2.1]},
            "Bayern Munich": {"league": "Bundesliga", "xg": 2.40, "xga": 0.95, "possession": 63, "tactical_style": "HIGH_PRESS", "formation": "4-2-3-1", "conversion_rate": 0.16, "fatigue_days": 5, "recent_form": [2.5, 2.3, 2.6, 2.2, 2.4]},
            "Inter Milan": {"league": "Serie A", "xg": 2.15, "xga": 0.75, "possession": 57, "tactical_style": "HIGH_PRESS", "formation": "3-5-2", "conversion_rate": 0.14, "fatigue_days": 8, "recent_form": [2.2, 2.0, 2.3, 1.9, 2.1]},
            "PSG": {"league": "Ligue 1", "xg": 2.30, "xga": 0.90, "possession": 64, "tactical_style": "POSSESSION", "formation": "4-3-3", "conversion_rate": 0.15, "fatigue_days": 6, "recent_form": [2.4, 2.2, 2.5, 2.1, 2.3]},
        }
        
        # Enhanced tactical style effects
        self.tactical_effects = {
            ('LOW_BLOCK', 'HIGH_PRESS'): {'home_xg_mod': +0.05, 'away_xg_mod': -0.10, 'explanation': "Home low block reduces away attacking efficiency"},
            ('DEFENSIVE', 'POSSESSION'): {'home_xg_mod': -0.05, 'away_xg_mod': +0.05, 'explanation': "Away possession dominance increases their attacking chances"},
            ('COUNTER_ATTACK', 'HIGH_LINE'): {'home_xg_mod': +0.15, 'away_xg_mod': 0, 'explanation': "Home counter attack perfectly suits high defensive line"},
            ('GEGENPRESS', 'POSSESSION'): {'home_xg_mod': +0.08, 'away_xg_mod': -0.05, 'explanation': "High press disrupts possession-based build-up"},
            ('HIGH_PRESS', 'DEFENSIVE'): {'home_xg_mod': +0.10, 'away_xg_mod': -0.08, 'explanation': "High press forces errors from defensive team"},
            ('POSSESSION', 'COUNTER_ATTACK'): {'home_xg_mod': -0.07, 'away_xg_mod': +0.12, 'explanation': "Away team can exploit spaces in possession system"},
            ('HIGH_PRESS', 'TRANSITION'): {'home_xg_mod': +0.10, 'away_xg_mod': -0.05, 'explanation': "Home high press disrupts away transitions"},
            ('COUNTER_ATTACK', 'HIGH_PRESS'): {'home_xg_mod': +0.15, 'away_xg_mod': 0, 'explanation': "Home counter attack perfectly suits high press"},
        }
        
        # Position weights for injury impact
        self.position_weights = {
            'GK': {'xg_impact': 0.0, 'xga_impact': 0.15},
            'CB': {'xg_impact': 0.0, 'xga_impact': 0.10},
            'FB': {'xg_impact': 0.02, 'xga_impact': 0.05},
            'CM': {'xg_impact': 0.05, 'xga_impact': 0.06},
            'AM': {'xg_impact': 0.08, 'xga_impact': 0.03},
            'FW': {'xg_impact': 0.12, 'xga_impact': 0.00}
        }
    
    def get_team_data(self, team_name):
        return self.team_database.get(team_name, {
            "league": "EPL", "xg": 1.50, "xga": 1.50, "possession": 50, 
            "tactical_style": "BALANCED", "formation": "4-4-2",
            "conversion_rate": 0.10, "fatigue_days": 7,
            "recent_form": [1.5, 1.5, 1.5, 1.5, 1.5]
        })
    
    def calculate_form_trend(self, recent_form):
        """Calculate form trend from recent xG data"""
        if len(recent_form) < 2:
            return 0.0
        return np.polyfit(range(len(recent_form)), recent_form, 1)[0]  # Slope of trend line
    
    def calculate_fatigue_modifier(self, fatigue_days):
        """Calculate fatigue modifier based on days since last match"""
        if fatigue_days <= 3:
            return 0.85  # Heavy fatigue
        elif fatigue_days <= 5:
            return 0.92  # Moderate fatigue
        elif fatigue_days <= 7:
            return 0.98  # Light fatigue
        else:
            return 1.02  # Rested advantage
    
    def apply_position_weighted_injuries(self, team_xg, team_xga, injury_data):
        """Apply position-weighted injury impacts"""
        xg_modifier = 1.0
        xga_modifier = 1.0
        
        for position, count in injury_data.items():
            if position in self.position_weights:
                weight = self.position_weights[position]
                xg_modifier -= weight['xg_impact'] * count
                xga_modifier += weight['xga_impact'] * count
        
        return team_xg * max(0.5, xg_modifier), team_xga * max(1.0, xga_modifier)
    
    def team_strength_snapshot(self, xg, xga):
        attack_strength = min(10, max(1, 5 + 5 * ((xg - self.league_avg_xg) / self.league_avg_xg)))
        defense_strength = min(10, max(1, 5 - 5 * ((xga - self.league_avg_xga) / self.league_avg_xga)))
        return round(defense_strength, 1), round(attack_strength, 1)
    
    def apply_injury_modifier(self, team_xg, team_xga, injury_tier, position_data=None):
        injury_weights = {0: 0.00, 1: -0.05, 2: -0.10, 3: -0.20, 4: -0.35}
        base_modifier = 1 + injury_weights[injury_tier]
        
        # Apply position-weighted injuries if data provided
        if position_data:
            return self.apply_position_weighted_injuries(team_xg * base_modifier, team_xga / base_modifier, position_data)
        
        return team_xg * base_modifier, team_xga / base_modifier
    
    def calculate_confidence_score(self, probabilities):
        probs = np.array(list(probabilities.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(probabilities))
        normalized_conf = 1 - (entropy / max_entropy)
        return round(normalized_conf * 100, 1)
    
    def get_confidence_label(self, score):
        if score >= 80: return "confidence-high", "HIGH CONFIDENCE"
        elif score >= 60: return "confidence-medium", "MEDIUM CONFIDENCE"
        else: return "confidence-low", "LOW CONFIDENCE"
    
    def apply_tactical_modifiers(self, home_style, away_style, home_xg, away_xg):
        explanations = []
        style_key = (home_style, away_style)
        if style_key in self.tactical_effects:
            effects = self.tactical_effects[style_key]
            home_xg *= (1 + effects['home_xg_mod'])
            away_xg *= (1 + effects['away_xg_mod'])
            explanations.append(f"Tactical: {effects['explanation']}")
        return home_xg, away_xg, explanations
    
    def calculate_value_bets(self, probabilities, odds):
        value_bets = {}
        
        home_implied_prob = 1 / odds['home']
        home_ev = (probabilities['home'] * odds['home']) - 1
        home_value = probabilities['home'] / home_implied_prob
        
        draw_implied_prob = 1 / odds['draw']
        draw_ev = (probabilities['draw'] * odds['draw']) - 1
        draw_value = probabilities['draw'] / draw_implied_prob
        
        over_implied_prob = 1 / odds['over_2.5']
        over_ev = (probabilities['over_2.5'] * odds['over_2.5']) - 1
        over_value = probabilities['over_2.5'] / over_implied_prob
        
        value_bets['home'] = {'value': round(home_value, 2), 'ev': round(home_ev, 3)}
        value_bets['draw'] = {'value': round(draw_value, 2), 'ev': round(draw_ev, 3)}
        value_bets['over_2.5'] = {'value': round(over_value, 2), 'ev': round(over_ev, 3)}
        
        return value_bets
    
    def get_all_teams(self, league=None):
        if league:
            return [team for team, data in self.team_database.items() if data['league'] == league]
        return list(self.team_database.keys())

def display_input_section(engine, existing_data=None):
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #1f77b4;">⚽ Match Configuration</h2>', unsafe_allow_html=True)
    
    # League selection with team filtering
    col1, col2 = st.columns(2)
    
    with col1:
        league = st.selectbox("SELECT LEAGUE", ["All Leagues", "EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"], 
                            key="league_select")
    
    # Get teams based on league selection
    if league == "All Leagues":
        available_teams = engine.get_all_teams()
    else:
        available_teams = engine.get_all_teams(league)
    
    # Team selection - use existing data if available
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 HOME TEAM")
        
        # Set default index based on existing data
        home_default_index = available_teams.index("Tottenham") if "Tottenham" in available_teams else 0
        if existing_data and existing_data['home_team'] in available_teams:
            home_default_index = available_teams.index(existing_data['home_team'])
            
        home_team = st.selectbox("Select Home Team", available_teams, index=home_default_index, key="home_team_select")
        home_data = engine.get_team_data(home_team)
        
        # Enhanced input sections
        col1a, col1b = st.columns(2)
        with col1a:
            home_xg = st.number_input("Expected Goals (xG)", 
                                    value=existing_data['home_xg'] if existing_data else home_data["xg"], 
                                    min_value=0.0, step=0.1, key="home_xg")
        with col1b:
            home_xga = st.number_input("Expected Goals Against (xGA)", 
                                     value=existing_data['home_xga'] if existing_data else home_data["xga"], 
                                     min_value=0.0, step=0.1, key="home_xga")
        
        # NEW: Possession with numerical input
        home_possession = st.number_input("Average Possession %", 
                                        value=existing_data['home_possession'] if existing_data else home_data["possession"], 
                                        min_value=0, max_value=100, key="home_possession")
        
        # NEW: Formation selection
        formation_options = ["4-3-3", "4-2-3-1", "4-4-2", "3-5-2", "3-4-3", "4-1-4-1", "5-3-2"]
        home_formation = st.selectbox("Formation", formation_options,
                                    index=formation_options.index(existing_data['home_formation']) if existing_data and 'home_formation' in existing_data else formation_options.index(home_data["formation"]),
                                    key="home_formation")
        
        # NEW: Conversion Efficiency
        home_conversion = st.slider("Conversion Rate (Goals/xG)", 0.0, 0.2, 
                                  value=existing_data['home_conversion'] if existing_data else home_data["conversion_rate"],
                                  step=0.01, key="home_conversion")
        
        # NEW: Fatigue - Days since last match
        home_fatigue = st.slider("Days Since Last Match", 2, 14,
                               value=existing_data['home_fatigue'] if existing_data else home_data["fatigue_days"],
                               key="home_fatigue")
        
        # Tactical style
        tactical_options = ["DEFENSIVE", "LOW_BLOCK", "COUNTER_ATTACK", "POSSESSION", "HIGH_PRESS", "GEGENPRESS", "HIGH_LINE", "BALANCED", "TRANSITION"]
        home_tactical_index = tactical_options.index(home_data["tactical_style"]) if home_data["tactical_style"] in tactical_options else 0
        if existing_data:
            home_tactical_index = tactical_options.index(existing_data['home_tactical']) if existing_data['home_tactical'] in tactical_options else home_tactical_index
        home_tactical = st.selectbox("Home Tactical Style", tactical_options, index=home_tactical_index, key="home_tactical")
    
    with col2:
        st.subheader("✈️ AWAY TEAM")
        
        away_default_index = available_teams.index("Manchester United") if "Manchester United" in available_teams else 1
        if existing_data and existing_data['away_team'] in available_teams:
            away_default_index = available_teams.index(existing_data['away_team'])
            
        away_team = st.selectbox("Select Away Team", available_teams, index=away_default_index, key="away_team_select")
        away_data = engine.get_team_data(away_team)
        
        # Enhanced input sections for away team
        col2a, col2b = st.columns(2)
        with col2a:
            away_xg = st.number_input("Away Expected Goals (xG)", 
                                    value=existing_data['away_xg'] if existing_data else away_data["xg"], 
                                    min_value=0.0, step=0.1, key="away_xg")
        with col2b:
            away_xga = st.number_input("Away Expected Goals Against (xGA)", 
                                     value=existing_data['away_xga'] if existing_data else away_data["xga"], 
                                     min_value=0.0, step=0.1, key="away_xga")
        
        # NEW: Possession with numerical input
        away_possession = st.number_input("Away Average Possession %", 
                                        value=existing_data['away_possession'] if existing_data else away_data["possession"], 
                                        min_value=0, max_value=100, key="away_possession")
        
        # NEW: Formation selection
        away_formation = st.selectbox("Away Formation", formation_options,
                                    index=formation_options.index(existing_data['away_formation']) if existing_data and 'away_formation' in existing_data else formation_options.index(away_data["formation"]),
                                    key="away_formation")
        
        # NEW: Conversion Efficiency
        away_conversion = st.slider("Away Conversion Rate (Goals/xG)", 0.0, 0.2, 
                                  value=existing_data['away_conversion'] if existing_data else away_data["conversion_rate"],
                                  step=0.01, key="away_conversion")
        
        # NEW: Fatigue - Days since last match
        away_fatigue = st.slider("Away Days Since Last Match", 2, 14,
                               value=existing_data['away_fatigue'] if existing_data else away_data["fatigue_days"],
                               key="away_fatigue")
        
        away_tactical_index = tactical_options.index(away_data["tactical_style"]) if away_data["tactical_style"] in tactical_options else 4
        if existing_data:
            away_tactical_index = tactical_options.index(existing_data['away_tactical']) if existing_data['away_tactical'] in tactical_options else away_tactical_index
        away_tactical = st.selectbox("Away Tactical Style", tactical_options, index=away_tactical_index, key="away_tactical")
    
    # Enhanced Injury Section with Position-Weighted Inputs
    st.markdown("---")
    st.subheader("🩹 Enhanced Injury Analysis")
    
    col1, col2 = st.columns(2)
    
    injury_options = ["None", "Minor (1-2 rotational)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ starters)"]
    
    with col1:
        st.write("**Home Team Injuries**")
        home_injury_tier = st.selectbox("Overall Injury Impact", injury_options,
                                      index=injury_options.index(existing_data['home_injuries']) if existing_data and existing_data['home_injuries'] in injury_options else 0,
                                      key="home_injuries")
        
        # NEW: Position-specific injuries
        st.write("Position-Specific Injuries (Optional)")
        home_gk_inj = st.number_input("GK Injuries", 0, 3, 0, key="home_gk_inj")
        home_cb_inj = st.number_input("CB Injuries", 0, 3, 0, key="home_cb_inj")
        home_fb_inj = st.number_input("FB Injuries", 0, 3, 0, key="home_fb_inj")
        home_cm_inj = st.number_input("CM Injuries", 0, 3, 0, key="home_cm_inj")
        home_am_inj = st.number_input("AM Injuries", 0, 3, 0, key="home_am_inj")
        home_fw_inj = st.number_input("FW Injuries", 0, 3, 0, key="home_fw_inj")
    
    with col2:
        st.write("**Away Team Injuries**")
        away_injury_tier = st.selectbox("Away Overall Injury Impact", injury_options,
                                      index=injury_options.index(existing_data['away_injuries']) if existing_data and existing_data['away_injuries'] in injury_options else 0,
                                      key="away_injuries")
        
        # NEW: Position-specific injuries
        st.write("Position-Specific Injuries (Optional)")
        away_gk_inj = st.number_input("Away GK Injuries", 0, 3, 0, key="away_gk_inj")
        away_cb_inj = st.number_input("Away CB Injuries", 0, 3, 0, key="away_cb_inj")
        away_fb_inj = st.number_input("Away FB Injuries", 0, 3, 0, key="away_fb_inj")
        away_cm_inj = st.number_input("Away CM Injuries", 0, 3, 0, key="away_cm_inj")
        away_am_inj = st.number_input("Away AM Injuries", 0, 3, 0, key="away_am_inj")
        away_fw_inj = st.number_input("Away FW Injuries", 0, 3, 0, key="away_fw_inj")
    
    # Match Context Section
    st.markdown("---")
    st.subheader("🎭 Match Context")
    
    col1, col2, col3, col4 = st.columns(4)
    
    context_options = {
        "importance": ["Normal", "High", "Critical", "Cup Final", "Relegation Battle", "Title Decider"],
        "crowd": ["Normal", "Electric", "Hostile", "Quiet", "Volatile"],
        "referee": ["Lenient", "Normal", "Strict", "Card Happy", "VAR Heavy"],
        "weather": ["Normal", "Rainy", "Windy", "Hot", "Cold", "Poor Pitch"]
    }
    
    with col1:
        importance_index = context_options["importance"].index(existing_data['match_importance']) if existing_data and existing_data['match_importance'] in context_options["importance"] else 0
        match_importance = st.selectbox("Match Importance", context_options["importance"], 
                                      index=importance_index, key="importance")
    with col2:
        crowd_index = context_options["crowd"].index(existing_data['crowd_impact']) if existing_data and existing_data['crowd_impact'] in context_options["crowd"] else 0
        crowd_impact = st.selectbox("Home Crowd Impact", context_options["crowd"], 
                                  index=crowd_index, key="crowd")
    with col3:
        referee_index = context_options["referee"].index(existing_data['referee_style']) if existing_data and existing_data['referee_style'] in context_options["referee"] else 1
        referee_style = st.selectbox("Referee Style", context_options["referee"], 
                                   index=referee_index, key="referee")
    with col4:
        weather_index = context_options["weather"].index(existing_data['weather_conditions']) if existing_data and existing_data['weather_conditions'] in context_options["weather"] else 0
        weather_conditions = st.selectbox("Weather Conditions", context_options["weather"], 
                                        index=weather_index, key="weather")
    
    # Market Odds
    st.markdown("---")
    st.subheader("💰 Market Odds")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        home_odds = st.number_input("Home Win Odds", 
                                  value=existing_data['home_odds'] if existing_data else 7.50, 
                                  min_value=1.01, step=0.1, key="home_odds")
    with col2:
        draw_odds = st.number_input("Draw Odds", 
                                  value=existing_data['draw_odds'] if existing_data else 5.00, 
                                  min_value=1.01, step=0.1, key="draw_odds")
    with col3:
        away_odds = st.number_input("Away Win Odds", 
                                  value=existing_data['away_odds'] if existing_data else 1.38, 
                                  min_value=1.01, step=0.1, key="away_odds")
    with col4:
        over_odds = st.number_input("Over 2.5 Goals Odds", 
                                  value=existing_data['over_odds'] if existing_data else 2.00, 
                                  min_value=1.01, step=0.1, key="over_odds")
    
    # Generate Prediction Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_prediction = st.button("🚀 GENERATE PREDICTION", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Compile position data for injuries
    home_position_data = {
        'GK': home_gk_inj, 'CB': home_cb_inj, 'FB': home_fb_inj,
        'CM': home_cm_inj, 'AM': home_am_inj, 'FW': home_fw_inj
    }
    
    away_position_data = {
        'GK': away_gk_inj, 'CB': away_cb_inj, 'FB': away_fb_inj,
        'CM': away_cm_inj, 'AM': away_am_inj, 'FW': away_fw_inj
    }
    
    return {
        'generate_prediction': generate_prediction,
        'home_team': home_team, 'away_team': away_team,
        'home_data': home_data, 'away_data': away_data,
        'home_xg': home_xg, 'home_xga': home_xga,
        'away_xg': away_xg, 'away_xga': away_xga,
        'home_possession': home_possession, 'away_possession': away_possession,
        'home_formation': home_formation, 'away_formation': away_formation,
        'home_conversion': home_conversion, 'away_conversion': away_conversion,
        'home_fatigue': home_fatigue, 'away_fatigue': away_fatigue,
        'home_tactical': home_tactical, 'away_tactical': away_tactical,
        'home_injuries': home_injury_tier, 'away_injuries': away_injury_tier,
        'home_position_data': home_position_data, 'away_position_data': away_position_data,
        'home_odds': home_odds, 'draw_odds': draw_odds, 'away_odds': away_odds, 'over_odds': over_odds,
        'match_importance': match_importance, 'crowd_impact': crowd_impact,
        'referee_style': referee_style, 'weather_conditions': weather_conditions
    }

def display_prediction_section(engine, input_data):
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    
    # Convert injury tiers
    injury_tier_map = {"None": 0, "Minor (1-2 rotational)": 1, "Moderate (1-2 key starters)": 2, "Significant (3-4 key players)": 3, "Crisis (5+ starters)": 4}
    home_injury_tier = injury_tier_map[input_data['home_injuries']]
    away_injury_tier = injury_tier_map[input_data['away_injuries']]
    
    # Apply enhanced modifiers
    home_xg_adj, home_xga_adj = engine.apply_injury_modifier(
        input_data['home_xg'], input_data['home_xga'], home_injury_tier, input_data['home_position_data']
    )
    away_xg_adj, away_xga_adj = engine.apply_injury_modifier(
        input_data['away_xg'], input_data['away_xga'], away_injury_tier, input_data['away_position_data']
    )
    
    # Apply fatigue modifiers
    home_fatigue_mod = engine.calculate_fatigue_modifier(input_data['home_fatigue'])
    away_fatigue_mod = engine.calculate_fatigue_modifier(input_data['away_fatigue'])
    
    home_xg_adj *= home_fatigue_mod
    away_xg_adj *= away_fatigue_mod
    
    # Apply conversion rate adjustments
    home_conversion_boost = 1.0 + (input_data['home_conversion'] - 0.1) * 0.5  # Boost for efficient teams
    away_conversion_boost = 1.0 + (input_data['away_conversion'] - 0.1) * 0.5
    
    home_xg_adj *= home_conversion_boost
    away_xg_adj *= away_conversion_boost
    
    # Apply tactical modifiers
    home_xg_final, away_xg_final, tactical_explanations = engine.apply_tactical_modifiers(
        input_data['home_tactical'], input_data['away_tactical'], home_xg_adj, away_xg_adj
    )
    
    # Normalize and clamp
    MIN_XG, MAX_XG, MAX_TOTAL_XG = 0.15, 3.0, 6.0
    home_xg_final = max(MIN_XG, min(MAX_XG, home_xg_final))
    away_xg_final = max(MIN_XG, min(MAX_XG, away_xg_final))
    
    total_xg = home_xg_final + away_xg_final
    if total_xg > MAX_TOTAL_XG:
        damping = MAX_TOTAL_XG / total_xg
        home_xg_final *= damping
        away_xg_final *= damping
    
    # Calculate probabilities using normalized xG
    home_advantage = 1.1
    home_win_prob = (home_xg_final / (home_xg_final + away_xg_final)) * 45 * home_advantage
    away_win_prob = (away_xg_final / (home_xg_final + away_xg_final)) * 45
    draw_prob = 100 - home_win_prob - away_win_prob
    
    total = home_win_prob + draw_prob + away_win_prob
    home_win_prob = round((home_win_prob / total) * 100, 1)
    draw_prob = round((draw_prob / total) * 100, 1)
    away_win_prob = round((away_win_prob / total) * 100, 1)
    
    # Header
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; color: white; margin-bottom: 1rem;">🎯 PREDICTION RESULTS</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="text-align: center; color: white; margin-bottom: 2rem;">{input_data["home_team"]} vs {input_data["away_team"]}</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="expected-score">{round(home_xg_final, 1)} - {round(away_xg_final, 1)}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: white; font-size: 1.2rem;">Expected Final Score</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🏠 Home Win", f"{home_win_prob}%")
    with col2: st.metric("🤝 Draw", f"{draw_prob}%")
    with col3: st.metric("✈️ Away Win", f"{away_win_prob}%")
    with col4: 
        over_25_prob = round(min(95, max(25, (total_xg / 2.5) * 65)), 1)
        st.metric("⚽ Over 2.5", f"{over_25_prob}%")
    
    # Detailed Predictions
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Detailed Predictions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="team-analysis-card">', unsafe_allow_html=True)
        st.subheader("🏆 Match Outcome")
        st.markdown(f'<div style="text-align: center;">', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">{input_data["home_team"]}: {home_win_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">Draw: {draw_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">{input_data["away_team"]}: {away_win_prob}%</span>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="team-analysis-card">', unsafe_allow_html=True)
        st.subheader("⚽ Goals Market")
        under_25_prob = round(100 - over_25_prob, 1)
        btts_yes_prob = round(min(90, max(25, ((home_xg_final * 0.7 + away_xg_final * 0.7) / 2) * 85)), 1)
        btts_no_prob = round(100 - btts_yes_prob, 1)
        
        st.markdown(f'<div style="text-align: center;">', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">Over 2.5: {over_25_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">Under 2.5: {under_25_prob}%</span>', unsafe_allow_html=True)
        st.markdown('</div>')
        st.markdown(f'<div style="text-align: center; margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">BTTS Yes: {btts_yes_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">BTTS No: {btts_no_prob}%</span>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="value-bet-card">', unsafe_allow_html=True)
        st.subheader("💰 Value Bets")
        st.write("Recommended bets based on model vs market odds")
        st.markdown("---")
        
        odds_dict = {'home': input_data['home_odds'], 'draw': input_data['draw_odds'], 'over_2.5': input_data['over_odds']}
        probs_dict = {'home': home_win_prob/100, 'draw': draw_prob/100, 'over_2.5': over_25_prob/100}
        value_bets = engine.calculate_value_bets(probs_dict, odds_dict)
        
        if value_bets['home']['value'] > 1.1:
            st.markdown(f'**🏠 {input_data["home_team"]} Win**')
            st.markdown(f'Value: {value_bets["home"]["value"]}x | EV: {value_bets["home"]["ev"]}')
            st.markdown("---")
        if value_bets['draw']['value'] > 1.1:
            st.markdown(f'**🤝 Draw**')
            st.markdown(f'Value: {value_bets["draw"]["value"]}x | EV: {value_bets["draw"]["ev"]}')
            st.markdown("---")
        if value_bets['over_2.5']['value'] > 1.1:
            st.markdown(f'**⚽ Over 2.5 Goals**')
            st.markdown(f'Value: {value_bets["over_2.5"]["value"]}x | EV: {value_bets["over_2.5"]["ev"]}')
            st.markdown("---")
        if all(value['value'] <= 1.1 for value in value_bets.values()):
            st.markdown("**No strong value bets identified**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Team Analysis
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Enhanced Team Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        home_defense, home_attack = engine.team_strength_snapshot(input_data['home_xg'], input_data['home_xga'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(f"🏠 {input_data['home_team']}")
        
        col1a, col1b = st.columns(2)
        with col1a: 
            st.metric("Defensive Strength", f"{home_defense}/10")
            st.metric("Formation", input_data['home_formation'])
        with col1b: 
            st.metric("Attacking Strength", f"{home_attack}/10")
            st.metric("Conversion Rate", f"{input_data['home_conversion']:.1%}")
        
        st.write(f"**Style**: {input_data['home_tactical'].replace('_', ' ').title()}")
        st.write(f"**Injuries**: {input_data['home_injuries']}")
        st.write(f"**Fatigue**: {input_data['home_fatigue']} days rest")
        st.write(f"**Base xG**: {input_data['home_xg']} | **Base xGA**: {input_data['home_xga']}")
        
        # Fatigue indicator
        if input_data['home_fatigue'] <= 4:
            st.warning("⚠️ Heavy fatigue - performance may be impacted")
        elif input_data['home_fatigue'] >= 10:
            st.success("✅ Well rested - potential performance boost")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        away_defense, away_attack = engine.team_strength_snapshot(input_data['away_xg'], input_data['away_xga'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(f"✈️ {input_data['away_team']}")
        
        col2a, col2b = st.columns(2)
        with col2a: 
            st.metric("Defensive Strength", f"{away_defense}/10")
            st.metric("Formation", input_data['away_formation'])
        with col2b: 
            st.metric("Attacking Strength", f"{away_attack}/10")
            st.metric("Conversion Rate", f"{input_data['away_conversion']:.1%}")
        
        st.write(f"**Style**: {input_data['away_tactical'].replace('_', ' ').title()}")
        st.write(f"**Injuries**: {input_data['away_injuries']}")
        st.write(f"**Fatigue**: {input_data['away_fatigue']} days rest")
        st.write(f"**Base xG**: {input_data['away_xg']} | **Base xGA**: {input_data['away_xga']}")
        
        # Fatigue indicator
        if input_data['away_fatigue'] <= 4:
            st.warning("⚠️ Heavy fatigue - performance may be impacted")
        elif input_data['away_fatigue'] >= 10:
            st.success("✅ Well rested - potential performance boost")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Model Insights
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 Enhanced Model Insights</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.subheader("📈 Expected Goals Progression")
        progress_data = {
            'Stage': ['Base xG', 'Injuries', 'Fatigue', 'Conversion', 'Tactical', 'Final xG'],
            f'{input_data["home_team"]}': [
                input_data['home_xg'], 
                round(home_xg_adj, 2),
                round(home_xg_adj * home_fatigue_mod, 2),
                round(home_xg_adj * home_fatigue_mod * home_conversion_boost, 2),
                round(home_xg_final, 2),
                round(home_xg_final, 2)
            ],
            f'{input_data["away_team"]}': [
                input_data['away_xg'], 
                round(away_xg_adj, 2),
                round(away_xg_adj * away_fatigue_mod, 2),
                round(away_xg_adj * away_fatigue_mod * away_conversion_boost, 2),
                round(away_xg_final, 2),
                round(away_xg_final, 2)
            ]
        }
        st.dataframe(pd.DataFrame(progress_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.subheader("🔑 Key Factors & Adjustments")
        factors = [
            f"🏠 Home advantage: +10% boost",
            f"📊 Total expected goals: {round(total_xg, 2)}",
            f"⚡ Home fatigue: {home_fatigue_mod:.1%} modifier",
            f"⚡ Away fatigue: {away_fatigue_mod:.1%} modifier",
            f"🎯 Home conversion: {home_conversion_boost:.1%} boost",
            f"🎯 Away conversion: {away_conversion_boost:.1%} boost"
        ]
        
        if home_injury_tier > 0: 
            factors.append(f"🩹 {input_data['home_team']} injuries: {input_data['home_injuries']}")
        if away_injury_tier > 0: 
            factors.append(f"🩹 {input_data['away_team']} injuries: {input_data['away_injuries']}")
        if tactical_explanations: 
            factors.extend([f"🎯 {exp}" for exp in tactical_explanations])
        
        for factor in factors: st.write(f"• {factor}")
        
        # Enhanced confidence score
        outcome_probs = {'home': home_win_prob/100, 'draw': draw_prob/100, 'away': away_win_prob/100}
        confidence_score = engine.calculate_confidence_score(outcome_probs)
        conf_class, conf_label = engine.get_confidence_label(confidence_score)
        
        st.markdown("---")
        st.markdown(f'**Model Confidence**: <span class="{conf_class}">{confidence_score}/100 - {conf_label}</span>', unsafe_allow_html=True)
        
        # Data quality indicator
        data_quality = "HIGH" if confidence_score > 70 else "MEDIUM" if confidence_score > 50 else "LOW"
        st.markdown(f'**Data Quality**: {data_quality}')
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Action Buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.show_prediction = False
            st.session_state.existing_data = None
            st.rerun()
    with col2:
        if st.button("✏️ Refine Input", use_container_width=True, type="primary"):
            st.session_state.show_prediction = False
            st.session_state.existing_data = input_data
            st.rerun()
    with col3:
        if st.button("📊 Advanced Stats", use_container_width=True):
            st.info("Advanced statistics feature coming soon!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎯 Enhanced Hybrid Precision Prediction Engine</h1>', unsafe_allow_html=True)
    
    engine = EnhancedPredictionEngine()
    
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    if 'existing_data' not in st.session_state:
        st.session_state.existing_data = None
    
    if not st.session_state.show_prediction:
        input_data = display_input_section(engine, st.session_state.existing_data)
        if input_data['generate_prediction']:
            st.session_state.input_data = input_data
            st.session_state.show_prediction = True
            st.rerun()
    else:
        if st.session_state.input_data:
            display_prediction_section(engine, st.session_state.input_data)
        else:
            st.error("No input data available. Please configure the match first.")
            if st.button("← Back to Input"):
                st.session_state.show_prediction = False
                st.rerun()

if __name__ == "__main__":
    main()
