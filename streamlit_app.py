import streamlit as st
import pandas as pd
import numpy as np
from math import log
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        font-weight: bold;
    }
    .prediction-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .expected-score {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        height: 100%;
    }
    .value-bet-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .probability-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.3rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .team-analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 0.5rem 0;
        height: 100%;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .confidence-high {
        color: #00a86b;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff4444;
        font-weight: bold;
    }
    .insight-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedPredictionEngine:
    def __init__(self):
        self.league_avg_xg = 1.35
        self.league_avg_xga = 1.35
        
        # Comprehensive team database with all major leagues
        self.team_database = {
            # Premier League Teams
            "Arsenal": {"league": "EPL", "xg": 2.10, "xga": 0.95, "possession": 58, "tactical_style": "HIGH_PRESS"},
            "Manchester City": {"league": "EPL", "xg": 2.35, "xga": 0.85, "possession": 65, "tactical_style": "POSSESSION"},
            "Liverpool": {"league": "EPL", "xg": 2.25, "xga": 1.05, "possession": 62, "tactical_style": "GEGENPRESS"},
            "Aston Villa": {"league": "EPL", "xg": 1.85, "xga": 1.25, "possession": 55, "tactical_style": "HIGH_LINE"},
            "Tottenham": {"league": "EPL", "xg": 1.95, "xga": 1.45, "possession": 58, "tactical_style": "HIGH_PRESS"},
            "Newcastle": {"league": "EPL", "xg": 1.75, "xga": 1.35, "possession": 52, "tactical_style": "COUNTER_ATTACK"},
            "Brighton": {"league": "EPL", "xg": 1.90, "xga": 1.55, "possession": 60, "tactical_style": "POSSESSION"},
            "Manchester United": {"league": "EPL", "xg": 1.65, "xga": 1.40, "possession": 54, "tactical_style": "TRANSITION"},
            "West Ham": {"league": "EPL", "xg": 1.55, "xga": 1.65, "possession": 48, "tactical_style": "COUNTER_ATTACK"},
            "Chelsea": {"league": "EPL", "xg": 1.70, "xga": 1.50, "possession": 59, "tactical_style": "POSSESSION"},
            "Bournemouth": {"league": "EPL", "xg": 1.45, "xga": 1.70, "possession": 46, "tactical_style": "COUNTER_ATTACK"},
            "Crystal Palace": {"league": "EPL", "xg": 1.35, "xga": 1.60, "possession": 49, "tactical_style": "DEFENSIVE"},
            "Fulham": {"league": "EPL", "xg": 1.50, "xga": 1.55, "possession": 51, "tactical_style": "BALANCED"},
            "Wolves": {"league": "EPL", "xg": 1.40, "xga": 1.65, "possession": 47, "tactical_style": "COUNTER_ATTACK"},
            "Everton": {"league": "EPL", "xg": 1.30, "xga": 1.45, "possession": 45, "tactical_style": "DEFENSIVE"},
            "Brentford": {"league": "EPL", "xg": 1.55, "xga": 1.75, "possession": 50, "tactical_style": "HIGH_PRESS"},
            "Nottingham Forest": {"league": "EPL", "xg": 1.25, "xga": 1.80, "possession": 44, "tactical_style": "DEFENSIVE"},
            "Luton": {"league": "EPL", "xg": 1.20, "xga": 2.00, "possession": 42, "tactical_style": "DEFENSIVE"},
            "Burnley": {"league": "EPL", "xg": 1.15, "xga": 2.10, "possession": 55, "tactical_style": "POSSESSION"},
            "Sheffield United": {"league": "EPL", "xg": 1.05, "xga": 2.30, "possession": 40, "tactical_style": "DEFENSIVE"},
            
            # La Liga Teams
            "Real Madrid": {"league": "La Liga", "xg": 2.20, "xga": 0.80, "possession": 60, "tactical_style": "COUNTER_ATTACK"},
            "Barcelona": {"league": "La Liga", "xg": 2.15, "xga": 0.90, "possession": 68, "tactical_style": "POSSESSION"},
            "Atletico Madrid": {"league": "La Liga", "xg": 1.80, "xga": 1.10, "possession": 48, "tactical_style": "DEFENSIVE"},
            "Girona": {"league": "La Liga", "xg": 1.75, "xga": 1.30, "possession": 54, "tactical_style": "HIGH_PRESS"},
            "Athletic Bilbao": {"league": "La Liga", "xg": 1.70, "xga": 1.20, "possession": 52, "tactical_style": "HIGH_PRESS"},
            "Real Sociedad": {"league": "La Liga", "xg": 1.65, "xga": 1.25, "possession": 56, "tactical_style": "POSSESSION"},
            "Real Betis": {"league": "La Liga", "xg": 1.60, "xga": 1.40, "possession": 55, "tactical_style": "POSSESSION"},
            "Valencia": {"league": "La Liga", "xg": 1.45, "xga": 1.35, "possession": 49, "tactical_style": "DEFENSIVE"},
            "Getafe": {"league": "La Liga", "xg": 1.30, "xga": 1.50, "possession": 42, "tactical_style": "DEFENSIVE"},
            "Las Palmas": {"league": "La Liga", "xg": 1.25, "xga": 1.45, "possession": 58, "tactical_style": "POSSESSION"},
            "Osasuna": {"league": "La Liga", "xg": 1.35, "xga": 1.55, "possession": 46, "tactical_style": "DEFENSIVE"},
            "Villarreal": {"league": "La Liga", "xg": 1.55, "xga": 1.65, "possession": 53, "tactical_style": "HIGH_PRESS"},
            "Alaves": {"league": "La Liga", "xg": 1.20, "xga": 1.60, "possession": 44, "tactical_style": "DEFENSIVE"},
            "Sevilla": {"league": "La Liga", "xg": 1.50, "xga": 1.70, "possession": 54, "tactical_style": "POSSESSION"},
            "Mallorca": {"league": "La Liga", "xg": 1.15, "xga": 1.55, "possession": 45, "tactical_style": "DEFENSIVE"},
            "Rayo Vallecano": {"league": "La Liga", "xg": 1.25, "xga": 1.65, "possession": 47, "tactical_style": "HIGH_PRESS"},
            "Celta Vigo": {"league": "La Liga", "xg": 1.40, "xga": 1.75, "possession": 52, "tactical_style": "POSSESSION"},
            "Cadiz": {"league": "La Liga", "xg": 1.10, "xga": 1.80, "possession": 40, "tactical_style": "DEFENSIVE"},
            "Granada": {"league": "La Liga", "xg": 1.05, "xga": 2.10, "possession": 48, "tactical_style": "DEFENSIVE"},
            "Almeria": {"league": "La Liga", "xg": 1.00, "xga": 2.20, "possession": 50, "tactical_style": "POSSESSION"},
        }
        
        # Enhanced tactical style effects
        self.tactical_effects = {
            ('LOW_BLOCK', 'HIGH_PRESS'): {'home_xg_mod': +0.05, 'away_xg_mod': -0.10, 'explanation': "Home low block reduces away attacking efficiency"},
            ('DEFENSIVE', 'POSSESSION'): {'home_xg_mod': -0.05, 'away_xg_mod': +0.05, 'explanation': "Away possession dominance increases their attacking chances"},
            ('COUNTER_ATTACK', 'HIGH_LINE'): {'home_xg_mod': +0.15, 'away_xg_mod': 0, 'explanation': "Home counter attack perfectly suits high defensive line"},
            ('GEGENPRESS', 'POSSESSION'): {'home_xg_mod': +0.08, 'away_xg_mod': -0.05, 'explanation': "High press disrupts possession-based build-up"},
            ('HIGH_PRESS', 'DEFENSIVE'): {'home_xg_mod': +0.10, 'away_xg_mod': -0.08, 'explanation': "High press forces errors from defensive team"},
            ('POSSESSION', 'COUNTER_ATTACK'): {'home_xg_mod': -0.07, 'away_xg_mod': +0.12, 'explanation': "Away team can exploit spaces in possession system"},
        }
    
    def get_team_data(self, team_name):
        """Get team data from database or return default if not found"""
        return self.team_database.get(team_name, {
            "league": "EPL", "xg": 1.50, "xga": 1.50, "possession": 50, "tactical_style": "BALANCED"
        })
    
    def team_strength_snapshot(self, xg, xga):
        """Calculate team strength scores (1-10 scale)"""
        attack_strength = min(10, max(1, 5 + 5 * ((xg - self.league_avg_xg) / self.league_avg_xg)))
        defense_strength = min(10, max(1, 5 - 5 * ((xga - self.league_avg_xga) / self.league_avg_xga)))
        return round(defense_strength, 1), round(attack_strength, 1)
    
    def apply_injury_modifier(self, team_xg, team_xga, injury_tier):
        """Apply injury modifiers with position-weighted impact"""
        injury_weights = {
            0: 0.00,   # None
            1: -0.05,  # Minor
            2: -0.10,  # Moderate
            3: -0.20,  # Significant
            4: -0.35   # Crisis
        }
        modifier = 1 + injury_weights[injury_tier]
        
        adjusted_xg = team_xg * modifier
        adjusted_xga = team_xga / modifier
        
        return adjusted_xg, adjusted_xga
    
    def calculate_confidence_score(self, probabilities):
        """Calculate confidence score (0-100) based on prediction entropy"""
        probs = np.array(list(probabilities.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(probabilities))
        normalized_conf = 1 - (entropy / max_entropy)
        return round(normalized_conf * 100, 1)
    
    def get_confidence_label(self, score):
        """Get confidence label based on score"""
        if score >= 80:
            return "confidence-high", "HIGH CONFIDENCE"
        elif score >= 60:
            return "confidence-medium", "MEDIUM CONFIDENCE"
        else:
            return "confidence-low", "LOW CONFIDENCE"
    
    def apply_tactical_modifiers(self, home_style, away_style, home_xg, away_xg, home_xga, away_xga):
        """Apply tactical style modifiers to xG calculations"""
        explanations = []
        
        # Apply modifiers if style combination exists
        style_key = (home_style, away_style)
        if style_key in self.tactical_effects:
            effects = self.tactical_effects[style_key]
            home_xg *= (1 + effects['home_xg_mod'])
            away_xg *= (1 + effects['away_xg_mod'])
            explanations.append(f"Tactical: {effects['explanation']}")
        
        return home_xg, away_xg, home_xga, away_xga, explanations
    
    def calculate_value_bets(self, probabilities, odds):
        """Calculate value bets and expected value"""
        value_bets = {}
        
        # Home win value
        home_implied_prob = 1 / odds['home']
        home_ev = (probabilities['home'] * odds['home']) - 1
        home_value = probabilities['home'] / home_implied_prob
        
        # Draw value
        draw_implied_prob = 1 / odds['draw']
        draw_ev = (probabilities['draw'] * odds['draw']) - 1
        draw_value = probabilities['draw'] / draw_implied_prob
        
        # Over 2.5 value
        over_implied_prob = 1 / odds['over_2.5']
        over_ev = (probabilities['over_2.5'] * odds['over_2.5']) - 1
        over_value = probabilities['over_2.5'] / over_implied_prob
        
        value_bets['home'] = {'value': round(home_value, 2), 'ev': round(home_ev, 3)}
        value_bets['draw'] = {'value': round(draw_value, 2), 'ev': round(draw_ev, 3)}
        value_bets['over_2.5'] = {'value': round(over_value, 2), 'ev': round(over_ev, 3)}
        
        return value_bets
    
    def get_all_teams(self, league=None):
        """Get list of all teams, optionally filtered by league"""
        if league:
            return [team for team, data in self.team_database.items() if data['league'] == league]
        return list(self.team_database.keys())

def display_input_section(engine):
    """Display the main input section"""
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #1f77b4;">⚽ Match Configuration</h2>', unsafe_allow_html=True)
    
    # League selection with team filtering
    col1, col2 = st.columns(2)
    
    with col1:
        league = st.selectbox("SELECT LEAGUE", ["All Leagues", "EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"], key="league_select")
    
    # Get teams based on league selection
    if league == "All Leagues":
        available_teams = engine.get_all_teams()
    else:
        available_teams = engine.get_all_teams(league)
    
    # Team selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 HOME TEAM")
        home_team = st.selectbox("Select Home Team", available_teams, index=available_teams.index("Tottenham") if "Tottenham" in available_teams else 0, key="home_team_select")
        
        # Auto-fill home team data
        home_data = engine.get_team_data(home_team)
        
        home_xg = st.number_input("Expected Goals (xG)", value=home_data["xg"], min_value=0.0, key="home_xg")
        home_xga = st.number_input("Expected Goals Against (xGA)", value=home_data["xga"], min_value=0.0, key="home_xga")
        home_possession = st.slider("Average Possession %", 0, 100, home_data["possession"], key="home_possession")
        home_tactical = st.selectbox("Home Tactical Style", 
                                   ["DEFENSIVE", "LOW_BLOCK", "COUNTER_ATTACK", "POSSESSION", "HIGH_PRESS", "GEGENPRESS", "HIGH_LINE", "BALANCED", "TRANSITION"],
                                   index=["DEFENSIVE", "LOW_BLOCK", "COUNTER_ATTACK", "POSSESSION", "HIGH_PRESS", "GEGENPRESS", "HIGH_LINE", "BALANCED", "TRANSITION"].index(home_data["tactical_style"]) if home_data["tactical_style"] in ["DEFENSIVE", "LOW_BLOCK", "COUNTER_ATTACK", "POSSESSION", "HIGH_PRESS", "GEGENPRESS", "HIGH_LINE", "BALANCED", "TRANSITION"] else 0, key="home_tactical")
    
    with col2:
        st.subheader("✈️ AWAY TEAM")
        away_team = st.selectbox("Select Away Team", available_teams, index=available_teams.index("Manchester United") if "Manchester United" in available_teams else 1, key="away_team_select")
        
        # Auto-fill away team data
        away_data = engine.get_team_data(away_team)
        
        away_xg = st.number_input("Away Expected Goals (xG)", value=away_data["xg"], min_value=0.0, key="away_xg")
        away_xga = st.number_input("Away Expected Goals Against (xGA)", value=away_data["xga"], min_value=0.0, key="away_xga")
        away_possession = st.slider("Away Average Possession %", 0, 100, away_data["possession"], key="away_possession")
        away_tactical = st.selectbox("Away Tactical Style",
                                   ["DEFENSIVE", "LOW_BLOCK", "COUNTER_ATTACK", "POSSESSION", "HIGH_PRESS", "GEGENPRESS", "HIGH_LINE", "BALANCED", "TRANSITION"],
                                   index=["DEFENSIVE", "LOW_BLOCK", "COUNTER_ATTACK", "POSSESSION", "HIGH_PRESS", "GEGENPRESS", "HIGH_LINE", "BALANCED", "TRANSITION"].index(away_data["tactical_style"]) if away_data["tactical_style"] in ["DEFENSIVE", "LOW_BLOCK", "COUNTER_ATTACK", "POSSESSION", "HIGH_PRESS", "GEGENPRESS", "HIGH_LINE", "BALANCED", "TRANSITION"] else 4, key="away_tactical")
    
    # Match Context Section
    st.markdown("---")
    st.subheader("🎭 Match Context")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        match_importance = st.selectbox("Match Importance", ["Normal", "High", "Critical", "Cup Final", "Relegation Battle", "Title Decider"], key="importance")
    with col2:
        crowd_impact = st.selectbox("Home Crowd Impact", ["Normal", "Electric", "Hostile", "Quiet", "Volatile"], key="crowd")
    with col3:
        referee_style = st.selectbox("Referee Style", ["Lenient", "Normal", "Strict", "Card Happy", "VAR Heavy"], key="referee")
    with col4:
        weather_conditions = st.selectbox("Weather Conditions", ["Normal", "Rainy", "Windy", "Hot", "Cold", "Poor Pitch"], key="weather")
    
    # Injury input with tiered system
    st.subheader("🩹 Injury Status")
    col1, col2 = st.columns(2)
    with col1:
        home_injuries = st.selectbox("Home Key Injuries", 
                                   ["None", "Minor (1-2 rotational)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ starters)"],
                                   key="home_injuries")
    with col2:
        away_injuries = st.selectbox("Away Key Injuries",
                                   ["None", "Minor (1-2 rotational)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ starters)"],
                                   key="away_injuries")
    
    # Market Odds
    st.markdown("---")
    st.subheader("💰 Market Odds")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        home_odds = st.number_input("Home Win Odds", value=7.50, min_value=1.01, key="home_odds")
    with col2:
        draw_odds = st.number_input("Draw Odds", value=5.00, min_value=1.01, key="draw_odds")
    with col3:
        away_odds = st.number_input("Away Win Odds", value=1.38, min_value=1.01, key="away_odds")
    with col4:
        over_odds = st.number_input("Over 2.5 Goals Odds", value=2.00, min_value=1.01, key="over_odds")
    
    # Generate Prediction Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_prediction = st.button("🚀 GENERATE PREDICTION", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'generate_prediction': generate_prediction,
        'home_team': home_team,
        'away_team': away_team,
        'home_data': home_data,
        'away_data': away_data,
        'home_xg': home_xg,
        'home_xga': home_xga,
        'away_xg': away_xg,
        'away_xga': away_xga,
        'home_tactical': home_tactical,
        'away_tactical': away_tactical,
        'home_injuries': home_injuries,
        'away_injuries': away_injuries,
        'home_odds': home_odds,
        'draw_odds': draw_odds,
        'away_odds': away_odds,
        'over_odds': over_odds,
        'match_importance': match_importance,
        'crowd_impact': crowd_impact,
        'referee_style': referee_style,
        'weather_conditions': weather_conditions
    }

def display_prediction_section(engine, input_data):
    """Display the prediction results section"""
    
    # Convert injury tiers to numerical values
    injury_tier_map = {
        "None": 0, "Minor (1-2 rotational)": 1, "Moderate (1-2 key starters)": 2,
        "Significant (3-4 key players)": 3, "Crisis (5+ starters)": 4
    }
    
    home_injury_tier = injury_tier_map[input_data['home_injuries']]
    away_injury_tier = injury_tier_map[input_data['away_injuries']]
    
    # Apply injury modifiers
    home_xg_adj, home_xga_adj = engine.apply_injury_modifier(input_data['home_xg'], input_data['home_xga'], home_injury_tier)
    away_xg_adj, away_xga_adj = engine.apply_injury_modifier(input_data['away_xg'], input_data['away_xga'], away_injury_tier)
    
    # Apply tactical modifiers
    home_xg_final, away_xg_final, home_xga_final, away_xga_final, tactical_explanations = engine.apply_tactical_modifiers(
        input_data['home_tactical'], input_data['away_tactical'], home_xg_adj, away_xg_adj, home_xga_adj, away_xga_adj
    )
    
    # Apply normalization and clamping (CRITICAL FIX)
    MIN_XG = 0.15
    MAX_XG = 3.0
    MAX_TOTAL_XG = 6.0
    
    def clamp_xg(x):
        return max(MIN_XG, min(MAX_XG, x))
    
    home_xg_final = clamp_xg(home_xg_final)
    away_xg_final = clamp_xg(away_xg_final)
    
    # Apply total xG cap
    total_xg = home_xg_final + away_xg_final
    if total_xg > MAX_TOTAL_XG:
        damping = MAX_TOTAL_XG / total_xg
        home_xg_final *= damping
        away_xg_final *= damping
    
    # Calculate probabilities using normalized xG
    home_advantage = 1.1  # Home advantage factor
    home_win_prob = (home_xg_final / (home_xg_final + away_xg_final)) * 45 * home_advantage
    away_win_prob = (away_xg_final / (home_xg_final + away_xg_final)) * 45
    draw_prob = 100 - home_win_prob - away_win_prob
    
    # Normalize probabilities
    total = home_win_prob + draw_prob + away_win_prob
    home_win_prob = round((home_win_prob / total) * 100, 1)
    draw_prob = round((draw_prob / total) * 100, 1)
    away_win_prob = round((away_win_prob / total) * 100, 1)
    
    # Over/Under and BTTS probabilities
    over_25_prob = round(min(95, max(25, (total_xg / 2.5) * 65)), 1)
    under_25_prob = round(100 - over_25_prob, 1)
    
    btts_yes_prob = round(min(90, max(25, ((home_xg_final * 0.7 + away_xg_final * 0.7) / 2) * 85)), 1)
    btts_no_prob = round(100 - btts_yes_prob, 1)
    
    # Expected Score
    expected_home_goals = round(home_xg_final, 1)
    expected_away_goals = round(away_xg_final, 1)
    
    # HEADER SECTION
    st.markdown('<div class="prediction-header">', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; color: white; margin-bottom: 1rem;">🎯 PREDICTION RESULTS</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="text-align: center; color: white; margin-bottom: 2rem;">{input_data["home_team"]} vs {input_data["away_team"]}</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="expected-score">{expected_home_goals} - {expected_away_goals}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: white; font-size: 1.2rem;">Expected Final Score</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # QUICK STATS ROW
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Home Win", f"{home_win_prob}%")
    with col2:
        st.metric("🤝 Draw", f"{draw_prob}%")
    with col3:
        st.metric("✈️ Away Win", f"{away_win_prob}%")
    with col4:
        st.metric("⚽ Over 2.5", f"{over_25_prob}%")
    
    # MAIN PREDICTION CARDS
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Detailed Predictions</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("🏆 Match Outcome")
        st.markdown(f'<div style="text-align: center; margin: 1rem 0;">')
        st.markdown(f'<span class="probability-badge" style="background-color: #1f77b4;">{input_data["home_team"]}: {home_win_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge" style="background-color: #ff7f0e;">Draw: {draw_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge" style="background-color: #d62728;">{input_data["away_team"]}: {away_win_prob}%</span>', unsafe_allow_html=True)
        st.markdown('</div>')
        
        # Outcome pie chart
        fig_outcome = go.Figure(data=[go.Pie(
            labels=[f'{input_data["home_team"]}', 'Draw', f'{input_data["away_team"]}'],
            values=[home_win_prob, draw_prob, away_win_prob],
            hole=.3,
            marker_colors=['#1f77b4', '#ff7f0e', '#d62728']
        )])
        fig_outcome.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_outcome, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("⚽ Goals Market")
        st.markdown(f'<div style="text-align: center; margin: 1rem 0;">')
        st.markdown(f'<span class="probability-badge" style="background-color: #2ca02c;">Over 2.5: {over_25_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge" style="background-color: #ff7f0e;">Under 2.5: {under_25_prob}%</span>', unsafe_allow_html=True)
        st.markdown('</div>')
        
        st.markdown(f'<div style="text-align: center; margin: 1rem 0;">')
        st.markdown(f'<span class="probability-badge" style="background-color: #9467bd;">BTTS Yes: {btts_yes_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge" style="background-color: #8c564b;">BTTS No: {btts_no_prob}%</span>', unsafe_allow_html=True)
        st.markdown('</div>')
        
        # Goals market chart
        fig_goals = go.Figure()
        fig_goals.add_trace(go.Bar(
            x=['Over 2.5', 'Under 2.5', 'BTTS Yes', 'BTTS No'],
            y=[over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob],
            marker_color=['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
        ))
        fig_goals.update_layout(
            height=300,
            yaxis_title='Probability (%)',
            showlegend=False
        )
        st.plotly_chart(fig_goals, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Calculate value bets
        odds_dict = {'home': input_data['home_odds'], 'draw': input_data['draw_odds'], 'over_2.5': input_data['over_odds']}
        probs_dict = {'home': home_win_prob/100, 'draw': draw_prob/100, 'over_2.5': over_25_prob/100}
        value_bets = engine.calculate_value_bets(probs_dict, odds_dict)
        
        st.markdown('<div class="value-bet-card">', unsafe_allow_html=True)
        st.subheader("💰 Value Bets")
        st.markdown('<p style="color: white; text-align: center;">Recommended bets based on model vs market odds</p>', unsafe_allow_html=True)
        
        value_found = False
        if value_bets['home']['value'] > 1.1:
            st.markdown(f'<div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">', unsafe_allow_html=True)
            st.markdown(f'<h4 style="color: white; margin: 0;">🏠 {input_data["home_team"]} Win</h4>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: white; margin: 0;">Value: {value_bets["home"]["value"]}x | EV: {value_bets["home"]["ev"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            value_found = True
        
        if value_bets['draw']['value'] > 1.1:
            st.markdown(f'<div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">', unsafe_allow_html=True)
            st.markdown(f'<h4 style="color: white; margin: 0;">🤝 Draw</h4>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: white; margin: 0;">Value: {value_bets["draw"]["value"]}x | EV: {value_bets["draw"]["ev"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            value_found = True
        
        if value_bets['over_2.5']['value'] > 1.1:
            st.markdown(f'<div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">', unsafe_allow_html=True)
            st.markdown(f'<h4 style="color: white; margin: 0;">⚽ Over 2.5 Goals</h4>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: white; margin: 0;">Value: {value_bets["over_2.5"]["value"]}x | EV: {value_bets["over_2.5"]["ev"]}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            value_found = True
        
        if not value_found:
            st.markdown(f'<div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">', unsafe_allow_html=True)
            st.markdown(f'<h4 style="color: white; margin: 0;">📊 No Strong Value</h4>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: white; margin: 0;">Market odds align closely with predictions</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TEAM ANALYSIS SECTION
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Team Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_defense, home_attack = engine.team_strength_snapshot(input_data['home_xg'], input_data['home_xga'])
        st.markdown('<div class="team-analysis-card">', unsafe_allow_html=True)
        st.subheader(f"🏠 {input_data['home_team']}")
        
        # Home team stats
        col1a, col2a = st.columns(2)
        with col1a:
            st.metric("Defensive Strength", f"{home_defense}/10")
        with col2a:
            st.metric("Attacking Strength", f"{home_attack}/10")
        
        st.write(f"**Tactical Style**: {input_data['home_tactical'].replace('_', ' ').title()}")
        st.write(f"**Injuries**: {input_data['home_injuries']}")
        st.write(f"**Base xG**: {input_data['home_xg']}")
        st.write(f"**Base xGA**: {input_data['home_xga']}")
        
        # Home team gauge
        fig_home = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = home_attack,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Attack Rating"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 4], 'color': "lightgray"},
                    {'range': [4, 7], 'color': "lightyellow"},
                    {'range': [7, 10], 'color': "lightgreen"}
                ]
            }
        ))
        fig_home.update_layout(height=250)
        st.plotly_chart(fig_home, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        away_defense, away_attack = engine.team_strength_snapshot(input_data['away_xg'], input_data['away_xga'])
        st.markdown('<div class="team-analysis-card">', unsafe_allow_html=True)
        st.subheader(f"✈️ {input_data['away_team']}")
        
        # Away team stats
        col1a, col2a = st.columns(2)
        with col1a:
            st.metric("Defensive Strength", f"{away_defense}/10")
        with col2a:
            st.metric("Attacking Strength", f"{away_attack}/10")
        
        st.write(f"**Tactical Style**: {input_data['away_tactical'].replace('_', ' ').title()}")
        st.write(f"**Injuries**: {input_data['away_injuries']}")
        st.write(f"**Base xG**: {input_data['away_xg']}")
        st.write(f"**Base xGA**: {input_data['away_xga']}")
        
        # Away team gauge
        fig_away = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = away_attack,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Attack Rating"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 4], 'color': "lightgray"},
                    {'range': [4, 7], 'color': "lightyellow"},
                    {'range': [7, 10], 'color': "lightgreen"}
                ]
            }
        ))
        fig_away.update_layout(height=250)
        st.plotly_chart(fig_away, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # MODEL INSIGHTS SECTION
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 Model Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("📈 Expected Goals Progression")
        
        progress_data = {
            'Stage': ['Base xG', 'After Injuries', 'Tactical Adjust', 'Final xG'],
            f'{input_data["home_team"]}': [input_data['home_xg'], round(home_xg_adj, 2), round(home_xg_final, 2), round(home_xg_final, 2)],
            f'{input_data["away_team"]}': [input_data['away_xg'], round(away_xg_adj, 2), round(away_xg_final, 2), round(away_xg_final, 2)]
        }
        
        df_progress = pd.DataFrame(progress_data)
        st.dataframe(df_progress, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🔑 Key Factors")
        
        factors = []
        factors.append("🏠 Home advantage: +10% boost")
        factors.append(f"📊 Total expected goals: {round(total_xg, 2)}")
        
        if home_injury_tier > 0:
            factors.append(f"🩹 {input_data['home_team']} injuries: {input_data['home_injuries']}")
        if away_injury_tier > 0:
            factors.append(f"🩹 {input_data['away_team']} injuries: {input_data['away_injuries']}")
        
        if tactical_explanations:
            for explanation in tactical_explanations:
                factors.append(f"🎯 {explanation}")
        
        for factor in factors:
            st.write(f"• {factor}")
        
        # Confidence calculation
        outcome_probs = {'home': home_win_prob/100, 'draw': draw_prob/100, 'away': away_win_prob/100}
        confidence = engine.calculate_confidence_score(outcome_probs)
        conf_class, conf_label = engine.get_confidence_label(confidence)
        
        st.markdown(f"**Model Confidence**: <span class='{conf_class}'>{confidence}/100 - {conf_label}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # NEW ANALYSIS BUTTON
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 PERFORM NEW ANALYSIS", use_container_width=True, type="primary"):
            st.session_state.show_prediction = False
            st.rerun()

def main():
    st.markdown('<h1 class="main-header">🎯 Enhanced Hybrid Precision Prediction Engine</h1>', unsafe_allow_html=True)
    
    # Initialize prediction engine
    engine = EnhancedPredictionEngine()
    
    # Initialize session state
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    
    # Main app flow
    if not st.session_state.show_prediction:
        # Show input section
        input_data = display_input_section(engine)
        
        if input_data['generate_prediction']:
            st.session_state.input_data = input_data
            st.session_state.show_prediction = True
            st.rerun()
    else:
        # Show prediction section
        if st.session_state.input_data:
            display_prediction_section(engine, st.session_state.input_data)
        else:
            st.error("No input data available. Please configure the match first.")
            if st.button("← Back to Input"):
                st.session_state.show_prediction = False
                st.rerun()

if __name__ == "__main__":
    main()
