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
            
            # Bundesliga Teams
            "Bayer Leverkusen": {"league": "Bundesliga", "xg": 2.25, "xga": 0.85, "possession": 61, "tactical_style": "POSSESSION"},
            "Bayern Munich": {"league": "Bundesliga", "xg": 2.40, "xga": 0.95, "possession": 63, "tactical_style": "HIGH_PRESS"},
            "Stuttgart": {"league": "Bundesliga", "xg": 2.00, "xga": 1.15, "possession": 55, "tactical_style": "HIGH_PRESS"},
            "RB Leipzig": {"league": "Bundesliga", "xg": 2.10, "xga": 1.25, "possession": 58, "tactical_style": "GEGENPRESS"},
            "Borussia Dortmund": {"league": "Bundesliga", "xg": 2.05, "xga": 1.30, "possession": 59, "tactical_style": "HIGH_PRESS"},
            "Eintracht Frankfurt": {"league": "Bundesliga", "xg": 1.70, "xga": 1.40, "possession": 51, "tactical_style": "COUNTER_ATTACK"},
            "Freiburg": {"league": "Bundesliga", "xg": 1.65, "xga": 1.45, "possession": 50, "tactical_style": "BALANCED"},
            "Hoffenheim": {"league": "Bundesliga", "xg": 1.75, "xga": 1.60, "possession": 53, "tactical_style": "HIGH_PRESS"},
            "Heidenheim": {"league": "Bundesliga", "xg": 1.40, "xga": 1.55, "possession": 46, "tactical_style": "DEFENSIVE"},
            "Wolfsburg": {"league": "Bundesliga", "xg": 1.55, "xga": 1.65, "possession": 52, "tactical_style": "TRANSITION"},
            "Augsburg": {"league": "Bundesliga", "xg": 1.50, "xga": 1.70, "possession": 47, "tactical_style": "COUNTER_ATTACK"},
            "Borussia M'gladbach": {"league": "Bundesliga", "xg": 1.60, "xga": 1.75, "possession": 54, "tactical_style": "HIGH_PRESS"},
            "Werder Bremen": {"league": "Bundesliga", "xg": 1.45, "xga": 1.80, "possession": 49, "tactical_style": "COUNTER_ATTACK"},
            "Bochum": {"league": "Bundesliga", "xg": 1.30, "xga": 1.85, "possession": 44, "tactical_style": "DEFENSIVE"},
            "Union Berlin": {"league": "Bundesliga", "xg": 1.25, "xga": 1.90, "possession": 45, "tactical_style": "DEFENSIVE"},
            "Mainz": {"league": "Bundesliga", "xg": 1.35, "xga": 1.95, "possession": 48, "tactical_style": "HIGH_PRESS"},
            "Koln": {"league": "Bundesliga", "xg": 1.20, "xga": 2.00, "possession": 50, "tactical_style": "POSSESSION"},
            "Darmstadt": {"league": "Bundesliga", "xg": 1.15, "xga": 2.20, "possession": 47, "tactical_style": "DEFENSIVE"},
            
            # Serie A Teams
            "Inter Milan": {"league": "Serie A", "xg": 2.15, "xga": 0.75, "possession": 57, "tactical_style": "HIGH_PRESS"},
            "Juventus": {"league": "Serie A", "xg": 1.85, "xga": 0.95, "possession": 52, "tactical_style": "DEFENSIVE"},
            "AC Milan": {"league": "Serie A", "xg": 2.00, "xga": 1.10, "possession": 56, "tactical_style": "HIGH_PRESS"},
            "Fiorentina": {"league": "Serie A", "xg": 1.75, "xga": 1.25, "possession": 58, "tactical_style": "POSSESSION"},
            "Atalanta": {"league": "Serie A", "xg": 1.90, "xga": 1.35, "possession": 54, "tactical_style": "HIGH_PRESS"},
            "Lazio": {"league": "Serie A", "xg": 1.70, "xga": 1.20, "possession": 55, "tactical_style": "POSSESSION"},
            "Napoli": {"league": "Serie A", "xg": 1.80, "xga": 1.40, "possession": 53, "tactical_style": "COUNTER_ATTACK"},
            "Roma": {"league": "Serie A", "xg": 1.65, "xga": 1.30, "possession": 52, "tactical_style": "DEFENSIVE"},
            "Bologna": {"league": "Serie A", "xg": 1.60, "xga": 1.25, "possession": 51, "tactical_style": "HIGH_PRESS"},
            "Monza": {"league": "Serie A", "xg": 1.35, "xga": 1.45, "possession": 48, "tactical_style": "DEFENSIVE"},
            "Torino": {"league": "Serie A", "xg": 1.40, "xga": 1.35, "possession": 49, "tactical_style": "DEFENSIVE"},
            "Genoa": {"league": "Serie A", "xg": 1.30, "xga": 1.50, "possession": 47, "tactical_style": "DEFENSIVE"},
            "Lecce": {"league": "Serie A", "xg": 1.25, "xga": 1.55, "possession": 46, "tactical_style": "DEFENSIVE"},
            "Sassuolo": {"league": "Serie A", "xg": 1.45, "xga": 1.80, "possession": 53, "tactical_style": "POSSESSION"},
            "Frosinone": {"league": "Serie A", "xg": 1.20, "xga": 1.85, "possession": 50, "tactical_style": "HIGH_PRESS"},
            "Udinese": {"league": "Serie A", "xg": 1.35, "xga": 1.65, "possession": 49, "tactical_style": "DEFENSIVE"},
            "Empoli": {"league": "Serie A", "xg": 1.15, "xga": 1.75, "possession": 48, "tactical_style": "DEFENSIVE"},
            "Cagliari": {"league": "Serie A", "xg": 1.10, "xga": 1.90, "possession": 46, "tactical_style": "DEFENSIVE"},
            "Verona": {"league": "Serie A", "xg": 1.05, "xga": 1.95, "possession": 44, "tactical_style": "DEFENSIVE"},
            "Salernitana": {"league": "Serie A", "xg": 1.00, "xga": 2.10, "possession": 45, "tactical_style": "DEFENSIVE"},
            
            # Ligue 1 Teams
            "PSG": {"league": "Ligue 1", "xg": 2.30, "xga": 0.90, "possession": 64, "tactical_style": "POSSESSION"},
            "Nice": {"league": "Ligue 1", "xg": 1.80, "xga": 0.85, "possession": 52, "tactical_style": "DEFENSIVE"},
            "Monaco": {"league": "Ligue 1", "xg": 2.00, "xga": 1.40, "possession": 56, "tactical_style": "HIGH_PRESS"},
            "Lille": {"league": "Ligue 1", "xg": 1.75, "xga": 1.20, "possession": 53, "tactical_style": "DEFENSIVE"},
            "Brest": {"league": "Ligue 1", "xg": 1.60, "xga": 1.25, "possession": 49, "tactical_style": "DEFENSIVE"},
            "Lens": {"league": "Ligue 1", "xg": 1.70, "xga": 1.35, "possession": 51, "tactical_style": "HIGH_PRESS"},
            "Marseille": {"league": "Ligue 1", "xg": 1.85, "xga": 1.50, "possession": 55, "tactical_style": "HIGH_PRESS"},
            "Rennes": {"league": "Ligue 1", "xg": 1.65, "xga": 1.45, "possession": 54, "tactical_style": "POSSESSION"},
            "Reims": {"league": "Ligue 1", "xg": 1.55, "xga": 1.55, "possession": 50, "tactical_style": "HIGH_PRESS"},
            "Strasbourg": {"league": "Ligue 1", "xg": 1.45, "xga": 1.60, "possession": 48, "tactical_style": "DEFENSIVE"},
            "Montpellier": {"league": "Ligue 1", "xg": 1.50, "xga": 1.70, "possession": 49, "tactical_style": "COUNTER_ATTACK"},
            "Nantes": {"league": "Ligue 1", "xg": 1.35, "xga": 1.65, "possession": 47, "tactical_style": "DEFENSIVE"},
            "Le Havre": {"league": "Ligue 1", "xg": 1.25, "xga": 1.55, "possession": 46, "tactical_style": "DEFENSIVE"},
            "Toulouse": {"league": "Ligue 1", "xg": 1.40, "xga": 1.75, "possession": 51, "tactical_style": "HIGH_PRESS"},
            "Lorient": {"league": "Ligue 1", "xg": 1.30, "xga": 1.85, "possession": 49, "tactical_style": "COUNTER_ATTACK"},
            "Clermont Foot": {"league": "Ligue 1", "xg": 1.15, "xga": 1.90, "possession": 47, "tactical_style": "DEFENSIVE"},
            "Metz": {"league": "Ligue 1", "xg": 1.20, "xga": 2.00, "possession": 45, "tactical_style": "DEFENSIVE"},
            "Lyon": {"league": "Ligue 1", "xg": 1.45, "xga": 1.80, "possession": 53, "tactical_style": "POSSESSION"},
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

def display_input_section(engine, existing_data=None):
    """Display the main input section with existing data if provided"""
    st.markdown('<div style="background-color: #f8f9fa; padding: 2rem; border-radius: 15px; border: 2px solid #e9ecef; margin-bottom: 2rem;">', unsafe_allow_html=True)
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
        home_default_index = 0
        if existing_data and existing_data['home_team'] in available_teams:
            home_default_index = available_teams.index(existing_data['home_team'])
            
        home_team = st.selectbox("Select Home Team", available_teams, 
                               index=home_default_index, key="home_team_select")
        
        # Auto-fill home team data
        home_data = engine.get_team_data(home_team)
        
        home_xg = st.number_input("Expected Goals (xG)", 
                                value=existing_data['home_xg'] if existing_data else home_data["xg"], 
                                min_value=0.0, key="home_xg")
        home_xga = st.number_input("Expected Goals Against (xGA)", 
                                 value=existing_data['home_xga'] if existing_data else home_data["xga"], 
                                 min_value=0.0, key="home_xga")
        home_possession = st.slider("Average Possession %", 0, 100, 
                                  existing_data['home_possession'] if existing_data else home_data["possession"], 
                                  key="home_possession")
        
        # Tactical style with existing data
        tactical_options = ["DEFENSIVE", "LOW_BLOCK", "COUNTER_ATTACK", "POSSESSION", "HIGH_PRESS", "GEGENPRESS", "HIGH_LINE", "BALANCED", "TRANSITION"]
        home_tactical_index = tactical_options.index(home_data["tactical_style"]) if home_data["tactical_style"] in tactical_options else 0
        if existing_data:
            home_tactical_index = tactical_options.index(existing_data['home_tactical']) if existing_data['home_tactical'] in tactical_options else home_tactical_index
            
        home_tactical = st.selectbox("Home Tactical Style", tactical_options,
                                   index=home_tactical_index, key="home_tactical")
    
    with col2:
        st.subheader("✈️ AWAY TEAM")
        
        # Set default index based on existing data
        away_default_index = 1 if len(available_teams) > 1 else 0
        if existing_data and existing_data['away_team'] in available_teams:
            away_default_index = available_teams.index(existing_data['away_team'])
            
        away_team = st.selectbox("Select Away Team", available_teams, 
                               index=away_default_index, key="away_team_select")
        
        # Auto-fill away team data
        away_data = engine.get_team_data(away_team)
        
        away_xg = st.number_input("Away Expected Goals (xG)", 
                                value=existing_data['away_xg'] if existing_data else away_data["xg"], 
                                min_value=0.0, key="away_xg")
        away_xga = st.number_input("Away Expected Goals Against (xGA)", 
                                 value=existing_data['away_xga'] if existing_data else away_data["xga"], 
                                 min_value=0.0, key="away_xga")
        away_possession = st.slider("Away Average Possession %", 0, 100, 
                                  existing_data['away_possession'] if existing_data else away_data["possession"], 
                                  key="away_possession")
        
        # Tactical style with existing data
        away_tactical_index = tactical_options.index(away_data["tactical_style"]) if away_data["tactical_style"] in tactical_options else 4
        if existing_data:
            away_tactical_index = tactical_options.index(existing_data['away_tactical']) if existing_data['away_tactical'] in tactical_options else away_tactical_index
            
        away_tactical = st.selectbox("Away Tactical Style", tactical_options,
                                   index=away_tactical_index, key="away_tactical")
    
    # Match Context Section - with existing data
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
    
    # Injury input with tiered system - with existing data
    st.subheader("🩹 Injury Status")
    col1, col2 = st.columns(2)
    
    injury_options = ["None", "Minor (1-2 rotational)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ starters)"]
    
    with col1:
        home_injury_index = injury_options.index(existing_data['home_injuries']) if existing_data and existing_data['home_injuries'] in injury_options else 0
        home_injuries = st.selectbox("Home Key Injuries", injury_options,
                                   index=home_injury_index, key="home_injuries")
    with col2:
        away_injury_index = injury_options.index(existing_data['away_injuries']) if existing_data and existing_data['away_injuries'] in injury_options else 0
        away_injuries = st.selectbox("Away Key Injuries", injury_options,
                                   index=away_injury_index, key="away_injuries")
    
    # Market Odds - with existing data
    st.markdown("---")
    st.subheader("💰 Market Odds")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        home_odds = st.number_input("Home Win Odds", 
                                  value=existing_data['home_odds'] if existing_data else 7.50, 
                                  min_value=1.01, key="home_odds")
    with col2:
        draw_odds = st.number_input("Draw Odds", 
                                  value=existing_data['draw_odds'] if existing_data else 5.00, 
                                  min_value=1.01, key="draw_odds")
    with col3:
        away_odds = st.number_input("Away Win Odds", 
                                  value=existing_data['away_odds'] if existing_data else 1.38, 
                                  min_value=1.01, key="away_odds")
    with col4:
        over_odds = st.number_input("Over 2.5 Goals Odds", 
                                  value=existing_data['over_odds'] if existing_data else 2.00, 
                                  min_value=1.01, key="over_odds")
    
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
        'home_possession': home_possession,
        'away_possession': away_possession,
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
    st.markdown('<div style="background-color: #ffffff; padding: 2rem; border-radius: 15px; border: 2px solid #1f77b4; margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    # Convert injury tiers to numerical values
    injury_tier_map = {
        "None": 0, "Minor (1-2 rotational)": 1, "Moderate (1-2 key starters)": 2,
        "Significant (3-4 key players)": 3, "Crisis (5+ starters)": 4
    }
    
    home_injury_tier = injury_tier_map[input_data['home_injuries']]
    away_injury_tier = injury_tier_map[input_data['away_injuries']]
    
    # Main Prediction Header
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; color: white; margin-bottom: 1rem;">🎯 PREDICTION RESULTS</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="text-align: center; color: white; margin-bottom: 2rem;">{input_data["home_team"]} vs {input_data["away_team"]}</h2>', unsafe_allow_html=True)
    
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
    
    # Expected Score Display
    expected_home_goals = round(home_xg_final, 1)
    expected_away_goals = round(away_xg_final, 1)
    
    st.markdown(f'<div class="expected-score">{expected_home_goals} - {expected_away_goals}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: white; font-size: 1.2rem;">Expected Final Score</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Stats Row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Home Win", f"{home_win_prob}%")
    with col2:
        st.metric("🤝 Draw", f"{draw_prob}%")
    with col3:
        st.metric("✈️ Away Win", f"{away_win_prob}%")
    with col4:
        over_25_prob = round(min(95, max(25, (total_xg / 2.5) * 65)), 1)
        st.metric("⚽ Over 2.5", f"{over_25_prob}%")
    
    # Detailed Predictions Section
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
        st.markdown('</div>')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="team-analysis-card">', unsafe_allow_html=True)
        st.subheader("⚽ Goals Market")
        under_25_prob = round(100 - over_25_prob, 1)
        st.markdown(f'<div style="text-align: center;">', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">Over 2.5: {over_25_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">Under 2.5: {under_25_prob}%</span>', unsafe_allow_html=True)
        st.markdown('</div>')
        
        btts_yes_prob = round(min(90, max(25, ((home_xg_final * 0.7 + away_xg_final * 0.7) / 2) * 85)), 1)
        btts_no_prob = round(100 - btts_yes_prob, 1)
        st.markdown(f'<div style="text-align: center; margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">BTTS Yes: {btts_yes_prob}%</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="probability-badge">BTTS No: {btts_no_prob}%</span>', unsafe_allow_html=True)
        st.markdown('</div>')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="value-bet-card">', unsafe_allow_html=True)
        st.subheader("💰 Value Bets")
        st.write("Recommended bets based on model vs market odds")
        
        # Calculate value bets
        odds_dict = {'home': input_data['home_odds'], 'draw': input_data['draw_odds'], 'over_2.5': input_data['over_odds']}
        probs_dict = {'home': home_win_prob/100, 'draw': draw_prob/100, 'over_2.5': over_25_prob/100}
        value_bets = engine.calculate_value_bets(probs_dict, odds_dict)
        
        st.markdown("---")
        
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
            st.markdown("All market odds appear efficient")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Team Analysis
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Team Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_defense, home_attack = engine.team_strength_snapshot(input_data['home_xg'], input_data['home_xga'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(f"🏠 {input_data['home_team']}")
        col1a, col2a = st.columns(2)
        with col1a:
            st.metric("Defensive Strength", f"{home_defense}/10")
        with col2a:
            st.metric("Attacking Strength", f"{home_attack}/10")
        st.write(f"**Tactical Style**: {input_data['home_tactical'].replace('_', ' ').title()}")
        st.write(f"**Injuries**: {input_data['home_injuries']}")
        st.write(f"**Base xG**: {input_data['home_xg']}")
        st.write(f"**Base xGA**: {input_data['home_xga']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        away_defense, away_attack = engine.team_strength_snapshot(input_data['away_xg'], input_data['away_xga'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(f"✈️ {input_data['away_team']}")
        col1a, col2a = st.columns(2)
        with col1a:
            st.metric("Defensive Strength", f"{away_defense}/10")
        with col2a:
            st.metric("Attacking Strength", f"{away_attack}/10")
        st.write(f"**Tactical Style**: {input_data['away_tactical'].replace('_', ' ').title()}")
        st.write(f"**Injuries**: {input_data['away_injuries']}")
        st.write(f"**Base xG**: {input_data['away_xg']}")
        st.write(f"**Base xGA**: {input_data['away_xga']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Insights
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 Model Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.subheader("📈 Expected Goals Progression")
        progress_data = {
            'Stage': ['Base xG', 'After Injuries', 'Tactical Adjust', 'Final xG'],
            f'{input_data["home_team"]}': [input_data['home_xg'], round(home_xg_adj, 2), round(home_xg_final, 2), round(home_xg_final, 2)],
            f'{input_data["away_team"]}': [input_data['away_xg'], round(away_xg_adj, 2), round(away_xg_final, 2), round(away_xg_final, 2)]
        }
        st.dataframe(pd.DataFrame(progress_data), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
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
        
        # Confidence score
        outcome_probs = {'home': home_win_prob/100, 'draw': draw_prob/100, 'away': away_win_prob/100}
        confidence_score = engine.calculate_confidence_score(outcome_probs)
        conf_class, conf_label = engine.get_confidence_label(confidence_score)
        
        st.markdown("---")
        st.markdown(f'**Model Confidence**: <span class="{conf_class}">{confidence_score}/100 - {conf_label}</span>', unsafe_allow_html=True)
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
    
    # Initialize prediction engine
    engine = EnhancedPredictionEngine()
    
    # Initialize session state
    if 'show_prediction' not in st.session_state:
        st.session_state.show_prediction = False
    if 'existing_data' not in st.session_state:
        st.session_state.existing_data = None
    
    # Main app flow
    if not st.session_state.show_prediction:
        # Show input section with existing data if available
        input_data = display_input_section(engine, st.session_state.existing_data)
        
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
