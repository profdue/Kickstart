import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Professional Football Prediction Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Professional Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.4rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .input-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
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
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .value-good {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .value-poor {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #856404;
    }
    .success-box {
        background-color: #d1edff;
        border: 1px solid #b3d9ff;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #004085;
    }
    .understat-format {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-family: monospace;
        font-weight: bold;
        text-align: center;
    }
    .contradiction-flag {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #ff4757;
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalPredictionEngine:
    def __init__(self):
        self.league_avg_xg = 1.35
        self.league_avg_xga = 1.35
        self.rho = -0.13  # Dixon-Coles correlation parameter
        
        # Comprehensive team database with all top 5 leagues
        self.team_database = self._initialize_complete_database()
        
        # Enhanced injury impact weights with performance multipliers
        self.injury_weights = {
            "None": {"attack_mult": 1.00, "defense_mult": 1.00, "description": "No impact"},
            "Minor (1-2 rotational)": {"attack_mult": 0.92, "defense_mult": 0.95, "description": "Slight impact"},
            "Moderate (1-2 key starters)": {"attack_mult": 0.85, "defense_mult": 0.88, "description": "Moderate impact"},
            "Significant (3-4 key players)": {"attack_mult": 0.75, "defense_mult": 0.80, "description": "Significant impact"},
            "Crisis (5+ starters)": {"attack_mult": 0.60, "defense_mult": 0.70, "description": "Severe impact"}
        }
        
        # Enhanced fatigue multipliers with progressive scaling
        self.fatigue_multipliers = {
            2: 0.82, 3: 0.85, 4: 0.88, 5: 0.91, 6: 0.94, 
            7: 0.97, 8: 1.00, 9: 1.02, 10: 1.03, 11: 1.04,
            12: 1.04, 13: 1.04, 14: 1.04
        }
        
        # Opponent strength adjustment factors
        self.opponent_strength = {
            "Elite": 1.2, "Strong": 1.1, "Average": 1.0, "Weak": 0.9, "Very Weak": 0.8
        }

    def _initialize_complete_database(self):
        """Initialize complete database for all top 5 leagues"""
        return {
            # Premier League (20 teams)
            "Arsenal": {"league": "Premier League", "last_5_xg_total": 10.25, "last_5_xga_total": 1.75, "form_trend": 0.08, "last_5_opponents": ["Bournemouth", "Tottenham", "Burnley", "Newcastle", "West Ham"]},
            "Aston Villa": {"league": "Premier League", "last_5_xg_total": 9.25, "last_5_xga_total": 6.25, "form_trend": 0.10, "last_5_opponents": ["Man City", "Arsenal", "Brentford", "West Ham", "Wolves"]},
            "Bournemouth": {"league": "Premier League", "last_5_xg_total": 5.77, "last_5_xga_total": 2.30, "form_trend": 0.12, "last_5_opponents": ["Arsenal", "Brighton", "Wolves", "Crystal Palace", "Everton"]},
            "Brentford": {"league": "Premier League", "last_5_xg_total": 7.50, "last_5_xga_total": 8.25, "form_trend": -0.05, "last_5_opponents": ["Chelsea", "Brighton", "West Ham", "Fulham", "Crystal Palace"]},
            "Brighton": {"league": "Premier League", "last_5_xg_total": 8.75, "last_5_xga_total": 7.25, "form_trend": 0.03, "last_5_opponents": ["Man City", "Bournemouth", "Chelsea", "Brentford", "Sheffield Utd"]},
            "Burnley": {"league": "Premier League", "last_5_xg_total": 4.50, "last_5_xga_total": 9.75, "form_trend": -0.15, "last_5_opponents": ["Arsenal", "Everton", "Chelsea", "Wolves", "Crystal Palace"]},
            "Chelsea": {"league": "Premier League", "last_5_xg_total": 8.50, "last_5_xga_total": 7.50, "form_trend": 0.06, "last_5_opponents": ["Man City", "Tottenham", "Brighton", "Man United", "Burnley"]},
            "Crystal Palace": {"league": "Premier League", "last_5_xg_total": 6.25, "last_5_xga_total": 8.75, "form_trend": -0.08, "last_5_opponents": ["Tottenham", "West Ham", "Newcastle", "Bournemouth", "Burnley"]},
            "Everton": {"league": "Premier League", "last_5_xg_total": 7.00, "last_5_xga_total": 8.50, "form_trend": 0.02, "last_5_opponents": ["Burnley", "Newcastle", "Man United", "Liverpool", "Bournemouth"]},
            "Fulham": {"league": "Premier League", "last_5_xg_total": 7.75, "last_5_xga_total": 7.25, "form_trend": 0.04, "last_5_opponents": ["West Ham", "Newcastle", "Brentford", "Forest", "Liverpool"]},
            "Liverpool": {"league": "Premier League", "last_5_xg_total": 11.25, "last_5_xga_total": 5.25, "form_trend": 0.10, "last_5_opponents": ["Brighton", "Sheffield Utd", "Crystal Palace", "Man United", "Everton"]},
            "Luton": {"league": "Premier League", "last_5_xg_total": 6.50, "last_5_xga_total": 10.25, "form_trend": -0.12, "last_5_opponents": ["Tottenham", "Arsenal", "Bournemouth", "Forest", "Brentford"]},
            "Manchester City": {"league": "Premier League", "last_5_xg_total": 11.44, "last_5_xga_total": 5.00, "form_trend": 0.15, "last_5_opponents": ["Chelsea", "Liverpool", "Brighton", "Man United", "Aston Villa"]},
            "Manchester United": {"league": "Premier League", "last_5_xg_total": 10.64, "last_5_xga_total": 4.88, "form_trend": -0.05, "last_5_opponents": ["Chelsea", "Liverpool", "West Ham", "Everton", "Newcastle"]},
            "Newcastle": {"league": "Premier League", "last_5_xg_total": 8.75, "last_5_xga_total": 6.75, "form_trend": 0.03, "last_5_opponents": ["Arsenal", "West Ham", "Everton", "Crystal Palace", "Fulham"]},
            "Nottingham Forest": {"league": "Premier League", "last_5_xg_total": 6.25, "last_5_xga_total": 9.50, "form_trend": -0.10, "last_5_opponents": ["Fulham", "Wolves", "Spurs", "Brighton", "Luton"]},
            "Sheffield United": {"league": "Premier League", "last_5_xg_total": 4.25, "last_5_xga_total": 12.75, "form_trend": -0.20, "last_5_opponents": ["Liverpool", "Brentford", "Chelsea", "Burnley", "Brighton"]},
            "Tottenham": {"league": "Premier League", "last_5_xg_total": 9.75, "last_5_xga_total": 7.25, "form_trend": -0.02, "last_5_opponents": ["Arsenal", "Chelsea", "Wolves", "Crystal Palace", "Luton"]},
            "West Ham": {"league": "Premier League", "last_5_xg_total": 7.75, "last_5_xga_total": 8.25, "form_trend": -0.08, "last_5_opponents": ["Arsenal", "Newcastle", "Tottenham", "Wolves", "Fulham"]},
            "Wolves": {"league": "Premier League", "last_5_xg_total": 7.25, "last_5_xga_total": 7.75, "form_trend": 0.01, "last_5_opponents": ["Tottenham", "Bournemouth", "Forest", "West Ham", "Burnley"]},

            # La Liga (20 teams)
            "Real Madrid": {"league": "La Liga", "last_5_xg_total": 12.50, "last_5_xga_total": 4.00, "form_trend": 0.15, "last_5_opponents": ["Barcelona", "Atletico", "Sevilla", "Valencia", "Girona"]},
            "Barcelona": {"league": "La Liga", "last_5_xg_total": 10.75, "last_5_xga_total": 4.50, "form_trend": 0.09, "last_5_opponents": ["Real Madrid", "Atletico", "Sevilla", "Valencia", "Betis"]},
            "Atletico Madrid": {"league": "La Liga", "last_5_xg_total": 9.75, "last_5_xga_total": 5.25, "form_trend": 0.07, "last_5_opponents": ["Real Madrid", "Barcelona", "Athletic", "Villarreal", "Sevilla"]},
            "Girona": {"league": "La Liga", "last_5_xg_total": 9.25, "last_5_xga_total": 6.75, "form_trend": 0.12, "last_5_opponents": ["Real Madrid", "Barcelona", "Betis", "Valencia", "Sevilla"]},
            "Athletic Bilbao": {"league": "La Liga", "last_5_xg_total": 8.75, "last_5_xga_total": 5.50, "form_trend": 0.08, "last_5_opponents": ["Atletico", "Real Sociedad", "Villarreal", "Betis", "Valencia"]},
            "Real Sociedad": {"league": "La Liga", "last_5_xg_total": 8.25, "last_5_xga_total": 5.75, "form_trend": 0.05, "last_5_opponents": ["Athletic", "Real Madrid", "Barcelona", "Sevilla", "Villarreal"]},
            "Real Betis": {"league": "La Liga", "last_5_xg_total": 7.75, "last_5_xga_total": 6.25, "form_trend": 0.03, "last_5_opponents": ["Barcelona", "Girona", "Athletic", "Valencia", "Sevilla"]},
            "Valencia": {"league": "La Liga", "last_5_xg_total": 7.25, "last_5_xga_total": 6.75, "form_trend": 0.02, "last_5_opponents": ["Real Madrid", "Barcelona", "Girona", "Betis", "Villarreal"]},
            "Villarreal": {"league": "La Liga", "last_5_xg_total": 8.50, "last_5_xga_total": 7.50, "form_trend": 0.04, "last_5_opponents": ["Atletico", "Athletic", "Real Sociedad", "Betis", "Valencia"]},
            "Sevilla": {"league": "La Liga", "last_5_xg_total": 7.00, "last_5_xga_total": 8.25, "form_trend": -0.05, "last_5_opponents": ["Real Madrid", "Barcelona", "Atletico", "Real Sociedad", "Betis"]},
            "Osasuna": {"league": "La Liga", "last_5_xg_total": 6.75, "last_5_xga_total": 7.50, "form_trend": -0.02, "last_5_opponents": ["Real Madrid", "Barcelona", "Athletic", "Betis", "Valencia"]},
            "Getafe": {"league": "La Liga", "last_5_xg_total": 6.25, "last_5_xga_total": 7.75, "form_trend": -0.03, "last_5_opponents": ["Atletico", "Real Sociedad", "Sevilla", "Villarreal", "Betis"]},
            "Alaves": {"league": "La Liga", "last_5_xg_total": 5.75, "last_5_xga_total": 8.25, "form_trend": -0.08, "last_5_opponents": ["Real Madrid", "Barcelona", "Athletic", "Sevilla", "Valencia"]},
            "Mallorca": {"league": "La Liga", "last_5_xg_total": 5.50, "last_5_xga_total": 8.50, "form_trend": -0.10, "last_5_opponents": ["Real Madrid", "Barcelona", "Atletico", "Betis", "Sevilla"]},
            "Rayo Vallecano": {"league": "La Liga", "last_5_xg_total": 6.00, "last_5_xga_total": 8.00, "form_trend": -0.06, "last_5_opponents": ["Real Madrid", "Barcelona", "Athletic", "Villarreal", "Betis"]},
            "Celta Vigo": {"league": "La Liga", "last_5_xg_total": 7.25, "last_5_xga_total": 8.75, "form_trend": -0.04, "last_5_opponents": ["Real Madrid", "Barcelona", "Atletico", "Sevilla", "Valencia"]},
            "Cadiz": {"league": "La Liga", "last_5_xg_total": 5.25, "last_5_xga_total": 9.25, "form_trend": -0.12, "last_5_opponents": ["Real Madrid", "Barcelona", "Atletico", "Betis", "Sevilla"]},
            "Granada": {"league": "La Liga", "last_5_xg_total": 4.75, "last_5_xga_total": 10.50, "form_trend": -0.15, "last_5_opponents": ["Real Madrid", "Barcelona", "Athletic", "Villarreal", "Betis"]},
            "Las Palmas": {"league": "La Liga", "last_5_xg_total": 6.50, "last_5_xga_total": 7.25, "form_trend": 0.01, "last_5_opponents": ["Real Madrid", "Barcelona", "Atletico", "Real Sociedad", "Betis"]},
            "Almeria": {"league": "La Liga", "last_5_xg_total": 5.00, "last_5_xga_total": 10.75, "form_trend": -0.18, "last_5_opponents": ["Real Madrid", "Barcelona", "Atletico", "Sevilla", "Valencia"]},

            # Bundesliga (18 teams)
            "Bayern Munich": {"league": "Bundesliga", "last_5_xg_total": 12.00, "last_5_xga_total": 4.75, "form_trend": 0.11, "last_5_opponents": ["Dortmund", "Leverkusen", "Leipzig", "Stuttgart", "Frankfurt"]},
            "Bayer Leverkusen": {"league": "Bundesliga", "last_5_xg_total": 11.75, "last_5_xga_total": 4.50, "form_trend": 0.13, "last_5_opponents": ["Bayern", "Dortmund", "Leipzig", "Stuttgart", "Frankfurt"]},
            "RB Leipzig": {"league": "Bundesliga", "last_5_xg_total": 10.50, "last_5_xga_total": 5.25, "form_trend": 0.09, "last_5_opponents": ["Bayern", "Leverkusen", "Dortmund", "Stuttgart", "Frankfurt"]},
            "Borussia Dortmund": {"league": "Bundesliga", "last_5_xg_total": 10.25, "last_5_xga_total": 5.75, "form_trend": 0.07, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Stuttgart", "Frankfurt"]},
            "Stuttgart": {"league": "Bundesliga", "last_5_xg_total": 9.75, "last_5_xga_total": 6.25, "form_trend": 0.10, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Dortmund", "Frankfurt"]},
            "Eintracht Frankfurt": {"league": "Bundesliga", "last_5_xg_total": 8.50, "last_5_xga_total": 6.75, "form_trend": 0.05, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Dortmund", "Stuttgart"]},
            "Freiburg": {"league": "Bundesliga", "last_5_xg_total": 8.25, "last_5_xga_total": 7.00, "form_trend": 0.03, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Dortmund", "Frankfurt"]},
            "Hoffenheim": {"league": "Bundesliga", "last_5_xg_total": 8.75, "last_5_xga_total": 8.25, "form_trend": 0.02, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Stuttgart", "Frankfurt"]},
            "Wolfsburg": {"league": "Bundesliga", "last_5_xg_total": 7.50, "last_5_xga_total": 7.75, "form_trend": -0.02, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Dortmund", "Stuttgart"]},
            "Augsburg": {"league": "Bundesliga", "last_5_xg_total": 7.25, "last_5_xga_total": 8.50, "form_trend": -0.04, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Frankfurt", "Freiburg"]},
            "Borussia Mönchengladbach": {"league": "Bundesliga", "last_5_xg_total": 8.00, "last_5_xga_total": 8.75, "form_trend": -0.03, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Dortmund", "Stuttgart"]},
            "Werder Bremen": {"league": "Bundesliga", "last_5_xg_total": 7.75, "last_5_xga_total": 8.25, "form_trend": -0.01, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Frankfurt", "Freiburg"]},
            "Heidenheim": {"league": "Bundesliga", "last_5_xg_total": 6.50, "last_5_xga_total": 9.25, "form_trend": -0.08, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Dortmund", "Stuttgart"]},
            "Union Berlin": {"league": "Bundesliga", "last_5_xg_total": 6.25, "last_5_xga_total": 9.50, "form_trend": -0.10, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Frankfurt", "Freiburg"]},
            "Mainz": {"league": "Bundesliga", "last_5_xg_total": 6.75, "last_5_xga_total": 8.75, "form_trend": -0.06, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Dortmund", "Stuttgart"]},
            "Köln": {"league": "Bundesliga", "last_5_xg_total": 6.00, "last_5_xga_total": 9.75, "form_trend": -0.12, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Frankfurt", "Freiburg"]},
            "Bochum": {"league": "Bundesliga", "last_5_xg_total": 5.75, "last_5_xga_total": 10.25, "form_trend": -0.14, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Dortmund", "Stuttgart"]},
            "Darmstadt": {"league": "Bundesliga", "last_5_xg_total": 5.25, "last_5_xga_total": 11.50, "form_trend": -0.18, "last_5_opponents": ["Bayern", "Leverkusen", "Leipzig", "Frankfurt", "Freiburg"]},

            # Serie A (20 teams)
            "Inter Milan": {"league": "Serie A", "last_5_xg_total": 10.75, "last_5_xga_total": 3.75, "form_trend": 0.13, "last_5_opponents": ["Juventus", "Milan", "Napoli", "Roma", "Lazio"]},
            "Juventus": {"league": "Serie A", "last_5_xg_total": 9.50, "last_5_xga_total": 4.25, "form_trend": 0.10, "last_5_opponents": ["Inter", "Milan", "Napoli", "Roma", "Lazio"]},
            "AC Milan": {"league": "Serie A", "last_5_xg_total": 9.75, "last_5_xga_total": 5.00, "form_trend": 0.08, "last_5_opponents": ["Inter", "Juventus", "Napoli", "Roma", "Lazio"]},
            "Napoli": {"league": "Serie A", "last_5_xg_total": 9.25, "last_5_xga_total": 5.50, "form_trend": 0.06, "last_5_opponents": ["Inter", "Juventus", "Milan", "Roma", "Lazio"]},
            "Atalanta": {"league": "Serie A", "last_5_xg_total": 9.00, "last_5_xga_total": 5.75, "form_trend": 0.07, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Roma": {"league": "Serie A", "last_5_xg_total": 8.75, "last_5_xga_total": 6.25, "form_trend": 0.05, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Lazio"]},
            "Lazio": {"league": "Serie A", "last_5_xg_total": 8.50, "last_5_xga_total": 6.50, "form_trend": 0.04, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Fiorentina": {"league": "Serie A", "last_5_xg_total": 8.25, "last_5_xga_total": 6.75, "form_trend": 0.03, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Bologna": {"league": "Serie A", "last_5_xg_total": 7.75, "last_5_xga_total": 7.00, "form_trend": 0.06, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Torino": {"league": "Serie A", "last_5_xg_total": 7.50, "last_5_xga_total": 7.25, "form_trend": 0.02, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Lazio"]},
            "Monza": {"league": "Serie A", "last_5_xg_total": 7.25, "last_5_xga_total": 7.50, "form_trend": 0.01, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Genoa": {"league": "Serie A", "last_5_xg_total": 6.75, "last_5_xga_total": 8.25, "form_trend": -0.04, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Lazio"]},
            "Lecce": {"league": "Serie A", "last_5_xg_total": 6.50, "last_5_xga_total": 8.50, "form_trend": -0.06, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Sassuolo": {"league": "Serie A", "last_5_xg_total": 7.00, "last_5_xga_total": 9.25, "form_trend": -0.08, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Lazio"]},
            "Frosinone": {"league": "Serie A", "last_5_xg_total": 6.25, "last_5_xga_total": 9.75, "form_trend": -0.10, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Udinese": {"league": "Serie A", "last_5_xg_total": 6.00, "last_5_xga_total": 8.75, "form_trend": -0.05, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Lazio"]},
            "Cagliari": {"league": "Serie A", "last_5_xg_total": 5.75, "last_5_xga_total": 9.50, "form_trend": -0.12, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Empoli": {"league": "Serie A", "last_5_xg_total": 5.50, "last_5_xga_total": 10.25, "form_trend": -0.14, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Lazio"]},
            "Verona": {"league": "Serie A", "last_5_xg_total": 5.25, "last_5_xga_total": 10.75, "form_trend": -0.16, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Roma"]},
            "Salernitana": {"league": "Serie A", "last_5_xg_total": 4.75, "last_5_xga_total": 11.50, "form_trend": -0.20, "last_5_opponents": ["Inter", "Juventus", "Milan", "Napoli", "Lazio"]},

            # Ligue 1 (18 teams)
            "Paris Saint-Germain": {"league": "Ligue 1", "last_5_xg_total": 11.50, "last_5_xga_total": 4.50, "form_trend": 0.14, "last_5_opponents": ["Monaco", "Lyon", "Marseille", "Lille", "Rennes"]},
            "Monaco": {"league": "Ligue 1", "last_5_xg_total": 10.25, "last_5_xga_total": 5.75, "form_trend": 0.11, "last_5_opponents": ["PSG", "Lyon", "Marseille", "Lille", "Rennes"]},
            "Lille": {"league": "Ligue 1", "last_5_xg_total": 9.75, "last_5_xga_total": 5.25, "form_trend": 0.09, "last_5_opponents": ["PSG", "Monaco", "Lyon", "Marseille", "Rennes"]},
            "Brest": {"league": "Ligue 1", "last_5_xg_total": 9.25, "last_5_xga_total": 5.50, "form_trend": 0.12, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Rennes"]},
            "Nice": {"league": "Ligue 1", "last_5_xg_total": 8.75, "last_5_xga_total": 4.75, "form_trend": 0.08, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Lyon"]},
            "Lens": {"league": "Ligue 1", "last_5_xg_total": 9.00, "last_5_xga_total": 6.25, "form_trend": 0.07, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Rennes"]},
            "Marseille": {"league": "Ligue 1", "last_5_xg_total": 8.50, "last_5_xga_total": 6.75, "form_trend": 0.05, "last_5_opponents": ["PSG", "Monaco", "Lille", "Lyon", "Rennes"]},
            "Rennes": {"league": "Ligue 1", "last_5_xg_total": 8.25, "last_5_xga_total": 7.00, "form_trend": 0.04, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Lyon"]},
            "Lyon": {"league": "Ligue 1", "last_5_xg_total": 8.00, "last_5_xga_total": 7.25, "form_trend": 0.06, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Rennes"]},
            "Reims": {"league": "Ligue 1", "last_5_xg_total": 7.75, "last_5_xga_total": 7.50, "form_trend": 0.03, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Rennes"]},
            "Toulouse": {"league": "Ligue 1", "last_5_xg_total": 7.50, "last_5_xga_total": 7.75, "form_trend": 0.01, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Lyon"]},
            "Montpellier": {"league": "Ligue 1", "last_5_xg_total": 7.25, "last_5_xga_total": 8.25, "form_trend": -0.02, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Rennes"]},
            "Strasbourg": {"league": "Ligue 1", "last_5_xg_total": 7.00, "last_5_xga_total": 8.50, "form_trend": -0.04, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Lyon"]},
            "Nantes": {"league": "Ligue 1", "last_5_xg_total": 6.75, "last_5_xga_total": 8.75, "form_trend": -0.06, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Rennes"]},
            "Le Havre": {"league": "Ligue 1", "last_5_xg_total": 6.50, "last_5_xga_total": 9.00, "form_trend": -0.08, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Lyon"]},
            "Lorient": {"league": "Ligue 1", "last_5_xg_total": 6.25, "last_5_xga_total": 9.25, "form_trend": -0.10, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Rennes"]},
            "Metz": {"league": "Ligue 1", "last_5_xg_total": 5.75, "last_5_xga_total": 9.75, "form_trend": -0.12, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Lyon"]},
            "Clermont Foot": {"league": "Ligue 1", "last_5_xg_total": 5.25, "last_5_xga_total": 10.50, "form_trend": -0.16, "last_5_opponents": ["PSG", "Monaco", "Lille", "Marseille", "Rennes"]}
        }

    def get_team_data(self, team_name):
        """Get team data with fallback defaults and enhanced validation"""
        default_data = {
            "league": "Unknown", 
            "last_5_xg_total": 7.50,
            "last_5_xga_total": 7.50,
            "form_trend": 0.00,
            "last_5_opponents": ["Unknown"] * 5
        }
        
        team_data = self.team_database.get(team_name, default_data).copy()
        
        # Calculate per-match averages with validation
        team_data['last_5_xg_per_match'] = team_data['last_5_xg_total'] / 5
        team_data['last_5_xga_per_match'] = team_data['last_5_xga_total'] / 5
        
        # Validate data ranges
        if not (0.1 <= team_data['last_5_xg_per_match'] <= 4.0):
            team_data['last_5_xg_per_match'] = max(0.1, min(4.0, team_data['last_5_xg_per_match']))
        if not (0.1 <= team_data['last_5_xga_per_match'] <= 4.0):
            team_data['last_5_xga_per_match'] = max(0.1, min(4.0, team_data['last_5_xga_per_match']))
        
        return team_data

    def validate_inputs(self, inputs):
        """Comprehensive input validation with enhanced checks"""
        errors = []
        warnings = []
        
        # Required field validation
        required_fields = ['home_team', 'away_team', 'home_xg_total', 'home_xga_total', 'away_xg_total', 'away_xga_total']
        for field in required_fields:
            if not inputs.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Enhanced numerical value validation
        numerical_fields = ['home_xg_total', 'home_xga_total', 'away_xg_total', 'away_xga_total', 'home_rest', 'away_rest']
        for field in numerical_fields:
            value = inputs.get(field)
            if value is not None:
                if field in ['home_xg_total', 'home_xga_total', 'away_xg_total', 'away_xga_total']:
                    if not (0.0 <= value <= 25.0):
                        errors.append(f"{field} must be between 0.0 and 25.0 (Understat format)")
                    elif value == 0:
                        warnings.append(f"{field} is 0 - this seems unusual for a team's last 5 matches")
                    elif value > 20.0:
                        warnings.append(f"{field} is very high ({value}) - please verify data quality")
                elif field in ['home_rest', 'away_rest']:
                    if not (2 <= value <= 14):
                        errors.append(f"{field} must be between 2 and 14 days")
        
        # Enhanced team validation
        if inputs.get('home_team') and inputs.get('away_team'):
            if inputs['home_team'] == inputs['away_team']:
                errors.append("Home and away teams cannot be the same")
            
            # Check if teams are from same league for context
            home_data = self.get_team_data(inputs['home_team'])
            away_data = self.get_team_data(inputs['away_team'])
            if home_data['league'] != away_data['league']:
                warnings.append(f"Teams from different leagues: {home_data['league']} vs {away_data['league']}")
        
        # Enhanced odds validation
        odds_fields = ['home_odds', 'draw_odds', 'away_odds', 'over_odds']
        for field in odds_fields:
            value = inputs.get(field)
            if value is not None and value < 1.01:
                errors.append(f"{field} must be at least 1.01")
            elif value is not None and value > 100.0:
                warnings.append(f"{field} is very high ({value}) - please verify")
        
        # Data quality and contradiction checks
        if inputs.get('home_xg_total') and inputs.get('home_xga_total'):
            home_xg_per_match = inputs['home_xg_total'] / 5
            home_xga_per_match = inputs['home_xga_total'] / 5
            
            # Check for data entry errors (swapped xG/xGA)
            if home_xg_per_match < 0.5 and home_xga_per_match > 2.5:
                warnings.append(f"Potential data entry error: {inputs['home_team']} has very low xG and high xGA - check if values are swapped")
            
            if home_xg_per_match > 3.0:
                warnings.append(f"{inputs['home_team']} has very high xG ({home_xg_per_match:.2f} per match) - please verify data")
            if home_xg_per_match < 0.5:
                warnings.append(f"{inputs['home_team']} has very low xG ({home_xg_per_match:.2f} per match) - please verify data")
        
        return errors, warnings

    def apply_modifiers(self, base_xg, base_xga, injury_level, rest_days, form_trend, is_home=True):
        """Apply all modifiers with enhanced impact modeling"""
        injury_data = self.injury_weights[injury_level]
        
        # Enhanced injury impact (different for attack vs defense)
        attack_injury_mult = injury_data["attack_mult"]
        defense_injury_mult = injury_data["defense_mult"]
        
        # Enhanced fatigue impact
        fatigue_mult = self.fatigue_multipliers.get(rest_days, 1.0)
        
        # Enhanced form trend impact
        form_mult = 1 + (form_trend * 0.25)  # Increased form impact
        
        # Home advantage
        home_mult = 1.1 if is_home else 1.0
        
        # Apply modifiers with position-specific impacts
        xg_modified = base_xg * attack_injury_mult * fatigue_mult * form_mult * home_mult
        xga_modified = base_xga * defense_injury_mult * (1/fatigue_mult) * (1/form_mult) * (1/home_mult if is_home else home_mult)
        
        return max(0.1, xg_modified), max(0.1, xga_modified)

    def calculate_rest_advantage(self, home_rest, away_rest):
        """Calculate rest advantage impact"""
        rest_diff = home_rest - away_rest
        advantage_mult = 1.0 + (rest_diff * 0.03)  # 3% per day advantage
        return max(0.85, min(1.15, advantage_mult))

    def dixon_coles_probabilities(self, home_exp, away_exp, max_goals=8):
        """Calculate match probabilities with Dixon-Coles adjustment"""
        # Basic Poisson probabilities
        home_probs = [poisson.pmf(i, home_exp) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, away_exp) for i in range(max_goals)]
        
        # Create joint probability matrix
        joint_probs = np.outer(home_probs, away_probs)
        
        # Apply Dixon-Coles correlation adjustment
        for i in range(max_goals):
            for j in range(max_goals):
                if i == 0 and j == 0:
                    joint_probs[i,j] *= 1 - (self.rho * np.sqrt(poisson.pmf(0, home_exp) * poisson.pmf(0, away_exp)))
                elif i == 0 and j == 1:
                    joint_probs[i,j] *= 1 + (self.rho * np.sqrt(poisson.pmf(0, home_exp) * poisson.pmf(1, away_exp)))
                elif i == 1 and j == 0:
                    joint_probs[i,j] *= 1 + (self.rho * np.sqrt(poisson.pmf(1, home_exp) * poisson.pmf(0, away_exp)))
                elif i == 1 and j == 1:
                    joint_probs[i,j] *= 1 - (self.rho * np.sqrt(poisson.pmf(1, home_exp) * poisson.pmf(1, away_exp)))
        
        # Normalize probabilities
        joint_probs = joint_probs / joint_probs.sum()
        
        # Calculate outcome probabilities
        home_win = np.sum(np.triu(joint_probs, 1))
        draw = np.sum(np.diag(joint_probs))
        away_win = np.sum(np.tril(joint_probs, -1))
        
        # Calculate over/under probabilities
        over_25 = 1 - np.sum(joint_probs[:3, :3])  # Sum probabilities for 0-2 goals
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'over_2.5': over_25,
            'under_2.5': 1 - over_25,
            'expected_home_goals': home_exp,
            'expected_away_goals': away_exp,
            'goal_matrix': joint_probs
        }

    def calculate_confidence(self, home_xg_per_match, away_xg_per_match, home_xga_per_match, away_xga_per_match, inputs):
        """Enhanced confidence calculation with multiple factors"""
        factors = {}
        
        # Data quality factor
        data_quality = min(1.0, (home_xg_per_match + away_xg_per_match + home_xga_per_match + away_xga_per_match) / 5.4)
        factors['data_quality'] = data_quality
        
        # Predictability factor
        predictability = 1 - (abs(home_xg_per_match - away_xg_per_match) / max(home_xg_per_match, away_xg_per_match, 0.1))
        factors['predictability'] = predictability
        
        # Injury impact factor
        home_injury_severity = 1 - self.injury_weights[inputs['home_injuries']]['attack_mult']
        away_injury_severity = 1 - self.injury_weights[inputs['away_injuries']]['attack_mult']
        injury_factor = 1 - (home_injury_severity + away_injury_severity) / 2
        factors['injury_stability'] = injury_factor
        
        # Rest advantage factor
        rest_diff = abs(inputs['home_rest'] - inputs['away_rest'])
        rest_factor = 1 - (rest_diff * 0.05)  # More rest difference = less confidence
        factors['rest_balance'] = rest_factor
        
        # Calculate weighted confidence
        weights = {
            'data_quality': 0.3,
            'predictability': 0.3, 
            'injury_stability': 0.25,
            'rest_balance': 0.15
        }
        
        confidence = sum(factors[factor] * weights[factor] for factor in factors) * 100
        
        return min(95, max(50, confidence)), factors

    def calculate_value_bets(self, probabilities, odds):
        """Enhanced value betting calculation with thresholds"""
        value_bets = {}
        
        # Home win value
        home_implied = 1 / odds['home']
        home_value = probabilities['home_win'] / home_implied
        home_ev = (probabilities['home_win'] * odds['home']) - 1
        
        # Draw value
        draw_implied = 1 / odds['draw']
        draw_value = probabilities['draw'] / draw_implied
        draw_ev = (probabilities['draw'] * odds['draw']) - 1
        
        # Away win value
        away_implied = 1 / odds['away']
        away_value = probabilities['away_win'] / away_implied
        away_ev = (probabilities['away_win'] * odds['away']) - 1
        
        # Over 2.5 value
        over_implied = 1 / odds['over_2.5']
        over_value = probabilities['over_2.5'] / over_implied
        over_ev = (probabilities['over_2.5'] * odds['over_2.5']) - 1
        
        value_bets['home'] = {
            'value_ratio': home_value,
            'ev': home_ev,
            'implied_prob': home_implied,
            'model_prob': probabilities['home_win'],
            'rating': self._get_value_rating(home_value, home_ev)
        }
        value_bets['draw'] = {
            'value_ratio': draw_value,
            'ev': draw_ev,
            'implied_prob': draw_implied,
            'model_prob': probabilities['draw'],
            'rating': self._get_value_rating(draw_value, draw_ev)
        }
        value_bets['away'] = {
            'value_ratio': away_value,
            'ev': away_ev,
            'implied_prob': away_implied,
            'model_prob': probabilities['away_win'],
            'rating': self._get_value_rating(away_value, away_ev)
        }
        value_bets['over_2.5'] = {
            'value_ratio': over_value,
            'ev': over_ev,
            'implied_prob': over_implied,
            'model_prob': probabilities['over_2.5'],
            'rating': self._get_value_rating(over_value, over_ev)
        }
        
        return value_bets

    def _get_value_rating(self, value_ratio, ev):
        """Get value bet rating"""
        if value_ratio > 1.15 and ev > 0.15:
            return "excellent"
        elif value_ratio > 1.08 and ev > 0.08:
            return "good" 
        elif value_ratio > 1.02 and ev > 0.02:
            return "fair"
        else:
            return "poor"

    def detect_contradictions(self, inputs, probabilities, home_expected, away_expected):
        """Detect contradictions in predictions and insights"""
        contradictions = []
        
        total_goals = home_expected + away_expected
        
        # Check if high xG but low Over probability
        if total_goals > 3.0 and probabilities['over_2.5'] < 0.4:
            contradictions.append(f"CONTRADICTION: High expected goals ({total_goals:.2f}) but low Over 2.5 probability ({probabilities['over_2.5']:.1%})")
        
        # Check if strong home advantage but low home win probability
        home_adv = probabilities['home_win'] - probabilities['away_win']
        if home_adv < -0.1:  # Home team is underdog despite advantage
            contradictions.append(f"CONTRADICTION: Home team has advantage but is significant underdog")
        
        # Check injury impact vs prediction
        home_injury_severity = 1 - self.injury_weights[inputs['home_injuries']]['attack_mult']
        away_injury_severity = 1 - self.injury_weights[inputs['away_injuries']]['attack_mult']
        
        if home_injury_severity > 0.2 and probabilities['home_win'] > 0.6:
            contradictions.append(f"CONTRADICTION: {inputs['home_team']} has significant injuries but high win probability")
        
        return contradictions

    def predict_match(self, inputs):
        """Enhanced main prediction function"""
        # Validate inputs first
        errors, warnings = self.validate_inputs(inputs)
        if errors:
            return None, errors, warnings
        
        # Calculate per-match averages from Understat totals
        home_xg_per_match = inputs['home_xg_total'] / 5
        home_xga_per_match = inputs['home_xga_total'] / 5
        away_xg_per_match = inputs['away_xg_total'] / 5
        away_xga_per_match = inputs['away_xga_total'] / 5
        
        # Get team data for form trends
        home_data = self.get_team_data(inputs['home_team'])
        away_data = self.get_team_data(inputs['away_team'])
        
        # Apply enhanced modifiers
        home_xg_adj, home_xga_adj = self.apply_modifiers(
            home_xg_per_match, home_xga_per_match,
            inputs['home_injuries'], inputs['home_rest'],
            home_data['form_trend'], is_home=True
        )
        
        away_xg_adj, away_xga_adj = self.apply_modifiers(
            away_xg_per_match, away_xga_per_match,
            inputs['away_injuries'], inputs['away_rest'],
            away_data['form_trend'], is_home=False
        )
        
        # Apply rest advantage
        rest_advantage = self.calculate_rest_advantage(inputs['home_rest'], inputs['away_rest'])
        home_xg_adj *= rest_advantage
        home_xga_adj /= rest_advantage
        
        # Calculate expected goals
        home_expected = (home_xg_adj + away_xga_adj) / 2
        away_expected = (away_xg_adj + home_xga_adj) / 2
        
        # Clamp values to reasonable ranges
        home_expected = max(0.1, min(4.0, home_expected))
        away_expected = max(0.1, min(4.0, away_expected))
        
        # Calculate probabilities
        probabilities = self.dixon_coles_probabilities(home_expected, away_expected)
        
        # Calculate enhanced confidence
        confidence, confidence_factors = self.calculate_confidence(
            home_xg_per_match, away_xg_per_match,
            home_xga_per_match, away_xga_per_match, inputs
        )
        
        # Calculate value bets
        odds = {
            'home': inputs['home_odds'],
            'draw': inputs['draw_odds'],
            'away': inputs['away_odds'],
            'over_2.5': inputs['over_odds']
        }
        value_bets = self.calculate_value_bets(probabilities, odds)
        
        # Generate insights and detect contradictions
        insights = self.generate_insights(inputs, probabilities, home_expected, away_expected, home_xg_per_match, away_xg_per_match, home_xga_per_match, away_xga_per_match)
        contradictions = self.detect_contradictions(inputs, probabilities, home_expected, away_expected)
        
        result = {
            'probabilities': probabilities,
            'expected_goals': {'home': home_expected, 'away': away_expected},
            'value_bets': value_bets,
            'confidence': confidence,
            'confidence_factors': confidence_factors,
            'insights': insights,
            'contradictions': contradictions,
            'per_match_stats': {
                'home_xg': home_xg_per_match,
                'home_xga': home_xga_per_match,
                'away_xg': away_xg_per_match,
                'away_xga': away_xga_per_match
            }
        }
        
        return result, errors, warnings

    def generate_insights(self, inputs, probabilities, home_expected, away_expected, home_xg_per_match, away_xg_per_match, home_xga_per_match, away_xga_per_match):
        """Generate enhanced insightful analysis"""
        insights = []
        
        # Home advantage insight
        home_adv = probabilities['home_win'] - probabilities['away_win']
        if home_adv > 0.15:
            insights.append(f"🏠 Strong home advantage for {inputs['home_team']} (+{home_adv:.1%})")
        elif home_adv < -0.1:
            insights.append(f"✈️ Strong away advantage for {inputs['away_team']} ({home_adv:+.1%})")
        
        # Enhanced injury impact
        home_injury_data = self.injury_weights[inputs['home_injuries']]
        away_injury_data = self.injury_weights[inputs['away_injuries']]
        
        if inputs['home_injuries'] != "None":
            insights.append(f"🩹 {inputs['home_team']} affected by {inputs['home_injuries'].lower()} ({home_injury_data['description']})")
        if inputs['away_injuries'] != "None":
            insights.append(f"🩹 {inputs['away_team']} affected by {inputs['away_injuries'].lower()} ({away_injury_data['description']})")
        
        # Enhanced fatigue analysis
        rest_diff = inputs['home_rest'] - inputs['away_rest']
        if rest_diff >= 3:
            insights.append(f"🕐 {inputs['home_team']} has {rest_diff} extra rest days (significant advantage)")
        elif rest_diff <= -3:
            insights.append(f"🕐 {inputs['away_team']} has {-rest_diff} extra rest days (significant advantage)")
        elif abs(rest_diff) >= 2:
            insights.append(f"⚖️ {abs(rest_diff)} day rest difference between teams")
        
        # Enhanced expected goals analysis
        total_goals = home_expected + away_expected
        if total_goals > 3.5:
            insights.append(f"⚽ Very high-scoring match expected ({total_goals:.2f} total xG)")
        elif total_goals > 3.0:
            insights.append(f"⚽ High-scoring match expected ({total_goals:.2f} total xG)")
        elif total_goals < 2.0:
            insights.append(f"🔒 Defensive battle anticipated ({total_goals:.2f} total xG)")
        
        # Enhanced form analysis
        if home_xg_per_match > 2.0:
            insights.append(f"📈 {inputs['home_team']} in strong attacking form ({home_xg_per_match:.2f} xG/match)")
        elif home_xg_per_match < 1.0:
            insights.append(f"📉 {inputs['home_team']} in poor attacking form ({home_xg_per_match:.2f} xG/match)")
            
        if away_xg_per_match > 2.0:
            insights.append(f"📈 {inputs['away_team']} in strong attacking form ({away_xg_per_match:.2f} xG/match)")
        elif away_xg_per_match < 1.0:
            insights.append(f"📉 {inputs['away_team']} in poor attacking form ({away_xg_per_match:.2f} xG/match)")
        
        # Enhanced defensive form analysis (FIXED: using xGA instead of xG)
        if home_xga_per_match < 1.0:
            insights.append(f"🛡️ {inputs['home_team']} showing excellent defense ({home_xga_per_match:.2f} xGA/match)")
        elif home_xga_per_match > 2.0:
            insights.append(f"🚨 {inputs['home_team']} defensive concerns ({home_xga_per_match:.2f} xGA/match)")
            
        if away_xga_per_match < 1.0:
            insights.append(f"🛡️ {inputs['away_team']} showing excellent defense ({away_xga_per_match:.2f} xGA/match)")
        elif away_xga_per_match > 2.0:
            insights.append(f"🚨 {inputs['away_team']} defensive concerns ({away_xga_per_match:.2f} xGA/match)")
        
        # Value bet insights
        excellent_bets = [k for k, v in probabilities.items() if isinstance(v, dict) and v.get('rating') == 'excellent']
        if excellent_bets:
            insights.append("💰 Excellent value betting opportunities identified")
        elif any(v.get('rating') == 'good' for k, v in probabilities.items() if isinstance(v, dict)):
            insights.append("💰 Good value betting opportunities available")
        
        return insights

def initialize_session_state():
    """Initialize session state variables"""
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'input_data' not in st.session_state:
        st.session_state.input_data = {}
    if 'show_edit' not in st.session_state:
        st.session_state.show_edit = False

def get_default_inputs():
    """Get default input values based on Understat format"""
    return {
        'home_team': 'Arsenal',
        'away_team': 'Liverpool',
        'home_xg_total': 10.25,
        'home_xga_total': 1.75,
        'away_xg_total': 11.25,
        'away_xga_total': 5.25,
        'home_injuries': 'None',
        'away_injuries': 'None',
        'home_rest': 7,
        'away_rest': 7,
        'home_odds': 2.50,
        'draw_odds': 3.40,
        'away_odds': 2.80,
        'over_odds': 1.90
    }

def display_understat_input_form(engine):
    """Display the main input form with Understat format"""
    st.markdown('<div class="main-header">🎯 Professional Football Prediction Engine</div>', unsafe_allow_html=True)
    
    # Use existing inputs or defaults
    if st.session_state.input_data:
        current_inputs = st.session_state.input_data
    else:
        current_inputs = get_default_inputs()
    
    st.markdown('<div class="section-header">🏆 Match Configuration</div>', unsafe_allow_html=True)
    
    # League and Team Selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🏠 Home Team")
        home_team = st.selectbox(
            "Select Home Team",
            list(engine.team_database.keys()),
            index=list(engine.team_database.keys()).index(current_inputs['home_team']),
            key="home_team_input"
        )
        home_data = engine.get_team_data(home_team)
        
        # Display team info
        st.write(f"**League:** {home_data['league']}")
        st.write(f"**Form Trend:** {'↗️ Improving' if home_data['form_trend'] > 0 else '↘️ Declining' if home_data['form_trend'] < 0 else '➡️ Stable'}")
        st.write(f"**Last 5 Opponents:** {', '.join(home_data['last_5_opponents'])}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("✈️ Away Team")
        away_team = st.selectbox(
            "Select Away Team",
            list(engine.team_database.keys()),
            index=list(engine.team_database.keys()).index(current_inputs['away_team']),
            key="away_team_input"
        )
        away_data = engine.get_team_data(away_team)
        
        # Display team info
        st.write(f"**League:** {away_data['league']}")
        st.write(f"**Form Trend:** {'↗️ Improving' if away_data['form_trend'] > 0 else '↘️ Declining' if away_data['form_trend'] < 0 else '➡️ Stable'}")
        st.write(f"**Last 5 Opponents:** {', '.join(away_data['last_5_opponents'])}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📊 Understat Last 5 Matches Data</div>', unsafe_allow_html=True)
    
    # Understat Format Explanation
    st.markdown("""
    <div class="warning-box">
    <strong>📝 Understat Format Guide:</strong><br>
    Enter data in the format shown on Understat.com: <strong>"10.25-1.75"</strong><br>
    - <strong>First number</strong>: Total xG scored in last 5 matches<br>
    - <strong>Second number</strong>: Total xGA conceded in last 5 matches<br>
    Example: Arsenal's "10.25-1.75" means 10.25 xG scored and 1.75 xGA conceded in last 5 matches.
    </div>
    """, unsafe_allow_html=True)
    
    # Understat Data Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader(f"📈 {home_team} - Last 5 Matches")
        
        # Understat format display
        current_home_format = f"{current_inputs['home_xg_total']}-{current_inputs['home_xga_total']}"
        st.markdown(f'<div class="understat-format">Understat Format: {current_home_format}</div>', unsafe_allow_html=True)
        
        col1a, col1b = st.columns(2)
        with col1a:
            home_xg_total = st.number_input(
                "Total xG Scored",
                min_value=0.0,
                max_value=25.0,
                value=current_inputs['home_xg_total'],
                step=0.1,
                key="home_xg_total_input",
                help="Total expected goals scored in last 5 matches (e.g., 10.25)"
            )
        with col1b:
            home_xga_total = st.number_input(
                "Total xGA Conceded",
                min_value=0.0,
                max_value=25.0,
                value=current_inputs['home_xga_total'],
                step=0.1,
                key="home_xga_total_input",
                help="Total expected goals against in last 5 matches (e.g., 1.75)"
            )
        
        # Calculate and show per-match averages
        home_xg_per_match = home_xg_total / 5
        home_xga_per_match = home_xga_total / 5
        
        st.metric("xG per match", f"{home_xg_per_match:.2f}")
        st.metric("xGA per match", f"{home_xga_per_match:.2f}")
        
        # Show comparison to database
        db_home_xg = home_data['last_5_xg_per_match']
        db_home_xga = home_data['last_5_xga_per_match']
        
        if home_xg_per_match != db_home_xg:
            diff = home_xg_per_match - db_home_xg
            color = "green" if diff > 0 else "red"
            st.markdown(f"<small style='color:{color}'>📊 {diff:+.2f} from database average</small>", unsafe_allow_html=True)
        
        if home_xga_per_match != db_home_xga:
            diff = home_xga_per_match - db_home_xga
            color = "red" if diff > 0 else "green"
            st.markdown(f"<small style='color:{color}'>🛡️ {diff:+.2f} from database average</small>", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader(f"📈 {away_team} - Last 5 Matches")
        
        # Understat format display
        current_away_format = f"{current_inputs['away_xg_total']}-{current_inputs['away_xga_total']}"
        st.markdown(f'<div class="understat-format">Understat Format: {current_away_format}</div>', unsafe_allow_html=True)
        
        col2a, col2b = st.columns(2)
        with col2a:
            away_xg_total = st.number_input(
                "Total xG Scored",
                min_value=0.0,
                max_value=25.0,
                value=current_inputs['away_xg_total'],
                step=0.1,
                key="away_xg_total_input",
                help="Total expected goals scored in last 5 matches (e.g., 11.25)"
            )
        with col2b:
            away_xga_total = st.number_input(
                "Total xGA Conceded",
                min_value=0.0,
                max_value=25.0,
                value=current_inputs['away_xga_total'],
                step=0.1,
                key="away_xga_total_input",
                help="Total expected goals against in last 5 matches (e.g., 5.25)"
            )
        
        # Calculate and show per-match averages
        away_xg_per_match = away_xg_total / 5
        away_xga_per_match = away_xga_total / 5
        
        st.metric("xG per match", f"{away_xg_per_match:.2f}")
        st.metric("xGA per match", f"{away_xga_per_match:.2f}")
        
        # Show comparison to database
        db_away_xg = away_data['last_5_xg_per_match']
        db_away_xga = away_data['last_5_xga_per_match']
        
        if away_xg_per_match != db_away_xg:
            diff = away_xg_per_match - db_away_xg
            color = "green" if diff > 0 else "red"
            st.markdown(f"<small style='color:{color}'>📊 {diff:+.2f} from database average</small>", unsafe_allow_html=True)
        
        if away_xga_per_match != db_away_xga:
            diff = away_xga_per_match - db_away_xga
            color = "red" if diff > 0 else "green"
            st.markdown(f"<small style='color:{color}'>🛡️ {diff:+.2f} from database average</small>", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🎭 Match Context</div>', unsafe_allow_html=True)
    
    # Context Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🩹 Injury Status")
        
        injury_options = ["None", "Minor (1-2 rotational)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ starters)"]
        
        home_injuries = st.selectbox(
            f"{home_team} Injuries",
            injury_options,
            index=injury_options.index(current_inputs['home_injuries']),
            key="home_injuries_input"
        )
        
        away_injuries = st.selectbox(
            f"{away_team} Injuries",
            injury_options,
            index=injury_options.index(current_inputs['away_injuries']),
            key="away_injuries_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🕐 Fatigue & Recovery")
        
        home_rest = st.number_input(
            f"{home_team} Rest Days",
            min_value=2,
            max_value=14,
            value=current_inputs['home_rest'],
            key="home_rest_input",
            help="Days since last match"
        )
        
        away_rest = st.number_input(
            f"{away_team} Rest Days",
            min_value=2,
            max_value=14,
            value=current_inputs['away_rest'],
            key="away_rest_input",
            help="Days since last match"
        )
        
        # Show rest comparison
        rest_diff = home_rest - away_rest
        if rest_diff > 0:
            st.success(f"🏠 {home_team} has {rest_diff} more rest days")
        elif rest_diff < 0:
            st.warning(f"✈️ {away_team} has {-rest_diff} more rest days")
        else:
            st.info("⚖️ Both teams have equal rest")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">💰 Market Odds</div>', unsafe_allow_html=True)
    
    # Odds Inputs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🏠 Home Win")
        home_odds = st.number_input(
            "Home Odds",
            min_value=1.01,
            max_value=100.0,
            value=current_inputs['home_odds'],
            step=0.1,
            key="home_odds_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🤝 Draw")
        draw_odds = st.number_input(
            "Draw Odds",
            min_value=1.01,
            max_value=100.0,
            value=current_inputs['draw_odds'],
            step=0.1,
            key="draw_odds_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("✈️ Away Win")
        away_odds = st.number_input(
            "Away Odds",
            min_value=1.01,
            max_value=100.0,
            value=current_inputs['away_odds'],
            step=0.1,
            key="away_odds_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("⚽ Over 2.5")
        over_odds = st.number_input(
            "Over 2.5 Odds",
            min_value=1.01,
            max_value=100.0,
            value=current_inputs['over_odds'],
            step=0.1,
            key="over_odds_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Compile all inputs
    inputs = {
        'home_team': home_team,
        'away_team': away_team,
        'home_xg_total': home_xg_total,
        'home_xga_total': home_xga_total,
        'away_xg_total': away_xg_total,
        'away_xga_total': away_xga_total,
        'home_injuries': home_injuries,
        'away_injuries': away_injuries,
        'home_rest': home_rest,
        'away_rest': away_rest,
        'home_odds': home_odds,
        'draw_odds': draw_odds,
        'away_odds': away_odds,
        'over_odds': over_odds
    }
    
    return inputs

def display_prediction_results(engine, result, inputs):
    """Display enhanced prediction results"""
    st.markdown('<div class="main-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    
    # Match header
    st.markdown(f'<h2 style="text-align: center; color: #1f77b4;">{inputs["home_team"]} vs {inputs["away_team"]}</h2>', unsafe_allow_html=True)
    
    # Expected score card
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    expected_home = result['expected_goals']['home']
    expected_away = result['expected_goals']['away']
    st.markdown(f'<h1 style="font-size: 4rem; margin: 1rem 0;">{expected_home:.1f} - {expected_away:.1f}</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem;">Expected Final Score</p>', unsafe_allow_html=True)
    
    # Enhanced confidence badge with factors
    confidence = result['confidence']
    confidence_stars = "★" * int(confidence / 20) + "☆" * (5 - int(confidence / 20))
    confidence_text = "Low" if confidence < 60 else "Medium" if confidence < 75 else "High" if confidence < 85 else "Very High"
    
    st.markdown(f'<div style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown(f'<span style="background: rgba(255,255,255,0.3); padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">Confidence: {confidence_stars} ({confidence:.0f}% - {confidence_text})</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show confidence factors on hover/expand
    with st.expander("Confidence Breakdown"):
        factors = result['confidence_factors']
        st.write("**Confidence Factors:**")
        for factor, value in factors.items():
            st.write(f"- {factor.replace('_', ' ').title()}: {value:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show contradictions if any
    if result['contradictions']:
        st.markdown('<div class="section-header">🚨 Contradiction Alerts</div>', unsafe_allow_html=True)
        for contradiction in result['contradictions']:
            st.markdown(f'<div class="contradiction-flag">{contradiction}</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    # Outcome Probabilities
    st.markdown('<div class="section-header">📊 Match Outcome Probabilities</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    probs = result['probabilities']
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        home_prob = probs['home_win']
        home_color = "🟢" if home_prob > 0.45 else "🟡" if home_prob > 0.35 else "🔴"
        st.metric(f"{home_color} {inputs['home_team']} Win", f"{home_prob:.1%}")
        st.write(f"**Odds:** {inputs['home_odds']:.2f}")
        
        value_home = result['value_bets']['home']
        _display_value_analysis(value_home)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        draw_prob = probs['draw']
        draw_color = "🟢" if draw_prob > 0.30 else "🟡" if draw_prob > 0.25 else "🔴"
        st.metric(f"{draw_color} Draw", f"{draw_prob:.1%}")
        st.write(f"**Odds:** {inputs['draw_odds']:.2f}")
        
        value_draw = result['value_bets']['draw']
        _display_value_analysis(value_draw)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        away_prob = probs['away_win']
        away_color = "🟢" if away_prob > 0.45 else "🟡" if away_prob > 0.35 else "🔴"
        st.metric(f"{away_color} {inputs['away_team']} Win", f"{away_prob:.1%}")
        st.write(f"**Odds:** {inputs['away_odds']:.2f}")
        
        value_away = result['value_bets']['away']
        _display_value_analysis(value_away)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Goals Market
    st.markdown('<div class="section-header">⚽ Goals Market</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        over_prob = probs['over_2.5']
        over_color = "🟢" if over_prob > 0.60 else "🟡" if over_prob > 0.50 else "🔴"
        st.metric(f"{over_color} Over 2.5 Goals", f"{over_prob:.1%}")
        st.write(f"**Odds:** {inputs['over_odds']:.2f}")
        
        value_over = result['value_bets']['over_2.5']
        _display_value_analysis(value_over)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        under_prob = probs['under_2.5']
        under_color = "🟢" if under_prob > 0.60 else "🟡" if under_prob > 0.50 else "🔴"
        st.metric(f"{under_color} Under 2.5 Goals", f"{under_prob:.1%}")
        st.write(f"**Implied Odds:** {1/under_prob:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommended Bets
    st.markdown('<div class="section-header">💰 Recommended Value Bets</div>', unsafe_allow_html=True)
    
    # Find best value bets
    value_bets = []
    for bet_type, data in result['value_bets'].items():
        if data['rating'] in ['excellent', 'good']:  # Only show good+ value bets
            value_bets.append({
                'type': bet_type,
                'value_ratio': data['value_ratio'],
                'ev': data['ev'],
                'odds': inputs[f"{bet_type}_odds" if bet_type != 'over_2.5' else 'over_odds'],
                'model_prob': data['model_prob'],
                'implied_prob': data['implied_prob'],
                'rating': data['rating']
            })
    
    # Sort by value ratio
    value_bets.sort(key=lambda x: x['value_ratio'], reverse=True)
    
    if value_bets:
        for bet in value_bets:
            if bet['rating'] == 'excellent':
                st.markdown('<div class="value-good">', unsafe_allow_html=True)
            else:
                st.markdown('<div class="value-poor">', unsafe_allow_html=True)
            
            bet_name = {
                'home': f"{inputs['home_team']} Win",
                'draw': "Draw",
                'away': f"{inputs['away_team']} Win", 
                'over_2.5': "Over 2.5 Goals"
            }[bet['type']]
            
            st.markdown(f"**✅ {bet_name} @ {bet['odds']:.2f}**")
            st.markdown(f"**Model Probability:** {bet['model_prob']:.1%} | **Market Implied:** {bet['implied_prob']:.1%}")
            st.markdown(f"**Value Ratio:** {bet['value_ratio']:.2f}x | **Expected Value:** {bet['ev']:.1%}")
            
            if bet['rating'] == 'excellent':
                st.markdown("**🎯 EXCELLENT VALUE BET**")
            else:
                st.markdown("**👍 GOOD VALUE OPPORTUNITY**")
                
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**No strong value bets identified**")
        st.markdown("All market odds appear efficient for this match. Consider waiting for line movement.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Insights
    st.markdown('<div class="section-header">🧠 Key Insights & Analysis</div>', unsafe_allow_html=True)
    
    for insight in result['insights']:
        st.markdown(f'<div class="metric-card">• {insight}</div>', unsafe_allow_html=True)
    
    # Additional statistical insights
    total_xg = expected_home + expected_away
    per_match = result['per_match_stats']
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**📈 Statistical Summary:**")
    st.markdown(f"- **Total Expected Goals:** {total_xg:.2f}")
    st.markdown(f"- **{inputs['home_team']} Form:** {per_match['home_xg']:.2f} xG, {per_match['home_xga']:.2f} xGA per match")
    st.markdown(f"- **{inputs['away_team']} Form:** {per_match['away_xg']:.2f} xG, {per_match['away_xga']:.2f} xGA per match")
    st.markdown(f"- **Goal Expectancy:** {total_xg/2.5:.1%} of average match")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 New Prediction", use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.input_data = {}
            st.rerun()
    
    with col2:
        if st.button("✏️ Edit Inputs", use_container_width=True, type="primary"):
            st.session_state.show_edit = True
            st.session_state.input_data = inputs
            st.rerun()
    
    with col3:
        if st.button("📊 Advanced Analytics", use_container_width=True):
            st.info("Advanced analytics feature coming soon!")

def _display_value_analysis(value_data):
    """Display value analysis for a bet"""
    if value_data['rating'] == 'excellent':
        st.success(f"**Excellent Value:** {value_data['value_ratio']:.2f}x")
    elif value_data['rating'] == 'good':
        st.info(f"**Good Value:** {value_data['value_ratio']:.2f}x")
    elif value_data['rating'] == 'fair':
        st.warning(f"**Fair Value:** {value_data['value_ratio']:.2f}x")
    else:
        st.error(f"**Poor Value:** {value_data['value_ratio']:.2f}x")
        
    st.write(f"**Expected Value:** {value_data['ev']:.1%}")

def main():
    """Main application function"""
    initialize_session_state()
    engine = ProfessionalPredictionEngine()
    
    # Show edit form if requested
    if st.session_state.show_edit:
        st.markdown('<div class="main-header">✏️ Edit Match Inputs</div>', unsafe_allow_html=True)
        inputs = display_understat_input_form(engine)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Prediction", use_container_width=True, type="primary"):
                result, errors, warnings = engine.predict_match(inputs)
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    if warnings:
                        for warning in warnings:
                            st.warning(f"⚠️ {warning}")
                    
                    st.session_state.prediction_result = result
                    st.session_state.input_data = inputs
                    st.session_state.show_edit = False
                    st.rerun()
            
            if st.button("← Back to Results", use_container_width=True):
                st.session_state.show_edit = False
                st.rerun()
    
    # Show prediction results if available
    elif st.session_state.prediction_result:
        display_prediction_results(engine, st.session_state.prediction_result, st.session_state.input_data)
    
    # Show main input form
    else:
        inputs = display_understat_input_form(engine)
        
        # Generate Prediction Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Prediction", use_container_width=True, type="primary", key="main_predict"):
                result, errors, warnings = engine.predict_match(inputs)
                
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    if warnings:
                        for warning in warnings:
                            st.warning(f"⚠️ {warning}")
                    
                    st.session_state.prediction_result = result
                    st.session_state.input_data = inputs
                    st.rerun()

if __name__ == "__main__":
    main()
