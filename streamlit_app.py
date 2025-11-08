import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# Clear cache to ensure fresh start
st.cache_data.clear()
st.cache_resource.clear()

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
    .disclaimer-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
    .kelly-recommendation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .bankroll-advice {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .debug-info {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .league-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .home-badge {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    .away-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalPredictionEngine:
    def __init__(self):
        # Injury impact weights
        self.injury_weights = {
            "None": {"attack_mult": 1.00, "defense_mult": 1.00, "description": "No impact"},
            "Minor (1-2 rotational)": {"attack_mult": 0.95, "defense_mult": 0.97, "description": "Slight impact"},
            "Moderate (1-2 key starters)": {"attack_mult": 0.88, "defense_mult": 0.90, "description": "Moderate impact"},
            "Significant (3-4 key players)": {"attack_mult": 0.78, "defense_mult": 0.82, "description": "Significant impact"},
            "Crisis (5+ starters)": {"attack_mult": 0.65, "defense_mult": 0.72, "description": "Severe impact"}
        }
        
        # Fatigue multipliers
        self.fatigue_multipliers = {
            2: 0.85, 3: 0.88, 4: 0.91, 5: 0.94, 6: 0.96, 
            7: 0.98, 8: 1.00, 9: 1.01, 10: 1.02, 11: 1.03,
            12: 1.03, 13: 1.03, 14: 1.03
        }
        
        # Team database with PROPER home/away separation
        self.team_database = self._initialize_complete_database()
        self.leagues = self._get_available_leagues()

    def _get_available_leagues(self):
        """Get list of available leagues from the database"""
        leagues = set()
        for team_data in self.team_database.values():
            leagues.add(team_data["league"])
        return sorted(list(leagues))

    def _initialize_complete_database(self):
        """Initialize complete database with PROPER home/away separation"""
        return {
            # Premier League - Home Data
            "Arsenal Home": {"league": "Premier League", "last_5_xg_total": 10.25, "last_5_xga_total": 1.75, "form_trend": 0.08},
            "Bournemouth Home": {"league": "Premier League", "last_5_xg_total": 5.77, "last_5_xga_total": 2.30, "form_trend": 0.12},
            "Manchester City Home": {"league": "Premier League", "last_5_xg_total": 11.44, "last_5_xga_total": 5.00, "form_trend": 0.15},
            "Manchester United Home": {"league": "Premier League", "last_5_xg_total": 10.64, "last_5_xga_total": 4.88, "form_trend": -0.05},
            "Liverpool Home": {"league": "Premier League", "last_5_xg_total": 7.95, "last_5_xga_total": 5.75, "form_trend": 0.10},
            "Brighton Home": {"league": "Premier League", "last_5_xg_total": 8.55, "last_5_xga_total": 5.87, "form_trend": 0.03},
            "Fulham Home": {"league": "Premier League", "last_5_xg_total": 6.73, "last_5_xga_total": 5.49, "form_trend": 0.04},
            "Brentford Home": {"league": "Premier League", "last_5_xg_total": 9.06, "last_5_xga_total": 7.18, "form_trend": -0.05},
            "Aston Villa Home": {"league": "Premier League", "last_5_xg_total": 5.47, "last_5_xga_total": 6.82, "form_trend": 0.10},
            "Crystal Palace Home": {"league": "Premier League", "last_5_xg_total": 11.69, "last_5_xga_total": 7.65, "form_trend": -0.08},
            "Newcastle United Home": {"league": "Premier League", "last_5_xg_total": 10.48, "last_5_xga_total": 4.95, "form_trend": 0.03},
            "Everton Home": {"league": "Premier League", "last_5_xg_total": 8.15, "last_5_xga_total": 7.49, "form_trend": 0.02},
            "Burnley Home": {"league": "Premier League", "last_5_xg_total": 3.50, "last_5_xga_total": 9.59, "form_trend": -0.15},
            "Chelsea Home": {"league": "Premier League", "last_5_xg_total": 8.28, "last_5_xga_total": 7.86, "form_trend": 0.06},
            "Tottenham Home": {"league": "Premier League", "last_5_xg_total": 4.63, "last_5_xga_total": 7.71, "form_trend": -0.02},
            "Nottingham Forest Home": {"league": "Premier League", "last_5_xg_total": 7.50, "last_5_xga_total": 8.98, "form_trend": -0.10},
            "West Ham Home": {"league": "Premier League", "last_5_xg_total": 3.91, "last_5_xga_total": 11.79, "form_trend": -0.08},
            "Sunderland Home": {"league": "Premier League", "last_5_xg_total": 5.44, "last_5_xga_total": 4.63, "form_trend": 0.06},
            "Leeds Home": {"league": "Premier League", "last_5_xg_total": 8.81, "last_5_xga_total": 3.23, "form_trend": 0.02},
            "Wolverhampton Wanderers Home": {"league": "Premier League", "last_5_xg_total": 5.98, "last_5_xga_total": 7.73, "form_trend": 0.01},

            # Premier League - Away Data
            "Arsenal Away": {"league": "Premier League", "last_5_xg_total": 8.48, "last_5_xga_total": 3.78, "form_trend": 0.08},
            "Bournemouth Away": {"league": "Premier League", "last_5_xg_total": 7.70, "last_5_xga_total": 11.60, "form_trend": 0.12},
            "Manchester City Away": {"league": "Premier League", "last_5_xg_total": 8.02, "last_5_xga_total": 4.98, "form_trend": 0.15},
            "Manchester United Away": {"league": "Premier League", "last_5_xg_total": 8.75, "last_5_xga_total": 10.60, "form_trend": -0.05},
            "Liverpool Away": {"league": "Premier League", "last_5_xg_total": 10.72, "last_5_xga_total": 10.17, "form_trend": 0.10},
            "Brighton Away": {"league": "Premier League", "last_5_xg_total": 7.52, "last_5_xga_total": 7.60, "form_trend": 0.03},
            "Fulham Away": {"league": "Premier League", "last_5_xg_total": 4.96, "last_5_xga_total": 9.34, "form_trend": 0.04},
            "Brentford Away": {"league": "Premier League", "last_5_xg_total": 7.88, "last_5_xga_total": 5.85, "form_trend": -0.05},
            "Aston Villa Away": {"league": "Premier League", "last_5_xg_total": 3.06, "last_5_xga_total": 6.95, "form_trend": 0.10},
            "Crystal Palace Away": {"league": "Premier League", "last_5_xg_total": 8.50, "last_5_xga_total": 6.04, "form_trend": -0.08},
            "Newcastle United Away": {"league": "Premier League", "last_5_xg_total": 3.66, "last_5_xga_total": 3.82, "form_trend": 0.03},
            "Everton Away": {"league": "Premier League", "last_5_xg_total": 5.24, "last_5_xga_total": 8.05, "form_trend": 0.02},
            "Burnley Away": {"league": "Premier League", "last_5_xg_total": 4.62, "last_5_xga_total": 13.54, "form_trend": -0.15},
            "Chelsea Away": {"league": "Premier League", "last_5_xg_total": 11.08, "last_5_xga_total": 5.70, "form_trend": 0.06},
            "Tottenham Away": {"league": "Premier League", "last_5_xg_total": 6.77, "last_5_xga_total": 5.23, "form_trend": -0.02},
            "Nottingham Forest Away": {"league": "Premier League", "last_5_xg_total": 3.10, "last_5_xga_total": 9.60, "form_trend": -0.10},
            "West Ham Away": {"league": "Premier League", "last_5_xg_total": 4.93, "last_5_xga_total": 7.63, "form_trend": -0.08},
            "Sunderland Away": {"league": "Premier League", "last_5_xg_total": 5.77, "last_5_xga_total": 7.33, "form_trend": 0.06},
            "Leeds Away": {"league": "Premier League", "last_5_xg_total": 4.13, "last_5_xga_total": 8.59, "form_trend": 0.02},
            "Wolverhampton Wanderers Away": {"league": "Premier League", "last_5_xg_total": 2.80, "last_5_xga_total": 6.73, "form_trend": 0.01},

            # La Liga - Home Data
            "Real Madrid Home": {"league": "La Liga", "last_5_xg_total": 15.97, "last_5_xga_total": 4.09, "form_trend": 0.11},
            "Villarreal Home": {"league": "La Liga", "last_5_xg_total": 14.71, "last_5_xga_total": 6.50, "form_trend": 0.13},
            "Atletico Madrid Home": {"league": "La Liga", "last_5_xg_total": 16.16, "last_5_xga_total": 4.99, "form_trend": 0.09},
            "Barcelona Home": {"league": "La Liga", "last_5_xg_total": 14.66, "last_5_xga_total": 3.54, "form_trend": 0.07},
            "Espanyol Home": {"league": "La Liga", "last_5_xg_total": 12.54, "last_5_xga_total": 8.27, "form_trend": 0.10},
            "Real Betis Home": {"league": "La Liga", "last_5_xg_total": 10.74, "last_5_xga_total": 5.36, "form_trend": 0.05},
            "Elche Home": {"league": "La Liga", "last_5_xg_total": 9.67, "last_5_xga_total": 6.21, "form_trend": 0.03},
            "Alaves Home": {"league": "La Liga", "last_5_xg_total": 9.58, "last_5_xga_total": 4.62, "form_trend": -0.02},
            "Osasuna Home": {"league": "La Liga", "last_5_xg_total": 6.53, "last_5_xga_total": 5.02, "form_trend": -0.04},
            "Real Sociedad Home": {"league": "La Liga", "last_5_xg_total": 11.66, "last_5_xga_total": 7.12, "form_trend": -0.03},
            "Athletic Club Home": {"league": "La Liga", "last_5_xg_total": 9.35, "last_5_xga_total": 3.56, "form_trend": -0.01},
            "Getafe Home": {"league": "La Liga", "last_5_xg_total": 4.29, "last_5_xga_total": 6.55, "form_trend": -0.08},
            "Valencia Home": {"league": "La Liga", "last_5_xg_total": 7.50, "last_5_xga_total": 5.93, "form_trend": -0.10},
            "Mallorca Home": {"league": "La Liga", "last_5_xg_total": 4.64, "last_5_xga_total": 7.95, "form_trend": -0.06},
            "Rayo Vallecano Home": {"league": "La Liga", "last_5_xg_total": 6.63, "last_5_xga_total": 5.63, "form_trend": -0.12},
            "Celta Vigo Home": {"league": "La Liga", "last_5_xg_total": 7.32, "last_5_xga_total": 7.44, "form_trend": -0.14},
            "Girona Home": {"league": "La Liga", "last_5_xg_total": 7.71, "last_5_xga_total": 14.78, "form_trend": -0.18},
            "Sevilla Home": {"league": "La Liga", "last_5_xg_total": 6.17, "last_5_xga_total": 8.95, "form_trend": 0.00},
            "Real Oviedo Home": {"league": "La Liga", "last_5_xg_total": 5.26, "last_5_xga_total": 11.52, "form_trend": 0.00},
            "Levante Home": {"league": "La Liga", "last_5_xg_total": 7.59, "last_5_xga_total": 13.20, "form_trend": 0.00},

            # La Liga - Away Data
            "Real Madrid Away": {"league": "La Liga", "last_5_xg_total": 11.60, "last_5_xga_total": 8.51, "form_trend": 0.11},
            "Villarreal Away": {"league": "La Liga", "last_5_xg_total": 7.62, "last_5_xga_total": 8.19, "form_trend": 0.13},
            "Atletico Madrid Away": {"league": "La Liga", "last_5_xg_total": 5.93, "last_5_xga_total": 6.41, "form_trend": 0.09},
            "Barcelona Away": {"league": "La Liga", "last_5_xg_total": 12.54, "last_5_xga_total": 12.85, "form_trend": 0.07},
            "Espanyol Away": {"league": "La Liga", "last_5_xg_total": 9.30, "last_5_xga_total": 7.96, "form_trend": 0.10},
            "Real Betis Away": {"league": "La Liga", "last_5_xg_total": 6.84, "last_5_xga_total": 5.96, "form_trend": 0.05},
            "Elche Away": {"league": "La Liga", "last_5_xg_total": 4.47, "last_5_xga_total": 10.56, "form_trend": 0.03},
            "Alaves Away": {"league": "La Liga", "last_5_xg_total": 4.72, "last_5_xga_total": 6.18, "form_trend": -0.02},
            "Osasuna Away": {"league": "La Liga", "last_5_xg_total": 5.72, "last_5_xga_total": 10.51, "form_trend": -0.04},
            "Real Sociedad Away": {"league": "La Liga", "last_5_xg_total": 7.14, "last_5_xga_total": 9.11, "form_trend": -0.03},
            "Athletic Club Away": {"league": "La Liga", "last_5_xg_total": 4.54, "last_5_xga_total": 7.26, "form_trend": -0.01},
            "Getafe Away": {"league": "La Liga", "last_5_xg_total": 5.66, "last_5_xga_total": 7.14, "form_trend": -0.08},
            "Valencia Away": {"league": "La Liga", "last_5_xg_total": 4.56, "last_5_xga_total": 12.84, "form_trend": -0.10},
            "Mallorca Away": {"league": "La Liga", "last_5_xg_total": 6.14, "last_5_xga_total": 13.16, "form_trend": -0.06},
            "Rayo Vallecano Away": {"league": "La Liga", "last_5_xg_total": 10.65, "last_5_xga_total": 13.26, "form_trend": -0.12},
            "Celta Vigo Away": {"league": "La Liga", "last_5_xg_total": 6.49, "last_5_xga_total": 9.57, "form_trend": -0.14},
            "Girona Away": {"league": "La Liga", "last_5_xg_total": 5.65, "last_5_xga_total": 10.63, "form_trend": -0.18},
            "Sevilla Away": {"league": "La Liga", "last_5_xg_total": 4.62, "last_5_xga_total": 12.12, "form_trend": 0.00},
            "Real Oviedo Away": {"league": "La Liga", "last_5_xg_total": 5.55, "last_5_xga_total": 9.10, "form_trend": 0.00},
            "Levante Away": {"league": "La Liga", "last_5_xg_total": 11.47, "last_5_xga_total": 7.38, "form_trend": 0.00},

            # Bundesliga - Home Data
            "Bayern Munich Home": {"league": "Bundesliga", "last_5_xg_total": 15.76, "last_5_xga_total": 3.17, "form_trend": 0.11},
            "RasenBallsport Leipzig Home": {"league": "Bundesliga", "last_5_xg_total": 9.98, "last_5_xga_total": 4.73, "form_trend": 0.13},
            "VfB Stuttgart Home": {"league": "Bundesliga", "last_5_xg_total": 6.49, "last_5_xga_total": 4.36, "form_trend": 0.09},
            "Bayer Leverkusen Home": {"league": "Bundesliga", "last_5_xg_total": 11.14, "last_5_xga_total": 4.34, "form_trend": 0.07},
            "Borussia Dortmund Home": {"league": "Bundesliga", "last_5_xg_total": 5.83, "last_5_xga_total": 3.79, "form_trend": 0.10},
            "Werder Bremen Home": {"league": "Bundesliga", "last_5_xg_total": 8.22, "last_5_xga_total": 6.48, "form_trend": 0.05},
            "Union Berlin Home": {"league": "Bundesliga", "last_5_xg_total": 8.03, "last_5_xga_total": 8.05, "form_trend": 0.03},
            "FC Cologne Home": {"league": "Bundesliga", "last_5_xg_total": 7.32, "last_5_xga_total": 5.45, "form_trend": -0.02},
            "Eintracht Frankfurt Home": {"league": "Bundesliga", "last_5_xg_total": 7.85, "last_5_xga_total": 5.57, "form_trend": -0.03},
            "Freiburg Home": {"league": "Bundesliga", "last_5_xg_total": 6.34, "last_5_xga_total": 4.87, "form_trend": -0.01},
            "FC Heidenheim Home": {"league": "Bundesliga", "last_5_xg_total": 6.78, "last_5_xga_total": 9.13, "form_trend": -0.08},
            "Hoffenheim Home": {"league": "Bundesliga", "last_5_xg_total": 6.17, "last_5_xga_total": 6.75, "form_trend": -0.06},
            "Augsburg Home": {"league": "Bundesliga", "last_5_xg_total": 4.74, "last_5_xga_total": 10.45, "form_trend": -0.12},
            "Wolfsburg Home": {"league": "Bundesliga", "last_5_xg_total": 6.81, "last_5_xga_total": 10.29, "form_trend": -0.14},
            "Borussia M.Gladbach Home": {"league": "Bundesliga", "last_5_xg_total": 5.96, "last_5_xga_total": 9.66, "form_trend": -0.18},
            "Hamburger SV Home": {"league": "Bundesliga", "last_5_xg_total": 8.13, "last_5_xga_total": 5.78, "form_trend": -0.04},
            "St. Pauli Home": {"league": "Bundesliga", "last_5_xg_total": 5.91, "last_5_xga_total": 9.74, "form_trend": -0.10},
            "Mainz 05 Home": {"league": "Bundesliga", "last_5_xg_total": 6.14, "last_5_xga_total": 10.22, "form_trend": 0.00},

            # Bundesliga - Away Data
            "Bayern Munich Away": {"league": "Bundesliga", "last_5_xg_total": 10.97, "last_5_xga_total": 2.86, "form_trend": 0.11},
            "RasenBallsport Leipzig Away": {"league": "Bundesliga", "last_5_xg_total": 8.47, "last_5_xga_total": 7.97, "form_trend": 0.13},
            "VfB Stuttgart Away": {"league": "Bundesliga", "last_5_xg_total": 8.02, "last_5_xga_total": 5.42, "form_trend": 0.09},
            "Bayer Leverkusen Away": {"league": "Bundesliga", "last_5_xg_total": 8.14, "last_5_xga_total": 7.05, "form_trend": 0.07},
            "Borussia Dortmund Away": {"league": "Bundesliga", "last_5_xg_total": 8.71, "last_5_xga_total": 6.08, "form_trend": 0.10},
            "Werder Bremen Away": {"league": "Bundesliga", "last_5_xg_total": 5.87, "last_5_xga_total": 13.99, "form_trend": 0.05},
            "Union Berlin Away": {"league": "Bundesliga", "last_5_xg_total": 4.89, "last_5_xga_total": 8.08, "form_trend": 0.03},
            "FC Cologne Away": {"league": "Bundesliga", "last_5_xg_total": 6.78, "last_5_xga_total": 10.43, "form_trend": -0.02},
            "Eintracht Frankfurt Away": {"league": "Bundesliga", "last_5_xg_total": 8.40, "last_5_xga_total": 7.69, "form_trend": -0.03},
            "Freiburg Away": {"league": "Bundesliga", "last_5_xg_total": 5.69, "last_5_xga_total": 7.40, "form_trend": -0.01},
            "FC Heidenheim Away": {"league": "Bundesliga", "last_5_xg_total": 4.80, "last_5_xga_total": 10.15, "form_trend": -0.08},
            "Hoffenheim Away": {"league": "Bundesliga", "last_5_xg_total": 9.62, "last_5_xga_total": 7.67, "form_trend": -0.06},
            "Augsburg Away": {"league": "Bundesliga", "last_5_xg_total": 3.98, "last_5_xga_total": 6.54, "form_trend": -0.12},
            "Wolfsburg Away": {"league": "Bundesliga", "last_5_xg_total": 6.15, "last_5_xga_total": 7.88, "form_trend": -0.14},
            "Borussia M.Gladbach Away": {"league": "Bundesliga", "last_5_xg_total": 7.16, "last_5_xga_total": 5.21, "form_trend": -0.18},
            "Hamburger SV Away": {"league": "Bundesliga", "last_5_xg_total": 4.44, "last_5_xga_total": 10.63, "form_trend": -0.04},
            "St. Pauli Away": {"league": "Bundesliga", "last_5_xg_total": 4.34, "last_5_xga_total": 7.74, "form_trend": -0.10},
            "Mainz 05 Away": {"league": "Bundesliga", "last_5_xg_total": 6.39, "last_5_xga_total": 4.81, "form_trend": 0.00},

            # Serie A - Home Data
            "Napoli Home": {"league": "Serie A", "last_5_xg_total": 6.75, "last_5_xga_total": 7.10, "form_trend": 0.11},
            "AC Milan Home": {"league": "Serie A", "last_5_xg_total": 11.95, "last_5_xga_total": 6.95, "form_trend": 0.13},
            "Inter Home": {"league": "Serie A", "last_5_xg_total": 14.93, "last_5_xga_total": 4.23, "form_trend": 0.09},
            "Juventus Home": {"league": "Serie A", "last_5_xg_total": 9.23, "last_5_xga_total": 4.88, "form_trend": 0.10},
            "Lazio Home": {"league": "Serie A", "last_5_xg_total": 7.58, "last_5_xga_total": 6.74, "form_trend": 0.05},
            "Bologna Home": {"league": "Serie A", "last_5_xg_total": 7.34, "last_5_xga_total": 2.14, "form_trend": 0.03},
            "Roma Home": {"league": "Serie A", "last_5_xg_total": 7.12, "last_5_xga_total": 4.74, "form_trend": -0.02},
            "Udinese Home": {"league": "Serie A", "last_5_xg_total": 6.56, "last_5_xga_total": 3.93, "form_trend": -0.04},
            "Torino Home": {"league": "Serie A", "last_5_xg_total": 7.32, "last_5_xga_total": 8.47, "form_trend": -0.03},
            "Atalanta Home": {"league": "Serie A", "last_5_xg_total": 10.08, "last_5_xga_total": 3.37, "form_trend": -0.01},
            "Sassuolo Home": {"league": "Serie A", "last_5_xg_total": 6.09, "last_5_xga_total": 7.70, "form_trend": -0.10},
            "Cagliari Home": {"league": "Serie A", "last_5_xg_total": 4.97, "last_5_xga_total": 9.15, "form_trend": -0.14},
            "Verona Home": {"league": "Serie A", "last_5_xg_total": 6.77, "last_5_xga_total": 4.13, "form_trend": -0.18},
            "Lecce Home": {"league": "Serie A", "last_5_xg_total": 5.41, "last_5_xga_total": 7.63, "form_trend": 0.00},
            "Genoa Home": {"league": "Serie A", "last_5_xg_total": 6.20, "last_5_xga_total": 7.28, "form_trend": 0.00},
            "Como Home": {"league": "Serie A", "last_5_xg_total": 7.72, "last_5_xga_total": 5.97, "form_trend": 0.07},
            "Cremonese Home": {"league": "Serie A", "last_5_xg_total": 6.51, "last_5_xga_total": 7.43, "form_trend": -0.08},
            "Pisa Home": {"league": "Serie A", "last_5_xg_total": 5.32, "last_5_xga_total": 7.07, "form_trend": -0.06},
            "Parma Calcio 1913 Home": {"league": "Serie A", "last_5_xg_total": 5.47, "last_5_xga_total": 6.39, "form_trend": -0.12},
            "Fiorentina Home": {"league": "Serie A", "last_5_xg_total": 8.20, "last_5_xga_total": 9.08, "form_trend": 0.00},

            # Serie A - Away Data
            "Napoli Away": {"league": "Serie A", "last_5_xg_total": 9.99, "last_5_xga_total": 5.58, "form_trend": 0.11},
            "AC Milan Away": {"league": "Serie A", "last_5_xg_total": 6.02, "last_5_xga_total": 2.63, "form_trend": 0.13},
            "Inter Away": {"league": "Serie A", "last_5_xg_total": 9.16, "last_5_xga_total": 4.96, "form_trend": 0.09},
            "Juventus Away": {"league": "Serie A", "last_5_xg_total": 6.85, "last_5_xga_total": 5.46, "form_trend": 0.10},
            "Lazio Away": {"league": "Serie A", "last_5_xg_total": 5.48, "last_5_xga_total": 7.14, "form_trend": 0.05},
            "Bologna Away": {"league": "Serie A", "last_5_xg_total": 7.79, "last_5_xga_total": 9.59, "form_trend": 0.03},
            "Roma Away": {"league": "Serie A", "last_5_xg_total": 7.62, "last_5_xga_total": 7.16, "form_trend": -0.02},
            "Udinese Away": {"league": "Serie A", "last_5_xg_total": 5.69, "last_5_xga_total": 9.80, "form_trend": -0.04},
            "Torino Away": {"league": "Serie A", "last_5_xg_total": 5.26, "last_5_xga_total": 6.87, "form_trend": -0.03},
            "Atalanta Away": {"league": "Serie A", "last_5_xg_total": 5.65, "last_5_xga_total": 5.58, "form_trend": -0.01},
            "Sassuolo Away": {"league": "Serie A", "last_5_xg_total": 5.40, "last_5_xga_total": 8.98, "form_trend": -0.10},
            "Cagliari Away": {"league": "Serie A", "last_5_xg_total": 5.97, "last_5_xga_total": 8.48, "form_trend": -0.14},
            "Verona Away": {"league": "Serie A", "last_5_xg_total": 6.11, "last_5_xga_total": 8.64, "form_trend": -0.18},
            "Lecce Away": {"league": "Serie A", "last_5_xg_total": 3.40, "last_5_xga_total": 7.18, "form_trend": 0.00},
            "Genoa Away": {"league": "Serie A", "last_5_xg_total": 7.84, "last_5_xga_total": 8.37, "form_trend": 0.00},
            "Como Away": {"league": "Serie A", "last_5_xg_total": 5.13, "last_5_xga_total": 6.01, "form_trend": 0.07},
            "Cremonese Away": {"league": "Serie A", "last_5_xg_total": 4.52, "last_5_xga_total": 12.87, "form_trend": -0.08},
            "Pisa Away": {"league": "Serie A", "last_5_xg_total": 6.70, "last_5_xga_total": 10.13, "form_trend": -0.06},
            "Parma Calcio 1913 Away": {"league": "Serie A", "last_5_xg_total": 4.91, "last_5_xga_total": 8.06, "form_trend": -0.12},
            "Fiorentina Away": {"league": "Serie A", "last_5_xg_total": 4.88, "last_5_xga_total": 8.04, "form_trend": 0.00},

            # Ligue 1 - Home Data
            "Lens Home": {"league": "Ligue 1", "last_5_xg_total": 17.18, "last_5_xga_total": 7.81, "form_trend": 0.11},
            "Marseille Home": {"league": "Ligue 1", "last_5_xg_total": 13.74, "last_5_xga_total": 3.51, "form_trend": 0.13},
            "Lille Home": {"league": "Ligue 1", "last_5_xg_total": 13.19, "last_5_xga_total": 6.33, "form_trend": 0.09},
            "Monaco Home": {"league": "Ligue 1", "last_5_xg_total": 14.38, "last_5_xga_total": 7.01, "form_trend": 0.07},
            "Paris Saint Germain Home": {"league": "Ligue 1", "last_5_xg_total": 10.01, "last_5_xga_total": 4.44, "form_trend": 0.10},
            "Nice Home": {"league": "Ligue 1", "last_5_xg_total": 10.24, "last_5_xga_total": 8.64, "form_trend": 0.05},
            "Strasbourg Home": {"league": "Ligue 1", "last_5_xg_total": 12.73, "last_5_xga_total": 3.93, "form_trend": 0.03},
            "Lyon Home": {"league": "Ligue 1", "last_5_xg_total": 10.28, "last_5_xga_total": 4.24, "form_trend": -0.02},
            "Rennes Home": {"league": "Ligue 1", "last_5_xg_total": 11.65, "last_5_xga_total": 9.97, "form_trend": -0.04},
            "Toulouse Home": {"league": "Ligue 1", "last_5_xg_total": 11.07, "last_5_xga_total": 7.53, "form_trend": -0.03},
            "Le Havre Home": {"league": "Ligue 1", "last_5_xg_total": 8.57, "last_5_xga_total": 4.86, "form_trend": -0.01},
            "Brest Home": {"league": "Ligue 1", "last_5_xg_total": 11.15, "last_5_xga_total": 9.51, "form_trend": -0.12},
            "Angers Home": {"league": "Ligue 1", "last_5_xg_total": 5.79, "last_5_xga_total": 7.44, "form_trend": -0.08},
            "Lorient Home": {"league": "Ligue 1", "last_5_xg_total": 8.48, "last_5_xga_total": 7.53, "form_trend": -0.10},
            "Paris FC Home": {"league": "Ligue 1", "last_5_xg_total": 10.90, "last_5_xga_total": 8.45, "form_trend": -0.06},
            "Auxerre Home": {"league": "Ligue 1", "last_5_xg_total": 6.21, "last_5_xga_total": 7.67, "form_trend": -0.14},
            "Metz Home": {"league": "Ligue 1", "last_5_xg_total": 5.62, "last_5_xga_total": 7.02, "form_trend": -0.18},
            "Nantes Home": {"league": "Ligue 1", "last_5_xg_total": 7.17, "last_5_xga_total": 12.13, "form_trend": 0.00},

            # Ligue 1 - Away Data
            "Lens Away": {"league": "Ligue 1", "last_5_xg_total": 5.74, "last_5_xga_total": 9.07, "form_trend": 0.11},
            "Marseille Away": {"league": "Ligue 1", "last_5_xg_total": 9.11, "last_5_xga_total": 8.69, "form_trend": 0.13},
            "Lille Away": {"league": "Ligue 1", "last_5_xg_total": 10.87, "last_5_xga_total": 8.99, "form_trend": 0.09},
            "Monaco Away": {"league": "Ligue 1", "last_5_xg_total": 9.80, "last_5_xga_total": 9.12, "form_trend": 0.07},
            "Paris Saint Germain Away": {"league": "Ligue 1", "last_5_xg_total": 10.70, "last_5_xga_total": 6.96, "form_trend": 0.10},
            "Nice Away": {"league": "Ligue 1", "last_5_xg_total": 5.38, "last_5_xga_total": 9.72, "form_trend": 0.05},
            "Strasbourg Away": {"league": "Ligue 1", "last_5_xg_total": 9.78, "last_5_xga_total": 11.51, "form_trend": 0.03},
            "Lyon Away": {"league": "Ligue 1", "last_5_xg_total": 8.26, "last_5_xga_total": 9.12, "form_trend": -0.02},
            "Rennes Away": {"league": "Ligue 1", "last_5_xg_total": 7.09, "last_5_xga_total": 13.52, "form_trend": -0.04},
            "Toulouse Away": {"league": "Ligue 1", "last_5_xg_total": 5.19, "last_5_xga_total": 9.05, "form_trend": -0.03},
            "Le Havre Away": {"league": "Ligue 1", "last_5_xg_total": 4.30, "last_5_xga_total": 9.56, "form_trend": -0.01},
            "Brest Away": {"league": "Ligue 1", "last_5_xg_total": 4.99, "last_5_xga_total": 8.39, "form_trend": -0.12},
            "Angers Away": {"league": "Ligue 1", "last_5_xg_total": 4.41, "last_5_xga_total": 14.13, "form_trend": -0.08},
            "Lorient Away": {"league": "Ligue 1", "last_5_xg_total": 8.14, "last_5_xga_total": 12.92, "form_trend": -0.10},
            "Paris FC Away": {"league": "Ligue 1", "last_5_xg_total": 6.87, "last_5_xga_total": 10.93, "form_trend": -0.06},
            "Auxerre Away": {"league": "Ligue 1", "last_5_xg_total": 6.44, "last_5_xga_total": 11.24, "form_trend": -0.14},
            "Metz Away": {"league": "Ligue 1", "last_5_xg_total": 6.54, "last_5_xga_total": 16.91, "form_trend": -0.18},
            "Nantes Away": {"league": "Ligue 1", "last_5_xg_total": 4.44, "last_5_xga_total": 8.55, "form_trend": 0.00},

            # Russian Premier League - Home Data
            "CSKA Moscow Home": {"league": "RFPL", "last_5_xg_total": 14.39, "last_5_xga_total": 7.32, "form_trend": 0.11},
            "Zenit St. Petersburg Home": {"league": "RFPL", "last_5_xg_total": 13.87, "last_5_xga_total": 4.95, "form_trend": 0.13},
            "FC Krasnodar Home": {"league": "RFPL", "last_5_xg_total": 13.37, "last_5_xga_total": 7.15, "form_trend": 0.09},
            "Spartak Moscow Home": {"league": "RFPL", "last_5_xg_total": 10.73, "last_5_xga_total": 6.13, "form_trend": 0.07},
            "Lokomotiv Moscow Home": {"league": "RFPL", "last_5_xg_total": 11.88, "last_5_xga_total": 5.92, "form_trend": 0.10},
            "Rubin Kazan Home": {"league": "RFPL", "last_5_xg_total": 8.50, "last_5_xga_total": 7.24, "form_trend": 0.05},
            "Baltika Home": {"league": "RFPL", "last_5_xg_total": 8.76, "last_5_xga_total": 4.39, "form_trend": 0.03},
            "Dinamo Moscow Home": {"league": "RFPL", "last_5_xg_total": 14.39, "last_5_xga_total": 11.36, "form_trend": -0.03},
            "FC Rostov Home": {"league": "RFPL", "last_5_xg_total": 8.61, "last_5_xga_total": 6.68, "form_trend": -0.01},
            "Dynamo Makhachkala Home": {"league": "RFPL", "last_5_xg_total": 7.88, "last_5_xga_total": 4.78, "form_trend": -0.02},
            "FK Akhmat Home": {"league": "RFPL", "last_5_xg_total": 8.36, "last_5_xga_total": 5.35, "form_trend": -0.04},
            "FC Orenburg Home": {"league": "RFPL", "last_5_xg_total": 8.20, "last_5_xga_total": 9.42, "form_trend": -0.10},
            "Nizhny Novgorod Home": {"league": "RFPL", "last_5_xg_total": 8.22, "last_5_xga_total": 8.82, "form_trend": -0.06},
            "Akron Home": {"league": "RFPL", "last_5_xg_total": 8.18, "last_5_xga_total": 9.35, "form_trend": -0.12},
            "PFC Sochi Home": {"league": "RFPL", "last_5_xg_total": 3.31, "last_5_xga_total": 11.51, "form_trend": -0.14},

            # Russian Premier League - Away Data
            "CSKA Moscow Away": {"league": "RFPL", "last_5_xg_total": 7.35, "last_5_xga_total": 8.30, "form_trend": 0.11},
            "Zenit St. Petersburg Away": {"league": "RFPL", "last_5_xg_total": 8.41, "last_5_xga_total": 6.59, "form_trend": 0.13},
            "FC Krasnodar Away": {"league": "RFPL", "last_5_xg_total": 9.44, "last_5_xga_total": 3.10, "form_trend": 0.09},
            "Spartak Moscow Away": {"league": "RFPL", "last_5_xg_total": 9.96, "last_5_xga_total": 10.09, "form_trend": 0.07},
            "Lokomotiv Moscow Away": {"league": "RFPL", "last_5_xg_total": 9.75, "last_5_xga_total": 12.94, "form_trend": 0.10},
            "Rubin Kazan Away": {"league": "RFPL", "last_5_xg_total": 5.75, "last_5_xga_total": 9.25, "form_trend": 0.05},
            "Baltika Away": {"league": "RFPL", "last_5_xg_total": 9.95, "last_5_xga_total": 7.90, "form_trend": 0.03},
            "Dinamo Moscow Away": {"league": "RFPL", "last_5_xg_total": 8.72, "last_5_xga_total": 6.44, "form_trend": -0.03},
            "FC Rostov Away": {"league": "RFPL", "last_5_xg_total": 6.45, "last_5_xga_total": 7.47, "form_trend": -0.01},
            "Dynamo Makhachkala Away": {"league": "RFPL", "last_5_xg_total": 5.29, "last_5_xga_total": 7.79, "form_trend": -0.02},
            "FK Akhmat Away": {"league": "RFPL", "last_5_xg_total": 9.43, "last_5_xga_total": 13.90, "form_trend": -0.04},
            "FC Orenburg Away": {"league": "RFPL", "last_5_xg_total": 7.61, "last_5_xga_total": 10.75, "form_trend": -0.10},
            "Nizhny Novgorod Away": {"league": "RFPL", "last_5_xg_total": 3.56, "last_5_xga_total": 12.30, "form_trend": -0.06},
            "Akron Away": {"league": "RFPL", "last_5_xg_total": 7.89, "last_5_xga_total": 11.22, "form_trend": -0.12},
            "PFC Sochi Away": {"league": "RFPL", "last_5_xg_total": 5.75, "last_5_xga_total": 13.51, "form_trend": -0.14},
        }

    def get_teams_by_league(self, league, team_type="all"):
        """Get teams in a specific league, filtered by type (home/away/all)"""
        teams = []
        for team_name, data in self.team_database.items():
            if data["league"] == league:
                if team_type == "all":
                    teams.append(team_name)
                elif team_type == "home" and "Home" in team_name:
                    teams.append(team_name)
                elif team_type == "away" and "Away" in team_name:
                    teams.append(team_name)
        return sorted(teams)

    def get_team_base_name(self, team_name):
        """Extract base team name without Home/Away suffix"""
        if " Home" in team_name:
            return team_name.replace(" Home", "")
        elif " Away" in team_name:
            return team_name.replace(" Away", "")
        return team_name

    def get_team_data(self, team_name):
        """Get team data with fallback defaults"""
        default_data = {
            "league": "Unknown", 
            "last_5_xg_total": 7.50,
            "last_5_xga_total": 7.50,
            "form_trend": 0.00
        }
        
        team_data = self.team_database.get(team_name, default_data).copy()
        
        # Calculate per-match averages
        team_data['last_5_xg_per_match'] = team_data['last_5_xg_total'] / 5
        team_data['last_5_xga_per_match'] = team_data['last_5_xga_total'] / 5
        
        return team_data

    def validate_team_selection(self, home_team, away_team):
        """Validate that teams are from the same base team and league"""
        home_base = self.get_team_base_name(home_team)
        away_base = self.get_team_base_name(away_team)
        home_league = self.get_team_data(home_team)["league"]
        away_league = self.get_team_data(away_team)["league"]
        
        errors = []
        if home_base == away_base:
            errors.append("Cannot select the same team for both home and away")
        if home_league != away_league:
            errors.append(f"Teams must be from the same league. {home_base} is in {home_league}, {away_base} is in {away_league}")
        
        return errors

    def apply_modifiers(self, base_xg, base_xga, injury_level, rest_days, form_trend):
        """Apply modifiers with improved defense fatigue calculation"""
        injury_data = self.injury_weights[injury_level]
        
        # Apply injury impacts
        attack_mult = injury_data["attack_mult"]
        defense_mult = injury_data["defense_mult"]
        
        # Apply fatigue impact - IMPROVED: symmetric approach
        fatigue_mult = self.fatigue_multipliers.get(rest_days, 1.0)
        
        # Apply form trend
        form_mult = 1 + (form_trend * 0.2)
        
        # Apply all modifiers - IMPROVED: defense uses same fatigue multiplier
        xg_modified = base_xg * attack_mult * fatigue_mult * form_mult
        xga_modified = base_xga * defense_mult * fatigue_mult * form_mult  # Same fatigue impact
        
        return max(0.1, xg_modified), max(0.1, xga_modified)

    def calculate_poisson_probabilities(self, home_xg, away_xg):
        """Calculate probabilities using Poisson distribution"""
        max_goals = 8  # Reasonable maximum for calculation
        
        # Initialize probability arrays
        home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]
        
        # Calculate outcome probabilities
        home_win = 0
        draw = 0
        away_win = 0
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = home_probs[i] * away_probs[j]
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        # Normalize to account for truncated distribution
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total
        
        # Calculate over/under probabilities using Poisson
        total_goals_lambda = home_xg + away_xg
        over_25 = 1 - sum(poisson.pmf(i, total_goals_lambda) for i in range(3))
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'over_2.5': over_25,
            'under_2.5': 1 - over_25,
            'expected_home_goals': home_xg,
            'expected_away_goals': away_xg
        }

    def calculate_advantage_adjusted_probabilities(self, home_xg, home_xga, away_xg, away_xga):
        """Calculate probabilities with advantage-based mean adjustment"""
        # Calculate advantage
        home_attack_advantage = home_xg - away_xg
        home_defense_advantage = away_xga - home_xga
        total_advantage = home_attack_advantage + home_defense_advantage
        
        # Apply advantage adjustment to means (α = 0.3 based on calibration)
        alpha = 0.3
        home_xg_adj = home_xg * np.exp(alpha * total_advantage)
        away_xg_adj = away_xg * np.exp(-alpha * total_advantage)
        
        # Use Poisson with adjusted means
        return self.calculate_poisson_probabilities(home_xg_adj, home_xga_adj, away_xg_adj, away_xga_adj)

    def calculate_confidence(self, home_xg, away_xg, home_xga, away_xga, inputs):
        """Calculate confidence based on data quality"""
        factors = {}
        
        # Data quality factor
        data_quality = min(1.0, (home_xg + away_xg + home_xga + away_xga) / 5.4)
        factors['data_quality'] = data_quality
        
        # Predictability factor
        predictability = 1 - (abs(home_xg - away_xg) / max(home_xg, away_xg, 0.1))
        factors['predictability'] = predictability
        
        # Injury impact factor
        home_injury_severity = 1 - self.injury_weights[inputs['home_injuries']]['attack_mult']
        away_injury_severity = 1 - self.injury_weights[inputs['away_injuries']]['attack_mult']
        injury_factor = 1 - (home_injury_severity + away_injury_severity) / 2
        factors['injury_stability'] = injury_factor
        
        # Rest balance factor
        rest_diff = abs(inputs['home_rest'] - inputs['away_rest'])
        rest_factor = 1 - (rest_diff * 0.03)
        factors['rest_balance'] = rest_factor
        
        # Calculate weighted confidence using logistic scaling
        weights = {
            'data_quality': 0.25,
            'predictability': 0.25, 
            'injury_stability': 0.20,
            'rest_balance': 0.15
        }
        
        # Logistic scaling to avoid edge saturation
        weighted_sum = sum(factors[factor] * weights[factor] for factor in factors)
        confidence_score = 1 / (1 + np.exp(-10 * (weighted_sum - 0.5)))  # Logistic transform
        
        base_confidence = 55
        adjustment = confidence_score * 30
        
        confidence = base_confidence + adjustment
        return min(85, max(45, confidence)), factors

    def calculate_true_value(self, probability, odds):
        """Proper value calculation"""
        if odds <= 1.0:
            return {'ev': -1, 'kelly_fraction': 0, 'value_ratio': 0, 'rating': 'invalid', 'implied_prob': 0, 'model_prob': probability}
        
        # Proper Expected Value calculation
        ev = (probability * (odds - 1)) - (1 - probability)
        
        # Kelly Criterion
        kelly_fraction = (probability * odds - 1) / (odds - 1) if probability * odds > 1 else 0
        
        value_ratio = probability * odds
        
        # Realistic value rating
        if ev > 0.08 and value_ratio > 1.12:
            rating = "excellent"
        elif ev > 0.04 and value_ratio > 1.06:
            rating = "good"
        elif ev > 0.01 and value_ratio > 1.02:
            rating = "fair"
        else:
            rating = "poor"
        
        return {
            'ev': ev,
            'kelly_fraction': max(0, kelly_fraction),
            'value_ratio': value_ratio,
            'rating': rating,
            'implied_prob': 1 / odds,
            'model_prob': probability
        }

    def calculate_value_bets(self, probabilities, odds):
        """Calculate value bets for all markets"""
        value_bets = {}
        
        value_bets['home'] = self.calculate_true_value(probabilities['home_win'], odds['home'])
        value_bets['draw'] = self.calculate_true_value(probabilities['draw'], odds['draw'])
        value_bets['away'] = self.calculate_true_value(probabilities['away_win'], odds['away'])
        value_bets['over_2.5'] = self.calculate_true_value(probabilities['over_2.5'], odds['over_2.5'])
        
        return value_bets

    def predict_match(self, inputs):
        """MAIN PREDICTION FUNCTION - IMPROVED WITH POISSON"""
        # Validate team selection
        validation_errors = self.validate_team_selection(inputs['home_team'], inputs['away_team'])
        if validation_errors:
            return None, validation_errors, []
        
        # Calculate per-match averages from user inputs
        home_xg_per_match = inputs['home_xg_total'] / 5
        home_xga_per_match = inputs['home_xga_total'] / 5
        away_xg_per_match = inputs['away_xg_total'] / 5
        away_xga_per_match = inputs['away_xga_total'] / 5
        
        # Apply modifiers
        home_xg_adj, home_xga_adj = self.apply_modifiers(
            home_xg_per_match, home_xga_per_match,
            inputs['home_injuries'], inputs['home_rest'],
            self.get_team_data(inputs['home_team'])['form_trend']
        )
        
        away_xg_adj, away_xga_adj = self.apply_modifiers(
            away_xg_per_match, away_xga_per_match,
            inputs['away_injuries'], inputs['away_rest'],
            self.get_team_data(inputs['away_team'])['form_trend']
        )
        
        # Calculate probabilities - IMPROVED: Poisson with advantage adjustment
        probabilities = self.calculate_advantage_adjusted_probabilities(
            home_xg_adj, home_xga_adj, away_xg_adj, away_xga_adj
        )
        
        # Calculate confidence
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
        
        # Generate insights
        insights = self.generate_insights(inputs, probabilities, home_xg_per_match, away_xg_per_match, home_xga_per_match, away_xga_per_match)
        
        # Store calculation details for transparency
        calculation_details = {
            'home_xg_raw': home_xg_per_match,
            'home_xg_modified': home_xg_adj,
            'away_xg_raw': away_xg_per_match, 
            'away_xg_modified': away_xg_adj,
            'home_xga_raw': home_xga_per_match,
            'home_xga_modified': home_xga_adj,
            'away_xga_raw': away_xga_per_match,
            'away_xga_modified': away_xga_adj,
            'total_goals_lambda': home_xg_adj + away_xg_adj
        }
        
        result = {
            'probabilities': probabilities,
            'expected_goals': {'home': probabilities['expected_home_goals'], 'away': probabilities['expected_away_goals']},
            'value_bets': value_bets,
            'confidence': confidence,
            'confidence_factors': confidence_factors,
            'insights': insights,
            'per_match_stats': {
                'home_xg': home_xg_per_match,
                'home_xga': home_xga_per_match,
                'away_xg': away_xg_per_match,
                'away_xga': away_xga_per_match
            },
            'calculation_details': calculation_details
        }
        
        return result, [], []

    def generate_insights(self, inputs, probabilities, home_xg, away_xg, home_xga, away_xga):
        """Generate insights based on football statistics"""
        insights = []
        
        # Get base team names for display
        home_base = self.get_team_base_name(inputs['home_team'])
        away_base = self.get_team_base_name(inputs['away_team'])
        
        # Injury insights
        home_injury_data = self.injury_weights[inputs['home_injuries']]
        away_injury_data = self.injury_weights[inputs['away_injuries']]
        
        if inputs['home_injuries'] != "None":
            insights.append(f"🩹 {home_base} affected by {inputs['home_injuries'].lower()} ({home_injury_data['description']})")
        if inputs['away_injuries'] != "None":
            insights.append(f"🩹 {away_base} affected by {inputs['away_injuries'].lower()} ({away_injury_data['description']})")
        
        # Rest insights
        rest_diff = inputs['home_rest'] - inputs['away_rest']
        if abs(rest_diff) >= 3:
            insights.append(f"⚖️ Rest difference: {abs(rest_diff)} days")
        
        # Team strength insights
        if home_xg > away_xg:
            insights.append(f"📈 {home_base} stronger attack ({home_xg:.2f} vs {away_xg:.2f} xG)")
        elif away_xg > home_xg:
            insights.append(f"📈 {away_base} stronger attack ({away_xg:.2f} vs {home_xg:.2f} xG)")
        
        if home_xga < away_xga:
            insights.append(f"🛡️ {home_base} better defense ({home_xga:.2f} vs {away_xga:.2f} xGA)")
        elif away_xga < home_xga:
            insights.append(f"🛡️ {away_base} better defense ({away_xga:.2f} vs {home_xga:.2f} xGA)")
        
        # Match type analysis
        total_goals = probabilities['expected_home_goals'] + probabilities['expected_away_goals']
        if total_goals > 3.0:
            insights.append(f"⚽ High-scoring match expected ({total_goals:.2f} total xG)")
        elif total_goals < 2.0:
            insights.append(f"🔒 Defensive battle anticipated ({total_goals:.2f} total xG)")
        
        # Value insights
        excellent_bets = [k for k, v in self.calculate_value_bets(probabilities, {
            'home': inputs['home_odds'], 'draw': inputs['draw_odds'], 
            'away': inputs['away_odds'], 'over_2.5': inputs['over_odds']
        }).items() if v['rating'] == 'excellent']
        
        good_bets = [k for k, v in self.calculate_value_bets(probabilities, {
            'home': inputs['home_odds'], 'draw': inputs['draw_odds'], 
            'away': inputs['away_odds'], 'over_2.5': inputs['over_odds']
        }).items() if v['rating'] == 'good']
        
        if excellent_bets:
            insights.append("💰 Excellent value betting opportunities identified")
        elif good_bets:
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
    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = "Premier League"

def get_default_inputs():
    """Get default input values"""
    return {
        'home_team': 'Arsenal Home',
        'away_team': 'Manchester United Away',
        'home_xg_total': 10.25,
        'home_xga_total': 1.75,
        'away_xg_total': 8.75,
        'away_xga_total': 10.60,
        'home_injuries': 'None',
        'away_injuries': 'None',
        'home_rest': 7,
        'away_rest': 7,
        'home_odds': 2.15,
        'draw_odds': 3.25,
        'away_odds': 2.80,
        'over_odds': 1.57
    }

def display_understat_input_form(engine):
    """Display the main input form with Understat format and league selection"""
    st.markdown('<div class="main-header">🎯 Professional Football Prediction Engine</div>', unsafe_allow_html=True)
    
    # CRITICAL DISCLAIMER
    st.markdown("""
    <div class="disclaimer-box">
    <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
    This tool is for <strong>EDUCATIONAL AND ANALYTICAL PURPOSES ONLY</strong>. Sports prediction is inherently uncertain.<br>
    <strong>NEVER bet more than you can afford to lose.</strong> Past performance does not guarantee future results.<br>
    Always practice responsible gambling and seek help if needed.
    </div>
    """, unsafe_allow_html=True)
    
    # Use existing inputs or defaults
    if st.session_state.input_data:
        current_inputs = st.session_state.input_data
    else:
        current_inputs = get_default_inputs()
    
    st.markdown('<div class="section-header">🏆 League & Match Configuration</div>', unsafe_allow_html=True)
    
    # League Selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🌍 Select League")
        selected_league = st.selectbox(
            "Choose League",
            options=engine.leagues,
            index=engine.leagues.index(st.session_state.selected_league),
            key="league_select"
        )
        st.session_state.selected_league = selected_league
        
        # Show league info
        home_teams = engine.get_teams_by_league(selected_league, "home")
        away_teams = engine.get_teams_by_league(selected_league, "away")
        st.write(f"**Home Teams:** {len(home_teams)}")
        st.write(f"**Away Teams:** {len(away_teams)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("📊 League Overview")
        st.write(f"**Selected:** <span class='league-badge'>{selected_league}</span>", unsafe_allow_html=True)
        st.write(f"**Data Quality:** ✅ Home/Away specific xG data")
        st.write(f"**Teams:** Complete home/away separation")
        st.write(f"**Validation:** Same-team and cross-league prevention")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">⚽ Team Selection</div>', unsafe_allow_html=True)
    
    # Team Selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🏠 Home Team")
        
        # Get home teams for selected league
        home_teams = engine.get_teams_by_league(selected_league, "home")
        
        home_team = st.selectbox(
            "Select Home Team",
            home_teams,
            index=home_teams.index(current_inputs['home_team']) if current_inputs['home_team'] in home_teams else 0,
            key="home_team_input"
        )
        home_data = engine.get_team_data(home_team)
        home_base = engine.get_team_base_name(home_team)
        
        # Display team info
        st.write(f"**Team:** {home_base} <span class='home-badge'>HOME</span>", unsafe_allow_html=True)
        st.write(f"**League:** {home_data['league']}")
        st.write(f"**Form Trend:** {'↗️ Improving' if home_data['form_trend'] > 0 else '↘️ Declining' if home_data['form_trend'] < 0 else '➡️ Stable'}")
        st.write(f"**Last 5 Home:** {home_data['last_5_xg_total']:.2f} xG, {home_data['last_5_xga_total']:.2f} xGA")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("✈️ Away Team")
        
        # Get away teams for selected league
        away_teams = engine.get_teams_by_league(selected_league, "away")
        
        away_team = st.selectbox(
            "Select Away Team",
            away_teams,
            index=away_teams.index(current_inputs['away_team']) if current_inputs['away_team'] in away_teams else min(1, len(away_teams)-1),
            key="away_team_input"
        )
        away_data = engine.get_team_data(away_team)
        away_base = engine.get_team_base_name(away_team)
        
        # Display team info
        st.write(f"**Team:** {away_base} <span class='away-badge'>AWAY</span>", unsafe_allow_html=True)
        st.write(f"**League:** {away_data['league']}")
        st.write(f"**Form Trend:** {'↗️ Improving' if away_data['form_trend'] > 0 else '↘️ Declining' if away_data['form_trend'] < 0 else '➡️ Stable'}")
        st.write(f"**Last 5 Away:** {away_data['last_5_xg_total']:.2f} xG, {away_data['last_5_xga_total']:.2f} xGA")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation check
    validation_errors = engine.validate_team_selection(home_team, away_team)
    if validation_errors:
        for error in validation_errors:
            st.error(f"🚫 {error}")
    
    st.markdown('<div class="section-header">📊 Understat Last 5 Matches Data</div>', unsafe_allow_html=True)
    
    # Understat Format Explanation
    st.markdown("""
    <div class="warning-box">
    <strong>📝 Understat Format Guide:</strong><br>
    Enter data in the format shown on Understat.com: <strong>"10.25-1.75"</strong><br>
    - <strong>First number</strong>: Total xG scored in last 5 matches<br>
    - <strong>Second number</strong>: Total xGA conceded in last 5 matches<br>
    <strong>Note:</strong> Using context-specific home/away data for maximum accuracy
    </div>
    """, unsafe_allow_html=True)
    
    # Understat Data Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader(f"📈 {home_base} - Last 5 HOME Matches")
        
        # Understat format display
        current_home_format = f"{current_inputs['home_xg_total']}-{current_inputs['home_xga_total']}"
        st.markdown(f'<div class="understat-format">Understat Format: {current_home_format}</div>', unsafe_allow_html=True)
        
        col1a, col1b = st.columns(2)
        with col1a:
            home_xg_total = st.number_input(
                "Total xG Scored",
                min_value=0.0,
                max_value=20.0,
                value=current_inputs['home_xg_total'],
                step=0.1,
                key="home_xg_total_input",
                help="Total expected goals scored in last 5 HOME matches"
            )
        with col1b:
            home_xga_total = st.number_input(
                "Total xGA Conceded",
                min_value=0.0,
                max_value=20.0,
                value=current_inputs['home_xga_total'],
                step=0.1,
                key="home_xga_total_input",
                help="Total expected goals against in last 5 HOME matches"
            )
        
        # Calculate and show per-match averages
        home_xg_per_match = home_xg_total / 5
        home_xga_per_match = home_xga_total / 5
        
        st.metric("xG per match", f"{home_xg_per_match:.2f}")
        st.metric("xGA per match", f"{home_xga_per_match:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader(f"📈 {away_base} - Last 5 AWAY Matches")
        
        # Understat format display
        current_away_format = f"{current_inputs['away_xg_total']}-{current_inputs['away_xga_total']}"
        st.markdown(f'<div class="understat-format">Understat Format: {current_away_format}</div>', unsafe_allow_html=True)
        
        col2a, col2b = st.columns(2)
        with col2a:
            away_xg_total = st.number_input(
                "Total xG Scored",
                min_value=0.0,
                max_value=20.0,
                value=current_inputs['away_xg_total'],
                step=0.1,
                key="away_xg_total_input",
                help="Total expected goals scored in last 5 AWAY matches"
            )
        with col2b:
            away_xga_total = st.number_input(
                "Total xGA Conceded",
                min_value=0.0,
                max_value=20.0,
                value=current_inputs['away_xga_total'],
                step=0.1,
                key="away_xga_total_input",
                help="Total expected goals against in last 5 AWAY matches"
            )
        
        # Calculate and show per-match averages
        away_xg_per_match = away_xg_total / 5
        away_xga_per_match = away_xga_total / 5
        
        st.metric("xG per match", f"{away_xg_per_match:.2f}")
        st.metric("xGA per match", f"{away_xga_per_match:.2f}")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🎭 Match Context</div>', unsafe_allow_html=True)
    
    # Context Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🩹 Injury Status")
        
        injury_options = ["None", "Minor (1-2 rotational)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ starters)"]
        
        home_injuries = st.selectbox(
            f"{home_base} Injuries",
            injury_options,
            index=injury_options.index(current_inputs['home_injuries']),
            key="home_injuries_input"
        )
        
        away_injuries = st.selectbox(
            f"{away_base} Injuries",
            injury_options,
            index=injury_options.index(current_inputs['away_injuries']),
            key="away_injuries_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🕐 Fatigue & Recovery")
        
        home_rest = st.number_input(
            f"{home_base} Rest Days",
            min_value=2,
            max_value=14,
            value=current_inputs['home_rest'],
            key="home_rest_input",
            help="Days since last match"
        )
        
        away_rest = st.number_input(
            f"{away_base} Rest Days",
            min_value=2,
            max_value=14,
            value=current_inputs['away_rest'],
            key="away_rest_input",
            help="Days since last match"
        )
        
        # Show rest comparison
        rest_diff = home_rest - away_rest
        if rest_diff > 0:
            st.success(f"🏠 {home_base} has {rest_diff} more rest days")
        elif rest_diff < 0:
            st.warning(f"✈️ {away_base} has {-rest_diff} more rest days")
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
    
    return inputs, validation_errors

# [The display_prediction_results, _display_value_analysis, and main functions remain the same as before]

def display_prediction_results(engine, result, inputs):
    """Display prediction results with improved transparency"""
    st.markdown('<div class="main-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    
    # Get base team names for display
    home_base = engine.get_team_base_name(inputs['home_team'])
    away_base = engine.get_team_base_name(inputs['away_team'])
    
    # Match header
    st.markdown(f'<h2 style="text-align: center; color: #1f77b4;">{home_base} vs {away_base}</h2>', unsafe_allow_html=True)
    
    # League badge
    home_league = engine.get_team_data(inputs['home_team'])['league']
    st.markdown(f'<div style="text-align: center; margin-bottom: 1rem;"><span class="league-badge">{home_league}</span></div>', unsafe_allow_html=True)
    
    # Expected score card
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    expected_home = result['expected_goals']['home']
    expected_away = result['expected_goals']['away']
    st.markdown(f'<h1 style="font-size: 4rem; margin: 1rem 0;">{expected_home:.2f} - {expected_away:.2f}</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem;">Expected Final Score (Poisson-based)</p>', unsafe_allow_html=True)
    
    # Confidence badge
    confidence = result['confidence']
    confidence_stars = "★" * int((confidence - 40) / 8) + "☆" * (5 - int((confidence - 40) / 8))
    confidence_text = "Low" if confidence < 55 else "Medium" if confidence < 65 else "High" if confidence < 75 else "Very High"
    
    st.markdown(f'<div style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown(f'<span style="background: rgba(255,255,255,0.3); padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">Confidence: {confidence_stars} ({confidence:.0f}% - {confidence_text})</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show confidence factors on hover/expand
    with st.expander("Confidence Breakdown"):
        factors = result['confidence_factors']
        st.write("**Confidence Factors:**")
        for factor, value in factors.items():
            st.write(f"- {factor.replace('_', ' ').title()}: {value:.1%}")
    
    # Show calculation details - WITH ERROR HANDLING
    if 'calculation_details' in result:
        with st.expander("Calculation Details"):
            details = result['calculation_details']
            st.write("**Raw vs Modified Values:**")
            st.write(f"- Home xG: {details['home_xg_raw']:.3f} → {details['home_xg_modified']:.3f}")
            st.write(f"- Away xG: {details['away_xg_raw']:.3f} → {details['away_xg_modified']:.3f}")
            st.write(f"- Home xGA: {details['home_xga_raw']:.3f} → {details['home_xga_modified']:.3f}")
            st.write(f"- Away xGA: {details['away_xga_raw']:.3f} → {details['away_xga_modified']:.3f}")
            st.write(f"- Total Goals λ: {details['total_goals_lambda']:.3f}")
            st.write("**Method:** Poisson distribution with advantage-based mean adjustment")
    else:
        # Fallback if calculation_details is missing
        with st.expander("Calculation Details"):
            st.write("**Method:** Poisson distribution with advantage-based mean adjustment")
            st.write("Calculation details not available in this session.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Outcome Probabilities
    st.markdown('<div class="section-header">📊 Match Outcome Probabilities</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    probs = result['probabilities']
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        home_prob = probs['home_win']
        home_color = "🟢" if home_prob > 0.45 else "🟡" if home_prob > 0.35 else "🔴"
        st.metric(f"{home_color} {home_base} Win", f"{home_prob:.1%}")
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
        st.metric(f"{away_color} {away_base} Win", f"{away_prob:.1%}")
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
        if data['rating'] in ['excellent', 'good']:
            value_bets.append({
                'type': bet_type,
                'value_ratio': data['value_ratio'],
                'ev': data['ev'],
                'odds': inputs[f"{bet_type}_odds" if bet_type != 'over_2.5' else 'over_odds'],
                'model_prob': data['model_prob'],
                'implied_prob': data['implied_prob'],
                'rating': data['rating'],
                'kelly_fraction': data['kelly_fraction']
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
                'home': f"{home_base} Win",
                'draw': "Draw",
                'away': f"{away_base} Win", 
                'over_2.5': "Over 2.5 Goals"
            }[bet['type']]
            
            st.markdown(f"**✅ {bet_name} @ {bet['odds']:.2f}**")
            st.markdown(f"**Model Probability:** {bet['model_prob']:.1%} | **Market Implied:** {bet['implied_prob']:.1%}")
            st.markdown(f"**Value Ratio:** {bet['value_ratio']:.2f}x | **Expected Value:** {bet['ev']:.1%}")
            
            if bet['kelly_fraction'] > 0:
                st.markdown(f'<div class="kelly-recommendation">', unsafe_allow_html=True)
                st.markdown(f"**Kelly Recommended Stake:** {bet['kelly_fraction']:.1%} of bankroll")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if bet['rating'] == 'excellent':
                st.markdown("**🎯 EXCELLENT VALUE BET**")
            else:
                st.markdown("**👍 GOOD VALUE OPPORTUNITY**")
                
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**No strong value bets identified**")
        st.markdown("All market odds appear efficient for this match.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Insights
    st.markdown('<div class="section-header">🧠 Key Insights & Analysis</div>', unsafe_allow_html=True)
    
    for insight in result['insights']:
        st.markdown(f'<div class="metric-card">• {insight}</div>', unsafe_allow_html=True)
    
    # Statistical insights
    total_xg = expected_home + expected_away
    per_match = result['per_match_stats']
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("**📈 Statistical Summary:**")
    st.markdown(f"- **Total Expected Goals:** {total_xg:.2f}")
    st.markdown(f"- **{home_base} Home Form:** {per_match['home_xg']:.2f} xG, {per_match['home_xga']:.2f} xGA per match")
    st.markdown(f"- **{away_base} Away Form:** {per_match['away_xg']:.2f} xG, {per_match['away_xga']:.2f} xGA per match")
    st.markdown(f"- **Goal Expectancy:** {total_xg/2.5:.1%} of average match")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bankroll Management Advice
    st.markdown('<div class="section-header">💼 Bankroll Management</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="bankroll-advice">
    <strong>Responsible Betting Guidelines:</strong><br>
    • <strong>Never bet more than 1-2% of your total bankroll on a single bet</strong><br>
    • Use Kelly Criterion fractions as maximum stakes, not recommendations<br>
    • Maintain detailed records of all bets and results<br>
    • Set stop-loss limits and stick to them<br>
    • Remember: Even the best models have losing streaks
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Performance Expectations
    st.markdown('<div class="section-header">📊 Realistic Performance Expectations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
    <strong>Professional Betting Reality Check:</strong><br>
    • <strong>Realistic Accuracy:</strong> 52-57% for match outcomes<br>
    • <strong>Sustainable Edge:</strong> 2-5% in efficient markets<br>
    • <strong>Value Bet Frequency:</strong> 5-15% of matches<br>
    • <strong>Long-term Success:</strong> Requires discipline and proper bankroll management<br>
    • <strong>Variance:</strong> Even with positive EV, losing streaks of 5-10 bets are normal
    </div>
    """, unsafe_allow_html=True)
    
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
        st.success(f"**Excellent Value:** EV {value_data['ev']:.1%}")
    elif value_data['rating'] == 'good':
        st.info(f"**Good Value:** EV {value_data['ev']:.1%}")
    elif value_data['rating'] == 'fair':
        st.warning(f"**Fair Value:** EV {value_data['ev']:.1%}")
    else:
        st.error(f"**Poor Value:** EV {value_data['ev']:.1%}")
        
    if value_data['kelly_fraction'] > 0:
        st.write(f"**Kelly Stake:** {value_data['kelly_fraction']:.1%}")

def main():
    """Main application function"""
    initialize_session_state()
    engine = ProfessionalPredictionEngine()
    
    # Show edit form if requested
    if st.session_state.show_edit:
        st.markdown('<div class="main-header">✏️ Edit Match Inputs</div>', unsafe_allow_html=True)
        inputs, validation_errors = display_understat_input_form(engine)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Prediction", use_container_width=True, type="primary") and not validation_errors:
                result, errors, warnings = engine.predict_match(inputs)
                if result:
                    st.session_state.prediction_result = result
                    st.session_state.input_data = inputs
                    st.session_state.show_edit = False
                    st.rerun()
                else:
                    for error in errors:
                        st.error(f"🚫 {error}")
            
            if st.button("← Back to Results", use_container_width=True):
                st.session_state.show_edit = False
                st.rerun()
    
    # Show prediction results if available
    elif st.session_state.prediction_result:
        display_prediction_results(engine, st.session_state.prediction_result, st.session_state.input_data)
    
    # Show main input form
    else:
        inputs, validation_errors = display_understat_input_form(engine)
        
        # Generate Prediction Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Prediction", use_container_width=True, type="primary", key="main_predict") and not validation_errors:
                result, errors, warnings = engine.predict_match(inputs)
                if result:
                    st.session_state.prediction_result = result
                    st.session_state.input_data = inputs
                    st.rerun()
                else:
                    for error in errors:
                        st.error(f"🚫 {error}")

if __name__ == "__main__":
    main()
