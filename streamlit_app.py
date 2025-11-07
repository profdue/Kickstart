import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Professional Football Prediction Engine",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .team-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.25rem 0;
    }
    .value-bet-good {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .value-bet-poor {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .confidence-stars {
        color: #ffc107;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Database of teams for top 5 leagues
TEAMS_DATA = {
    "Premier League": {
        "Arsenal": {"league": "Premier League"},
        "Aston Villa": {"league": "Premier League"},
        "Bournemouth": {"league": "Premier League"},
        "Brentford": {"league": "Premier League"},
        "Brighton": {"league": "Premier League"},
        "Chelsea": {"league": "Premier League"},
        "Crystal Palace": {"league": "Premier League"},
        "Everton": {"league": "Premier League"},
        "Fulham": {"league": "Premier League"},
        "Liverpool": {"league": "Premier League"},
        "Luton": {"league": "Premier League"},
        "Manchester City": {"league": "Premier League"},
        "Manchester United": {"league": "Premier League"},
        "Newcastle": {"league": "Premier League"},
        "Nottingham Forest": {"league": "Premier League"},
        "Sheffield United": {"league": "Premier League"},
        "Tottenham": {"league": "Premier League"},
        "West Ham": {"league": "Premier League"},
        "Wolves": {"league": "Premier League"}
    },
    "La Liga": {
        "Alaves": {"league": "La Liga"},
        "Almeria": {"league": "La Liga"},
        "Athletic Bilbao": {"league": "La Liga"},
        "Atletico Madrid": {"league": "La Liga"},
        "Barcelona": {"league": "La Liga"},
        "Betis": {"league": "La Liga"},
        "Celta Vigo": {"league": "La Liga"},
        "Cadiz": {"league": "La Liga"},
        "Getafe": {"league": "La Liga"},
        "Girona": {"league": "La Liga"},
        "Granada": {"league": "La Liga"},
        "Las Palmas": {"league": "La Liga"},
        "Mallorca": {"league": "La Liga"},
        "Osasuna": {"league": "La Liga"},
        "Rayo Vallecano": {"league": "La Liga"},
        "Real Madrid": {"league": "La Liga"},
        "Real Sociedad": {"league": "La Liga"},
        "Sevilla": {"league": "La Liga"},
        "Valencia": {"league": "La Liga"},
        "Villarreal": {"league": "La Liga"}
    },
    "Serie A": {
        "AC Milan": {"league": "Serie A"},
        "AS Roma": {"league": "Serie A"},
        "Atalanta": {"league": "Serie A"},
        "Bologna": {"league": "Serie A"},
        "Cagliari": {"league": "Serie A"},
        "Empoli": {"league": "Serie A"},
        "Fiorentina": {"league": "Serie A"},
        "Frosinone": {"league": "Serie A"},
        "Genoa": {"league": "Serie A"},
        "Inter Milan": {"league": "Serie A"},
        "Juventus": {"league": "Serie A"},
        "Lazio": {"league": "Serie A"},
        "Lecce": {"league": "Serie A"},
        "Monza": {"league": "Serie A"},
        "Napoli": {"league": "Serie A"},
        "Salernitana": {"league": "Serie A"},
        "Sassuolo": {"league": "Serie A"},
        "Torino": {"league": "Serie A"},
        "Udinese": {"league": "Serie A"},
        "Verona": {"league": "Serie A"}
    },
    "Bundesliga": {
        "Augsburg": {"league": "Bundesliga"},
        "Bayer Leverkusen": {"league": "Bundesliga"},
        "Bayern Munich": {"league": "Bundesliga"},
        "Bochum": {"league": "Bundesliga"},
        "Borussia Dortmund": {"league": "Bundesliga"},
        "Borussia M'gladbach": {"league": "Bundesliga"},
        "Darmstadt": {"league": "Bundesliga"},
        "Eintracht Frankfurt": {"league": "Bundesliga"},
        "Freiburg": {"league": "Bundesliga"},
        "Heidenheim": {"league": "Bundesliga"},
        "Hoffenheim": {"league": "Bundesliga"},
        "Koln": {"league": "Bundesliga"},
        "Mainz": {"league": "Bundesliga"},
        "RB Leipzig": {"league": "Bundesliga"},
        "Stuttgart": {"league": "Bundesliga"},
        "Union Berlin": {"league": "Bundesliga"},
        "Werder Bremen": {"league": "Bundesliga"},
        "Wolfsburg": {"league": "Bundesliga"}
    },
    "Ligue 1": {
        "AS Monaco": {"league": "Ligue 1"},
        "Lens": {"league": "Ligue 1"},
        "Lille": {"league": "Ligue 1"},
        "Marseille": {"league": "Ligue 1"},
        "Paris Saint-Germain": {"league": "Ligue 1"},
        "Brest": {"league": "Ligue 1"},
        "Nice": {"league": "Ligue 1"},
        "Lorient": {"league": "Ligue 1"},
        "Reims": {"league": "Ligue 1"},
        "Montpellier": {"league": "Ligue 1"},
        "Toulouse": {"league": "Ligue 1"},
        "Clermont": {"league": "Ligue 1"},
        "Strasbourg": {"league": "Ligue 1"},
        "Nantes": {"league": "Ligue 1"},
        "Le Havre": {"league": "Ligue 1"},
        "Metz": {"league": "Ligue 1"},
        "Rennes": {"league": "Ligue 1"},
        "Lyon": {"league": "Ligue 1"}
    }
}

# Database averages for xG and xGA
LEAGUE_AVERAGES = {
    "Premier League": {"xG": 1.43, "xGA": 1.43},
    "La Liga": {"xG": 1.38, "xGA": 1.38},
    "Serie A": {"xG": 1.41, "xGA": 1.41},
    "Bundesliga": {"xG": 1.52, "xGA": 1.52},
    "Ligue 1": {"xG": 1.39, "xGA": 1.39}
}

class FootballPredictionEngine:
    def __init__(self):
        self.teams_data = TEAMS_DATA
        self.league_averages = LEAGUE_AVERAGES
    
    def calculate_xg_per_match(self, understat_format):
        """Parse Understat format and calculate per match xG and xGA"""
        try:
            xg_total, xga_total = map(float, understat_format.split('-'))
            xg_per_match = xg_total / 5
            xga_per_match = xga_total / 5
            return xg_per_match, xga_per_match, xg_total, xga_total
        except:
            return 0, 0, 0, 0
    
    def get_league_average(self, league):
        """Get league average xG and xGA"""
        return self.league_averages.get(league, {"xG": 1.4, "xGA": 1.4})
    
    def calculate_injury_impact(self, injury_status):
        """Calculate performance multiplier based on injury status"""
        injury_impact = {
            "None": 1.0,
            "Minor (bench players)": 0.95,
            "Moderate (1-2 key starters)": 0.85,
            "Significant (3-4 key players)": 0.70,
            "Crisis (5+ players)": 0.55
        }
        return injury_impact.get(injury_status, 1.0)
    
    def calculate_rest_advantage(self, home_rest_days, away_rest_days):
        """Calculate rest advantage multiplier"""
        rest_difference = away_rest_days - home_rest_days
        advantage_multiplier = 1.0 + (rest_difference * 0.05)
        return max(0.8, min(1.2, advantage_multiplier)), rest_difference
    
    def calculate_team_strength(self, xg_per_match, xga_per_match, league, injury_status, rest_days, is_home=True):
        """Calculate comprehensive team strength score"""
        league_avg = self.get_league_average(league)
        
        # Base strength from xG performance vs league average
        xg_strength = xg_per_match / league_avg["xG"]
        xga_strength = league_avg["xGA"] / xga_per_match if xga_per_match > 0 else 1.0
        
        # Apply injury impact
        injury_multiplier = self.calculate_injury_impact(injury_status)
        
        # Home advantage
        home_advantage = 1.05 if is_home else 1.0
        
        # Combine factors
        strength_score = (xg_strength * 0.6 + xga_strength * 0.4) * injury_multiplier * home_advantage
        
        return strength_score
    
    def predict_match(self, home_team, away_team, home_xg_data, away_xg_data, 
                     home_injuries, away_injuries, home_rest_days, away_rest_days,
                     home_odds, draw_odds, away_odds, over_odds):
        """Main prediction function with all improvements"""
        
        # Calculate xG per match
        home_xg_pm, home_xga_pm, home_xg_total, home_xga_total = self.calculate_xg_per_match(home_xg_data)
        away_xg_pm, away_xga_pm, away_xg_total, away_xga_total = self.calculate_xg_per_match(away_xg_data)
        
        # Get leagues
        home_league = self._find_team_league(home_team)
        away_league = self._find_team_league(away_team)
        
        # Calculate rest advantage
        rest_multiplier, rest_difference = self.calculate_rest_advantage(home_rest_days, away_rest_days)
        
        # Calculate team strengths
        home_strength = self.calculate_team_strength(
            home_xg_pm, home_xga_pm, home_league, home_injuries, home_rest_days, True
        )
        
        away_strength = self.calculate_team_strength(
            away_xg_pm, away_xga_pm, away_league, away_injuries, away_rest_days, False
        ) * rest_multiplier
        
        # Calculate win probabilities
        total_strength = home_strength + away_strength
        home_win_prob = home_strength / total_strength
        away_win_prob = away_strength / total_strength
        draw_prob = 0.25 * (1 - abs(home_win_prob - away_win_prob))
        
        # Normalize probabilities
        total = home_win_prob + away_win_prob + draw_prob
        home_win_prob /= total
        away_win_prob /= total
        draw_prob /= total
        
        # Calculate expected goals
        total_xg = home_xg_pm + away_xg_pm
        total_xga = home_xga_pm + away_xga_pm
        avg_xg = (total_xg + total_xga) / 2
        
        # Goal expectancy
        goal_expectancy = (avg_xg / 1.4) * 100  # Compared to average 1.4 xG per team
        
        # Over/Under probability
        over_prob = min(0.95, max(0.05, avg_xg * 0.3))
        under_prob = 1 - over_prob
        
        # Expected score (simplified)
        home_expected_goals = (home_xg_pm * 0.7 + away_xga_pm * 0.3)
        away_expected_goals = (away_xg_pm * 0.7 + home_xga_pm * 0.3)
        
        # Value bets calculation
        home_value = home_win_prob * home_odds
        draw_value = draw_prob * draw_odds
        away_value = away_win_prob * away_odds
        over_value = over_prob * over_odds
        
        # Confidence calculation
        confidence_factors = []
        
        # Data quality
        if home_xg_total > 0 and away_xg_total > 0:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Form reliability (consistency in recent performances)
        form_confidence = min(1.0, (home_xg_total + away_xg_total) / 20)
        confidence_factors.append(form_confidence)
        
        # Injury impact confidence
        injury_confidence = 1.0
        if home_injuries in ["Significant", "Crisis"] or away_injuries in ["Significant", "Crisis"]:
            injury_confidence = 0.7
        confidence_factors.append(injury_confidence)
        
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return {
            "home_win_prob": home_win_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_win_prob,
            "over_prob": over_prob,
            "under_prob": under_prob,
            "home_expected_goals": home_expected_goals,
            "away_expected_goals": away_expected_goals,
            "total_expected_goals": avg_xg * 2,
            "goal_expectancy": goal_expectancy,
            "home_value": home_value,
            "draw_value": draw_value,
            "away_value": away_value,
            "over_value": over_value,
            "confidence": overall_confidence,
            "rest_difference": rest_difference,
            "home_xg_pm": home_xg_pm,
            "home_xga_pm": home_xga_pm,
            "away_xg_pm": away_xg_pm,
            "away_xga_pm": away_xga_pm,
            "home_league_avg": self.get_league_average(home_league),
            "away_league_avg": self.get_league_average(away_league)
        }
    
    def _find_team_league(self, team_name):
        """Find which league a team belongs to"""
        for league, teams in self.teams_data.items():
            if team_name in teams:
                return league
        return "Premier League"  # Default
    
    def get_value_bet_recommendation(self, value_ratio, threshold=1.0):
        """Determine value bet recommendation"""
        if value_ratio >= 1.1:
            return "Strong Value", "good"
        elif value_ratio >= 1.0:
            return "Fair Value", "neutral"
        else:
            return "Poor Value", "poor"

def main():
    st.markdown('<div class="main-header">🎯 Professional Football Prediction Engine</div>', unsafe_allow_html=True)
    
    # Initialize prediction engine
    engine = FootballPredictionEngine()
    
    # Match Configuration Section
    st.markdown('<div class="section-header">🏆 Match Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Home Team")
        
        # League selection for home team
        home_league = st.selectbox(
            "Select Home Team League",
            list(TEAMS_DATA.keys()),
            key="home_league"
        )
        
        home_team = st.selectbox(
            "Select Home Team",
            list(TEAMS_DATA[home_league].keys()),
            key="home_team"
        )
        
        st.info(f"**League:** {home_league}")
        
        home_form = st.selectbox(
            "Form Trend",
            ["↗️ Improving", "➡️ Stable", "↘️ Declining"],
            key="home_form"
        )
        
        home_last_opponents = st.text_area(
            "Last 5 Opponents",
            "Arsenal, Chelsea, Wolves, Crystal Palace, Luton",
            key="home_opponents"
        )
    
    with col2:
        st.markdown("### ✈️ Away Team")
        
        # League selection for away team
        away_league = st.selectbox(
            "Select Away Team League",
            list(TEAMS_DATA.keys()),
            key="away_league"
        )
        
        away_team = st.selectbox(
            "Select Away Team",
            list(TEAMS_DATA[away_league].keys()),
            key="away_team"
        )
        
        st.info(f"**League:** {away_league}")
        
        away_form = st.selectbox(
            "Form Trend",
            ["↗️ Improving", "➡️ Stable", "↘️ Declining"],
            key="away_form"
        )
        
        away_last_opponents = st.text_area(
            "Last 5 Opponents",
            "Chelsea, Liverpool, West Ham, Everton, Newcastle",
            key="away_opponents"
        )
    
    # Understat Data Section
    st.markdown('<div class="section-header">📊 Understat Last 5 Matches Data</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **📝 Understat Format Guide:**
    Enter data in the format shown on Understat.com: "10.25-1.75"
    - First number: Total xG scored in last 5 matches
    - Second number: Total xGA conceded in last 5 matches
    
    *Example: Arsenal's "10.25-1.75" means 10.25 xG scored and 1.75 xGA conceded in last 5 matches.*
    """)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(f"### 📈 {home_team} - Last 5 Matches")
        home_understat = st.text_input("Understat Format", "4.63-7.71", key="home_understat")
        
        if home_understat:
            try:
                home_xg_pm, home_xga_pm, home_xg_total, home_xga_total = engine.calculate_xg_per_match(home_understat)
                home_league_avg = engine.get_league_average(home_league)
                
                col3a, col3b, col3c = st.columns(3)
                with col3a:
                    st.metric("Total xG Scored", f"{home_xg_total:.2f}")
                with col3b:
                    st.metric("Total xGA Conceded", f"{home_xga_total:.2f}")
                with col3c:
                    st.metric("xG per match", f"{home_xg_pm:.2f}")
                
                col3d, col3e = st.columns(2)
                with col3d:
                    xg_vs_avg = home_xg_pm - home_league_avg["xG"]
                    st.metric("xGA per match", f"{home_xga_pm:.2f}", 
                             delta=f"{xg_vs_avg:+.2f} from database average")
                with col3e:
                    xga_vs_avg = home_league_avg["xGA"] - home_xga_pm
                    st.metric("", "", 
                             delta=f"{xga_vs_avg:+.2f} from database average")
            except:
                st.error("Invalid Understat format for home team")
    
    with col4:
        st.markdown(f"### 📈 {away_team} - Last 5 Matches")
        away_understat = st.text_input("Understat Format", "8.75-10.60", key="away_understat")
        
        if away_understat:
            try:
                away_xg_pm, away_xga_pm, away_xg_total, away_xga_total = engine.calculate_xg_per_match(away_understat)
                away_league_avg = engine.get_league_average(away_league)
                
                col4a, col4b, col4c = st.columns(3)
                with col4a:
                    st.metric("Total xG Scored", f"{away_xg_total:.2f}")
                with col4b:
                    st.metric("Total xGA Conceded", f"{away_xga_total:.2f}")
                with col4c:
                    st.metric("xG per match", f"{away_xg_pm:.2f}")
                
                col4d, col4e = st.columns(2)
                with col4d:
                    xg_vs_avg = away_xg_pm - away_league_avg["xG"]
                    st.metric("xGA per match", f"{away_xga_pm:.2f}", 
                             delta=f"{xg_vs_avg:+.2f} from database average")
                with col4e:
                    xga_vs_avg = away_league_avg["xGA"] - away_xga_pm
                    st.metric("", "", 
                             delta=f"{xga_vs_avg:+.2f} from database average")
            except:
                st.error("Invalid Understat format for away team")
    
    # Match Context Section
    st.markdown('<div class="section-header">🎭 Match Context</div>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("### 🩹 Injury Status")
        home_injuries = st.selectbox(
            f"{home_team} Injuries",
            ["None", "Minor (bench players)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ players)"],
            key="home_injuries"
        )
        
        away_injuries = st.selectbox(
            f"{away_team} Injuries",
            ["None", "Minor (bench players)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ players)"],
            key="away_injuries"
        )
    
    with col6:
        st.markdown("### 🕐 Fatigue & Recovery")
        home_rest = st.number_input(f"{home_team} Rest Days", min_value=1, max_value=14, value=3, key="home_rest")
        away_rest = st.number_input(f"{away_team} Rest Days", min_value=1, max_value=14, value=6, key="away_rest")
        
        rest_diff = away_rest - home_rest
        if rest_diff > 0:
            st.info(f"✈️ {away_team} has {rest_diff} more rest days")
        elif rest_diff < 0:
            st.info(f"🏠 {home_team} has {abs(rest_diff)} more rest days")
        else:
            st.info("⚖️ Both teams have equal rest days")
    
    # Market Odds Section
    st.markdown('<div class="section-header">💰 Market Odds</div>', unsafe_allow_html=True)
    
    col7, col8, col9, col10 = st.columns(4)
    
    with col7:
        st.markdown("### 🏠 Home Win")
        home_odds = st.number_input("Home Odds", min_value=1.01, max_value=10.0, value=2.63, step=0.01, key="home_odds")
    
    with col8:
        st.markdown("### 🤝 Draw")
        draw_odds = st.number_input("Draw Odds", min_value=1.01, max_value=10.0, value=3.60, step=0.01, key="draw_odds")
    
    with col9:
        st.markdown("### ✈️ Away Win")
        away_odds = st.number_input("Away Odds", min_value=1.01, max_value=10.0, value=2.50, step=0.01, key="away_odds")
    
    with col10:
        st.markdown("### ⚽ Over 2.5")
        over_odds = st.number_input("Over 2.5 Odds", min_value=1.01, max_value=10.0, value=1.67, step=0.01, key="over_odds")
    
    # Prediction Button
    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
        if home_understat and away_understat:
            try:
                # Generate prediction
                prediction = engine.predict_match(
                    home_team, away_team, home_understat, away_understat,
                    home_injuries, away_injuries, home_rest, away_rest,
                    home_odds, draw_odds, away_odds, over_odds
                )
                
                # Display Results
                st.markdown("---")
                st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
                
                # Expected Score
                col11, col12, col13 = st.columns([1,2,1])
                with col12:
                    st.markdown(f"### {home_team} vs {away_team}")
                    expected_score = f"{prediction['home_expected_goals']:.1f} - {prediction['away_expected_goals']:.1f}"
                    st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'>{expected_score}</h2>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center;'>Expected Final Score</p>", unsafe_allow_html=True)
                    
                    # Confidence stars
                    confidence_stars = "★" * int(prediction['confidence'] * 5) + "☆" * (5 - int(prediction['confidence'] * 5))
                    confidence_percent = int(prediction['confidence'] * 100)
                    st.markdown(f"<p style='text-align: center;' class='confidence-stars'>Confidence: {confidence_stars} ({confidence_percent}%)</p>", unsafe_allow_html=True)
                
                # Match Outcome Probabilities
                st.markdown("### 📊 Match Outcome Probabilities")
                
                col14, col15, col16 = st.columns(3)
                
                with col14:
                    home_value_rec, home_value_class = engine.get_value_bet_recommendation(prediction['home_value'])
                    st.markdown(f"**🟡 {home_team} Win**")
                    st.metric("Probability", f"{prediction['home_win_prob']*100:.1f}%")
                    st.metric("Odds", f"{home_odds:.2f}")
                    if home_value_class == "good":
                        st.markdown(f'<div class="value-bet-good">{home_value_rec}: {prediction["home_value"]:.2f}x</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="value-bet-poor">{home_value_rec}: {prediction["home_value"]:.2f}x</div>', unsafe_allow_html=True)
                
                with col15:
                    draw_value_rec, draw_value_class = engine.get_value_bet_recommendation(prediction['draw_value'])
                    st.markdown("**🔴 Draw**")
                    st.metric("Probability", f"{prediction['draw_prob']*100:.1f}%")
                    st.metric("Odds", f"{draw_odds:.2f}")
                    if draw_value_class == "good":
                        st.markdown(f'<div class="value-bet-good">{draw_value_rec}: {prediction["draw_value"]:.2f}x</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="value-bet-poor">{draw_value_rec}: {prediction["draw_value"]:.2f}x</div>', unsafe_allow_html=True)
                
                with col16:
                    away_value_rec, away_value_class = engine.get_value_bet_recommendation(prediction['away_value'])
                    st.markdown(f"**🟡 {away_team} Win**")
                    st.metric("Probability", f"{prediction['away_win_prob']*100:.1f}%")
                    st.metric("Odds", f"{away_odds:.2f}")
                    if away_value_class == "good":
                        st.markdown(f'<div class="value-bet-good">{away_value_rec}: {prediction["away_value"]:.2f}x</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="value-bet-poor">{away_value_rec}: {prediction["away_value"]:.2f}x</div>', unsafe_allow_html=True)
                
                # Goals Market
                st.markdown("### ⚽ Goals Market")
                
                col17, col18 = st.columns(2)
                
                with col17:
                    over_value_rec, over_value_class = engine.get_value_bet_recommendation(prediction['over_value'])
                    st.markdown("**🔴 Over 2.5 Goals**")
                    st.metric("Probability", f"{prediction['over_prob']*100:.1f}%")
                    st.metric("Odds", f"{over_odds:.2f}")
                    if over_value_class == "good":
                        st.markdown(f'<div class="value-bet-good">{over_value_rec}: {prediction["over_value"]:.2f}x</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="value-bet-poor">{over_value_rec}: {prediction["over_value"]:.2f}x</div>', unsafe_allow_html=True)
                
                with col18:
                    under_implied_odds = 1 / prediction['under_prob'] if prediction['under_prob'] > 0 else 0
                    st.markdown("**🟡 Under 2.5 Goals**")
                    st.metric("Probability", f"{prediction['under_prob']*100:.1f}%")
                    st.metric("Implied Odds", f"{under_implied_odds:.2f}")
                
                # Value Bets Recommendation
                st.markdown("### 💰 Recommended Value Bets")
                
                value_bets = []
                if prediction['home_value'] >= 1.1:
                    value_bets.append(f"**{home_team} Win** (Value: {prediction['home_value']:.2f}x)")
                if prediction['draw_value'] >= 1.1:
                    value_bets.append(f"**Draw** (Value: {prediction['draw_value']:.2f}x)")
                if prediction['away_value'] >= 1.1:
                    value_bets.append(f"**{away_team} Win** (Value: {prediction['away_value']:.2f}x)")
                if prediction['over_value'] >= 1.1:
                    value_bets.append(f"**Over 2.5 Goals** (Value: {prediction['over_value']:.2f}x)")
                
                if value_bets:
                    for bet in value_bets:
                        st.success(f"🎯 {bet}")
                else:
                    st.info("📊 No strong value bets identified. All market odds appear efficient for this match. Consider waiting for line movement.")
                
                # Key Insights & Analysis
                st.markdown("### 🧠 Key Insights & Analysis")
                
                insights = []
                
                # Injury insights
                if home_injuries != "None":
                    insights.append(f"🩹 {home_team} affected by {home_injuries.lower()}")
                if away_injuries != "None":
                    insights.append(f"🩹 {away_team} affected by {away_injuries.lower()}")
                
                # Rest advantage insights
                if prediction['rest_difference'] > 0:
                    insights.append(f"🕐 {away_team} has {prediction['rest_difference']} extra rest days")
                elif prediction['rest_difference'] < 0:
                    insights.append(f"🕐 {home_team} has {abs(prediction['rest_difference'])} extra rest days")
                
                # Goal expectation insights
                if prediction['total_expected_goals'] > 3.0:
                    insights.append("⚽ High-scoring match expected")
                elif prediction['total_expected_goals'] < 2.0:
                    insights.append("⚽ Low-scoring match expected")
                else:
                    insights.append("⚽ Moderate-scoring match expected")
                
                # Defensive performance insights
                if prediction['home_xga_pm'] < prediction['home_league_avg']['xGA']:
                    insights.append(f"🛡️ {home_team} showing solid defense ({prediction['home_xga_pm']:.2f} xGA/match)")
                else:
                    insights.append(f"🎯 {home_team} defense vulnerable ({prediction['home_xga_pm']:.2f} xGA/match)")
                
                if prediction['away_xga_pm'] < prediction['away_league_avg']['xGA']:
                    insights.append(f"🛡️ {away_team} showing solid defense ({prediction['away_xga_pm']:.2f} xGA/match)")
                else:
                    insights.append(f"🎯 {away_team} defense vulnerable ({prediction['away_xga_pm']:.2f} xGA/match)")
                
                # Display insights
                for insight in insights:
                    st.write(f"• {insight}")
                
                # Statistical Summary
                st.markdown("### 📈 Statistical Summary")
                
                col19, col20, col21 = st.columns(3)
                
                with col19:
                    st.metric("Total Expected Goals", f"{prediction['total_expected_goals']:.2f}")
                
                with col20:
                    st.metric("Goal Expectancy", f"{prediction['goal_expectancy']:.1f}% of average match")
                
                with col21:
                    st.metric("Confidence Score", f"{prediction['confidence']*100:.1f}%")
                
                # Team form summary
                st.markdown("#### Team Form Analysis")
                col22, col23 = st.columns(2)
                
                with col22:
                    st.write(f"**{home_team} Form:** {prediction['home_xg_pm']:.2f} xG, {prediction['home_xga_pm']:.2f} xGA per match")
                
                with col23:
                    st.write(f"**{away_team} Form:** {prediction['away_xg_pm']:.2f} xG, {prediction['away_xga_pm']:.2f} xGA per match")
                
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
        else:
            st.error("Please enter valid Understat data for both teams")

if __name__ == "__main__":
    main()
