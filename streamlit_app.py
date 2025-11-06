import streamlit as st
import pandas as pd
import numpy as np
from prediction_engine.league_manager import LEAGUE_CONFIGS
from prediction_engine.data_processor import prepare_match_data
from prediction_engine.statistical_model import predict_match

# Page configuration
st.set_page_config(
    page_title="Football Prediction Engine",
    page_icon="⚽",
    layout="wide"
)

# Title and description
st.title("⚽ Football Prediction Engine")
st.markdown("Predict match outcomes using expected goals (xG) data")

# Initialize session state for team data
if 'team_data' not in st.session_state:
    st.session_state.team_data = {}

# League and team selection
col1, col2, col3 = st.columns(3)

with col1:
    league = st.selectbox(
        "Select League",
        list(LEAGUE_CONFIGS.keys())
    )

with col2:
    home_team = st.selectbox(
        "Home Team",
        LEAGUE_CONFIGS[league]["teams"]
    )

with col3:
    away_team = st.selectbox(
        "Away Team", 
        LEAGUE_CONFIGS[league]["teams"]
    )

# Display league context
league_info = LEAGUE_CONFIGS[league]["baselines"]
st.info(
    f"**{league} Context** | "
    f"Avg Goals: {league_info['avg_goals']} | "
    f"Home Advantage: +{int((league_info['home_advantage']-1)*100)}% | "
    f"BTTS Frequency: {int(league_info['avg_btts_prob']*100)}%"
)

# Team data input section
st.header("📊 Team Statistics Input")

# Create tabs for home and away teams
tab1, tab2 = st.tabs([f"🏠 {home_team} Data", f"✈️ {away_team} Data"])

with tab1:
    st.subheader(f"{home_team} - Current Season Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Overall Stats**")
        home_overall_m = st.number_input("Matches", min_value=1, value=10, key="home_overall_m")
        home_overall_g = st.number_input("Goals Scored", min_value=0, value=18, key="home_overall_g")
        home_overall_ga = st.number_input("Goals Conceded", min_value=0, value=3, key="home_overall_ga")
        home_overall_xg = st.number_input("xG", min_value=0.0, value=18.7, key="home_overall_xg")
        home_overall_xga = st.number_input("xGA", min_value=0.0, value=6.6, key="home_overall_xga")
    
    with col2:
        st.markdown("**Home Stats**")
        home_home_m = st.number_input("Matches", min_value=1, value=5, key="home_home_m")
        home_home_g = st.number_input("Goals Scored", min_value=0, value=12, key="home_home_g")
        home_home_ga = st.number_input("Goals Conceded", min_value=0, value=2, key="home_home_ga")
        home_home_xg = st.number_input("xG", min_value=0.0, value=8.1, key="home_home_xg")
        home_home_xga = st.number_input("xGA", min_value=0.0, value=3.2, key="home_home_xga")
    
    with col3:
        st.markdown("**Last 5 Matches (Overall)**")
        home_last5_xg = st.number_input("xG Total", min_value=0.0, value=10.25, key="home_last5_xg")
        home_last5_points = st.number_input("Points", min_value=0, value=13, key="home_last5_points")
        home_last5_goals = st.number_input("Goals Scored", min_value=0, value=12, key="home_last5_goals")

with tab2:
    st.subheader(f"{away_team} - Current Season Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Overall Stats**")
        away_overall_m = st.number_input("Matches", min_value=1, value=10, key="away_overall_m")
        away_overall_g = st.number_input("Goals Scored", min_value=0, value=20, key="away_overall_g")
        away_overall_ga = st.number_input("Goals Conceded", min_value=0, value=8, key="away_overall_ga")
        away_overall_xg = st.number_input("xG", min_value=0.0, value=19.5, key="away_overall_xg")
        away_overall_xga = st.number_input("xGA", min_value=0.0, value=10.0, key="away_overall_xga")
    
    with col2:
        st.markdown("**Away Stats**")
        away_away_m = st.number_input("Matches", min_value=1, value=5, key="away_away_m")
        away_away_g = st.number_input("Goals Scored", min_value=0, value=8, key="away_away_g")
        away_away_ga = st.number_input("Goals Conceded", min_value=0, value=6, key="away_away_ga")
        away_away_xg = st.number_input("xG", min_value=0.0, value=7.9, key="away_away_xg")
        away_away_xga = st.number_input("xGA", min_value=0.0, value=5.1, key="away_away_xga")
    
    with col3:
        st.markdown("**Last 5 Matches (Overall)**")
        away_last5_xg = st.number_input("xG Total", min_value=0.0, value=11.44, key="away_last5_xg")
        away_last5_points = st.number_input("Points", min_value=0, value=12, key="away_last5_points")
        away_last5_goals = st.number_input("Goals Scored", min_value=0, value=13, key="away_last5_goals")

# Contextual factors
st.header("🎯 Contextual Factors")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Team Context")
    home_injuries = st.text_input(f"{home_team} Key Injuries/Suspensions", placeholder="e.g., Saliba, Saka")
    away_injuries = st.text_input(f"{away_team} Key Injuries/Suspensions", placeholder="e.g., De Bruyne")
    
    home_days_rest = st.number_input(f"{home_team} Days Rest", min_value=2, max_value=14, value=4)
    away_days_rest = st.number_input(f"{away_team} Days Rest", min_value=2, max_value=14, value=6)

with col2:
    st.subheader("Match Context")
    match_importance = st.slider(
        "Match Importance",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="0.0 = Friendly, 1.0 = Cup Final/Title Decider"
    )
    
    # Recent form arrays (simplified)
    st.markdown("**Recent Form (Last 5 xG - Optional)**")
    home_recent_xg = st.text_input(f"{home_team} Last 5 xG", placeholder="2.1,1.8,2.3,1.5,2.0")
    away_recent_xg = st.text_input(f"{away_team} Last 5 xG", placeholder="1.4,1.9,1.2,2.1,1.6")

# Prediction button
if st.button("🎯 Generate Prediction", type="primary"):
    st.header("📊 Prediction Results")
    
    # Display loading while processing
    with st.spinner("Calculating predictions..."):
        # Prepare data structures
        home_data = {
            'name': home_team,
            'overall': {'matches': home_overall_m, 'goals_scored': home_overall_g, 
                       'goals_conceded': home_overall_ga, 'xG': home_overall_xg, 'xGA': home_overall_xga},
            'home': {'matches': home_home_m, 'goals_scored': home_home_g, 
                    'goals_conceded': home_home_ga, 'xG': home_home_xg, 'xGA': home_home_xga},
            'last5': {'xG_total': home_last5_xg, 'points': home_last5_points, 'goals_scored': home_last5_goals},
            'context': {'injuries': home_injuries, 'days_rest': home_days_rest}
        }
        
        away_data = {
            'name': away_team,
            'overall': {'matches': away_overall_m, 'goals_scored': away_overall_g, 
                       'goals_conceded': away_overall_ga, 'xG': away_overall_xg, 'xGA': away_overall_xga},
            'away': {'matches': away_away_m, 'goals_scored': away_away_g, 
                    'goals_conceded': away_away_ga, 'xG': away_away_xg, 'xGA': away_away_xga},
            'last5': {'xG_total': away_last5_xg, 'points': away_last5_points, 'goals_scored': away_last5_goals},
            'context': {'injuries': away_injuries, 'days_rest': away_days_rest}
        }
        
        match_context = {
            'importance': match_importance,
            'home_recent_xg_array': [float(x) for x in home_recent_xg.split(',')] if home_recent_xg else None,
            'away_recent_xg_array': [float(x) for x in away_recent_xg.split(',')] if away_recent_xg else None
        }
        
        # Generate prediction
        try:
            prediction = predict_match(home_data, away_data, league, match_context)
            
            # Display results
            display_prediction_results(prediction, home_team, away_team)
            
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")
            st.info("Please check your input data and try again.")

def display_prediction_results(prediction, home_team, away_team):
    # This will be implemented in Phase 2
    st.success("Prediction engine connected successfully!")
    st.write("Full prediction results will be displayed here in Phase 2")
    st.json(prediction)  # Temporary to see the data structure

# Footer
st.markdown("---")
st.markdown("*Data sources: Understat.com | Model: Bivariate Poisson with league-specific baselines*")
