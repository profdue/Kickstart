import streamlit as st
import pandas as pd
import numpy as np
from prediction_engine.league_manager import LEAGUE_CONFIGS
from prediction_engine.data_processor import prepare_match_data
from prediction_engine.statistical_model import predict_match
from prediction_engine.confidence_calculator import calculate_confidence
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Football Prediction Engine",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-high { color: #00a650; font-weight: bold; }
    .confidence-medium { color: #ffa500; font-weight: bold; }
    .confidence-low { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">⚽ Football Prediction Engine</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictions_log' not in st.session_state:
        st.session_state.predictions_log = []
    
    # Main layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("🎯 Match Configuration")
        
        # League selection
        league = st.selectbox(
            "Select League",
            list(LEAGUE_CONFIGS.keys()),
            index=0
        )
        
        # Team selection based on league
        teams = LEAGUE_CONFIGS[league]["teams"]
        home_team = st.selectbox("Home Team", teams, index=0)
        away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)
        
        if home_team == away_team:
            st.error("Home and Away teams cannot be the same!")
            return
    
    with col2:
        st.subheader("📊 Team Statistics - Home")
        
        # Home team stats
        with st.expander("Home Team Data", expanded=True):
            st.write("**Overall Stats**")
            col1, col2, col3 = st.columns(3)
            with col1:
                home_matches = st.number_input("Matches", min_value=1, max_value=50, value=10, key="home_m")
            with col2:
                home_goals = st.number_input("Goals", min_value=0, max_value=100, value=18, key="home_g")
            with col3:
                home_goals_against = st.number_input("GA", min_value=0, max_value=100, value=3, key="home_ga")
            
            col1, col2 = st.columns(2)
            with col1:
                home_xg = st.number_input("xG", min_value=0.0, max_value=50.0, value=18.7, key="home_xg")
            with col2:
                home_xga = st.number_input("xGA", min_value=0.0, max_value=50.0, value=6.6, key="home_xga")
            
            st.write("**Home Stats Only**")
            col1, col2, col3 = st.columns(3)
            with col1:
                home_home_matches = st.number_input("Home Matches", min_value=1, max_value=25, value=5, key="home_hm")
            with col2:
                home_home_goals = st.number_input("Home Goals", min_value=0, max_value=50, value=12, key="home_hg")
            with col3:
                home_home_ga = st.number_input("Home GA", min_value=0, max_value=50, value=2, key="home_hga")
            
            col1, col2 = st.columns(2)
            with col1:
                home_home_xg = st.number_input("Home xG", min_value=0.0, max_value=25.0, value=8.1, key="home_hxg")
            with col2:
                home_home_xga = st.number_input("Home xGA", min_value=0.0, max_value=25.0, value=3.2, key="home_hxga")
            
            st.write("**Last 5 Matches**")
            home_last5_xg = st.number_input("Last 5 xG Total", min_value=0.0, max_value=25.0, value=10.25, key="home_l5xg")
            home_last5_points = st.number_input("Last 5 Points", min_value=0, max_value=15, value=13, key="home_l5p")
    
    with col3:
        st.subheader("📊 Team Statistics - Away")
        
        # Away team stats
        with st.expander("Away Team Data", expanded=True):
            st.write("**Overall Stats**")
            col1, col2, col3 = st.columns(3)
            with col1:
                away_matches = st.number_input("Matches", min_value=1, max_value=50, value=10, key="away_m")
            with col2:
                away_goals = st.number_input("Goals", min_value=0, max_value=100, value=20, key="away_g")
            with col3:
                away_goals_against = st.number_input("GA", min_value=0, max_value=100, value=8, key="away_ga")
            
            col1, col2 = st.columns(2)
            with col1:
                away_xg = st.number_input("xG", min_value=0.0, max_value=50.0, value=19.5, key="away_xg")
            with col2:
                away_xga = st.number_input("xGA", min_value=0.0, max_value=50.0, value=10.0, key="away_xga")
            
            st.write("**Away Stats Only**")
            col1, col2, col3 = st.columns(3)
            with col1:
                away_away_matches = st.number_input("Away Matches", min_value=1, max_value=25, value=5, key="away_am")
            with col2:
                away_away_goals = st.number_input("Away Goals", min_value=0, max_value=50, value=8, key="away_ag")
            with col3:
                away_away_ga = st.number_input("Away GA", min_value=0, max_value=50, value=6, key="away_aga")
            
            col1, col2 = st.columns(2)
            with col1:
                away_away_xg = st.number_input("Away xG", min_value=0.0, max_value=25.0, value=7.9, key="away_axg")
            with col2:
                away_away_xga = st.number_input("Away xGA", min_value=0.0, max_value=25.0, value=5.1, key="away_axga")
            
            st.write("**Last 5 Matches**")
            away_last5_xg = st.number_input("Last 5 xG Total", min_value=0.0, max_value=25.0, value=11.44, key="away_l5xg")
            away_last5_points = st.number_input("Last 5 Points", min_value=0, max_value=15, value=12, key="away_l5p")
    
    # Contextual factors
    st.subheader("🎭 Contextual Factors")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_injuries = st.text_input("Home Team Key Injuries", placeholder="e.g., Saliba, Saka")
        away_injuries = st.text_input("Away Team Key Injuries", placeholder="e.g., De Bruyne")
    
    with col2:
        home_days_rest = st.number_input("Home Days Since Last Match", min_value=2, max_value=14, value=4)
        away_days_rest = st.number_input("Away Days Since Last Match", min_value=2, max_value=14, value=6)
    
    with col3:
        match_importance = st.slider("Match Importance", 0.0, 1.0, 0.7, 0.1,
                                   format="%.1f (Friendly - Cup Final)")
    
    # Prediction button
    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Calculating predictions..."):
            # Prepare data
            home_data = {
                'name': home_team,
                'overall': {
                    'matches': home_matches,
                    'goals_scored': home_goals,
                    'goals_conceded': home_goals_against,
                    'xG': home_xg,
                    'xGA': home_xga
                },
                'home': {
                    'matches': home_home_matches,
                    'goals_scored': home_home_goals,
                    'goals_conceded': home_home_ga,
                    'xG': home_home_xg,
                    'xGA': home_home_xga
                },
                'last_5': {
                    'xG_total': home_last5_xg,
                    'points': home_last5_points
                }
            }
            
            away_data = {
                'name': away_team,
                'overall': {
                    'matches': away_matches,
                    'goals_scored': away_goals,
                    'goals_conceded': away_goals_against,
                    'xG': away_xg,
                    'xGA': away_xga
                },
                'away': {
                    'matches': away_away_matches,
                    'goals_scored': away_away_goals,
                    'goals_conceded': away_away_ga,
                    'xG': away_away_xg,
                    'xGA': away_away_xga
                },
                'last_5': {
                    'xG_total': away_last5_xg,
                    'points': away_last5_points
                }
            }
            
            context = {
                'home_injuries': [inj.strip() for inj in home_injuries.split(',')] if home_injuries else [],
                'away_injuries': [inj.strip() for inj in away_injuries.split(',')] if away_injuries else [],
                'home_days_rest': home_days_rest,
                'away_days_rest': away_days_rest,
                'match_importance': match_importance
            }
            
            # Generate prediction
            prediction = predict_match(home_data, away_data, context, league)
            
            # Display results
            display_prediction_results(prediction, home_team, away_team, league)
            
            # Log prediction
            st.session_state.predictions_log.append({
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'prediction': prediction,
                'timestamp': pd.Timestamp.now()
            })

def display_prediction_results(prediction, home_team, away_team, league):
    st.markdown("---")
    st.markdown('<div class="main-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    
    # Main predictions in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏆 Match Outcome")
        outcome = prediction['match_outcome']
        display_confidence(outcome['confidence'])
        
        # Create bar chart for outcome probabilities
        fig_outcome = go.Figure(data=[
            go.Bar(x=['Home', 'Draw', 'Away'],
                  y=[outcome['home_win'], outcome['draw'], outcome['away_win']],
                  marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ])
        fig_outcome.update_layout(
            height=300,
            showlegend=False,
            yaxis_title="Probability",
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig_outcome, use_container_width=True)
        
        # Display percentages
        st.write(f"**{home_team}**: {outcome['home_win']:.1%}")
        st.write(f"**Draw**: {outcome['draw']:.1%}")
        st.write(f"**{away_team}**: {outcome['away_win']:.1%}")
    
    with col2:
        st.subheader("📊 Over/Under 2.5")
        over_under = prediction['over_under']
        display_confidence(over_under['confidence'])
        
        fig_ou = go.Figure(data=[
            go.Bar(x=['Over 2.5', 'Under 2.5'],
                  y=[over_under['over_2.5'], over_under['under_2.5']],
                  marker_color=['#ff6b6b', '#4ecdc4'])
        ])
        fig_ou.update_layout(
            height=300,
            showlegend=False,
            yaxis_title="Probability",
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig_ou, use_container_width=True)
        
        st.write(f"**Over 2.5**: {over_under['over_2.5']:.1%}")
        st.write(f"**Under 2.5**: {over_under['under_2.5']:.1%}")
    
    with col3:
        st.subheader("⚽ Both Teams to Score")
        btts = prediction['both_teams_score']
        display_confidence(btts['confidence'])
        
        fig_btts = go.Figure(data=[
            go.Bar(x=['Yes', 'No'],
                  y=[btts['yes'], btts['no']],
                  marker_color=['#a05195', '#f95d6a'])
        ])
        fig_btts.update_layout(
            height=300,
            showlegend=False,
            yaxis_title="Probability",
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig_btts, use_container_width=True)
        
        st.write(f"**Yes**: {btts['yes']:.1%}")
        st.write(f"**No**: {btts['no']:.1%}")
    
    # Additional insights
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Expected Score")
        expected_score = prediction['expected_score']
        st.metric(
            label="Expected Goals",
            value=f"{expected_score['home']:.1f} - {expected_score['away']:.1f}",
            delta=f"Total: {expected_score['home'] + expected_score['away']:.1f} goals"
        )
        
        st.subheader("🎯 Most Likely Scores")
        for score in prediction['most_likely_scores'][:3]:
            st.write(f"**{score['score']}**: {score['probability']:.1%}")
    
    with col2:
        st.subheader("🔍 Key Factors")
        for factor in prediction['key_factors']:
            emoji = "✅" if factor['impact'] == 'positive' else "⚠️" if factor['impact'] == 'negative' else "➖"
            st.write(f"{emoji} {factor['factor']}")

def display_confidence(confidence_level):
    if confidence_level == "HIGH":
        st.markdown('<p class="confidence-high">🟢 High Confidence</p>', unsafe_allow_html=True)
    elif confidence_level == "MEDIUM":
        st.markdown('<p class="confidence-medium">🟡 Medium Confidence</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="confidence-low">🔴 Low Confidence</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()