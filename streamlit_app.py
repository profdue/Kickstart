import streamlit as st
import pandas as pd
import numpy as np

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'team_selection'
if 'selected_teams' not in st.session_state:
    st.session_state.selected_teams = {}

def main():
    st.set_page_config(
        page_title="Football Prediction Engine",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .value-bet-good {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .value-bet-poor {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Page routing
    if st.session_state.current_page == 'team_selection':
        show_team_selection()
    elif st.session_state.current_page == 'advanced_settings':
        show_advanced_settings()
    elif st.session_state.current_page == 'prediction_results':
        show_prediction_results()

def show_team_selection():
    st.markdown('<h1 style="text-align: center;">🎯 Football Prediction Engine</h1>', unsafe_allow_html=True)
    
    # League Selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_league = st.selectbox(
            "SELECT LEAGUE",
            list(LEAGUES.keys()),
            key="league_select"
        )
    
    st.markdown("---")
    
    # Team Selection Cards
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.subheader("🏠 HOME TEAM")
        home_team = st.selectbox(
            "Select Home Team",
            LEAGUES[selected_league],
            key="home_team_select"
        )
        
        # Show team stats
        home_data = get_team_data(home_team)
        st.metric("Expected Goals (xG)", f"{home_data['xg']:.2f}")
        st.metric("Expected Goals Against (xGA)", f"{home_data['xga']:.2f}")
        st.metric("Recent Form", home_data['form'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="text-align: center; padding: 4rem 0;">', unsafe_allow_html=True)
        st.markdown('<h1>VS</h1>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.subheader("✈️ AWAY TEAM")
        away_team = st.selectbox(
            "Select Away Team", 
            LEAGUES[selected_league],
            key="away_team_select"
        )
        
        # Show team stats
        away_data = get_team_data(away_team)
        st.metric("Expected Goals (xG)", f"{away_data['xg']:.2f}")
        st.metric("Expected Goals Against (xGA)", f"{away_data['xga']:.2f}")
        st.metric("Recent Form", away_data['form'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action Buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
        
        if st.button("🚀 QUICK PREDICTION", use_container_width=True, type="primary"):
            st.session_state.selected_teams = {
                'league': selected_league,
                'home_team': home_team,
                'away_team': away_team,
                'home_data': home_data,
                'away_data': away_data
            }
            st.session_state.current_page = 'prediction_results'
            st.rerun()
        
        if st.button("⚙️ CUSTOMIZE SETTINGS", use_container_width=True):
            st.session_state.selected_teams = {
                'league': selected_league,
                'home_team': home_team,
                'away_team': away_team,
                'home_data': home_data,
                'away_data': away_data
            }
            st.session_state.current_page = 'advanced_settings'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_advanced_settings():
    st.markdown('<h1 style="text-align: center;">⚙️ Advanced Settings</h1>', unsafe_allow_html=True)
    
    teams = st.session_state.selected_teams
    home_data = teams['home_data']
    away_data = teams['away_data']
    
    st.markdown(f"### {teams['home_team']} vs {teams['away_team']} - {teams['league']}")
    st.markdown("---")
    
    # Team Metrics Adjustment
    st.subheader("📊 Team Metrics Adjustment")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.write("**METRIC**")
        st.write("Expected Goals")
        st.write("Expected Goals Against")
    
    with col2:
        st.write(f"**HOME ({teams['home_team']})**")
        home_xg = st.slider(
            "Home xG", 0.5, 4.0, home_data['xg'], 0.1,
            key="home_xg_adv",
            label_visibility="collapsed"
        )
        home_xga = st.slider(
            "Home xGA", 0.5, 4.0, home_data['xga'], 0.1,
            key="home_xga_adv", 
            label_visibility="collapsed"
        )
        
        # Show impact
        xg_diff = home_xg - home_data['xg']
        xga_diff = home_xga - home_data['xga']
        if xg_diff != 0:
            color = "green" if xg_diff > 0 else "red"
            st.markdown(f"<small style='color:{color}'>📊 {xg_diff:+.2f} from average</small>", unsafe_allow_html=True)
        if xga_diff != 0:
            color = "red" if xga_diff > 0 else "green"
            st.markdown(f"<small style='color:{color}'>🛡️ {xga_diff:+.2f} from average</small>", unsafe_allow_html=True)
    
    with col3:
        st.write(f"**AWAY ({teams['away_team']})**")
        away_xg = st.slider(
            "Away xG", 0.5, 4.0, away_data['xg'], 0.1,
            key="away_xg_adv",
            label_visibility="collapsed"
        )
        away_xga = st.slider(
            "Away xGA", 0.5, 4.0, away_data['xga'], 0.1,
            key="away_xga_adv",
            label_visibility="collapsed"
        )
        
        # Show impact
        xg_diff = away_xg - away_data['xg']
        xga_diff = away_xga - away_data['xga']
        if xg_diff != 0:
            color = "green" if xg_diff > 0 else "red"
            st.markdown(f"<small style='color:{color}'>📊 {xg_diff:+.2f} from average</small>", unsafe_allow_html=True)
        if xga_diff != 0:
            color = "red" if xga_diff > 0 else "green"
            st.markdown(f"<small style='color:{color}'>🛡️ {xga_diff:+.2f} from average</small>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Match Context
    st.subheader("🎭 Match Context")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🩹 Key Injuries**")
        home_injuries = st.select_slider(
            f"{teams['home_team']} Injuries",
            options=["None", "1-2 Rotational", "1-2 Starters", "3+ Starters"],
            key="home_injuries"
        )
        away_injuries = st.select_slider(
            f"{teams['away_team']} Injuries", 
            options=["None", "1-2 Rotational", "1-2 Starters", "3+ Starters"],
            key="away_injuries"
        )
    
    with col2:
        st.write("**🕐 Days Rest**")
        home_rest = st.slider(
            f"{teams['home_team']} Rest Days",
            2, 14, 7,
            key="home_rest"
        )
        away_rest = st.slider(
            f"{teams['away_team']} Rest Days",
            2, 14, 7, 
            key="away_rest"
        )
        
        # Fatigue indicator
        rest_diff = home_rest - away_rest
        if rest_diff >= 3:
            st.success(f"🏠 {teams['home_team']} has {rest_diff} extra rest days")
        elif rest_diff <= -3:
            st.warning(f"✈️ {teams['away_team']} has {-rest_diff} extra rest days")
    
    with col3:
        st.write("**📈 Recent Form**")
        home_form = st.radio(
            f"{teams['home_team']} Form",
            ["Declining 🔻", "Stable ➡️", "Improving 🔺"],
            horizontal=False,
            key="home_form"
        )
        away_form = st.radio(
            f"{teams['away_team']} Form",
            ["Declining 🔻", "Stable ➡️", "Improving 🔺"], 
            horizontal=False,
            key="away_form"
        )
    
    st.markdown("---")
    
    # Market Odds
    st.subheader("💰 Market Odds")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        home_odds = st.number_input(
            "Home Win Odds",
            min_value=1.01,
            max_value=100.0,
            value=2.50,
            step=0.1,
            key="home_odds"
        )
    with col2:
        draw_odds = st.number_input(
            "Draw Odds",
            min_value=1.01, 
            max_value=100.0,
            value=3.40,
            step=0.1,
            key="draw_odds"
        )
    with col3:
        away_odds = st.number_input(
            "Away Win Odds",
            min_value=1.01,
            max_value=100.0, 
            value=2.80,
            step=0.1,
            key="away_odds"
        )
    with col4:
        over_odds = st.number_input(
            "Over 2.5 Goals Odds",
            min_value=1.01,
            max_value=100.0,
            value=1.90,
            step=0.1,
            key="over_odds"
        )
    
    st.markdown("---")
    
    # Action Buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 GENERATE PREDICTION", use_container_width=True, type="primary"):
            # Store all inputs
            st.session_state.prediction_inputs = {
                'home_xg': home_xg, 'home_xga': home_xga,
                'away_xg': away_xg, 'away_xga': away_xga,
                'home_injuries': home_injuries, 'away_injuries': away_injuries,
                'home_rest': home_rest, 'away_rest': away_rest,
                'home_form': home_form, 'away_form': away_form,
                'home_odds': home_odds, 'draw_odds': draw_odds,
                'away_odds': away_odds, 'over_odds': over_odds
            }
            st.session_state.current_page = 'prediction_results'
            st.rerun()
        
        if st.button("← BACK TO TEAM SELECTION", use_container_width=True):
            st.session_state.current_page = 'team_selection'
            st.rerun()

def show_prediction_results():
    st.markdown('<h1 style="text-align: center;">🎯 Prediction Results</h1>', unsafe_allow_html=True)
    
    teams = st.session_state.selected_teams
    inputs = st.session_state.prediction_inputs
    
    # Generate prediction
    result = generate_prediction(teams, inputs)
    
    # Header with expected score
    st.markdown(f'<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown(f'<h2>{teams["home_team"]} vs {teams["away_team"]}</h2>', unsafe_allow_html=True)
    st.markdown(f'<h3>{teams["league"]}</h3>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size: 4rem; margin: 1rem 0;">{result["expected_score"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p>Expected Final Score</p>', unsafe_allow_html=True)
    
    # Confidence badge
    confidence_stars = "★" * result['confidence_stars'] + "☆" * (5 - result['confidence_stars'])
    st.markdown(f'<p>Confidence: {confidence_stars} ({result["confidence_score"]}% - {result["confidence_text"]})</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Outcome Probabilities
    st.subheader("📊 Match Outcome Probabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_color = "🟢" if result['probabilities']['home'] > 0.4 else "🟡" if result['probabilities']['home'] > 0.3 else "🔴"
        st.metric(f"{home_color} {teams['home_team']} Win", 
                 f"{result['probabilities']['home']:.1%}",
                 f"Odds: {inputs['home_odds']}")
        
        value_home = result['value_bets']['home']['value_ratio']
        ev_home = result['value_bets']['home']['ev']
        if value_home > 1.1:
            st.success(f"Value: {value_home:.2f}x | EV: {ev_home:.1%}")
        else:
            st.error(f"Value: {value_home:.2f}x | EV: {ev_home:.1%}")
    
    with col2:
        draw_color = "🟢" if result['probabilities']['draw'] > 0.3 else "🟡" if result['probabilities']['draw'] > 0.25 else "🔴"
        st.metric(f"{draw_color} Draw", 
                 f"{result['probabilities']['draw']:.1%}",
                 f"Odds: {inputs['draw_odds']}")
        
        value_draw = result['value_bets']['draw']['value_ratio']
        ev_draw = result['value_bets']['draw']['ev']
        if value_draw > 1.1:
            st.success(f"Value: {value_draw:.2f}x | EV: {ev_draw:.1%}")
        else:
            st.error(f"Value: {value_draw:.2f}x | EV: {ev_draw:.1%}")
    
    with col3:
        away_color = "🟢" if result['probabilities']['away'] > 0.4 else "🟡" if result['probabilities']['away'] > 0.3 else "🔴"
        st.metric(f"{away_color} {teams['away_team']} Win", 
                 f"{result['probabilities']['away']:.1%}",
                 f"Odds: {inputs['away_odds']}")
        
        value_away = result['value_bets']['away']['value_ratio']
        ev_away = result['value_bets']['away']['ev']
        if value_away > 1.1:
            st.success(f"Value: {value_away:.2f}x | EV: {ev_away:.1%}")
        else:
            st.error(f"Value: {value_away:.2f}x | EV: {ev_away:.1%}")
    
    st.markdown("---")
    
    # Recommended Bets
    st.subheader("💰 Recommended Bets")
    
    good_bets = [bet for bet in result['value_bets'].values() if bet['value_ratio'] > 1.1]
    
    if good_bets:
        for bet in good_bets:
            st.markdown(f'<div class="value-bet-good">', unsafe_allow_html=True)
            st.markdown(f"**✅ {bet['type']} @ {bet['odds']}**")
            st.markdown(f"Model Probability: {bet['probability']:.1%} vs Market: {bet['implied_prob']:.1%}")
            st.markdown(f"Value: {bet['value_ratio']:.2f}x | Expected Value: {bet['ev']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="value-bet-poor">', unsafe_allow_html=True)
        st.markdown("**No strong value bets identified**")
        st.markdown("All market odds are efficient for this match")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Insights
    st.subheader("🧠 Key Insights")
    
    for insight in result['insights']:
        st.markdown(f'<div class="insight-card">• {insight}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 NEW PREDICTION", use_container_width=True):
            st.session_state.current_page = 'team_selection'
            st.rerun()
    
    with col2:
        if st.button("⚙️ ADJUST SETTINGS", use_container_width=True, type="primary"):
            st.session_state.current_page = 'advanced_settings'
            st.rerun()
    
    with col3:
        if st.button("📊 ADVANCED STATS", use_container_width=True):
            st.info("Advanced statistics feature coming soon!")

# Helper functions
def get_team_data(team_name):
    # This would connect to your team database
    return {
        'xg': np.random.uniform(1.0, 2.5),
        'xga': np.random.uniform(0.8, 1.8),
        'form': np.random.choice(['↗️ Improving', '→ Stable', '↘️ Declining'])
    }

def generate_prediction(teams, inputs):
    # This would use your prediction engine
    return {
        'expected_score': f"{np.random.uniform(1.5, 3.0):.1f} - {np.random.uniform(0.5, 2.5):.1f}",
        'probabilities': {
            'home': np.random.uniform(0.3, 0.6),
            'draw': np.random.uniform(0.2, 0.35),
            'away': np.random.uniform(0.2, 0.4)
        },
        'confidence_score': np.random.randint(70, 95),
        'confidence_stars': np.random.randint(3, 6),
        'confidence_text': np.random.choice(['High', 'Very High', 'Excellent']),
        'value_bets': {
            'home': {'value_ratio': np.random.uniform(0.8, 1.3), 'ev': np.random.uniform(-0.1, 0.3), 'type': 'Home Win', 'odds': inputs['home_odds'], 'probability': np.random.uniform(0.3, 0.6), 'implied_prob': 1/inputs['home_odds']},
            'draw': {'value_ratio': np.random.uniform(0.8, 1.3), 'ev': np.random.uniform(-0.1, 0.3), 'type': 'Draw', 'odds': inputs['draw_odds'], 'probability': np.random.uniform(0.2, 0.35), 'implied_prob': 1/inputs['draw_odds']},
            'away': {'value_ratio': np.random.uniform(0.8, 1.3), 'ev': np.random.uniform(-0.1, 0.3), 'type': 'Away Win', 'odds': inputs['away_odds'], 'probability': np.random.uniform(0.2, 0.4), 'implied_prob': 1/inputs['away_odds']}
        },
        'insights': [
            f"{teams['home_team']}'s home advantage provides +0.3 xG boost",
            f"{teams['away_team']} missing 2 key defenders in back line",
            "Both teams in good scoring form recently",
            "High probability of Both Teams to Score (64%)",
            f"{teams['home_team']} has {inputs['home_rest'] - inputs['away_rest']} extra rest days"
        ]
    }

# League data
LEAGUES = {
    "Premier League": ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool", "Luton Town", "Manchester City", "Manchester United", "Newcastle", "Nottingham Forest", "Sheffield United", "Tottenham", "West Ham", "Wolves"],
    "La Liga": ["Alaves", "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Betis", "Celta Vigo", "Cadiz", "Getafe", "Girona", "Granada", "Las Palmas", "Mallorca", "Osasuna", "Rayo Vallecano", "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal"],
    "Serie A": ["AC Milan", "Atalanta", "Bologna", "Cagliari", "Empoli", "Fiorentina", "Frosinone", "Genoa", "Inter Milan", "Juventus", "Lazio", "Lecce", "Monza", "Napoli", "Roma", "Salernitana", "Sassuolo", "Torino", "Udinese", "Verona"],
    "Bundesliga": ["Augsburg", "Bayer Leverkusen", "Bayern Munich", "Bochum", "Borussia Dortmund", "Borussia M'gladbach", "Darmstadt", "Eintracht Frankfurt", "Freiburg", "Heidenheim", "Hoffenheim", "Koln", "Mainz", "RB Leipzig", "Stuttgart", "Union Berlin", "Werder Bremen", "Wolfsburg"],
    "Ligue 1": ["AS Monaco", "Brest", "Clermont Foot", "Le Havre", "Lens", "Lille", "Lorient", "Lyon", "Marseille", "Metz", "Montpellier", "Nantes", "Nice", "Paris Saint-Germain", "Reims", "Rennes", "Strasbourg", "Toulouse"]
}

if __name__ == "__main__":
    main()
