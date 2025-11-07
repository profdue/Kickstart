import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
</style>
""", unsafe_allow_html=True)

class EnhancedPredictionEngine:
    def __init__(self):
        self.league_avg_xg = 1.35
        self.league_avg_xga = 1.35
        
        # Comprehensive team database
        self.team_database = {
            "Tottenham": {"league": "EPL", "xg": 1.95, "xga": 1.45, "possession": 58, "tactical_style": "HIGH_PRESS"},
            "Manchester United": {"league": "EPL", "xg": 1.65, "xga": 1.40, "possession": 54, "tactical_style": "TRANSITION"},
            "Arsenal": {"league": "EPL", "xg": 2.10, "xga": 0.95, "possession": 58, "tactical_style": "HIGH_PRESS"},
            "Manchester City": {"league": "EPL", "xg": 2.35, "xga": 0.85, "possession": 65, "tactical_style": "POSSESSION"},
            "Liverpool": {"league": "EPL", "xg": 2.25, "xga": 1.05, "possession": 62, "tactical_style": "GEGENPRESS"},
        }
        
        # Enhanced tactical style effects
        self.tactical_effects = {
            ('HIGH_PRESS', 'TRANSITION'): {'home_xg_mod': +0.10, 'away_xg_mod': -0.05, 'explanation': "Home high press disrupts away transitions"},
            ('COUNTER_ATTACK', 'HIGH_PRESS'): {'home_xg_mod': +0.15, 'away_xg_mod': 0, 'explanation': "Home counter attack perfectly suits high press"},
        }
    
    def get_team_data(self, team_name):
        return self.team_database.get(team_name, {
            "league": "EPL", "xg": 1.50, "xga": 1.50, "possession": 50, "tactical_style": "BALANCED"
        })
    
    def team_strength_snapshot(self, xg, xga):
        attack_strength = min(10, max(1, 5 + 5 * ((xg - self.league_avg_xg) / self.league_avg_xg)))
        defense_strength = min(10, max(1, 5 - 5 * ((xga - self.league_avg_xga) / self.league_avg_xga)))
        return round(defense_strength, 1), round(attack_strength, 1)
    
    def apply_injury_modifier(self, team_xg, team_xga, injury_tier):
        injury_weights = {0: 0.00, 1: -0.05, 2: -0.10, 3: -0.20, 4: -0.35}
        modifier = 1 + injury_weights[injury_tier]
        return team_xg * modifier, team_xga / modifier
    
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

def display_input_section(engine, existing_data=None):
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #1f77b4;">⚽ Match Configuration</h2>', unsafe_allow_html=True)
    
    available_teams = list(engine.team_database.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 HOME TEAM")
        home_default_index = available_teams.index("Tottenham") if "Tottenham" in available_teams else 0
        if existing_data and existing_data['home_team'] in available_teams:
            home_default_index = available_teams.index(existing_data['home_team'])
            
        home_team = st.selectbox("Select Home Team", available_teams, index=home_default_index, key="home_team_select")
        home_data = engine.get_team_data(home_team)
        
        home_xg = st.number_input("Expected Goals (xG)", value=existing_data['home_xg'] if existing_data else home_data["xg"], min_value=0.0, key="home_xg")
        home_xga = st.number_input("Expected Goals Against (xGA)", value=existing_data['home_xga'] if existing_data else home_data["xga"], min_value=0.0, key="home_xga")
        home_possession = st.slider("Average Possession %", 0, 100, existing_data['home_possession'] if existing_data else home_data["possession"], key="home_possession")
        
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
        
        away_xg = st.number_input("Away Expected Goals (xG)", value=existing_data['away_xg'] if existing_data else away_data["xg"], min_value=0.0, key="away_xg")
        away_xga = st.number_input("Away Expected Goals Against (xGA)", value=existing_data['away_xga'] if existing_data else away_data["xga"], min_value=0.0, key="away_xga")
        away_possession = st.slider("Away Average Possession %", 0, 100, existing_data['away_possession'] if existing_data else away_data["possession"], key="away_possession")
        
        away_tactical_index = tactical_options.index(away_data["tactical_style"]) if away_data["tactical_style"] in tactical_options else 4
        if existing_data:
            away_tactical_index = tactical_options.index(existing_data['away_tactical']) if existing_data['away_tactical'] in tactical_options else away_tactical_index
        away_tactical = st.selectbox("Away Tactical Style", tactical_options, index=away_tactical_index, key="away_tactical")
    
    st.markdown("---")
    st.subheader("🎭 Match Context")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        match_importance = st.selectbox("Match Importance", ["Normal", "High", "Critical", "Cup Final"], key="importance")
    with col2:
        crowd_impact = st.selectbox("Home Crowd Impact", ["Normal", "Electric", "Hostile"], key="crowd")
    with col3:
        referee_style = st.selectbox("Referee Style", ["Lenient", "Normal", "Strict"], key="referee")
    with col4:
        weather_conditions = st.selectbox("Weather Conditions", ["Normal", "Rainy", "Windy"], key="weather")
    
    st.subheader("🩹 Injury Status")
    col1, col2 = st.columns(2)
    injury_options = ["None", "Minor (1-2 rotational)", "Moderate (1-2 key starters)", "Significant (3-4 key players)", "Crisis (5+ starters)"]
    
    with col1:
        home_injuries = st.selectbox("Home Key Injuries", injury_options, key="home_injuries")
    with col2:
        away_injuries = st.selectbox("Away Key Injuries", injury_options, key="away_injuries")
    
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
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_prediction = st.button("🚀 GENERATE PREDICTION", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'generate_prediction': generate_prediction,
        'home_team': home_team, 'away_team': away_team,
        'home_data': home_data, 'away_data': away_data,
        'home_xg': home_xg, 'home_xga': home_xga,
        'away_xg': away_xg, 'away_xga': away_xga,
        'home_possession': home_possession, 'away_possession': away_possession,
        'home_tactical': home_tactical, 'away_tactical': away_tactical,
        'home_injuries': home_injuries, 'away_injuries': away_injuries,
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
    
    # Apply modifiers
    home_xg_adj, home_xga_adj = engine.apply_injury_modifier(input_data['home_xg'], input_data['home_xga'], home_injury_tier)
    away_xg_adj, away_xga_adj = engine.apply_injury_modifier(input_data['away_xg'], input_data['away_xga'], away_injury_tier)
    
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
    
    # Calculate probabilities
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
    
    # Team Analysis
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Team Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        home_defense, home_attack = engine.team_strength_snapshot(input_data['home_xg'], input_data['home_xga'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(f"🏠 {input_data['home_team']}")
        col1a, col2a = st.columns(2)
        with col1a: st.metric("Defensive Strength", f"{home_defense}/10")
        with col2a: st.metric("Attacking Strength", f"{home_attack}/10")
        st.write(f"**Style**: {input_data['home_tactical'].replace('_', ' ').title()}")
        st.write(f"**Injuries**: {input_data['home_injuries']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        away_defense, away_attack = engine.team_strength_snapshot(input_data['away_xg'], input_data['away_xga'])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader(f"✈️ {input_data['away_team']}")
        col1a, col2a = st.columns(2)
        with col1a: st.metric("Defensive Strength", f"{away_defense}/10")
        with col2a: st.metric("Attacking Strength", f"{away_attack}/10")
        st.write(f"**Style**: {input_data['away_tactical'].replace('_', ' ').title()}")
        st.write(f"**Injuries**: {input_data['away_injuries']}")
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
        factors = ["🏠 Home advantage: +10% boost", f"📊 Total expected goals: {round(total_xg, 2)}"]
        if home_injury_tier > 0: factors.append(f"🩹 {input_data['home_team']} injuries: {input_data['home_injuries']}")
        if away_injury_tier > 0: factors.append(f"🩹 {input_data['away_team']} injuries: {input_data['away_injuries']}")
        if tactical_explanations: factors.extend([f"🎯 {exp}" for exp in tactical_explanations])
        
        for factor in factors: st.write(f"• {factor}")
        
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
