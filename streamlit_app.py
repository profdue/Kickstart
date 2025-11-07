import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PRECISION PREDICTION ENGINE - REALISTIC DATA EDITION
# =============================================================================

class RealisticPrecisionEngine:
    def __init__(self):
        self.team_profiles = {}
        self.league_baselines = {
            'EPL': {'avg_goals': 2.75, 'home_advantage': 1.14},
            'La Liga': {'avg_goals': 2.55, 'home_advantage': 1.16},
            'Bundesliga': {'avg_goals': 3.10, 'home_advantage': 1.12},
            'Serie A': {'avg_goals': 2.55, 'home_advantage': 1.15},
            'Ligue 1': {'avg_goals': 2.45, 'home_advantage': 1.13}
        }
    
    def _build_team_profile(self, team_data, is_home=True):
        """Build team profile from realistic available data"""
        profile = {
            'name': team_data['name'],
            'is_home': is_home,
            
            # Core available metrics
            'xg_offense': team_data['overall']['xg_per_game'],
            'xg_defense': team_data['overall']['xga_per_game'],
            'goals_scored': team_data['overall']['goals_per_game'],
            'goals_conceded': team_data['overall']['goals_against_per_game'],
            'possession': team_data.get('possession', 50),
            
            # Inferred tactical style from observable patterns
            'tactical_style': self._infer_tactical_style(team_data),
            
            # Form and momentum from available data
            'recent_form': team_data.get('recent_form', 'average'),
            'finishing_efficiency': team_data['overall'].get('finishing_efficiency', 1.0),
            
            # Contextual factors
            'home_advantage': 1.14 if is_home else 1.0,
            'momentum': team_data.get('momentum', 0)
        }
        
        return profile
    
    def _infer_tactical_style(self, team_data):
        """Infer tactical style from available stats and observable patterns"""
        styles = []
        
        # Inference from possession stats
        possession = team_data.get('possession', 50)
        if possession > 60:
            styles.append('POSSESSION')
        elif possession < 45:
            styles.append('COUNTER_ATTACK')
        
        # Inference from goals vs xG (finishing efficiency)
        finishing_eff = team_data['overall'].get('finishing_efficiency', 1.0)
        if finishing_eff > 1.1:
            styles.append('EFFICIENT')
        elif finishing_eff < 0.9:
            styles.append('INEFFICIENT')
        
        # Inference from observable patterns (user input)
        observable = team_data.get('observable_patterns', [])
        if 'presses_high' in observable:
            styles.append('HIGH_PRESS')
        if 'sits_deep' in observable:
            styles.append('DEFENSIVE')
        if 'counters_quickly' in observable:
            styles.append('COUNTER_ATTACK')
        if 'wing_play' in observable:
            styles.append('WING_FOCUS')
        
        return styles if styles else ['BALANCED']
    
    def _calculate_expected_goals(self, home_profile, away_profile, context):
        """Calculate expected goals using realistic available data"""
        league = context.get('league', 'EPL')
        baselines = self.league_baselines[league]
        
        # Base expected goals from team strengths
        home_attack = home_profile['xg_offense']
        away_defense = away_profile['xg_defense']
        away_attack = away_profile['xg_offense'] 
        home_defense = home_profile['xg_defense']
        
        # Adjust for home advantage
        home_advantage = baselines['home_advantage']
        
        # Calculate expected goals
        lambda_home = (home_attack * away_defense) * home_advantage
        lambda_away = (away_attack * home_defense) / home_advantage
        
        # Apply finishing efficiency
        lambda_home *= home_profile['finishing_efficiency']
        lambda_away *= away_profile['finishing_efficiency']
        
        # Context adjustments
        lambda_home, lambda_away = self._apply_context_adjustments(
            lambda_home, lambda_away, home_profile, away_profile, context
        )
        
        return lambda_home, lambda_away
    
    def _apply_context_adjustments(self, lambda_home, lambda_away, home_profile, away_profile, context):
        """Apply context adjustments based on realistic factors"""
        
        # Recent form adjustments
        home_form = home_profile.get('recent_form', 'average')
        away_form = away_profile.get('recent_form', 'average')
        
        form_multipliers = {
            'excellent': 1.15,
            'good': 1.08,
            'average': 1.0,
            'poor': 0.92,
            'terrible': 0.85
        }
        
        lambda_home *= form_multipliers.get(home_form, 1.0)
        lambda_away *= form_multipliers.get(away_form, 1.0)
        
        # Injury impacts
        home_injuries = context.get('home_injuries', [])
        away_injuries = context.get('away_injuries', [])
        
        lambda_home *= (1 - len(home_injuries) * 0.05)
        lambda_away *= (1 - len(away_injuries) * 0.05)
        
        # Match importance
        importance = context.get('match_importance', 'normal')
        if importance == 'high':
            # High importance often reduces scoring
            lambda_home *= 0.95
            lambda_away *= 0.95
        
        return max(0.1, lambda_home), max(0.1, lambda_away)
    
    def _simulate_style_interactions(self, home_profile, away_profile):
        """Simulate how different playing styles interact"""
        home_styles = home_profile['tactical_style']
        away_styles = away_profile['tactical_style']
        
        home_boost = 1.0
        away_boost = 1.0
        btts_boost = 1.0
        
        # Style interaction logic based on realistic observations
        if 'HIGH_PRESS' in home_styles and 'POSSESSION' in away_styles:
            home_boost *= 1.1  # Pressing disrupts possession teams
        if 'COUNTER_ATTACK' in home_styles and 'HIGH_PRESS' in away_styles:
            home_boost *= 1.15  # Counter effective against high press
        if 'DEFENSIVE' in home_styles and 'POSSESSION' in away_styles:
            away_boost *= 0.9  # Defense can frustrate possession teams
        
        # Both teams pressing often leads to more goals
        if 'HIGH_PRESS' in home_styles and 'HIGH_PRESS' in away_styles:
            btts_boost *= 1.2
        
        return {
            'home_xg_boost': home_boost,
            'away_xg_boost': away_boost,
            'btts_boost': btts_boost
        }
    
    def predict_match(self, home_data, away_data, context):
        """Main prediction method using realistic data"""
        
        # Build profiles from available data
        home_profile = self._build_team_profile(home_data, is_home=True)
        away_profile = self._build_team_profile(away_data, is_home=False)
        
        # Calculate base expected goals
        lambda_home, lambda_away = self._calculate_expected_goals(
            home_profile, away_profile, context
        )
        
        # Apply style interactions
        style_effects = self._simulate_style_interactions(home_profile, away_profile)
        lambda_home *= style_effects['home_xg_boost']
        lambda_away *= style_effects['away_xg_boost']
        
        # Generate outcome probabilities
        outcome_probs = self._calculate_outcome_probabilities(lambda_home, lambda_away)
        
        # Calculate additional markets
        additional_markets = self._calculate_additional_markets(lambda_home, lambda_away)
        additional_markets['btts_yes'] *= style_effects['btts_boost']
        additional_markets['btts_no'] = 1 - additional_markets['btts_yes']
        
        # Calculate confidence
        confidence = self._calculate_confidence(home_profile, away_profile, outcome_probs)
        
        return {
            'match_outcome': outcome_probs,
            'additional_markets': additional_markets,
            'expected_goals': {'home': lambda_home, 'away': lambda_away},
            'confidence': confidence,
            'team_profiles': {'home': home_profile, 'away': away_profile},
            'style_effects': style_effects
        }
    
    def _calculate_outcome_probabilities(self, lambda_home, lambda_away):
        """Calculate match outcome probabilities using Poisson distribution"""
        home_win = 0
        draw = 0
        away_win = 0
        
        for i in range(8):  # Home goals
            for j in range(8):  # Away goals
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        # Normalize
        total = home_win + draw + away_win
        return {
            'home_win': home_win / total,
            'draw': draw / total,
            'away_win': away_win / total
        }
    
    def _calculate_additional_markets(self, lambda_home, lambda_away):
        """Calculate over/under and BTTS probabilities"""
        # Over/Under 2.5
        over_2_5 = 0
        for i in range(8):
            for j in range(8):
                if i + j > 2.5:
                    prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                    over_2_5 += prob
        
        # Both Teams to Score
        btts_yes = 0
        for i in range(1, 8):  # Home scores at least 1
            for j in range(1, 8):  # Away scores at least 1
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                btts_yes += prob
        
        return {
            'over_2.5': over_2_5,
            'under_2.5': 1 - over_2_5,
            'btts_yes': btts_yes,
            'btts_no': 1 - btts_yes
        }
    
    def _calculate_confidence(self, home_profile, away_profile, outcome_probs):
        """Calculate prediction confidence based on data quality and clarity"""
        confidence_factors = []
        
        # Data completeness
        if (home_profile['xg_offense'] > 0 and away_profile['xg_offense'] > 0):
            confidence_factors.append(0.8)
        
        # Outcome clarity
        max_prob = max(outcome_probs.values())
        if max_prob > 0.5:
            confidence_factors.append(0.7)
        elif max_prob > 0.4:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Form consistency
        if (home_profile.get('recent_form', 'average') != 'average' and 
            away_profile.get('recent_form', 'average') != 'average'):
            confidence_factors.append(0.6)
        
        avg_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        if avg_confidence >= 0.7:
            return "HIGH"
        elif avg_confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

# =============================================================================
# REALISTIC STREAMLIT INTERFACE
# =============================================================================

def create_realistic_interface():
    """Create interface using only realistically available data"""
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .team-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .confidence-high { color: #00a650; font-weight: bold; }
    .confidence-medium { color: #ffa500; font-weight: bold; }
    .confidence-low { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">⚽ Precision Prediction Engine</div>', unsafe_allow_html=True)
    
    # League selection
    league = st.selectbox("🏆 Select League", ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
    
    # Team selection
    teams = get_teams_for_league(league)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="team-section">🏠 Home Team</div>', unsafe_allow_html=True)
        home_team = st.selectbox("Home Team", teams, key="home")
        
        # Realistic stats input
        st.subheader("📊 Basic Stats (From ESPN/FotMob)")
        h_matches = st.number_input("Matches Played", min_value=1, max_value=38, value=10, key="h_m")
        h_goals = st.number_input("Goals Scored", min_value=0, max_value=100, value=22, key="h_g")
        h_conceded = st.number_input("Goals Conceded", min_value=0, max_value=100, value=8, key="h_c")
        h_xg = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=50.0, value=20.1, key="h_xg")
        h_xga = st.number_input("Expected Goals Against (xGA)", min_value=0.0, max_value=50.0, value=9.8, key="h_xga")
        h_possession = st.slider("Average Possession %", 0, 100, 65, key="h_poss")
        
        # Recent form
        st.subheader("📈 Recent Form (Last 5 Games)")
        h_form = st.selectbox("Form", ["Excellent", "Good", "Average", "Poor", "Terrible"], key="h_form")
        h_last5_goals = st.number_input("Goals in Last 5", min_value=0, max_value=30, value=11, key="h_l5g")
        h_last5_conceded = st.number_input("Conceded in Last 5", min_value=0, max_value=30, value=3, key="h_l5c")
        
        # Observable patterns
        st.subheader("👀 Observable Style")
        h_patterns = st.multiselect(
            "What you've seen in recent games:",
            ["Dominates possession", "Presses high", "Sits deep and counters", 
             "Creates many chances", "Struggles to score", "Strong at home",
             "Poor away form", "Vulnerable to counters", "Solid defense"],
            key="h_patterns"
        )
    
    with col2:
        st.markdown('<div class="team-section">✈️ Away Team</div>', unsafe_allow_html=True)
        away_team = st.selectbox("Away Team", teams, key="away")
        
        # Realistic stats input
        st.subheader("📊 Basic Stats (From ESPN/FotMob)")
        a_matches = st.number_input("Matches Played", min_value=1, max_value=38, value=10, key="a_m")
        a_goals = st.number_input("Goals Scored", min_value=0, max_value=100, value=18, key="a_g")
        a_conceded = st.number_input("Goals Conceded", min_value=0, max_value=100, value=12, key="a_c")
        a_xg = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=50.0, value=17.5, key="a_xg")
        a_xga = st.number_input("Expected Goals Against (xGA)", min_value=0.0, max_value=50.0, value=11.2, key="a_xga")
        a_possession = st.slider("Average Possession %", 0, 100, 55, key="a_poss")
        
        # Recent form
        st.subheader("📈 Recent Form (Last 5 Games)")
        a_form = st.selectbox("Form", ["Excellent", "Good", "Average", "Poor", "Terrible"], key="a_form")
        a_last5_goals = st.number_input("Goals in Last 5", min_value=0, max_value=30, value=8, key="a_l5g")
        a_last5_conceded = st.number_input("Conceded in Last 5", min_value=0, max_value=30, value=6, key="a_l5c")
        
        # Observable patterns
        st.subheader("👀 Observable Style")
        a_patterns = st.multiselect(
            "What you've seen in recent games:",
            ["Dominates possession", "Presses high", "Sits deep and counters", 
             "Creates many chances", "Struggles to score", "Strong away form",
             "Poor away form", "Vulnerable to counters", "Solid defense"],
            key="a_patterns"
        )
    
    # Context factors
    st.markdown("---")
    st.subheader("🎭 Match Context")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        match_importance = st.selectbox(
            "Match Importance",
            ["Normal", "Relegation Battle", "Title Decider", "Cup Final", "Dead Rubber"]
        )
        home_injuries = st.text_input("Home Key Injuries", placeholder="e.g., Davies, Kimmich")
    
    with col2:
        weather = st.selectbox(
            "Conditions",
            ["Normal", "Poor Pitch", "Extreme Weather", "Perfect Conditions"]
        )
        away_injuries = st.text_input("Away Key Injuries", placeholder="e.g., De Bruyne, Salah")
    
    with col3:
        crowd_impact = st.selectbox(
            "Crowd Impact", 
            ["Normal", "Electric Home Crowd", "Hostile Away"]
        )
        referee = st.selectbox("Referee Style", ["Normal", "Lenient", "Strict"])
    
    return {
        'league': league,
        'home_team': home_team,
        'away_team': away_team,
        'home_data': {
            'matches': h_matches,
            'goals': h_goals,
            'conceded': h_conceded,
            'xg': h_xg,
            'xga': h_xga,
            'possession': h_possession,
            'form': h_form,
            'last5_goals': h_last5_goals,
            'last5_conceded': h_last5_conceded,
            'patterns': h_patterns
        },
        'away_data': {
            'matches': a_matches,
            'goals': a_goals,
            'conceded': a_conceded,
            'xg': a_xg,
            'xga': a_xga,
            'possession': a_possession,
            'form': a_form,
            'last5_goals': a_last5_goals,
            'last5_conceded': a_last5_conceded,
            'patterns': a_patterns
        },
        'context': {
            'match_importance': match_importance,
            'home_injuries': [inj.strip() for inj in home_injuries.split(',')] if home_injuries else [],
            'away_injuries': [inj.strip() for inj in away_injuries.split(',')] if away_injuries else [],
            'weather': weather,
            'crowd_impact': crowd_impact,
            'referee': referee
        }
    }

def prepare_team_data(raw_data, is_home=True):
    """Convert raw input data to engine-ready format"""
    # Calculate per-game averages
    matches = raw_data['matches']
    
    # Calculate finishing efficiency (goals vs xG)
    finishing_efficiency = raw_data['goals'] / raw_data['xg'] if raw_data['xg'] > 0 else 1.0
    
    # Convert observable patterns to tactical indicators
    observable_patterns = []
    pattern_mapping = {
        'Dominates possession': 'high_possession',
        'Presses high': 'presses_high', 
        'Sits deep and counters': 'sits_deep',
        'Creates many chances': 'creative',
        'Struggles to score': 'inefficient',
        'Strong at home': 'strong_home' if is_home else 'strong_away',
        'Poor away form': 'poor_away',
        'Vulnerable to counters': 'vulnerable_counters',
        'Solid defense': 'solid_defense'
    }
    
    for pattern in raw_data['patterns']:
        if pattern in pattern_mapping:
            observable_patterns.append(pattern_mapping[pattern])
    
    return {
        'name': 'Home Team' if is_home else 'Away Team',
        'overall': {
            'matches': matches,
            'goals_scored': raw_data['goals'],
            'goals_conceded': raw_data['conceded'],
            'xG': raw_data['xg'],
            'xGA': raw_data['xga'],
            'goals_per_game': raw_data['goals'] / matches,
            'goals_against_per_game': raw_data['conceded'] / matches,
            'xg_per_game': raw_data['xg'] / matches,
            'xga_per_game': raw_data['xga'] / matches,
            'finishing_efficiency': finishing_efficiency
        },
        'possession': raw_data['possession'],
        'recent_form': raw_data['form'].lower(),
        'observable_patterns': observable_patterns,
        'momentum': 1 if raw_data['form'] in ['Excellent', 'Good'] else -1 if raw_data['form'] in ['Poor', 'Terrible'] else 0
    }

def display_realistic_results(prediction, input_data):
    """Display results using realistic data"""
    
    st.markdown("---")
    st.markdown('<div class="main-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    
    # Confidence display
    confidence = prediction['confidence']
    if confidence == "HIGH":
        st.markdown('<p class="confidence-high">🟢 High Confidence Prediction</p>', unsafe_allow_html=True)
    elif confidence == "MEDIUM":
        st.markdown('<p class="confidence-medium">🟡 Medium Confidence Prediction</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="confidence-low">🔴 Low Confidence Prediction</p>', unsafe_allow_html=True)
    
    # Main predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏆 Match Outcome")
        outcome = prediction['match_outcome']
        
        fig = go.Figure(data=[
            go.Bar(x=['Home', 'Draw', 'Away'],
                  y=[outcome['home_win'], outcome['draw'], outcome['away_win']],
                  marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ])
        fig.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**{input_data['home_team']}**: {outcome['home_win']:.1%}")
        st.write(f"**Draw**: {outcome['draw']:.1%}")
        st.write(f"**{input_data['away_team']}**: {outcome['away_win']:.1%}")
    
    with col2:
        st.subheader("📊 Over/Under 2.5")
        markets = prediction['additional_markets']
        
        fig = go.Figure(data=[
            go.Bar(x=['Over 2.5', 'Under 2.5'],
                  y=[markets['over_2.5'], markets['under_2.5']],
                  marker_color=['#ff6b6b', '#4ecdc4'])
        ])
        fig.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Over 2.5**: {markets['over_2.5']:.1%}")
        st.write(f"**Under 2.5**: {markets['under_2.5']:.1%}")
    
    with col3:
        st.subheader("⚽ Both Teams to Score")
        btts = prediction['additional_markets']
        
        fig = go.Figure(data=[
            go.Bar(x=['Yes', 'No'],
                  y=[btts['btts_yes'], btts['btts_no']],
                  marker_color=['#a05195', '#f95d6a'])
        ])
        fig.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Yes**: {btts['btts_yes']:.1%}")
        st.write(f"**No**: {btts['btts_no']:.1%}")
    
    # Additional insights
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Expected Score")
        exp_goals = prediction['expected_goals']
        st.metric(
            "Expected Goals",
            f"{exp_goals['home']:.1f} - {exp_goals['away']:.1f}",
            delta=f"Total: {exp_goals['home'] + exp_goals['away']:.1f} goals"
        )
        
        st.subheader("🔍 Style Analysis")
        home_styles = prediction['team_profiles']['home']['tactical_style']
        away_styles = prediction['team_profiles']['away']['tactical_style']
        
        st.write(f"**{input_data['home_team']}**: {', '.join(home_styles)}")
        st.write(f"**{input_data['away_team']}**: {', '.join(away_styles)}")
        
        effects = prediction['style_effects']
        st.write(f"**Style Impact**: Home {effects['home_xg_boost']:.2f}x, Away {effects['away_xg_boost']:.2f}x")
    
    with col2:
        st.subheader("🎯 Key Factors")
        factors = []
        
        # Home form advantage
        if input_data['home_data']['form'] in ['Excellent', 'Good']:
            factors.append(f"✅ {input_data['home_team']} in good form")
        elif input_data['home_data']['form'] in ['Poor', 'Terrible']:
            factors.append(f"⚠️ {input_data['home_team']} in poor form")
        
        # Away form
        if input_data['away_data']['form'] in ['Excellent', 'Good']:
            factors.append(f"✅ {input_data['away_team']} in good form")
        elif input_data['away_data']['form'] in ['Poor', 'Terrible']:
            factors.append(f"⚠️ {input_data['away_team']} in poor form")
        
        # Observable patterns
        if 'presses_high' in prediction['team_profiles']['home']['tactical_style']:
            factors.append(f"✅ {input_data['home_team']} presses high")
        if 'solid_defense' in prediction['team_profiles']['away']['tactical_style']:
            factors.append(f"🛡️ {input_data['away_team']} solid defensively")
        
        for factor in factors[:5]:  # Show top 5 factors
            st.write(factor)
        
        # Recommendation
        best_bet = max([
            ('Home Win', prediction['match_outcome']['home_win']),
            ('Draw', prediction['match_outcome']['draw']),
            ('Away Win', prediction['match_outcome']['away_win']),
            ('Over 2.5', prediction['additional_markets']['over_2.5']),
            ('Under 2.5', prediction['additional_markets']['under_2.5']),
            ('BTTS Yes', prediction['additional_markets']['btts_yes']),
            ('BTTS No', prediction['additional_markets']['btts_no'])
        ], key=lambda x: x[1])
        
        st.success(f"**Strongest Value**: {best_bet[0]} ({best_bet[1]:.1%})")

def get_teams_for_league(league):
    """Get teams for selected league"""
    leagues = {
        "EPL": ["Arsenal", "Man City", "Liverpool", "Chelsea", "Tottenham", "Man United", 
                "Newcastle", "Brighton", "West Ham", "Aston Villa", "Crystal Palace"],
        "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia",
                   "Villarreal", "Real Sociedad", "Athletic Bilbao", "Betis"],
        "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
                      "Eintracht Frankfurt", "Wolfsburg", "Borussia M'gladbach"],
        "Serie A": ["Inter", "Juventus", "AC Milan", "Napoli", "Roma", "Lazio", "Atalanta"],
        "Ligue 1": ["PSG", "Marseille", "Lyon", "Monaco", "Lille", "Nice", "Rennes"]
    }
    return leagues.get(league, leagues["EPL"])

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Precision Prediction Engine",
        page_icon="⚽",
        layout="wide"
    )
    
    # Initialize engine
    engine = RealisticPrecisionEngine()
    
    # Create interface
    input_data = create_realistic_interface()
    
    # Prediction button
    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing match data..."):
            # Prepare team data
            home_team_data = prepare_team_data(input_data['home_data'], is_home=True)
            away_team_data = prepare_team_data(input_data['away_data'], is_home=False)
            
            home_team_data['name'] = input_data['home_team']
            away_team_data['name'] = input_data['away_team']
            
            # Generate prediction
            prediction = engine.predict_match(home_team_data, away_team_data, input_data['context'])
            
            # Display results
            display_realistic_results(prediction, input_data)

if __name__ == "__main__":
    main()
