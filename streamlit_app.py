import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# LEAGUE CONFIGURATIONS - Complete with all 6 leagues
# =============================================================================

LEAGUE_CONFIGS = {
    "EPL": {
        "teams": [
            "Arsenal", "Man City", "Liverpool", "Chelsea", "Tottenham", "Man United",
            "Newcastle", "Brighton", "West Ham", "Crystal Palace", "Wolves", "Fulham",
            "Bournemouth", "Aston Villa", "Brentford", "Everton", "Nott'ham Forest",
            "Luton", "Burnley", "Sheffield Utd"
        ],
        "baselines": {
            "avg_home_xG": 1.65,
            "avg_away_xG": 1.35,
            "avg_goals": 2.75,
            "home_advantage": 1.15,
            "avg_btts_prob": 0.52
        },
        "characteristics": ["high_pace", "physical", "transition_heavy"]
    },
    "La Liga": {
        "teams": [
            "Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao", "Real Sociedad",
            "Villarreal", "Betis", "Sevilla", "Valencia", "Osasuna", "Getafe", "Girona",
            "Mallorca", "Celta Vigo", "Rayo Vallecano", "Alaves", "Cadiz", "Granada",
            "Almeria", "Las Palmas"
        ],
        "baselines": {
            "avg_home_xG": 1.55,
            "avg_away_xG": 1.25,
            "avg_goals": 2.55,
            "home_advantage": 1.20,
            "avg_btts_prob": 0.48
        },
        "characteristics": ["technical", "possession_based", "slower_pace"]
    },
    "Bundesliga": {
        "teams": [
            "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
            "Union Berlin", "Freiburg", "Wolfsburg", "Mainz", "Monchengladbach",
            "Eintracht Frankfurt", "Koln", "Werder Bremen", "Bochum", "Augsburg",
            "Stuttgart", "Heidenheim", "Darmstadt"
        ],
        "baselines": {
            "avg_home_xG": 1.75,
            "avg_away_xG": 1.45,
            "avg_goals": 3.10,
            "home_advantage": 1.12,
            "avg_btts_prob": 0.58
        },
        "characteristics": ["high_scoring", "pressing", "youth_development"]
    },
    "Serie A": {
        "teams": [
            "Inter", "Juventus", "AC Milan", "Napoli", "Atalanta", "Roma", "Lazio",
            "Fiorentina", "Bologna", "Torino", "Monza", "Udinese", "Sassuolo",
            "Empoli", "Salernitana", "Lecce", "Frosinone", "Genoa", "Verona", "Cagliari"
        ],
        "baselines": {
            "avg_home_xG": 1.50,
            "avg_away_xG": 1.20,
            "avg_goals": 2.55,
            "home_advantage": 1.18,
            "avg_btts_prob": 0.46
        },
        "characteristics": ["tactical", "defensive", "structured"]
    },
    "Ligue 1": {
        "teams": [
            "PSG", "Lens", "Marseille", "Monaco", "Rennes", "Lille", "Nice",
            "Lorient", "Reims", "Lyon", "Montpellier", "Toulouse", "Clermont",
            "Strasbourg", "Nantes", "Brest", "Le Havre", "Metz"
        ],
        "baselines": {
            "avg_home_xG": 1.45,
            "avg_away_xG": 1.15,
            "avg_goals": 2.45,
            "home_advantage": 1.16,
            "avg_btts_prob": 0.44
        },
        "characteristics": ["physical", "transition", "psg_dominated"]
    },
    "RFPL": {
        "teams": [
            "Zenit", "CSKA Moscow", "Spartak Moscow", "Dynamo Moscow", "Krasnodar",
            "Lokomotiv Moscow", "Rostov", "Akhmat", "Sochi", "Orenburg", "Krylya Sovetov",
            "Fakel", "Ural", "NN", "Baltika", "Rubin"
        ],
        "baselines": {
            "avg_home_xG": 1.40,
            "avg_away_xG": 1.10,
            "avg_goals": 2.35,
            "home_advantage": 1.22,
            "avg_btts_prob": 0.42
        },
        "characteristics": ["physical", "defensive", "strong_home_advantage"]
    }
}

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def calculate_per_game_averages(team_data):
    """Calculate per-game averages for team data."""
    team_data = team_data.copy()
    
    # Overall averages
    if team_data['overall']['matches'] > 0:
        team_data['overall']['goals_per_game'] = team_data['overall']['goals_scored'] / team_data['overall']['matches']
        team_data['overall']['goals_against_per_game'] = team_data['overall']['goals_conceded'] / team_data['overall']['matches']
        team_data['overall']['xg_per_game'] = team_data['overall']['xG'] / team_data['overall']['matches']
        team_data['overall']['xga_per_game'] = team_data['overall']['xGA'] / team_data['overall']['matches']
    
    # Home/Away averages
    if 'home' in team_data and team_data['home']['matches'] > 0:
        team_data['home']['xg_per_game'] = team_data['home']['xG'] / team_data['home']['matches']
        team_data['home']['xga_per_game'] = team_data['home']['xGA'] / team_data['home']['matches']
    
    if 'away' in team_data and team_data['away']['matches'] > 0:
        team_data['away']['xg_per_game'] = team_data['away']['xG'] / team_data['away']['matches']
        team_data['away']['xga_per_game'] = team_data['away']['xGA'] / team_data['away']['matches']
    
    return team_data

def validate_data_quality(home_data, away_data):
    """Validate the quality and completeness of input data."""
    quality_score = 100
    
    # Check minimum matches
    if home_data['overall']['matches'] < 5:
        quality_score -= 20
    if away_data['overall']['matches'] < 5:
        quality_score -= 20
    
    if home_data.get('home', {}).get('matches', 0) < 3:
        quality_score -= 15
    if away_data.get('away', {}).get('matches', 0) < 3:
        quality_score -= 15
    
    # Check data consistency
    if home_data['overall']['xg_per_game'] > 4.0 or home_data['overall']['xg_per_game'] < 0.5:
        quality_score -= 10
    if away_data['overall']['xg_per_game'] > 4.0 or away_data['overall']['xg_per_game'] < 0.5:
        quality_score -= 10
    
    return max(quality_score, 0)

def calculate_form_indicators(team_data):
    """Calculate form indicators from recent performance."""
    team_data = team_data.copy()
    
    # Calculate recent form vs season average
    if 'last_5' in team_data and team_data['last_5']['xG_total'] > 0:
        recent_xg_avg = team_data['last_5']['xG_total'] / 5
        season_xg_avg = team_data['overall']['xg_per_game']
        
        if season_xg_avg > 0:
            team_data['form_ratio'] = recent_xg_avg / season_xg_avg
        else:
            team_data['form_ratio'] = 1.0
        
        # Form classification
        if team_data['form_ratio'] > 1.15:
            team_data['form'] = 'excellent'
        elif team_data['form_ratio'] > 1.05:
            team_data['form'] = 'good'
        elif team_data['form_ratio'] > 0.95:
            team_data['form'] = 'average'
        elif team_data['form_ratio'] > 0.85:
            team_data['form'] = 'poor'
        else:
            team_data['form'] = 'very_poor'
    
    return team_data

def prepare_context_factors(context, league):
    """Prepare contextual factors for prediction."""
    factors = context.copy()
    
    # Calculate injury impact
    factors['home_injury_impact'] = len(factors.get('home_injuries', [])) * 0.08
    factors['away_injury_impact'] = len(factors.get('away_injuries', [])) * 0.08
    
    # Calculate fatigue impact
    factors['home_fatigue_impact'] = max(0, (7 - factors.get('home_days_rest', 7)) * 0.03)
    factors['away_fatigue_impact'] = max(0, (7 - factors.get('away_days_rest', 7)) * 0.03)
    
    # League-specific adjustments
    if league == "EPL":
        factors['injury_multiplier'] = 1.2  # Injuries matter more in physical league
    elif league == "La Liga":
        factors['injury_multiplier'] = 1.1  # Technical quality drops with injuries
    else:
        factors['injury_multiplier'] = 1.0
    
    return factors

def prepare_match_data(home_data, away_data, context, league):
    """
    Prepare and validate match data for prediction.
    """
    # Calculate per-game averages
    home_data = calculate_per_game_averages(home_data)
    away_data = calculate_per_game_averages(away_data)
    
    # Validate data completeness
    data_quality = validate_data_quality(home_data, away_data)
    
    # Calculate form indicators
    home_data = calculate_form_indicators(home_data)
    away_data = calculate_form_indicators(away_data)
    
    # Prepare contextual factors
    context_factors = prepare_context_factors(context, league)
    
    return {
        'home': home_data,
        'away': away_data,
        'context': context_factors,
        'data_quality': data_quality,
        'league': league
    }

# =============================================================================
# STATISTICAL MODEL FUNCTIONS
# =============================================================================

def calculate_expected_goals(match_data):
    """Calculate expected goals using team strengths and league baselines."""
    league_baselines = LEAGUE_CONFIGS[match_data['league']]["baselines"]
    
    home_team = match_data['home']
    away_team = match_data['away']
    
    # Use home/away specific data where available
    home_attack = home_team.get('home', {}).get('xg_per_game', home_team['overall']['xg_per_game'])
    home_defense = home_team.get('home', {}).get('xga_per_game', home_team['overall']['xga_per_game'])
    
    away_attack = away_team.get('away', {}).get('xg_per_game', away_team['overall']['xg_per_game'])
    away_defense = away_team.get('away', {}).get('xga_per_game', away_team['overall']['xga_per_game'])
    
    # Calculate expected goals using league baselines
    lambda_home = (home_attack * away_defense) / league_baselines['avg_home_xG']
    lambda_away = (away_attack * home_defense) / league_baselines['avg_away_xG']
    
    # Apply home advantage
    lambda_home *= league_baselines['home_advantage']
    
    # Ensure reasonable bounds
    lambda_home = max(0.1, min(4.0, lambda_home))
    lambda_away = max(0.1, min(4.0, lambda_away))
    
    return lambda_home, lambda_away

def apply_contextual_modifiers(lambda_home, lambda_away, context):
    """Apply contextual factors to expected goals."""
    # Injury impacts
    lambda_home *= (1 - context.get('home_injury_impact', 0) * context.get('injury_multiplier', 1.0))
    lambda_away *= (1 - context.get('away_injury_impact', 0) * context.get('injury_multiplier', 1.0))
    
    # Fatigue impacts
    lambda_home *= (1 - context.get('home_fatigue_impact', 0))
    lambda_away *= (1 - context.get('away_fatigue_impact', 0))
    
    # Match importance (affects both teams similarly)
    importance = context.get('match_importance', 0.5)
    if importance > 0.7:  # High importance matches often lower scoring
        lambda_home *= 0.95
        lambda_away *= 0.95
    
    return lambda_home, lambda_away

def generate_scoreline_probabilities(lambda_home, lambda_away, max_goals=8):
    """Generate probabilities for all possible scorelines."""
    scoreline_probs = {}
    
    for i in range(max_goals + 1):  # Home goals
        for j in range(max_goals + 1):  # Away goals
            prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            scoreline_probs[f"{i}-{j}"] = prob
    
    # Normalize probabilities
    total_prob = sum(scoreline_probs.values())
    if total_prob > 0:
        for scoreline in scoreline_probs:
            scoreline_probs[scoreline] /= total_prob
    
    return scoreline_probs

def calculate_outcome_probabilities(scoreline_probs):
    """Calculate match outcome probabilities from scorelines."""
    home_win = 0
    draw = 0
    away_win = 0
    
    for scoreline, prob in scoreline_probs.items():
        home_goals, away_goals = map(int, scoreline.split('-'))
        
        if home_goals > away_goals:
            home_win += prob
        elif home_goals == away_goals:
            draw += prob
        else:
            away_win += prob
    
    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win
    }

def calculate_over_under_probabilities(scoreline_probs):
    """Calculate over/under 2.5 goals probabilities."""
    over_2_5 = 0
    under_2_5 = 0
    
    for scoreline, prob in scoreline_probs.items():
        home_goals, away_goals = map(int, scoreline.split('-'))
        total_goals = home_goals + away_goals
        
        if total_goals > 2.5:
            over_2_5 += prob
        else:
            under_2_5 += prob
    
    return {
        'over_2.5': over_2_5,
        'under_2.5': under_2_5
    }

def calculate_btts_probabilities(scoreline_probs):
    """Calculate both teams to score probabilities."""
    yes = 0
    no = 0
    
    for scoreline, prob in scoreline_probs.items():
        home_goals, away_goals = map(int, scoreline.split('-'))
        
        if home_goals > 0 and away_goals > 0:
            yes += prob
        else:
            no += prob
    
    return {
        'yes': yes,
        'no': no
    }

def get_most_likely_scores(scoreline_probs, top_n=5):
    """Get the most likely scorelines."""
    sorted_scores = sorted(scoreline_probs.items(), key=lambda x: x[1], reverse=True)
    
    most_likely = []
    for scoreline, prob in sorted_scores[:top_n]:
        most_likely.append({
            'score': scoreline,
            'probability': prob
        })
    
    return most_likely

def identify_key_factors(match_data, lambda_home, lambda_away):
    """Identify key factors influencing the prediction."""
    factors = []
    
    home_team = match_data['home']
    away_team = match_data['away']
    context = match_data['context']
    
    # Home advantage
    factors.append({
        'factor': 'Home advantage',
        'impact': 'positive',
        'magnitude': 'medium'
    })
    
    # Form factors
    if home_team.get('form_ratio', 1.0) > 1.1:
        factors.append({
            'factor': f"{home_team['name']} good recent form",
            'impact': 'positive',
            'magnitude': 'medium'
        })
    elif home_team.get('form_ratio', 1.0) < 0.9:
        factors.append({
            'factor': f"{home_team['name']} poor recent form",
            'impact': 'negative',
            'magnitude': 'medium'
        })
    
    if away_team.get('form_ratio', 1.0) > 1.1:
        factors.append({
            'factor': f"{away_team['name']} good recent form",
            'impact': 'positive',
            'magnitude': 'medium'
        })
    elif away_team.get('form_ratio', 1.0) < 0.9:
        factors.append({
            'factor': f"{away_team['name']} poor recent form",
            'impact': 'negative',
            'magnitude': 'medium'
        })
    
    # Injury factors
    if context.get('home_injury_impact', 0) > 0.1:
        factors.append({
            'factor': f"{home_team['name']} key injuries",
            'impact': 'negative',
            'magnitude': 'high' if context['home_injury_impact'] > 0.15 else 'medium'
        })
    
    if context.get('away_injury_impact', 0) > 0.1:
        factors.append({
            'factor': f"{away_team['name']} key injuries",
            'impact': 'negative',
            'magnitude': 'high' if context['away_injury_impact'] > 0.15 else 'medium'
        })
    
    # Fatigue factors
    if context.get('home_fatigue_impact', 0) > 0.05:
        factors.append({
            'factor': f"{home_team['name']} fatigue (short rest)",
            'impact': 'negative',
            'magnitude': 'medium'
        })
    
    if context.get('away_fatigue_impact', 0) > 0.05:
        factors.append({
            'factor': f"{away_team['name']} fatigue (short rest)",
            'impact': 'negative',
            'magnitude': 'medium'
        })
    
    return factors[:5]  # Return top 5 factors

# =============================================================================
# CONFIDENCE CALCULATION FUNCTIONS
# =============================================================================

def calculate_entropy(probabilities):
    """Calculate entropy of probability distribution."""
    entropy = 0
    for prob in probabilities.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy

def adjust_confidence(base_confidence, max_probability, entropy):
    """Adjust confidence based on probability distribution."""
    # Higher max probability = higher confidence
    probability_boost = max_probability * 0.3
    
    # Lower entropy = higher confidence (more certain distribution)
    entropy_penalty = entropy * 0.2
    
    adjusted = base_confidence + probability_boost - entropy_penalty
    return max(0.1, min(0.95, adjusted))

def calculate_context_impact(context):
    """Calculate impact of contextual factors on confidence."""
    impact = 1.0
    
    # Injuries reduce confidence
    injury_impact = (context.get('home_injury_impact', 0) + context.get('away_injury_impact', 0)) * 0.3
    impact -= injury_impact
    
    # Fatigue reduces confidence
    fatigue_impact = (context.get('home_fatigue_impact', 0) + context.get('away_fatigue_impact', 0)) * 0.2
    impact -= fatigue_impact
    
    # Extreme match importance can reduce confidence
    importance = context.get('match_importance', 0.5)
    if importance > 0.8:
        impact *= 0.9
    
    return max(0.5, impact)

def probability_to_confidence_level(probability):
    """Convert probability to confidence level."""
    if probability >= 0.8:
        return "HIGH"
    elif probability >= 0.6:
        return "MEDIUM"
    else:
        return "LOW"

def calculate_confidence(match_data, outcome_probs):
    """
    Calculate confidence levels for different prediction markets.
    """
    data_quality = match_data['data_quality']
    
    # Base confidence from data quality
    base_confidence = data_quality / 100
    
    # Outcome confidence (1X2)
    max_outcome_prob = max(outcome_probs.values())
    outcome_entropy = calculate_entropy(outcome_probs)
    outcome_confidence = adjust_confidence(base_confidence, max_outcome_prob, outcome_entropy)
    
    # Over/Under confidence
    ou_probs = {
        'over': outcome_probs.get('over_2.5', 0.5),
        'under': outcome_probs.get('under_2.5', 0.5)
    }
    ou_entropy = calculate_entropy(ou_probs)
    ou_confidence = adjust_confidence(base_confidence, max(ou_probs.values()), ou_entropy)
    
    # BTTS confidence
    btts_probs = {
        'yes': outcome_probs.get('btts_yes', 0.5),
        'no': outcome_probs.get('btts_no', 0.5)
    }
    btts_entropy = calculate_entropy(btts_probs)
    btts_confidence = adjust_confidence(base_confidence, max(btts_probs.values()), btts_entropy)
    
    # Context impact
    context = match_data.get('context', {})
    context_impact = calculate_context_impact(context)
    
    # Apply context impact
    outcome_confidence *= context_impact
    ou_confidence *= context_impact
    btts_confidence *= context_impact
    
    return {
        'outcome': probability_to_confidence_level(outcome_confidence),
        'over_under': probability_to_confidence_level(ou_confidence),
        'btts': probability_to_confidence_level(btts_confidence)
    }

# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_match(home_data, away_data, context, league):
    """
    Generate match predictions using statistical models.
    """
    # Prepare data
    match_data = prepare_match_data(home_data, away_data, context, league)
    
    # Calculate expected goals using bivariate Poisson approximation
    lambda_home, lambda_away = calculate_expected_goals(match_data)
    
    # Apply contextual modifiers
    lambda_home, lambda_away = apply_contextual_modifiers(
        lambda_home, lambda_away, match_data['context']
    )
    
    # Generate scoreline probabilities
    scoreline_probs = generate_scoreline_probabilities(lambda_home, lambda_away)
    
    # Calculate market probabilities
    outcome_probs = calculate_outcome_probabilities(scoreline_probs)
    over_under_probs = calculate_over_under_probabilities(scoreline_probs)
    btts_probs = calculate_btts_probabilities(scoreline_probs)
    
    # Get most likely scores
    most_likely_scores = get_most_likely_scores(scoreline_probs)
    
    # Calculate confidence levels
    confidence = calculate_confidence(match_data, outcome_probs)
    
    # Identify key factors
    key_factors = identify_key_factors(match_data, lambda_home, lambda_away)
    
    return {
        'match_outcome': {
            'home_win': outcome_probs['home_win'],
            'draw': outcome_probs['draw'],
            'away_win': outcome_probs['away_win'],
            'confidence': confidence['outcome']
        },
        'over_under': {
            'over_2.5': over_under_probs['over_2.5'],
            'under_2.5': over_under_probs['under_2.5'],
            'confidence': confidence['over_under']
        },
        'both_teams_score': {
            'yes': btts_probs['yes'],
            'no': btts_probs['no'],
            'confidence': confidence['btts']
        },
        'expected_score': {
            'home': lambda_home,
            'away': lambda_away
        },
        'most_likely_scores': most_likely_scores,
        'key_factors': key_factors,
        'model_data': {
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'data_quality': match_data['data_quality']
        }
    }

# =============================================================================
# STREAMLIT UI COMPONENTS
# =============================================================================

def display_prediction_results(prediction, home_team, away_team, league):
    """Display prediction results in an organized layout."""
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
    """Display confidence level with appropriate color coding."""
    if confidence_level == "HIGH":
        st.markdown('<p class="confidence-high">🟢 High Confidence</p>', unsafe_allow_html=True)
    elif confidence_level == "MEDIUM":
        st.markdown('<p class="confidence-medium">🟡 Medium Confidence</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="confidence-low">🔴 Low Confidence</p>', unsafe_allow_html=True)

# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
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
        .team-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">⚽ Football Prediction Engine</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictions_log' not in st.session_state:
        st.session_state.predictions_log = []
    
    # Main layout - Match Configuration
    st.subheader("🎯 Match Configuration")
    config_col1, config_col2, config_col3 = st.columns([2, 2, 3])
    
    with config_col1:
        # League selection
        league = st.selectbox(
            "Select League",
            list(LEAGUE_CONFIGS.keys()),
            index=0
        )
        
    with config_col2:
        # Team selection based on league
        teams = LEAGUE_CONFIGS[league]["teams"]
        home_team = st.selectbox("Home Team", teams, index=0)
        
    with config_col3:
        away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)
        
        if home_team == away_team:
            st.error("Home and Away teams cannot be the same!")
            return
        
        # Show league context
        baseline = LEAGUE_CONFIGS[league]["baselines"]
        st.info(f"🇩🇪 **{league} Context** | Avg Goals: {baseline['avg_goals']} | Home Advantage: +{int((baseline['home_advantage']-1)*100)}%")
    
    # Team Statistics - BALANCED LAYOUT
    st.subheader("📊 Team Statistics")
    home_col, away_col = st.columns(2)
    
    # HOME TEAM SECTION - COMPLETE AND BALANCED
    with home_col:
        st.markdown('<div class="team-section">🏠 Home Team: ' + home_team + '</div>', unsafe_allow_html=True)
        
        # Overall Stats
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
        
        # Home Stats Only
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
        
        # Last 5 Matches
        st.write("**Last 5 Matches**")
        col1, col2 = st.columns(2)
        with col1:
            home_last5_xg = st.number_input("Last 5 xG Total", min_value=0.0, max_value=25.0, value=10.25, key="home_l5xg")
        with col2:
            home_last5_points = st.number_input("Last 5 Points", min_value=0, max_value=15, value=13, key="home_l5p")
    
    # AWAY TEAM SECTION - COMPLETE AND BALANCED (FIXED)
    with away_col:
        st.markdown('<div class="team-section">✈️ Away Team: ' + away_team + '</div>', unsafe_allow_html=True)
        
        # Overall Stats (ADDED - was missing)
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
        
        # Away Stats Only (RENAMED - was incorrectly "Home Stats")
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
        
        # Last 5 Matches (ADDED - was missing)
        st.write("**Last 5 Matches**")
        col1, col2 = st.columns(2)
        with col1:
            away_last5_xg = st.number_input("Last 5 xG Total", min_value=0.0, max_value=25.0, value=11.44, key="away_l5xg")
        with col2:
            away_last5_points = st.number_input("Last 5 Points", min_value=0, max_value=15, value=12, key="away_l5p")
    
    # Contextual factors
    st.subheader("🎭 Contextual Factors")
    context_col1, context_col2, context_col3 = st.columns(3)
    
    with context_col1:
        home_injuries = st.text_input("Home Team Key Injuries", placeholder="e.g., Saliba, Saka")
        away_injuries = st.text_input("Away Team Key Injuries", placeholder="e.g., De Bruyne")
    
    with context_col2:
        home_days_rest = st.number_input("Home Days Since Last Match", min_value=2, max_value=14, value=4)
        away_days_rest = st.number_input("Away Days Since Last Match", min_value=2, max_value=14, value=6)
    
    with context_col3:
        match_importance = st.slider("Match Importance", 0.0, 1.0, 0.7, 0.1,
                                   format="%.1f (Friendly - Cup Final)")
    
    # Prediction button
    if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Calculating predictions..."):
            # Prepare data - NOW WITH COMPLETE BALANCED DATA
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
                'timestamp': datetime.now()
            })

if __name__ == "__main__":
    main()