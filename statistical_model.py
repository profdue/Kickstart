import numpy as np
from scipy.stats import poisson
from .league_manager import get_league_baselines

def predict_match(home_data, away_data, context, league):
    """
    Generate match predictions using statistical models.
    """
    from .data_processor import prepare_match_data
    from .confidence_calculator import calculate_confidence
    
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

def calculate_expected_goals(match_data):
    """Calculate expected goals using team strengths and league baselines."""
    league_baselines = get_league_baselines(match_data['league'])
    
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
    
    # Form impacts (from data processor)
    # These are applied in the data preparation phase
    
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