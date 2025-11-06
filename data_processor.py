import numpy as np

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