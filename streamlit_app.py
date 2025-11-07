import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PRECISION PREDICTION ENGINE - PROFESSIONAL GRADE
# =============================================================================

class PrecisionPredictionEngine:
    def __init__(self):
        self.team_profiles = {}
        self.market_efficiency_data = {}
        self.context_weights = {
            'tactical': 0.25,
            'psychological': 0.20, 
            'environmental': 0.15,
            'historical': 0.10,
            'statistical': 0.30
        }
        self._initialize_market_data()
    
    def _initialize_market_data(self):
        """Initialize market efficiency benchmarks"""
        self.market_efficiency_data = {
            'closing_line_variance': 0.02,
            'public_bias_correction': 0.03,
            'sharp_money_indicators': {},
            'market_consensus_threshold': 0.68
        }
    
    def _build_team_profile(self, team_data, is_home=True):
        """Build comprehensive team profile with tactical intelligence"""
        profile = {
            'name': team_data['name'],
            'is_home': is_home,
            
            # Core metrics
            'xg_offense': team_data['overall']['xg_per_game'],
            'xg_defense': team_data['overall']['xga_per_game'],
            'finishing_efficiency': team_data['overall'].get('finishing_efficiency', 1.0),
            
            # Tactical style detection
            'tactical_style': self._detect_tactical_style(team_data),
            'pressing_intensity': self._calculate_pressing_intensity(team_data),
            'defensive_structure': self._assess_defensive_structure(team_data),
            
            # Psychological factors
            'momentum': team_data.get('momentum', 0),
            'consistency': self._calculate_consistency(team_data),
            
            # Market factors
            'public_perception_bias': 0.0,
            'market_efficiency': 1.0
        }
        
        return profile
    
    def _detect_tactical_style(self, team_data):
        """Automatically detect team's tactical style from data patterns"""
        xg_ratio = team_data['overall']['xg_per_game'] / team_data['overall']['xga_per_game']
        goal_ratio = team_data['overall']['goals_scored'] / max(team_data['overall']['goals_conceded'], 1)
        
        if xg_ratio > 1.8 and team_data['overall']['xg_per_game'] > 2.0:
            return "HIGH_PRESS"
        elif xg_ratio < 0.8 and team_data['overall']['xga_per_game'] < 1.2:
            return "PARK_BUS"
        elif team_data['overall']['xg_per_game'] > 1.8 and team_data['overall']['xga_per_game'] > 1.5:
            return "GEGENPRESS"
        elif goal_ratio > 1.5 and team_data['overall']['xga_per_game'] < 1.3:
            return "POSSESSION"
        else:
            return "BALANCED"
    
    def _calculate_pressing_intensity(self, team_data):
        """Calculate pressing intensity from defensive actions and territory"""
        # Higher xGA can indicate high press (more shots conceded but from worse positions)
        if team_data['overall']['xga_per_game'] > 1.8:
            return "VERY_HIGH"
        elif team_data['overall']['xga_per_game'] > 1.4:
            return "HIGH"
        elif team_data['overall']['xga_per_game'] < 1.0:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _assess_defensive_structure(self, team_data):
        """Assess defensive organization from xGA vs actual goals"""
        if team_data['overall']['goals_conceded'] < team_data['overall']['xga_per_game'] * 0.8:
            return "ORGANIZED"
        elif team_data['overall']['goals_conceded'] > team_data['overall']['xga_per_game'] * 1.2:
            return "DISORGANIZED"
        else:
            return "AVERAGE"
    
    def _calculate_consistency(self, team_data):
        """Calculate team consistency from recent form vs season average"""
        if 'last_5' in team_data:
            recent_xg = team_data['last_5']['xG_total'] / 5
            season_xg = team_data['overall']['xg_per_game']
            
            variance = abs(recent_xg - season_xg) / season_xg
            if variance < 0.1:
                return "VERY_CONSISTENT"
            elif variance < 0.2:
                return "CONSISTENT"
            else:
                return "VOLATILE"
        return "UNKNOWN"
    
    def _simulate_tactical_matchup(self, home_profile, away_profile):
        """Simulate tactical interactions with precise adjustments"""
        
        style_interactions = {
            'HIGH_PRESS_VS_BUILDUP': {'home_xg_boost': 1.15, 'away_xg_boost': 0.9, 'btts_boost': 1.1},
            'PARK_BUS_VS_POSSESSION': {'home_xg_boost': 0.8, 'away_xg_boost': 0.7, 'btts_boost': 0.8},
            'GEGENPRESS_VS_HIGH_LINE': {'home_xg_boost': 1.25, 'away_xg_boost': 1.1, 'btts_boost': 1.2},
            'BALANCED_VS_BALANCED': {'home_xg_boost': 1.0, 'away_xg_boost': 1.0, 'btts_boost': 1.0}
        }
        
        # Determine interaction key
        interaction_key = f"{home_profile['tactical_style']}_VS_{away_profile['tactical_style']}"
        effects = style_interactions.get(interaction_key, style_interactions['BALANCED_VS_BALANCED'])
        
        # Add pressing intensity effects
        if home_profile['pressing_intensity'] == "VERY_HIGH" and away_profile['defensive_structure'] == "DISORGANIZED":
            effects['home_xg_boost'] *= 1.1
        
        return effects
    
    def _calculate_dynamic_home_advantage(self, home_profile, away_profile, context):
        """Calculate context-aware home advantage"""
        base_advantage = 1.15  # Base 15% home advantage
        
        # Adjust based on team strengths
        strength_ratio = home_profile['xg_offense'] / away_profile['xg_offense']
        if strength_ratio > 1.5:
            base_advantage *= 1.1  # Strong home teams get bigger advantage
        elif strength_ratio < 0.67:
            base_advantage *= 0.9  # Weak home teams get reduced advantage
        
        # Crowd and travel adjustments
        if context.get('crowd_impact') == "HIGH":
            base_advantage *= 1.1
        if context.get('away_travel') == "LONG":
            base_advantage *= 1.05
        
        return base_advantage
    
    def _market_efficient_pricing(self, home_profile, away_profile, base_probs):
        """Apply market efficiency corrections"""
        
        # Public team bias adjustment (big clubs often overvalued)
        public_teams = ['Man United', 'Barcelona', 'Real Madrid', 'Bayern Munich', 'Liverpool']
        if home_profile['name'] in public_teams:
            base_probs['home_win'] *= 0.95
            base_probs['away_win'] *= 1.05
        if away_profile['name'] in public_teams:
            base_probs['away_win'] *= 0.95
            base_probs['home_win'] *= 1.05
        
        # Recent form overreaction correction
        if home_profile['momentum'] > 1:
            base_probs['home_win'] *= 0.98
        if home_profile['momentum'] < -1:
            base_probs['home_win'] *= 1.02
        
        # Normalize probabilities
        total = sum(base_probs.values())
        for key in base_probs:
            base_probs[key] /= total
        
        return base_probs
    
    def _bayesian_model_averaging(self, home_profile, away_profile, context):
        """Bayesian model averaging with multiple prediction approaches"""
        
        models = {}
        
        # 1. Poisson model (traditional)
        models['poisson'] = self._poisson_model(home_profile, away_profile, context)
        
        # 2. xG regression model (performance-based)
        models['xg_regression'] = self._xg_regression_model(home_profile, away_profile, context)
        
        # 3. Elo-style rating model
        models['rating'] = self._rating_model(home_profile, away_profile, context)
        
        # 4. Form-adjusted model
        models['form'] = self._form_model(home_profile, away_profile, context)
        
        # Calculate dynamic model weights based on context
        weights = self._calculate_model_weights(home_profile, away_profile, context, models)
        
        # Weighted average
        final_probs = {'home_win': 0, 'draw': 0, 'away_win': 0}
        for model_name, model_probs in models.items():
            for outcome in final_probs:
                final_probs[outcome] += model_probs[outcome] * weights[model_name]
        
        return final_probs, weights, models
    
    def _poisson_model(self, home_profile, away_profile, context):
        """Enhanced Poisson model with tactical adjustments"""
        home_advantage = self._calculate_dynamic_home_advantage(home_profile, away_profile, context)
        
        # Base expected goals
        lambda_home = home_profile['xg_offense'] * away_profile['xg_defense'] * home_advantage
        lambda_away = away_profile['xg_offense'] * home_profile['xg_defense'] / home_advantage
        
        # Apply finishing efficiency
        lambda_home *= home_profile['finishing_efficiency']
        lambda_away *= away_profile['finishing_efficiency']
        
        # Generate probabilities
        home_win = 0
        draw = 0
        away_win = 0
        
        for i in range(10):  # Home goals
            for j in range(10):  # Away goals
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        total = home_win + draw + away_win
        return {
            'home_win': home_win / total,
            'draw': draw / total,
            'away_win': away_win / total
        }
    
    def _xg_regression_model(self, home_profile, away_profile, context):
        """xG-based regression model"""
        # Simulate more sophisticated regression approach
        home_strength = home_profile['xg_offense'] - home_profile['xg_defense']
        away_strength = away_profile['xg_offense'] - away_profile['xg_defense']
        
        strength_diff = home_strength - away_strength + 0.3  # Home advantage
        
        # Logistic regression approximation
        home_win_prob = 1 / (1 + np.exp(-strength_diff))
        away_win_prob = 1 / (1 + np.exp(strength_diff + 0.2))
        draw_prob = 1 - home_win_prob - away_win_prob
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        return {
            'home_win': home_win_prob / total,
            'draw': draw_prob / total,
            'away_win': away_win_prob / total
        }
    
    def _rating_model(self, home_profile, away_profile, context):
        """Elo-style rating model"""
        home_rating = home_profile['xg_offense'] * 1000
        away_rating = away_profile['xg_offense'] * 1000
        
        # Home advantage
        home_rating += 70
        
        rating_diff = home_rating - away_rating
        
        # Elo probability calculation
        home_win_prob = 1 / (1 + 10 ** (-rating_diff / 400))
        away_win_prob = 1 / (1 + 10 ** (rating_diff / 400))
        
        # Draw probability estimation
        draw_prob = 0.2 * (1 - abs(home_win_prob - away_win_prob))
        
        home_win_adj = home_win_prob * (1 - draw_prob)
        away_win_adj = away_win_prob * (1 - draw_prob)
        
        return {
            'home_win': home_win_adj,
            'draw': draw_prob,
            'away_win': away_win_adj
        }
    
    def _form_model(self, home_profile, away_profile, context):
        """Recent form weighted model"""
        # Weight recent form heavily
        home_form = home_profile.get('momentum', 0) * 0.1 + 1.0
        away_form = away_profile.get('momentum', 0) * 0.1 + 1.0
        
        home_strength = home_profile['xg_offense'] * home_form
        away_strength = away_profile['xg_offense'] * away_form
        
        strength_diff = home_strength - away_strength + 0.25
        
        home_win_prob = 0.4 + strength_diff * 0.15
        away_win_prob = 0.3 - strength_diff * 0.15
        draw_prob = 0.3
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        return {
            'home_win': home_win_prob / total,
            'draw': draw_prob / total,
            'away_win': away_win_prob / total
        }
    
    def _calculate_model_weights(self, home_profile, away_profile, context, models):
        """Dynamic model weighting based on context"""
        base_weights = {
            'poisson': 0.25,
            'xg_regression': 0.30,
            'rating': 0.25,
            'form': 0.20
        }
        
        # Adjust weights based on context
        if context.get('data_quality') == "HIGH":
            base_weights['xg_regression'] += 0.1
            base_weights['poisson'] -= 0.05
            base_weights['form'] -= 0.05
        
        if home_profile['consistency'] == "VOLATILE" or away_profile['consistency'] == "VOLATILE":
            base_weights['form'] += 0.1
            base_weights['rating'] -= 0.1
        
        # Normalize
        total = sum(base_weights.values())
        for key in base_weights:
            base_weights[key] /= total
        
        return base_weights
    
    def predict_match(self, home_team, away_team, context):
        """Main prediction method with full precision pipeline"""
        
        # Build team profiles
        home_profile = self._build_team_profile(home_team, is_home=True)
        away_profile = self._build_team_profile(away_team, is_home=False)
        
        # Simulate tactical matchup
        tactical_effects = self._simulate_tactical_matchup(home_profile, away_profile)
        
        # Bayesian model averaging
        base_probs, model_weights, individual_models = self._bayesian_model_averaging(
            home_profile, away_profile, context
        )
        
        # Apply tactical effects
        adjusted_probs = self._apply_tactical_effects(base_probs, tactical_effects)
        
        # Market efficiency corrections
        final_probs = self._market_efficient_pricing(home_profile, away_profile, adjusted_probs)
        
        # Calculate additional markets
        additional_markets = self._calculate_additional_markets(home_profile, away_profile, final_probs, context)
        
        # Uncertainty quantification
        uncertainty = self._calculate_prediction_uncertainty(individual_models, model_weights)
        
        return {
            'match_outcome': final_probs,
            'additional_markets': additional_markets,
            'uncertainty': uncertainty,
            'model_weights': model_weights,
            'tactical_effects': tactical_effects,
            'team_profiles': {'home': home_profile, 'away': away_profile},
            'individual_models': individual_models
        }
    
    def _apply_tactical_effects(self, probs, tactical_effects):
        """Apply tactical matchup effects to probabilities"""
        adjusted = probs.copy()
        
        # Simple adjustment - in practice this would be more sophisticated
        home_strength = probs['home_win'] / (probs['home_win'] + probs['away_win'])
        adjustment = (tactical_effects['home_xg_boost'] - 1.0) * 0.3
        
        adjusted['home_win'] = home_strength + adjustment
        adjusted['away_win'] = (1 - home_strength) - adjustment
        adjusted['draw'] = probs['draw']  # Keep draw relatively stable
        
        # Normalize
        total = sum(adjusted.values())
        for key in adjusted:
            adjusted[key] /= total
        
        return adjusted
    
    def _calculate_additional_markets(self, home_profile, away_profile, outcome_probs, context):
        """Calculate over/under and BTTS probabilities"""
        # Estimate expected goals from outcome probabilities
        avg_home_goals = outcome_probs['home_win'] * 2.0 + outcome_probs['draw'] * 1.0 + outcome_probs['away_win'] * 0.5
        avg_away_goals = outcome_probs['away_win'] * 2.0 + outcome_probs['draw'] * 1.0 + outcome_probs['home_win'] * 0.5
        
        total_goals = avg_home_goals + avg_away_goals
        
        # Over/Under 2.5 calculation
        over_prob = 1 / (1 + np.exp(-(total_goals - 2.5) * 2))
        under_prob = 1 - over_prob
        
        # BTTS calculation
        home_score_prob = 1 - np.exp(-avg_home_goals)
        away_score_prob = 1 - np.exp(-avg_away_goals)
        btts_prob = home_score_prob * away_score_prob
        
        return {
            'over_2.5': over_prob,
            'under_2.5': under_prob,
            'btts_yes': btts_prob,
            'btts_no': 1 - btts_prob,
            'expected_goals': {'home': avg_home_goals, 'away': avg_away_goals}
        }
    
    def _calculate_prediction_uncertainty(self, models, weights):
        """Calculate prediction uncertainty from model variance"""
        outcomes = ['home_win', 'draw', 'away_win']
        variances = []
        
        for outcome in outcomes:
            weighted_mean = sum(models[model][outcome] * weights[model] for model in models)
            variance = sum(weights[model] * (models[model][outcome] - weighted_mean) ** 2 for model in models)
            variances.append(variance)
        
        avg_variance = np.mean(variances)
        uncertainty = min(1.0, avg_variance * 5)  # Scale to 0-1
        
        return uncertainty

# =============================================================================
# PRECISION CONFIDENCE ENGINE
# =============================================================================

class PrecisionConfidenceEngine:
    def __init__(self):
        self.confidence_thresholds = {
            'HIGH_PRECISION': 0.75,
            'MEDIUM_CONFIDENCE': 0.60,
            'LOW_CONFIDENCE': 0.40
        }
    
    def calculate_confidence(self, prediction, context):
        """Professional-grade confidence assessment"""
        
        confidence_factors = {
            'model_agreement': self._assess_model_agreement(prediction['individual_models']),
            'tactical_clarity': self._assess_tactical_clarity(prediction['tactical_effects']),
            'data_quality': self._assess_data_quality(context),
            'psychological_stability': self._assess_psychological_stability(prediction['team_profiles']),
            'market_efficiency': self._assess_market_efficiency(prediction),
            'historical_precedents': self._check_historical_precedents(prediction['team_profiles'])
        }
        
        # Weighted confidence score
        weights = {
            'model_agreement': 0.25,
            'tactical_clarity': 0.20,
            'data_quality': 0.20,
            'psychological_stability': 0.15,
            'market_efficiency': 0.10,
            'historical_precedents': 0.10
        }
        
        base_confidence = sum(confidence_factors[factor] * weights[factor] for factor in confidence_factors)
        
        # Uncertainty penalty
        uncertainty_penalty = prediction['uncertainty'] * 0.3
        final_confidence = base_confidence * (1 - uncertainty_penalty)
        
        # Determine confidence level
        if final_confidence >= self.confidence_thresholds['HIGH_PRECISION']:
            level = "HIGH PRECISION"
            color = "🟢"
            description = "Strong model agreement with clear tactical edge"
        elif final_confidence >= self.confidence_thresholds['MEDIUM_CONFIDENCE']:
            level = "MEDIUM CONFIDENCE"
            color = "🟡"
            description = "Reasonable prediction with some uncertainty factors"
        else:
            level = "LOW CONFIDENCE"
            color = "🔴"
            description = "High uncertainty - consider alternative approaches"
        
        return {
            'level': level,
            'score': final_confidence,
            'color': color,
            'description': description,
            'factors': confidence_factors
        }
    
    def _assess_model_agreement(self, individual_models):
        """Assess how much the different models agree"""
        outcomes = ['home_win', 'draw', 'away_win']
        variances = []
        
        for outcome in outcomes:
            probs = [individual_models[model][outcome] for model in individual_models]
            variance = np.var(probs)
            variances.append(variance)
        
        avg_variance = np.mean(variances)
        agreement = 1 - min(avg_variance * 10, 1.0)  # Convert to 0-1 scale
        
        return agreement
    
    def _assess_tactical_clarity(self, tactical_effects):
        """Assess how clear the tactical advantage is"""
        home_boost = tactical_effects.get('home_xg_boost', 1.0)
        clarity = min(abs(home_boost - 1.0) * 5, 1.0)  # Stronger effects = clearer
        
        return clarity
    
    def _assess_data_quality(self, context):
        """Assess quality and completeness of input data"""
        quality = context.get('data_quality_score', 70) / 100
        
        # Penalize for missing data
        if context.get('missing_key_metrics', False):
            quality *= 0.7
        
        return quality
    
    def _assess_psychological_stability(self, team_profiles):
        """Assess psychological stability of teams"""
        home_stability = 1.0 if team_profiles['home']['consistency'] in ['VERY_CONSISTENT', 'CONSISTENT'] else 0.6
        away_stability = 1.0 if team_profiles['away']['consistency'] in ['VERY_CONSISTENT', 'CONSISTENT'] else 0.6
        
        return (home_stability + away_stability) / 2
    
    def _assess_market_efficiency(self, prediction):
        """Assess if market is likely efficient for this match"""
        # Big public teams often have inefficient pricing
        public_teams = ['Man United', 'Barcelona', 'Real Madrid', 'Bayern Munich']
        home_public = prediction['team_profiles']['home']['name'] in public_teams
        away_public = prediction['team_profiles']['away']['name'] in public_teams
        
        if home_public or away_public:
            return 0.7  # Lower confidence in market efficiency
        else:
            return 0.9  # Higher confidence
    
    def _check_historical_precedents(self, team_profiles):
        """Check if historical patterns support prediction"""
        # Simplified - in practice would use historical data
        return 0.8

# =============================================================================
# STREAMLIT PRECISION INTERFACE
# =============================================================================

def create_precision_interface():
    """Create professional-grade input interface"""
    
    st.markdown("""
    <style>
    .precision-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .team-profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
    }
    .confidence-high { 
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    .confidence-medium { 
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    .confidence-low { 
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="precision-header">🎯 PRECISION PREDICTION ENGINE</div>', unsafe_allow_html=True)
    
    # League selection
    col1, col2 = st.columns([1, 1])
    with col1:
        league = st.selectbox("🏆 SELECT LEAGUE", ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
    
    # Team selection with auto-profiling
    teams = get_teams_for_league(league)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="team-profile-card">🏠 HOME TEAM</div>', unsafe_allow_html=True)
        home_team = st.selectbox("Home Team", teams, key="home")
        
        # Home team quick profile
        col1, col2 = st.columns(2)
        with col1:
            home_strength = st.slider("Offensive Strength", 1, 10, 7, key="home_off")
            home_momentum = st.slider("Current Momentum", -2, 2, 0, 
                                    format="%d (Collapsing ← → Surging)", key="home_mom")
        with col2:
            home_style = st.selectbox("Tactical Style", 
                                    ["High Press", "Possession", "Counter-Attack", "Defensive Block"], 
                                    key="home_style")
    
    with col2:
        st.markdown('<div class="team-profile-card">✈️ AWAY TEAM</div>', unsafe_allow_html=True)
        away_team = st.selectbox("Away Team", teams, key="away")
        
        col1, col2 = st.columns(2)
        with col1:
            away_strength = st.slider("Offensive Strength", 1, 10, 6, key="away_off")
            away_motivation = st.slider("Motivation", 0, 100, 70,
                                      format="%d%% (Nothing ← → Everything)", key="away_mot")
        with col2:
            away_style = st.selectbox("Tactical Style", 
                                    ["High Press", "Possession", "Counter-Attack", "Defensive Block"],
                                    key="away_style")
    
    # Precision context
    with st.expander("🎯 PRECISION CONTEXT SETTINGS", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_stakes = st.radio("🎪 MATCH STAKES", 
                                  ["Dead Rubber", "Normal League", "Relegation Battle", "Cup Final/Title Decider"])
            crowd_impact = st.radio("👥 CROWD IMPACT", ["Normal", "Hostile Away", "Electric Home"])
        
        with col2:
            weather_conditions = st.selectbox("🌤️ CONDITIONS", 
                                            ["Perfect", "Normal", "Heavy Pitch", "Extreme Weather"])
            referee_style = st.selectbox("⚖️ REFEREE STYLE", ["Lenient", "Normal", "Card Happy"])
        
        with col3:
            data_quality = st.slider("📊 DATA QUALITY", 50, 100, 85,
                                   format="%d%% (Poor ← → Excellent)")
            recent_form_weight = st.slider("📈 FORM WEIGHTING", 0, 100, 70,
                                         format="%d%% (Season Avg ← → Recent Form)")
    
    return {
        'league': league,
        'home_team': home_team,
        'away_team': away_team,
        'home_profile': {
            'offensive_strength': home_strength,
            'momentum': home_momentum,
            'tactical_style': home_style
        },
        'away_profile': {
            'offensive_strength': away_strength,
            'motivation': away_motivation,
            'tactical_style': away_style
        },
        'context': {
            'match_stakes': match_stakes,
            'crowd_impact': crowd_impact,
            'weather_conditions': weather_conditions,
            'referee_style': referee_style,
            'data_quality_score': data_quality,
            'recent_form_weight': recent_form_weight
        }
    }

def display_precision_results(prediction, confidence, input_data):
    """Display professional-grade results"""
    
    st.markdown("---")
    st.markdown('<div class="precision-header">🎯 PRECISION PREDICTION RESULTS</div>', unsafe_allow_html=True)
    
    # Confidence banner
    confidence_html = f"""
    <div class="confidence-{confidence['level'].split()[0].lower()}">
        {confidence['color']} {confidence['level']} - {confidence['description']}
    </div>
    """
    st.markdown(confidence_html, unsafe_allow_html=True)
    
    # Main predictions in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏆 MATCH OUTCOME")
        outcome = prediction['match_outcome']
        
        # Enhanced outcome visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['HOME', 'DRAW', 'AWAY'],
            y=[outcome['home_win'], outcome['draw'], outcome['away_win']],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[f"{outcome['home_win']:.1%}", f"{outcome['draw']:.1%}", f"{outcome['away_win']:.1%}"],
            textposition='auto',
        ))
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="Probability",
            yaxis_tickformat=".0%",
            title="Match Outcome Probabilities"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 ADDITIONAL MARKETS")
        markets = prediction['additional_markets']
        
        # Over/Under and BTTS
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Over/Under 2.5', 'Both Teams Score'))
        
        fig.add_trace(go.Bar(
            x=['OVER', 'UNDER'],
            y=[markets['over_2.5'], markets['under_2.5']],
            marker_color=['#ff6b6b', '#4ecdc4'],
            name='Over/Under'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=['YES', 'NO'],
            y=[markets['btts_yes'], markets['btts_no']],
            marker_color=['#a05195', '#f95d6a'],
            name='BTTS'
        ), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("🔍 MODEL INSIGHTS")
        
        # Expected goals
        exp_goals = markets['expected_goals']
        st.metric(
            "Expected Goals",
            f"{exp_goals['home']:.1f} - {exp_goals['away']:.1f}",
            delta=f"Total: {exp_goals['home'] + exp_goals['away']:.1f}"
        )
        
        # Model weights
        st.write("**Model Contributions:**")
        for model, weight in prediction['model_weights'].items():
            st.write(f"• {model.title()}: {weight:.1%}")
        
        # Tactical effects
        st.write("**Tactical Analysis:**")
        effects = prediction['tactical_effects']
        st.write(f"• Home XG Boost: {effects.get('home_xg_boost', 1.0):.2f}x")
        st.write(f"• BTTS Probability: {effects.get('btts_boost', 1.0):.2f}x")
    
    # Detailed analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 CONFIDENCE BREAKDOWN")
        factors = confidence['factors']
        
        for factor, score in factors.items():
            st.write(f"**{factor.replace('_', ' ').title()}:** {score:.1%}")
        
        st.progress(confidence['score'])
        st.write(f"Overall Confidence Score: {confidence['score']:.1%}")
    
    with col2:
        st.subheader("⚡ RECOMMENDATION")
        
        best_bet = max([
            ('Home Win', prediction['match_outcome']['home_win']),
            ('Draw', prediction['match_outcome']['draw']),
            ('Away Win', prediction['match_outcome']['away_win']),
            ('Over 2.5', prediction['additional_markets']['over_2.5']),
            ('Under 2.5', prediction['additional_markets']['under_2.5']),
            ('BTTS Yes', prediction['additional_markets']['btts_yes']),
            ('BTTS No', prediction['additional_markets']['btts_no'])
        ], key=lambda x: x[1])
        
        st.success(f"**STRONGEST VALUE:** {best_bet[0]} ({best_bet[1]:.1%} probability)")
        
        if confidence['score'] >= 0.75:
            st.info("💰 **HIGH PRECISION ALERT**: This prediction meets professional confidence thresholds")
        elif confidence['score'] >= 0.60:
            st.warning("📊 **MEDIUM CONFIDENCE**: Consider position sizing appropriately")
        else:
            st.error("🎲 **LOW CONFIDENCE**: High uncertainty - avoid large positions")

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

def convert_input_to_team_data(input_data):
    """Convert streamlined inputs to detailed team data"""
    home_profile = input_data['home_profile']
    away_profile = input_data['away_profile']
    
    # Convert simplified inputs to detailed data structures
    home_data = {
        'name': input_data['home_team'],
        'overall': {
            'matches': 10,
            'goals_scored': home_profile['offensive_strength'] * 1.8,
            'goals_conceded': (10 - home_profile['offensive_strength']) * 1.2,
            'xG': home_profile['offensive_strength'] * 1.9,
            'xGA': (10 - home_profile['offensive_strength']) * 1.3,
            'xg_per_game': home_profile['offensive_strength'] * 0.19,
            'xga_per_game': (10 - home_profile['offensive_strength']) * 0.13,
            'finishing_efficiency': 1.0 + (home_profile['momentum'] * 0.05)
        },
        'momentum': home_profile['momentum']
    }
    
    away_data = {
        'name': input_data['away_team'],
        'overall': {
            'matches': 10,
            'goals_scored': away_profile['offensive_strength'] * 1.6,
            'goals_conceded': (10 - away_profile['offensive_strength']) * 1.4,
            'xG': away_profile['offensive_strength'] * 1.7,
            'xGA': (10 - away_profile['offensive_strength']) * 1.5,
            'xg_per_game': away_profile['offensive_strength'] * 0.17,
            'xga_per_game': (10 - away_profile['offensive_strength']) * 0.15,
            'finishing_efficiency': 0.9 + (away_profile['motivation'] / 100 * 0.2)
        },
        'momentum': away_profile['offensive_strength'] / 10  # Simplified momentum
    }
    
    return home_data, away_data

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Precision Prediction Engine",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize engines
    prediction_engine = PrecisionPredictionEngine()
    confidence_engine = PrecisionConfidenceEngine()
    
    # Create interface
    input_data = create_precision_interface()
    
    # Prediction button
    if st.button("🎯 GENERATE PRECISION PREDICTION", type="primary", use_container_width=True):
        with st.spinner("🔄 Running precision analysis..."):
            # Convert inputs to team data
            home_data, away_data = convert_input_to_team_data(input_data)
            
            # Generate prediction
            prediction = prediction_engine.predict_match(home_data, away_data, input_data['context'])
            
            # Calculate confidence
            confidence = confidence_engine.calculate_confidence(prediction, input_data['context'])
            
            # Display results
            display_precision_results(prediction, confidence, input_data)

if __name__ == "__main__":
    main()
