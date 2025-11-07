import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ENHANCED HYBRID PRECISION ENGINE - COMPLETE IMPLEMENTATION
# =============================================================================

class InjuryImpactAssessor:
    def __init__(self):
        self.player_impact_db = {
            # Premier League
            'MARTIN ØDEGAARD': {'creativity': 0.85, 'xg_contribution': 0.80, 'possession_impact': 0.90},
            'BEN WHITE': {'defense': 0.90, 'xg_allowed_impact': 1.10},
            'JORDAN PICKFORD': {'goalkeeping': 0.70, 'xg_allowed_impact': 1.25},
            'KEVIN DE BRUYNE': {'creativity': 0.80, 'xg_contribution': 0.75, 'assist_impact': 0.70},
            'ERLING HAALAND': {'xg_contribution': 0.70, 'finishing': 0.65},
            'VIRGIL VAN DIJK': {'defense': 0.75, 'xg_allowed_impact': 1.20, 'leadership': 0.80},
            'ALISSON': {'goalkeeping': 0.70, 'xg_allowed_impact': 1.30},
            'HARRY KANE': {'xg_contribution': 0.75, 'finishing': 0.70},
            'SON HEUNG-MIN': {'xg_contribution': 0.80, 'creativity': 0.85},
            'BRUNO FERNANDES': {'creativity': 0.85, 'xg_contribution': 0.80},
            
            # La Liga
            'JUDE BELLINGHAM': {'xg_contribution': 0.80, 'creativity': 0.85},
            'VINICIUS JR': {'xg_contribution': 0.75, 'creativity': 0.80},
            'ROBERT LEWANDOWSKI': {'xg_contribution': 0.70, 'finishing': 0.75},
            'TER STEGEN': {'goalkeeping': 0.75, 'xg_allowed_impact': 1.25},
            'ANTOINE GRIEZMANN': {'creativity': 0.85, 'xg_contribution': 0.80},
            
            # Bundesliga
            'HARRY KANE': {'xg_contribution': 0.70, 'finishing': 0.75},
            'JAMAL MUSIALA': {'creativity': 0.85, 'xg_contribution': 0.80},
            'LEROY SANE': {'xg_contribution': 0.80, 'creativity': 0.85},
            
            # Serie A
            'LAUTARO MARTINEZ': {'xg_contribution': 0.75, 'finishing': 0.80},
            'KHVICHA KVARATSKHELIA': {'creativity': 0.85, 'xg_contribution': 0.80},
            'MIKE MAIGNAN': {'goalkeeping': 0.75, 'xg_allowed_impact': 1.25},
            
            # Ligue 1
            'KYLIAN MBAPPE': {'xg_contribution': 0.65, 'finishing': 0.70},
            'GIANLUIGI DONNARUMMA': {'goalkeeping': 0.80, 'xg_allowed_impact': 1.20},
        }
    
    def calculate_team_impact(self, injury_list, team_profile):
        if not injury_list:
            return {'xg_boost': 1.0, 'xga_boost': 1.0, 'possession_impact': 1.0}
        
        total_impact = {'xg_boost': 1.0, 'xga_boost': 1.0, 'possession_impact': 1.0}
        impact_count = 0
        
        for injury in injury_list:
            player_impact = self.player_impact_db.get(injury.upper())
            if player_impact:
                if 'xg_contribution' in player_impact:
                    total_impact['xg_boost'] *= player_impact['xg_contribution']
                if 'xg_allowed_impact' in player_impact:
                    total_impact['xga_boost'] *= player_impact['xg_allowed_impact']
                if 'possession_impact' in player_impact:
                    total_impact['possession_impact'] *= player_impact['possession_impact']
                impact_count += 1
        
        # Normalize if multiple impacts
        if impact_count > 1:
            for key in total_impact:
                total_impact[key] = 1 - (1 - total_impact[key]) * 0.8  # Diminishing returns
        
        return total_impact

class ContextModulator:
    def __init__(self):
        self.factors = {
            'crowd_impact': {
                'Electric Home Crowd': {'home_win_boost': 0.045, 'away_win_reduction': 0.035},
                'Hostile Away': {'away_win_reduction': 0.040, 'home_win_boost': 0.025},
                'Normal': {}
            },
            'referee_style': {
                'Lenient': {'total_goals_boost': 1.06, 'btts_boost': 1.04},
                'Strict': {'total_goals_dampen': 0.94, 'btts_dampen': 0.96},
                'Normal': {}
            },
            'match_importance': {
                'Title Decider': {'draw_boost': 0.055, 'total_goals_dampen': 0.92},
                'Relegation Battle': {'btts_dampen': 0.94, 'under_25_boost': 1.06},
                'Cup Final': {'draw_boost': 0.065, 'total_goals_dampen': 0.90},
                'Dead Rubber': {'total_goals_boost': 1.08, 'btts_boost': 1.05},
                'Normal': {}
            },
            'weather': {
                'Poor Pitch': {'total_goals_dampen': 0.88, 'btts_dampen': 0.92},
                'Extreme Weather': {'total_goals_dampen': 0.85, 'draw_boost': 0.06},
                'Perfect Conditions': {'total_goals_boost': 1.05, 'btts_boost': 1.03},
                'Normal': {}
            }
        }
    
    def apply_context_effects(self, prediction, context):
        """Apply context as final probability adjustment"""
        if not context:
            return prediction
        
        adjusted_prediction = prediction.copy()
        effects = self._aggregate_context_effects(context)
        
        if effects:
            adjusted_prediction = self._apply_aggregated_effects(adjusted_prediction, effects)
        
        return self._normalize_probabilities(adjusted_prediction)
    
    def _aggregate_context_effects(self, context):
        """Combine all context effects"""
        aggregated = {}
        
        for factor, value in context.items():
            if factor in self.factors and value in self.factors[factor]:
                factor_effects = self.factors[factor][value]
                for effect, magnitude in factor_effects.items():
                    if effect in aggregated:
                        aggregated[effect] *= magnitude
                    else:
                        aggregated[effect] = magnitude
        
        return aggregated
    
    def _apply_aggregated_effects(self, prediction, effects):
        """Apply all effects to prediction"""
        adjusted = prediction.copy()
        
        # Apply to match outcomes
        if 'home_win_boost' in effects:
            adjusted['match_outcome']['home_win'] *= (1 + effects['home_win_boost'])
        if 'away_win_reduction' in effects:
            adjusted['match_outcome']['away_win'] *= (1 - effects['away_win_reduction'])
        if 'draw_boost' in effects:
            adjusted['match_outcome']['draw'] *= (1 + effects['draw_boost'])
        
        # Apply to additional markets
        if 'total_goals_boost' in effects:
            boost = effects['total_goals_boost']
            adjusted['additional_markets']['over_2.5'] *= boost
            adjusted['additional_markets']['under_2.5'] = 1 - adjusted['additional_markets']['over_2.5']
        
        if 'total_goals_dampen' in effects:
            dampen = effects['total_goals_dampen']
            adjusted['additional_markets']['over_2.5'] *= dampen
            adjusted['additional_markets']['under_2.5'] = 1 - adjusted['additional_markets']['over_2.5']
        
        if 'btts_boost' in effects:
            boost = effects['btts_boost']
            adjusted['additional_markets']['btts_yes'] *= boost
            adjusted['additional_markets']['btts_no'] = 1 - adjusted['additional_markets']['btts_yes']
        
        if 'btts_dampen' in effects:
            dampen = effects['btts_dampen']
            adjusted['additional_markets']['btts_yes'] *= dampen
            adjusted['additional_markets']['btts_no'] = 1 - adjusted['additional_markets']['btts_yes']
        
        return adjusted
    
    def _normalize_probabilities(self, prediction):
        """Ensure all probabilities sum to 1"""
        # Normalize match outcomes
        total = sum(prediction['match_outcome'].values())
        for key in prediction['match_outcome']:
            prediction['match_outcome'][key] /= total
        
        # Ensure additional markets are valid
        prediction['additional_markets']['over_2.5'] = max(0.01, min(0.99, prediction['additional_markets']['over_2.5']))
        prediction['additional_markets']['under_2.5'] = 1 - prediction['additional_markets']['over_2.5']
        prediction['additional_markets']['btts_yes'] = max(0.01, min(0.99, prediction['additional_markets']['btts_yes']))
        prediction['additional_markets']['btts_no'] = 1 - prediction['additional_markets']['btts_yes']
        
        return prediction

class ValueFinder:
    def calculate_true_value(self, our_probability, market_odds):
        """Calculate if a bet offers positive expected value"""
        if not market_odds or market_odds <= 1.0:
            return None
        
        implied_probability = 1.0 / market_odds
        value_ratio = our_probability / implied_probability
        expected_value = (our_probability * (market_odds - 1)) - (1 - our_probability)
        
        if value_ratio > 1.20:
            recommendation = 'STRONG VALUE 💎'
        elif value_ratio > 1.10:
            recommendation = 'GOOD VALUE ✅'
        elif value_ratio > 1.05:
            recommendation = 'MARGINAL VALUE ⚠️'
        else:
            recommendation = 'NO VALUE ❌'
        
        return {
            'value_ratio': value_ratio,
            'expected_value': expected_value,
            'implied_probability': implied_probability,
            'recommendation': recommendation
        }
    
    def find_best_bets(self, prediction, market_odds=None):
        """Find true value bets based on market odds"""
        if not market_odds:
            return self._fallback_best_bet(prediction)
        
        value_analysis = {}
        markets = [
            ('home_win', prediction['match_outcome']['home_win']),
            ('draw', prediction['match_outcome']['draw']),
            ('away_win', prediction['match_outcome']['away_win']),
            ('over_2.5', prediction['additional_markets']['over_2.5']),
            ('under_2.5', prediction['additional_markets']['under_2.5']),
            ('btts_yes', prediction['additional_markets']['btts_yes']),
            ('btts_no', prediction['additional_markets']['btts_no'])
        ]
        
        for market, our_prob in markets:
            if market in market_odds:
                analysis = self.calculate_true_value(our_prob, market_odds[market])
                if analysis:
                    value_analysis[market] = analysis
        
        # Return sorted by value ratio (best first)
        return sorted([(k, v) for k, v in value_analysis.items()], 
                     key=lambda x: x[1]['value_ratio'], reverse=True)
    
    def _fallback_best_bet(self, prediction):
        """Fallback to highest probability if no odds provided"""
        markets = [
            ('Home Win', prediction['match_outcome']['home_win']),
            ('Draw', prediction['match_outcome']['draw']),
            ('Away Win', prediction['match_outcome']['away_win']),
            ('Over 2.5', prediction['additional_markets']['over_2.5']),
            ('Under 2.5', prediction['additional_markets']['under_2.5']),
            ('BTTS Yes', prediction['additional_markets']['btts_yes']),
            ('BTTS No', prediction['additional_markets']['btts_no'])
        ]
        return [(max(markets, key=lambda x: x[1]), {'recommendation': 'HIGHEST PROBABILITY'})]

class HybridPrecisionEngine:
    def __init__(self, constraints=None):
        self.constraints = {
            'max_transitions_per_game': 12,
            'possession_zero_sum': True,
            'game_state_damping': 0.7,
            'style_interaction_caps': {
                'HIGH_PRESS_VS_COUNTER': 1.25,
                'POSSESSION_VS_DEEP_DEFENSE': 0.85
            },
            'max_total_xg': 5.0,
            'uncertainty_compression_threshold': 0.55,
            'uncertainty_compression_factor': 0.80
        }
        if constraints:
            self.constraints.update(constraints)
        
        # Initialize enhancement modules
        self.injury_assessor = InjuryImpactAssessor()
        self.context_modulator = ContextModulator()
        self.value_finder = ValueFinder()

    def _poisson_prob(self, lam, k):
        return (lam ** k) * np.exp(-lam) / np.math.factorial(k)

    def _calculate_base_expected_goals(self, home_profile, away_profile, context=None):
        home_xg = home_profile.get('xg_per_game', 1.0)
        away_xg = away_profile.get('xg_per_game', 1.0)

        home_adv_multiplier = 1.14
        base_home = home_xg * home_adv_multiplier
        base_away = away_xg

        return {'home': base_home, 'away': base_away}

    def _apply_football_constraints(self, base_xg, home_profile, away_profile):
        constrained = base_xg.copy()

        # CONSTRAINT 1: possession zero-sum
        if self.constraints['possession_zero_sum']:
            home_pos = home_profile.get('possession', 50)
            away_pos = away_profile.get('possession', 50)
            if home_pos + away_pos > 100:
                scale = 100.0 / (home_pos + away_pos)
                constrained['home'] *= scale
                constrained['away'] *= scale

        # CONSTRAINT 2: style interaction caps
        home_style = home_profile.get('tactical_style', 'BALANCED')
        away_style = away_profile.get('tactical_style', 'BALANCED')

        if 'COUNTER' in home_style and 'HIGH_PRESS' in away_style:
            cap = self.constraints['style_interaction_caps'].get('HIGH_PRESS_VS_COUNTER', 1.25)
            style_boost = min(cap, 1.15)
            constrained['home'] *= style_boost

        if 'POSSESSION' in away_style and 'DEFENSIVE' in home_style:
            cap = self.constraints['style_interaction_caps'].get('POSSESSION_VS_DEEP_DEFENSE', 0.85)
            style_damp = max(cap, 0.9)
            constrained['away'] *= style_damp

        # CONSTRAINT 3: total xg cap
        total_xg = constrained['home'] + constrained['away']
        if total_xg > self.constraints['max_total_xg']:
            damping = self.constraints['max_total_xg'] / total_xg
            constrained['home'] *= damping
            constrained['away'] *= damping

        constrained['home'] = max(0.1, constrained['home'])
        constrained['away'] = max(0.1, constrained['away'])

        return constrained

    def _simulate_game_flow(self, xg_estimates, home_profile, away_profile):
        scenarios = []

        home_style = home_profile.get('tactical_style', 'BALANCED')
        away_style = away_profile.get('tactical_style', 'BALANCED')

        base_home_first = 0.34
        base_away_first = 0.33
        base_gfh = 0.33

        if 'COUNTER' in home_style and 'HIGH_PRESS' in away_style:
            base_home_first = 0.40
            base_away_first = 0.30
            base_gfh = 0.30

        total = base_home_first + base_away_first + base_gfh
        base_home_first /= total
        base_away_first /= total
        base_gfh /= total

        # Scenario 1: Home scores first
        scenarios.append({
            'prob': base_home_first,
            'home_xg': xg_estimates['home'] * 0.80,
            'away_xg': xg_estimates['away'] * 1.10
        })

        # Scenario 2: Away scores first
        scenarios.append({
            'prob': base_away_first,
            'home_xg': xg_estimates['home'] * 1.20,
            'away_xg': xg_estimates['away'] * 0.90
        })

        # Scenario 3: Goalless first half
        scenarios.append({
            'prob': base_gfh,
            'home_xg': xg_estimates['home'] * 1.10,
            'away_xg': xg_estimates['away'] * 1.10
        })

        final_home = sum(s['prob'] * s['home_xg'] for s in scenarios)
        final_away = sum(s['prob'] * s['away_xg'] for s in scenarios)

        total = final_home + final_away
        if total > 4.0:
            damping = 4.0 / total
            final_home *= damping
            final_away *= damping

        return {'home': final_home, 'away': final_away}

    def _xg_to_match_outcome_probs(self, home_xg, away_xg, max_goals=6):
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
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

    def _apply_correlation_adjustment(self, probs, home_xg, away_xg):
        xg_ratio = min(home_xg, away_xg) / max(home_xg, away_xg)
        if xg_ratio > 0.7:
            draw_boost = 0.08
            home_win_adj = probs['home_win'] * (1 - draw_boost/2)
            away_win_adj = probs['away_win'] * (1 - draw_boost/2)
            draw_adj = probs['draw'] + draw_boost
        else:
            return probs

        total = home_win_adj + draw_adj + away_win_adj
        return {
            'home_win': home_win_adj / total,
            'draw': draw_adj / total,
            'away_win': away_win_adj / total
        }

    def _apply_style_win_bias(self, probs, home_profile, away_profile):
        home_style = home_profile.get('tactical_style', 'BALANCED')
        away_style = away_profile.get('tactical_style', 'BALANCED')

        adjustment = 0.0

        if 'COUNTER' in home_style and 'HIGH_PRESS' in away_style:
            adjustment = 0.06
        elif 'DEFENSIVE' in home_style and 'POSSESSION' in away_style:
            adjustment = 0.04

        if adjustment > 0:
            probs['home_win'] += adjustment
            probs['away_win'] -= adjustment * 0.6
            probs['draw'] -= adjustment * 0.4

        total = sum(probs.values())
        for key in probs:
            probs[key] /= total

        return probs

    def _apply_uncertainty_calibration(self, probs):
        max_prob = max(probs.values())
        threshold = self.constraints['uncertainty_compression_threshold']
        factor = self.constraints['uncertainty_compression_factor']

        if max_prob > threshold:
            compressed = probs.copy()
            largest = max(probs, key=probs.get)
            compressed[largest] = probs[largest] * factor

            remaining_prob = 1 - compressed[largest]
            other_prob_total = sum(probs[k] for k in probs if k != largest)
            for key in compressed:
                if key != largest:
                    compressed[key] = probs[key] * (remaining_prob / other_prob_total)

            return compressed
        return probs

    def predict_match(self, home_profile, away_profile, context=None):
        # Apply injury impacts as pre-processing
        if context:
            home_impact = self.injury_assessor.calculate_team_impact(
                context.get('home_injuries', []), home_profile
            )
            away_impact = self.injury_assessor.calculate_team_impact(
                context.get('away_injuries', []), away_profile
            )
            
            # Adjust profiles based on injuries
            home_profile = home_profile.copy()
            away_profile = away_profile.copy()
            
            home_profile['xg_per_game'] *= home_impact['xg_boost']
            home_profile['xga_per_game'] *= home_impact['xga_boost']
            away_profile['xg_per_game'] *= away_impact['xg_boost']
            away_profile['xga_per_game'] *= away_impact['xga_boost']

        # Original core logic
        base = self._calculate_base_expected_goals(home_profile, away_profile, context)
        constrained = self._apply_football_constraints(base, home_profile, away_profile)
        dynamic = self._simulate_game_flow(constrained, home_profile, away_profile)

        probs = self._xg_to_match_outcome_probs(dynamic['home'], dynamic['away'])
        probs = self._apply_correlation_adjustment(probs, dynamic['home'], dynamic['away'])
        probs = self._apply_style_win_bias(probs, home_profile, away_profile)
        calibrated = self._apply_uncertainty_calibration(probs)

        additional_markets = self._calculate_additional_markets(dynamic['home'], dynamic['away'])

        prediction = {
            'match_outcome': calibrated,
            'additional_markets': additional_markets,
            'expected_goals': dynamic,
            'team_profiles': {'home': home_profile, 'away': away_profile},
            'model_data': {
                'base_xg': base,
                'constrained_xg': constrained,
                'dynamic_xg': dynamic
            }
        }

        # Apply context modulation as final layer
        final_prediction = self.context_modulator.apply_context_effects(prediction, context)
        
        return final_prediction

    def _calculate_additional_markets(self, home_xg, away_xg):
        over_2_5 = 0
        btts_yes = 0

        for i in range(8):
            for j in range(8):
                prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                if i + j > 2.5:
                    over_2_5 += prob
                if i > 0 and j > 0:
                    btts_yes += prob

        return {
            'over_2.5': over_2_5,
            'under_2.5': 1 - over_2_5,
            'btts_yes': btts_yes,
            'btts_no': 1 - btts_yes
        }

# =============================================================================
# ENHANCED LEAGUE CONFIGURATIONS
# =============================================================================

LEAGUE_CONFIGS = {
    "EPL": {
        "teams": [
            "Arsenal", "Man City", "Liverpool", "Chelsea", "Tottenham", "Man United",
            "Newcastle", "Brighton", "West Ham", "Aston Villa", "Crystal Palace", "Wolves",
            "Fulham", "Bournemouth", "Brentford", "Everton", "Nott'ham Forest",
            "Luton", "Sunderland", "Burnley", "Sheffield Utd"
        ],
        "baselines": {
            "avg_goals": 2.75,
            "home_advantage": 1.14
        }
    },
    "La Liga": {
        "teams": [
            "Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao", "Real Sociedad",
            "Villarreal", "Betis", "Sevilla", "Valencia", "Osasuna", "Getafe", "Girona",
            "Mallorca", "Celta Vigo", "Rayo Vallecano", "Alaves", "Cadiz", "Granada",
            "Almeria", "Las Palmas"
        ],
        "baselines": {
            "avg_goals": 2.55,
            "home_advantage": 1.16
        }
    },
    "Bundesliga": {
        "teams": [
            "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
            "Union Berlin", "Freiburg", "Wolfsburg", "Mainz", "Monchengladbach",
            "Eintracht Frankfurt", "Koln", "Werder Bremen", "Bochum", "Augsburg",
            "Stuttgart", "Heidenheim", "Darmstadt"
        ],
        "baselines": {
            "avg_goals": 3.10,
            "home_advantage": 1.12
        }
    },
    "Serie A": {
        "teams": [
            "Inter", "Juventus", "AC Milan", "Napoli", "Atalanta", "Roma", "Lazio",
            "Fiorentina", "Bologna", "Torino", "Monza", "Udinese", "Sassuolo",
            "Empoli", "Salernitana", "Lecce", "Frosinone", "Genoa", "Verona", "Cagliari"
        ],
        "baselines": {
            "avg_goals": 2.55,
            "home_advantage": 1.15
        }
    },
    "Ligue 1": {
        "teams": [
            "PSG", "Lens", "Marseille", "Monaco", "Rennes", "Lille", "Nice",
            "Lorient", "Reims", "Lyon", "Montpellier", "Toulouse", "Clermont",
            "Strasbourg", "Nantes", "Brest", "Le Havre", "Metz"
        ],
        "baselines": {
            "avg_goals": 2.45,
            "home_advantage": 1.13
        }
    }
}

# =============================================================================
# ENHANCED STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Hybrid Precision Prediction Engine",
        page_icon="🎯",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .team-section {
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
    .value-bet-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🎯 Enhanced Hybrid Precision Prediction Engine</div>', unsafe_allow_html=True)

    # Initialize engine
    engine = HybridPrecisionEngine()

    # League selection
    col1, col2 = st.columns([1, 1])
    with col1:
        league = st.selectbox("🏆 SELECT LEAGUE", list(LEAGUE_CONFIGS.keys()))

    # Team selection
    teams = LEAGUE_CONFIGS[league]["teams"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="team-section">🏠 HOME TEAM</div>', unsafe_allow_html=True)
        home_team = st.selectbox("Home Team", teams, key="home")

        # Home team stats
        st.subheader("📊 Basic Stats (From ESPN/FotMob)")
        h_matches = st.number_input("Matches Played", min_value=1, max_value=38, value=10, key="h_m")
        h_goals = st.number_input("Goals Scored", min_value=0, max_value=100, value=17, key="h_g")
        h_conceded = st.number_input("Goals Conceded", min_value=0, max_value=100, value=8, key="h_c")
        h_xg = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=50.0, value=10.1, key="h_xg")
        h_xga = st.number_input("Expected Goals Against (xGA)", min_value=0.0, max_value=50.0, value=14.6, key="h_xga")
        h_possession = st.slider("Average Possession %", 0, 100, 53, key="h_poss")

        # Home team style
        st.subheader("🎯 Tactical Style")
        h_style = st.multiselect(
            "Select observed playing style:",
            ["HIGH_PRESS", "POSSESSION", "COUNTER", "DEFENSIVE", "WING_PLAY", "BALANCED"],
            key="h_style"
        )

    with col2:
        st.markdown('<div class="team-section">✈️ AWAY TEAM</div>', unsafe_allow_html=True)
        away_team = st.selectbox("Away Team", teams, key="away")

        # Away team stats
        st.subheader("📊 Basic Stats (From ESPN/FotMob)")
        a_matches = st.number_input("Matches Played", min_value=1, max_value=38, value=10, key="a_m")
        a_goals = st.number_input("Goals Scored", min_value=0, max_value=100, value=17, key="a_g")
        a_conceded = st.number_input("Goals Conceded", min_value=0, max_value=100, value=16, key="a_c")
        a_xg = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=50.0, value=17.8, key="a_xg")
        a_xga = st.number_input("Expected Goals Against (xGA)", min_value=0.0, max_value=50.0, value=15.9, key="a_xga")
        a_possession = st.slider("Average Possession %", 0, 100, 51, key="a_poss")

        # Away team style
        st.subheader("🎯 Tactical Style")
        a_style = st.multiselect(
            "Select observed playing style:",
            ["HIGH_PRESS", "POSSESSION", "COUNTER", "DEFENSIVE", "WING_PLAY", "BALANCED"],
            key="a_style"
        )

    # Context factors
    st.markdown("---")
    st.subheader("🎭 Enhanced Match Context")

    col1, col2, col3 = st.columns(3)

    with col1:
        match_importance = st.selectbox(
            "Match Importance",
            ["Normal", "Relegation Battle", "Title Decider", "Cup Final", "Dead Rubber"]
        )
        home_injuries = st.text_input("Home Key Injuries", placeholder="e.g., De Bruyne, Salah", value="Jordan Pickford")

    with col2:
        weather = st.selectbox(
            "Conditions",
            ["Normal", "Poor Pitch", "Extreme Weather", "Perfect Conditions"]
        )
        away_injuries = st.text_input("Away Key Injuries", placeholder="e.g., Davies, Kimmich", value="Martin Ødegaard, Ben White")

    with col3:
        crowd_impact = st.selectbox(
            "Crowd Impact",
            ["Normal", "Electric Home Crowd", "Hostile Away"]
        )
        referee = st.selectbox("Referee Style", ["Normal", "Lenient", "Strict"])

    # Market Odds Input
    st.markdown("---")
    st.subheader("💰 Market Odds (For Value Analysis)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        home_win_odds = st.number_input(f"{home_team} Win Odds", min_value=1.0, max_value=100.0, value=2.5, step=0.1)
    with col2:
        draw_odds = st.number_input("Draw Odds", min_value=1.0, max_value=100.0, value=3.2, step=0.1)
    with col3:
        away_win_odds = st.number_input(f"{away_team} Win Odds", min_value=1.0, max_value=100.0, value=2.8, step=0.1)
    with col4:
        over_25_odds = st.number_input("Over 2.5 Goals Odds", min_value=1.0, max_value=100.0, value=1.9, step=0.1)

    market_odds = {
        'home_win': home_win_odds,
        'draw': draw_odds,
        'away_win': away_win_odds,
        'over_2.5': over_25_odds,
        'under_2.5': 1.0,  # Will be calculated
        'btts_yes': 1.85,  # Default values
        'btts_no': 1.95
    }

    # Prediction button
    if st.button("🎯 GENERATE ENHANCED PREDICTION", type="primary", use_container_width=True):
        with st.spinner("🔄 Running enhanced hybrid precision analysis..."):
            # Prepare team profiles
            home_profile = {
                'name': home_team,
                'xg_per_game': h_xg / h_matches,
                'xga_per_game': h_xga / h_matches,
                'goals_per_game': h_goals / h_matches,
                'goals_against_per_game': h_conceded / h_matches,
                'possession': h_possession,
                'tactical_style': h_style if h_style else ['BALANCED'],
                'is_home': True
            }

            away_profile = {
                'name': away_team,
                'xg_per_game': a_xg / a_matches,
                'xga_per_game': a_xga / a_matches,
                'goals_per_game': a_goals / a_matches,
                'goals_against_per_game': a_conceded / a_matches,
                'possession': a_possession,
                'tactical_style': a_style if a_style else ['BALANCED'],
                'is_home': False
            }

            context = {
                'match_importance': match_importance,
                'home_injuries': [inj.strip() for inj in home_injuries.split(',')] if home_injuries else [],
                'away_injuries': [inj.strip() for inj in away_injuries.split(',')] if away_injuries else [],
                'weather': weather,
                'crowd_impact': crowd_impact,
                'referee': referee
            }

            # Generate prediction
            prediction = engine.predict_match(home_profile, away_profile, context)

            # Display results
            display_enhanced_results(prediction, home_team, away_team, market_odds, context)

def display_enhanced_results(prediction, home_team, away_team, market_odds, context):
    st.markdown("---")
    st.markdown('<div class="main-header">🎯 ENHANCED PRECISION PREDICTION RESULTS</div>', unsafe_allow_html=True)

    # Calculate confidence
    max_prob = max(prediction['match_outcome'].values())
    if max_prob >= 0.45:
        confidence_level = "HIGH"
        confidence_class = "confidence-high"
    elif max_prob >= 0.35:
        confidence_level = "MEDIUM"
        confidence_class = "confidence-medium"
    else:
        confidence_level = "LOW"
        confidence_class = "confidence-low"

    st.markdown(f'<div class="{confidence_class}">🔍 {confidence_level} CONFIDENCE PREDICTION</div>', unsafe_allow_html=True)

    # Main predictions
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏆 Match Outcome")
        outcome = prediction['match_outcome']

        fig_outcome = go.Figure(data=[
            go.Bar(x=['Home', 'Draw', 'Away'],
                  y=[outcome['home_win'], outcome['draw'], outcome['away_win']],
                  marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ])
        fig_outcome.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig_outcome, use_container_width=True)

        st.write(f"**{home_team}**: {outcome['home_win']:.1%}")
        st.write(f"**Draw**: {outcome['draw']:.1%}")
        st.write(f"**{away_team}**: {outcome['away_win']:.1%}")

    with col2:
        st.subheader("📊 Over/Under 2.5")
        markets = prediction['additional_markets']

        fig_ou = go.Figure(data=[
            go.Bar(x=['Over 2.5', 'Under 2.5'],
                  y=[markets['over_2.5'], markets['under_2.5']],
                  marker_color=['#ff6b6b', '#4ecdc4'])
        ])
        fig_ou.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig_ou, use_container_width=True)

        st.write(f"**Over 2.5**: {markets['over_2.5']:.1%}")
        st.write(f"**Under 2.5**: {markets['under_2.5']:.1%}")

    with col3:
        st.subheader("⚽ Both Teams to Score")
        btts = prediction['additional_markets']

        fig_btts = go.Figure(data=[
            go.Bar(x=['Yes', 'No'],
                  y=[btts['btts_yes'], btts['btts_no']],
                  marker_color=['#a05195', '#f95d6a'])
        ])
        fig_btts.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig_btts, use_container_width=True)

        st.write(f"**Yes**: {btts['btts_yes']:.1%}")
        st.write(f"**No**: {btts['btts_no']:.1%}")

    # Value Betting Analysis
    st.markdown("---")
    st.subheader("💰 Smart Value Analysis")
    
    value_finder = HybridPrecisionEngine().value_finder
    best_bets = value_finder.find_best_bets(prediction, market_odds)
    
    if best_bets:
        st.success("**🎯 VALUE BETS IDENTIFIED**")
        
        for market, analysis in best_bets[:3]:  # Show top 3
            market_name = market.replace('_', ' ').title()
            if market == 'home_win':
                market_name = f"{home_team} Win"
            elif market == 'away_win':
                market_name = f"{away_team} Win"
            elif market == 'over_2.5':
                market_name = "Over 2.5 Goals"
            elif market == 'under_2.5':
                market_name = "Under 2.5 Goals"
            elif market == 'btts_yes':
                market_name = "BTTS Yes"
            elif market == 'btts_no':
                market_name = "BTTS No"
            
            col1, col2, col3, col4 = st.columns([2,1,1,1])
            with col1:
                st.write(f"**{market_name}**")
            with col2:
                st.write(f"Value: {analysis['value_ratio']:.2f}x")
            with col3:
                st.write(f"EV: {analysis['expected_value']:.3f}")
            with col4:
                st.write(analysis['recommendation'])
    else:
        st.info("No strong value bets identified based on current market odds")

    # Enhanced Insights
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Expected Score & Model Process")
        exp_goals = prediction['expected_goals']
        st.metric(
            "Expected Goals",
            f"{exp_goals['home']:.1f} - {exp_goals['away']:.1f}",
            delta=f"Total: {exp_goals['home'] + exp_goals['away']:.1f} goals"
        )

        model_data = prediction['model_data']
        st.write(f"**Base xG**: {model_data['base_xg']['home']:.2f} - {model_data['base_xg']['away']:.2f}")
        st.write(f"**Constrained xG**: {model_data['constrained_xg']['home']:.2f} - {model_data['constrained_xg']['away']:.2f}")
        st.write(f"**Dynamic xG**: {model_data['dynamic_xg']['home']:.2f} - {model_data['dynamic_xg']['away']:.2f}")

        # Injury Impact Analysis
        if context.get('home_injuries') or context.get('away_injuries'):
            st.subheader("🏥 Injury Impact Assessment")
            if context.get('home_injuries'):
                st.write(f"**{home_team} Injuries**: {', '.join(context['home_injuries'])}")
            if context.get('away_injuries'):
                st.write(f"**{away_team} Injuries**: {', '.join(context['away_injuries'])}")

    with col2:
        st.subheader("🎯 Tactical & Context Analysis")
        home_style = prediction['team_profiles']['home']['tactical_style']
        away_style = prediction['team_profiles']['away']['tactical_style']

        st.write(f"**{home_team} Style**: {', '.join(home_style)}")
        st.write(f"**{away_team} Style**: {', '.join(away_style)}")

        # Style interaction analysis
        tactical_insights = []
        if 'COUNTER' in home_style and 'HIGH_PRESS' in away_style:
            tactical_insights.append("✅ **Key Matchup**: Home counter-attacks vs away high press favors home team")
        if 'DEFENSIVE' in home_style and 'POSSESSION' in away_style:
            tactical_insights.append("🛡️ **Key Matchup**: Home defensive organization vs away possession")
        if 'HIGH_PRESS' in home_style and 'POSSESSION' in away_style:
            tactical_insights.append("⚡ **Key Matchup**: Home high press could disrupt away possession game")
        
        for insight in tactical_insights:
            st.write(insight)

        # Context factors applied
        st.subheader("🎭 Applied Context Factors")
        context_factors = []
        if context.get('crowd_impact') != 'Normal':
            context_factors.append(f"**Crowd**: {context['crowd_impact']}")
        if context.get('referee') != 'Normal':
            context_factors.append(f"**Referee**: {context['referee']}")
        if context.get('weather') != 'Normal':
            context_factors.append(f"**Weather**: {context['weather']}")
        if context.get('match_importance') != 'Normal':
            context_factors.append(f"**Importance**: {context['match_importance']}")
        
        if context_factors:
            for factor in context_factors:
                st.write(factor)
        else:
            st.write("Standard match conditions applied")

if __name__ == "__main__":
    main()
