import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FIXED HYBRID PRECISION ENGINE - WITH QUALITY AWARE CALIBRATION
# =============================================================================

class QualityAwareEngine:
    def __init__(self):
        self.team_tiers = {
            "ELITE": ["Arsenal", "Man City", "Liverpool", "Real Madrid", "Bayern Munich", "Barcelona", "PSG", "Inter", "Juventus", "AC Milan"],
            "TOP": ["Chelsea", "Tottenham", "Man United", "Newcastle", "Aston Villa", "Atletico Madrid", "Napoli", "Dortmund", "Leipzig", "Leverkusen"],
            "MID": ["Brighton", "West Ham", "Crystal Palace", "Wolves", "Fulham", "Real Sociedad", "Villarreal", "Betis", "Monchengladbach", "Frankfurt"],
            "LOWER": ["Sunderland", "Luton", "Burnley", "Sheffield Utd", "Nottingham Forest", "Bournemouth", "Brentford", "Everton", "Cadiz", "Almeria"]
        }
        
        self.constraints = {
            'max_upset_probability': 0.25,  # No minnow can have >25% win probability vs elite
            'min_elite_win_prob': 0.55,     # Elite teams minimum win probability vs lower
            'quality_xg_bounds': {
                'ELITE': {'min': 1.5, 'max': 3.5},
                'TOP': {'min': 1.2, 'max': 3.0},
                'MID': {'min': 0.8, 'max': 2.2},
                'LOWER': {'min': 0.5, 'max': 1.8}
            }
        }

    def get_team_tier(self, team_name):
        for tier, teams in self.team_tiers.items():
            if team_name in teams:
                return tier
        return "MID"  # Default

    def apply_quality_constraints(self, base_xg, home_team, away_team, base_probs):
        home_tier = self.get_team_tier(home_team)
        away_tier = self.get_team_tier(away_team)
        
        constrained_xg = base_xg.copy()
        constrained_probs = base_probs.copy()
        
        # Apply xG bounds based on team quality
        home_xg_bounds = self.constraints['quality_xg_bounds'][home_tier]
        away_xg_bounds = self.constraints['quality_xg_bounds'][away_tier]
        
        constrained_xg['home'] = max(home_xg_bounds['min'], min(home_xg_bounds['max'], constrained_xg['home']))
        constrained_xg['away'] = max(away_xg_bounds['min'], min(away_xg_bounds['max'], constrained_xg['away']))
        
        # Apply probability constraints for mismatches
        if home_tier == "LOWER" and away_tier == "ELITE":
            constrained_probs['home_win'] = min(constrained_probs['home_win'], self.constraints['max_upset_probability'])
            constrained_probs['away_win'] = max(constrained_probs['away_win'], self.constraints['min_elite_win_prob'])
        elif home_tier == "ELITE" and away_tier == "LOWER":
            constrained_probs['home_win'] = max(constrained_probs['home_win'], self.constraints['min_elite_win_prob'])
            constrained_probs['away_win'] = min(constrained_probs['away_win'], self.constraints['max_upset_probability'])
        
        # Re-normalize probabilities
        total = sum(constrained_probs.values())
        for key in constrained_probs:
            constrained_probs[key] /= total
            
        return constrained_xg, constrained_probs

class InjuryImpactAssessor:
    def __init__(self):
        self.player_impact_db = {
            # Reduced impacts for realism
            'MARTIN ØDEGAARD': {'xg_contribution': 0.90},
            'BEN WHITE': {'xg_allowed_impact': 1.08},
            'JORDAN PICKFORD': {'xg_allowed_impact': 1.15},
            'KEVIN DE BRUYNE': {'xg_contribution': 0.85},
            'ERLING HAALAND': {'xg_contribution': 0.80},
            'VIRGIL VAN DIJK': {'xg_allowed_impact': 1.15},
        }
    
    def calculate_team_impact(self, injury_list, team_profile, is_elite=False):
        if not injury_list:
            return {'xg_boost': 1.0, 'xga_boost': 1.0}
        
        total_impact = {'xg_boost': 1.0, 'xga_boost': 1.0}
        
        for injury in injury_list:
            player_impact = self.player_impact_db.get(injury.upper())
            if player_impact:
                if 'xg_contribution' in player_impact:
                    total_impact['xg_boost'] *= player_impact['xg_contribution']
                if 'xg_allowed_impact' in player_impact:
                    total_impact['xga_boost'] *= player_impact['xg_allowed_impact']
        
        # Elite teams suffer less from injuries
        if is_elite:
            for key in total_impact:
                if total_impact[key] != 1.0:
                    total_impact[key] = 1 - (1 - total_impact[key]) * 0.6  # 40% reduction
        
        return total_impact

class HybridPrecisionEngine:
    def __init__(self):
        self.quality_engine = QualityAwareEngine()
        self.injury_assessor = InjuryImpactAssessor()
        self.constraints = {
            'max_total_xg': 5.0,
            'possession_zero_sum': True,
        }

    def _calculate_base_expected_goals(self, home_profile, away_profile):
        home_xg = home_profile.get('xg_per_game', 1.0)
        away_xg = away_profile.get('xg_per_game', 1.0)

        home_adv_multiplier = 1.15
        base_home = home_xg * home_adv_multiplier
        base_away = away_xg

        return {'home': base_home, 'away': base_away}

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

    def predict_match(self, home_profile, away_profile, context=None):
        home_team = home_profile['name']
        away_team = away_profile['name']
        
        # Apply injury impacts
        home_elite = self.quality_engine.get_team_tier(home_team) in ["ELITE", "TOP"]
        away_elite = self.quality_engine.get_team_tier(away_team) in ["ELITE", "TOP"]
        
        if context:
            home_impact = self.injury_assessor.calculate_team_impact(
                context.get('home_injuries', []), home_profile, home_elite
            )
            away_impact = self.injury_assessor.calculate_team_impact(
                context.get('away_injuries', []), away_profile, away_elite
            )
            
            home_profile = home_profile.copy()
            away_profile = away_profile.copy()
            
            home_profile['xg_per_game'] *= home_impact['xg_boost']
            home_profile['xga_per_game'] *= home_impact['xga_boost']
            away_profile['xg_per_game'] *= away_impact['xg_boost']
            away_profile['xga_per_game'] *= away_impact['xga_boost']

        # Calculate base predictions
        base_xg = self._calculate_base_expected_goals(home_profile, away_profile)
        base_probs = self._xg_to_match_outcome_probs(base_xg['home'], base_xg['away'])
        
        # Apply quality constraints (THIS IS THE KEY FIX)
        constrained_xg, constrained_probs = self.quality_engine.apply_quality_constraints(
            base_xg, home_team, away_team, base_probs
        )
        
        additional_markets = self._calculate_additional_markets(constrained_xg['home'], constrained_xg['away'])

        return {
            'match_outcome': constrained_probs,
            'additional_markets': additional_markets,
            'expected_goals': constrained_xg,
            'team_profiles': {'home': home_profile, 'away': away_profile},
            'model_data': {
                'base_xg': base_xg,
                'constrained_xg': constrained_xg
            }
        }

# =============================================================================
# STREAMLIT APP
# =============================================================================

LEAGUE_CONFIGS = {
    "EPL": {
        "teams": ["Arsenal", "Man City", "Liverpool", "Chelsea", "Tottenham", "Man United",
                 "Newcastle", "Brighton", "West Ham", "Aston Villa", "Crystal Palace", "Wolves",
                 "Fulham", "Bournemouth", "Brentford", "Everton", "Nott'ham Forest",
                 "Luton", "Sunderland", "Burnley", "Sheffield Utd"]
    }
}

def main():
    st.set_page_config(page_title="Realistic Prediction Engine", layout="wide")
    
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; background: linear-gradient(45deg, #1f77b4, #ff7f0e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .realistic-prediction { background: linear-gradient(135deg, #00b09b, #96c93d); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">⚽ REALISTIC PREDICTION ENGINE</div>', unsafe_allow_html=True)

    # Team selection
    league = st.selectbox("🏆 SELECT LEAGUE", list(LEAGUE_CONFIGS.keys()))
    teams = LEAGUE_CONFIGS[league]["teams"]

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 HOME TEAM", teams, index=teams.index("Sunderland"))
    with col2:
        away_team = st.selectbox("✈️ AWAY TEAM", teams, index=teams.index("Arsenal"))

    # Quick stats (simplified)
    col1, col2 = st.columns(2)
    with col1:
        h_xg = st.number_input(f"{home_team} xG per game", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        h_style = st.selectbox(f"{home_team} Style", ["DEFENSIVE", "BALANCED", "ATTACKING"], key="h_style")
    with col2:
        a_xg = st.number_input(f"{away_team} xG per game", min_value=0.5, max_value=3.0, value=1.8, step=0.1)
        a_style = st.selectbox(f"{away_team} Style", ["DEFENSIVE", "BALANCED", "ATTACKING"], key="a_style")

    # Context
    st.subheader("🎭 Match Context")
    col1, col2 = st.columns(2)
    with col1:
        home_injuries = st.text_input("Home Key Injuries", value="Jordan Pickford")
        crowd_impact = st.selectbox("Crowd Impact", ["Normal", "Electric Home Crowd"])
    with col2:
        away_injuries = st.text_input("Away Key Injuries", value="Martin Ødegaard, Ben White")
        referee = st.selectbox("Referee Style", ["Normal", "Lenient", "Strict"])

    if st.button("🎯 GENERATE REALISTIC PREDICTION", type="primary", use_container_width=True):
        engine = HybridPrecisionEngine()
        
        home_profile = {
            'name': home_team,
            'xg_per_game': h_xg,
            'xga_per_game': 1.2,
            'tactical_style': [h_style]
        }
        
        away_profile = {
            'name': away_team, 
            'xg_per_game': a_xg,
            'xga_per_game': 0.9,
            'tactical_style': [a_style]
        }
        
        context = {
            'home_injuries': [inj.strip() for inj in home_injuries.split(',')] if home_injuries else [],
            'away_injuries': [inj.strip() for inj in away_injuries.split(',')] if away_injuries else [],
            'crowd_impact': crowd_impact,
            'referee': referee
        }
        
        prediction = engine.predict_match(home_profile, away_profile, context)
        display_realistic_results(prediction, home_team, away_team)

def display_realistic_results(prediction, home_team, away_team):
    st.markdown("---")
    st.markdown('<div class="main-header">🎯 REALISTIC PREDICTION RESULTS</div>', unsafe_allow_html=True)
    
    outcome = prediction['match_outcome']
    markets = prediction['additional_markets']
    xg = prediction['expected_goals']
    
    # Show realistic assessment
    st.markdown(f'<div class="realistic-prediction">'
                f'<h3>📊 REALISTIC ASSESSMENT</h3>'
                f'<p><b>{home_team}: {outcome["home_win"]:.1%}</b> | '
                f'<b>Draw: {outcome["draw"]:.1%}</b> | '
                f'<b>{away_team}: {outcome["away_win"]:.1%}</b></p>'
                f'<p>Expected Goals: {xg["home"]:.1f} - {xg["away"]:.1f}</p>'
                f'</div>', unsafe_allow_html=True)
    
    # Quality tier info
    engine = HybridPrecisionEngine()
    home_tier = engine.quality_engine.get_team_tier(home_team)
    away_tier = engine.quality_engine.get_team_tier(away_team)
    
    st.info(f"**Quality Assessment**: {home_team} ({home_tier}) vs {away_team} ({away_tier})")
    
    # Key factors
    st.subheader("🎯 Key Factors Applied")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Quality Constraints**:")
        st.write(f"- Max upset probability: 25%")
        st.write(f"- Min elite win probability: 55%")
        st.write(f"- Realistic xG bounds applied")
    
    with col2:
        st.write("**Context Adjustments**:")
        st.write(f"- Injuries with elite team discount")
        st.write(f"- Home advantage: +15% xG")
        st.write(f"- Realistic probability caps")

if __name__ == "__main__":
    main()
