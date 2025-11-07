import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# HYBRID PRECISION ENGINE - COMPLETE IMPLEMENTATION
# =============================================================================

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
        base = self._calculate_base_expected_goals(home_profile, away_profile, context)
        constrained = self._apply_football_constraints(base, home_profile, away_profile)
        dynamic = self._simulate_game_flow(constrained, home_profile, away_profile)

        probs = self._xg_to_match_outcome_probs(dynamic['home'], dynamic['away'])
        probs = self._apply_correlation_adjustment(probs, dynamic['home'], dynamic['away'])
        probs = self._apply_style_win_bias(probs, home_profile, away_profile)
        calibrated = self._apply_uncertainty_calibration(probs)

        additional_markets = self._calculate_additional_markets(dynamic['home'], dynamic['away'])

        return {
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
# LEAGUE CONFIGURATIONS
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
# STREAMLIT APP
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
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🎯 Hybrid Precision Prediction Engine</div>', unsafe_allow_html=True)

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

    # Prediction button
    if st.button("🎯 GENERATE PRECISION PREDICTION", type="primary", use_container_width=True):
        with st.spinner("🔄 Running hybrid precision analysis..."):
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
            display_results(prediction, home_team, away_team)

def display_results(prediction, home_team, away_team):
    st.markdown("---")
    st.markdown('<div class="main-header">🎯 PRECISION PREDICTION RESULTS</div>', unsafe_allow_html=True)

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

        st.subheader("🔍 Model Process")
        model_data = prediction['model_data']
        st.write(f"**Base xG**: {model_data['base_xg']['home']:.2f} - {model_data['base_xg']['away']:.2f}")
        st.write(f"**Constrained xG**: {model_data['constrained_xg']['home']:.2f} - {model_data['constrained_xg']['away']:.2f}")
        st.write(f"**Dynamic xG**: {model_data['dynamic_xg']['home']:.2f} - {model_data['dynamic_xg']['away']:.2f}")

    with col2:
        st.subheader("🎯 Key Factors")
        home_style = prediction['team_profiles']['home']['tactical_style']
        away_style = prediction['team_profiles']['away']['tactical_style']

        st.write(f"**{home_team} Style**: {', '.join(home_style)}")
        st.write(f"**{away_team} Style**: {', '.join(away_style)}")

        # Style interaction analysis
        if 'COUNTER' in home_style and 'HIGH_PRESS' in away_style:
            st.success("✅ **Key Matchup**: Home counter-attacks vs away high press favors home team")
        if 'DEFENSIVE' in home_style and 'POSSESSION' in away_style:
            st.info("🛡️ **Key Matchup**: Home defensive organization vs away possession")

        # Best value bet
        best_bet = max([
            (f'{home_team} Win', prediction['match_outcome']['home_win']),
            ('Draw', prediction['match_outcome']['draw']),
            (f'{away_team} Win', prediction['match_outcome']['away_win']),
            ('Over 2.5', prediction['additional_markets']['over_2.5']),
            ('Under 2.5', prediction['additional_markets']['under_2.5']),
            ('BTTS Yes', prediction['additional_markets']['btts_yes']),
            ('BTTS No', prediction['additional_markets']['btts_no'])
        ], key=lambda x: x[1])

        st.success(f"**💎 STRONGEST VALUE**: {best_bet[0]} ({best_bet[1]:.1%} probability)")

if __name__ == "__main__":
    main()
