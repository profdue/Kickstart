import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine")
st.markdown("""
    **Intelligent Prediction with Goal Source Analysis & Winner Identification**
    *Goals happen when capable attacks overcome relevant defenses.*
    *Winners are determined by net dominance in attack-defense balance.*
""")

# ========== ENHANCED CONSTANTS ==========
MAX_GOALS = 8
MAX_REGRESSION = 0.3
MIN_PROBABILITY = 0.1
MAX_PROBABILITY = 0.9

# Enhanced thresholds
DEFENSE_ELITE = -1.0
DEFENSE_STRONG = -0.5
DEFENSE_GOOD = -0.3
DEFENSE_AVERAGE = 0.0
DEFENSE_WEAK = 0.5
DEFENSE_VERY_WEAK = 1.0

ATTACK_ELITE_PLUS = 1.5
ATTACK_ELITE = 1.0
ATTACK_ABOVE_AVG = 0.5
ATTACK_AVERAGE = 0.0

DOMINANCE_THRESHOLD = 0.65
HIGH_XG_THRESHOLD = 2.4

# Winner prediction thresholds
DOMINANCE_MARGIN_HIGH = 0.8
DOMINANCE_MARGIN_MEDIUM = 0.5
DOMINANCE_MARGIN_LOW = 0.2
DRAW_PROBABILITY_THRESHOLD = 0.35

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

def factorial_cache(n):
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

@st.cache_data(ttl=3600)
def load_league_data(league_name):
    try:
        file_map = {
            "Premier League": "premier_league.csv",
            "Bundesliga": "bundesliga.csv",
            "Serie A": "serie_a.csv",
            "La Liga": "laliga.csv",
            "Ligue 1": "ligue_1.csv",
            "Eredivisie": "eredivisie.csv"
        }
        
        filename = file_map.get(league_name, f"{league_name.lower().replace(' ', '_')}.csv")
        file_path = f"leagues/{filename}"
        
        df = pd.read_csv(file_path)
        required = ['team', 'venue', 'matches', 'xg', 'xga', 'goals_vs_xg']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

def prepare_team_data(df):
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    return home_data.set_index('team'), away_data.set_index('team')

def calculate_league_baselines(df):
    home_xg = df[df['venue'] == 'home']['xg'] / df[df['venue'] == 'home']['matches']
    away_xg = df[df['venue'] == 'away']['xg'] / df[df['venue'] == 'away']['matches']
    home_xga = df[df['venue'] == 'home']['xga'] / df[df['venue'] == 'home']['matches']
    away_xga = df[df['venue'] == 'away']['xga'] / df[df['venue'] == 'away']['matches']
    
    all_xg = pd.concat([home_xg, away_xg])
    all_xga = pd.concat([home_xga, away_xga])
    
    return {
        'avg_xg': all_xg.mean(),
        'std_xg': max(all_xg.std(), 0.1),
        'avg_xga': all_xga.mean(),
        'std_xga': max(all_xga.std(), 0.1)
    }

class TeamIntelligence:
    @staticmethod
    def analyze_team(team_stats, league_baselines, venue):
        matches = team_stats['matches']
        
        xg_per_match = team_stats['xg'] / max(matches, 1)
        xga_per_match = team_stats['xga'] / max(matches, 1)
        goals_vs_xg_per_match = team_stats['goals_vs_xg'] / max(matches, 1)
        
        attack_score = (xg_per_match - league_baselines['avg_xg']) / league_baselines['std_xg']
        defense_score = (xga_per_match - league_baselines['avg_xga']) / league_baselines['std_xga']
        
        regression_factor = min(max(goals_vs_xg_per_match, -MAX_REGRESSION), MAX_REGRESSION)
        
        defense_tier = TeamIntelligence._classify_defense(defense_score)
        attack_tier = TeamIntelligence._classify_attack(attack_score)
        
        # Add dominance metrics
        net_dominance = attack_score - defense_score
        
        return {
            'venue': venue,
            'attack_score': attack_score,
            'defense_score': defense_score,
            'attack_tier': attack_tier,
            'defense_tier': defense_tier,
            'regression_factor': regression_factor,
            'net_dominance': net_dominance,
            'xg_per_match': xg_per_match,
            'xga_per_match': xga_per_match,
            'goals_vs_xg_per_match': goals_vs_xg_per_match
        }
    
    @staticmethod
    def _classify_defense(score):
        if score < DEFENSE_ELITE: return "ELITE"
        elif score < DEFENSE_STRONG: return "STRONG"
        elif score < DEFENSE_GOOD: return "GOOD"
        elif score < DEFENSE_AVERAGE: return "AVERAGE"
        elif score < DEFENSE_WEAK: return "WEAK"
        else: return "VERY_WEAK"
    
    @staticmethod
    def _classify_attack(score):
        if score > ATTACK_ELITE_PLUS: return "ELITE_PLUS"
        elif score > ATTACK_ELITE: return "ELITE"
        elif score > ATTACK_ABOVE_AVG: return "ABOVE_AVG"
        elif score > ATTACK_AVERAGE: return "AVERAGE"
        else: return "BELOW_AVG"

class MatchIntelligence:
    @staticmethod
    def analyze_match(home_analysis, away_analysis, home_xg, away_xg):
        total_xg = home_xg + away_xg
        
        # Goal source analysis
        if total_xg > 0.1:
            home_scoring_share = home_xg / total_xg
            away_scoring_share = away_xg / total_xg
            dominant_share = max(home_scoring_share, away_scoring_share)
            dominant_team = "HOME" if home_scoring_share > away_scoring_share else "AWAY"
        else:
            dominant_share = 0.5
            dominant_team = None
        
        is_dominance_match = (dominant_share > DOMINANCE_THRESHOLD and 
                             total_xg > HIGH_XG_THRESHOLD)
        
        # Winner analysis - NET DOMINANCE LOGIC
        home_net_dominance = home_analysis['attack_score'] - away_analysis['defense_score']
        away_net_dominance = away_analysis['attack_score'] - home_analysis['defense_score']
        net_dominance_difference = home_net_dominance - away_net_dominance
        
        # Match type classification
        home_def = home_analysis['defense_tier']
        away_def = away_analysis['defense_tier']
        home_att = home_analysis['attack_tier']
        away_att = away_analysis['attack_tier']
        
        # 1. DOMINANCE MATCH
        if is_dominance_match:
            return {
                'match_type': "DOMINANCE",
                'explanation': f"One team supplies {dominant_share:.0%} of expected goals",
                'is_dominance_match': True,
                'dominant_share': dominant_share,
                'dominant_team': dominant_team,
                'home_def_tier': home_def,
                'away_def_tier': away_def,
                'home_att_tier': home_att,
                'away_att_tier': away_att,
                'total_xg': total_xg,
                'home_net_dominance': home_net_dominance,
                'away_net_dominance': away_net_dominance,
                'net_dominance_difference': net_dominance_difference
            }
        
        # 2. ATTACK DOMINANCE
        elif ((home_att in ["ELITE", "ELITE_PLUS"] and away_def in ["WEAK", "VERY_WEAK"]) or
              (away_att in ["ELITE", "ELITE_PLUS"] and home_def in ["WEAK", "VERY_WEAK"])):
            return {
                'match_type': "ATTACK_DOMINANCE",
                'explanation': "Elite attack exploits weak defense",
                'is_dominance_match': False,
                'dominant_share': dominant_share,
                'dominant_team': dominant_team,
                'home_def_tier': home_def,
                'away_def_tier': away_def,
                'home_att_tier': home_att,
                'away_att_tier': away_att,
                'total_xg': total_xg,
                'home_net_dominance': home_net_dominance,
                'away_net_dominance': away_net_dominance,
                'net_dominance_difference': net_dominance_difference
            }
        
        # 3. GENUINE DEFENSIVE TACTICAL
        elif ((home_def in ["ELITE", "STRONG"] and away_att in ["ABOVE_AVG", "ELITE", "ELITE_PLUS"]) or
              (away_def in ["ELITE", "STRONG"] and home_att in ["ABOVE_AVG", "ELITE", "ELITE_PLUS"])):
            return {
                'match_type': "DEFENSIVE_TACTICAL",
                'explanation': "Strong defense faces capable attack",
                'is_dominance_match': False,
                'dominant_share': dominant_share,
                'dominant_team': dominant_team,
                'home_def_tier': home_def,
                'away_def_tier': away_def,
                'home_att_tier': home_att,
                'away_att_tier': away_att,
                'total_xg': total_xg,
                'home_net_dominance': home_net_dominance,
                'away_net_dominance': away_net_dominance,
                'net_dominance_difference': net_dominance_difference
            }
        
        # 4. FALSE DEFENSIVE
        elif ((home_def in ["ELITE", "STRONG", "GOOD"] and away_att in ["BELOW_AVG", "AVERAGE"]) or
              (away_def in ["ELITE", "STRONG", "GOOD"] and home_att in ["BELOW_AVG", "AVERAGE"])):
            return {
                'match_type': "FALSE_DEFENSIVE",
                'explanation': "Strong defense irrelevant against weak attack",
                'is_dominance_match': False,
                'dominant_share': dominant_share,
                'dominant_team': dominant_team,
                'home_def_tier': home_def,
                'away_def_tier': away_def,
                'home_att_tier': home_att,
                'away_att_tier': away_att,
                'total_xg': total_xg,
                'home_net_dominance': home_net_dominance,
                'away_net_dominance': away_net_dominance,
                'net_dominance_difference': net_dominance_difference
            }
        
        # 5. STANDARD
        else:
            return {
                'match_type': "STANDARD",
                'explanation': "Balanced matchup",
                'is_dominance_match': False,
                'dominant_share': dominant_share,
                'dominant_team': dominant_team,
                'home_def_tier': home_def,
                'away_def_tier': away_def,
                'home_att_tier': home_att,
                'away_att_tier': away_att,
                'total_xg': total_xg,
                'home_net_dominance': home_net_dominance,
                'away_net_dominance': away_net_dominance,
                'net_dominance_difference': net_dominance_difference
            }

class WinnerPrediction:
    @staticmethod
    def predict_winner(home_analysis, away_analysis, match_analysis, home_xg, away_xg):
        """
        Predict winner using net dominance logic
        """
        home_net = match_analysis['home_net_dominance']
        away_net = match_analysis['away_net_dominance']
        net_diff = match_analysis['net_dominance_difference']
        
        # Base probabilities from net dominance
        total_dominance = abs(home_net) + abs(away_net) + 0.01
        
        if net_diff > 0:
            home_win_prob = 0.5 + (min(abs(net_diff), 2.0) / 4.0)
            away_win_prob = 0.5 - (min(abs(net_diff), 2.0) / 4.0)
        else:
            home_win_prob = 0.5 - (min(abs(net_diff), 2.0) / 4.0)
            away_win_prob = 0.5 + (min(abs(net_diff), 2.0) / 4.0)
        
        # Apply match type adjustments
        match_type = match_analysis['match_type']
        
        if match_type == "DOMINANCE":
            if match_analysis['dominant_team'] == "HOME":
                home_win_prob += 0.2
                away_win_prob -= 0.2
            else:
                home_win_prob -= 0.2
                away_win_prob += 0.2
        
        elif match_type == "ATTACK_DOMINANCE":
            if home_analysis['attack_tier'] in ["ELITE", "ELITE_PLUS"] and away_analysis['defense_tier'] in ["WEAK", "VERY_WEAK"]:
                home_win_prob += 0.15
            elif away_analysis['attack_tier'] in ["ELITE", "ELITE_PLUS"] and home_analysis['defense_tier'] in ["WEAK", "VERY_WEAK"]:
                away_win_prob += 0.15
        
        elif match_type == "DEFENSIVE_TACTICAL":
            # Lower scoring, higher chance of draw
            draw_bonus = 0.1
            home_win_prob -= draw_bonus/2
            away_win_prob -= draw_bonus/2
        
        # Home advantage adjustment (typically 0.1-0.15 boost)
        home_advantage = 0.12
        home_win_prob += home_advantage
        away_win_prob -= home_advantage/2
        
        # Normalize probabilities
        total = home_win_prob + away_win_prob
        if total > 1.0:
            home_win_prob /= total
            away_win_prob /= total
        
        # Draw probability (inverse of dominance difference)
        draw_prob = max(0.1, min(0.35, 0.3 - abs(net_diff) * 0.1))
        
        # Final normalization
        remaining = 1.0 - draw_prob
        home_win_prob *= remaining
        away_win_prob *= remaining
        
        # Determine predicted winner
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            predicted_winner = "HOME"
            winner_confidence = WinnerPrediction._calculate_confidence(home_win_prob)
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            predicted_winner = "AWAY"
            winner_confidence = WinnerPrediction._calculate_confidence(away_win_prob)
        else:
            predicted_winner = "DRAW"
            winner_confidence = WinnerPrediction._calculate_confidence(draw_prob)
        
        # Most likely score
        most_likely_score = WinnerPrediction._predict_most_likely_score(home_xg, away_xg)
        
        return {
            'home_win_probability': home_win_prob,
            'away_win_probability': away_win_prob,
            'draw_probability': draw_prob,
            'predicted_winner': predicted_winner,
            'winner_confidence': winner_confidence,
            'most_likely_score': most_likely_score,
            'net_dominance_home': home_net,
            'net_dominance_away': away_net,
            'dominance_difference': net_diff
        }
    
    @staticmethod
    def _calculate_confidence(probability):
        if probability > 0.55:
            return "HIGH"
        elif probability > 0.45:
            return "MEDIUM"
        else:
            return "LOW"
    
    @staticmethod
    def _predict_most_likely_score(home_xg, away_xg):
        # Simple score prediction based on expected goals
        home_goals = round(home_xg)
        away_goals = round(away_xg)
        
        # Adjust for low-scoring matches
        if home_xg < 0.8 and away_xg < 0.8:
            return "0-0" if np.random.random() > 0.5 else "1-0"
        elif home_xg < 1.0:
            home_goals = max(0, home_goals)
        elif away_xg < 1.0:
            away_goals = max(0, away_goals)
        
        return f"{home_goals}-{away_goals}"

class PredictionEngine:
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
    
    def predict_expected_goals(self, home_analysis, away_analysis):
        home_attack = home_analysis['xg_per_match']
        home_defense = home_analysis['xga_per_match']
        away_attack = away_analysis['xg_per_match']
        away_defense = away_analysis['xga_per_match']
        
        home_expected = (home_attack * away_defense) / max(self.league_baselines['avg_xg'], 0.1)
        away_expected = (away_attack * home_defense) / max(self.league_baselines['avg_xg'], 0.1)
        
        home_final = home_expected * (1 + home_analysis['regression_factor'])
        away_final = away_expected * (1 + away_analysis['regression_factor'])
        
        home_final = max(min(home_final, 4.0), 0.3)
        away_final = max(min(away_final, 4.0), 0.3)
        
        return home_final, away_final
    
    def calculate_base_probability(self, total_xg):
        prob_under = (poisson_pmf(0, total_xg) + 
                     poisson_pmf(1, total_xg) + 
                     poisson_pmf(2, total_xg))
        return 1 - prob_under

class CorrectionIntelligence:
    @staticmethod
    def calculate_correction(match_analysis, base_prob):
        match_type = match_analysis['match_type']
        correction = 0.0
        rationale = []
        
        corrections = {
            "DOMINANCE": 0.0,
            "ATTACK_DOMINANCE": 0.15,
            "DEFENSIVE_TACTICAL": -0.20,
            "FALSE_DEFENSIVE": 0.0,
            "STANDARD": 0.0
        }
        
        if match_type in corrections:
            correction = corrections[match_type]
        
        # Elite attack override in dominance matches
        if match_type == "DOMINANCE" and match_analysis['dominant_team']:
            if match_analysis['dominant_team'] == "HOME":
                att_tier = match_analysis['home_att_tier']
            else:
                att_tier = match_analysis['away_att_tier']
            
            if att_tier in ["ELITE", "ELITE_PLUS"]:
                rationale.append(f"Dominant team has {att_tier} attack")
                correction = 0.0
        
        # Override protection
        if base_prob > 0.70 and correction < 0:
            correction = max(correction, -0.15)
        elif base_prob < 0.30 and correction > 0:
            correction = min(correction, 0.15)
        
        return correction, rationale, match_type

class FootballIntelligenceEngine:
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
        self.team_intel = TeamIntelligence()
        self.match_intel = MatchIntelligence()
        self.prediction_engine = PredictionEngine(league_baselines)
        self.correction_intel = CorrectionIntelligence()
        self.winner_predictor = WinnerPrediction()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        # Team analysis
        home_analysis = self.team_intel.analyze_team(home_stats, self.league_baselines, "home")
        away_analysis = self.team_intel.analyze_team(away_stats, self.league_baselines, "away")
        
        # Expected goals
        home_xg, away_xg = self.prediction_engine.predict_expected_goals(home_analysis, away_analysis)
        total_xg = home_xg + away_xg
        
        # Base probability
        base_prob = self.prediction_engine.calculate_base_probability(total_xg)
        
        # Match intelligence
        match_analysis = self.match_intel.analyze_match(home_analysis, away_analysis, home_xg, away_xg)
        
        # Correction
        correction, correction_rationale, correction_type = self.correction_intel.calculate_correction(
            match_analysis, base_prob
        )
        
        # Final probability
        final_prob = base_prob + correction
        final_prob = max(min(final_prob, MAX_PROBABILITY), MIN_PROBABILITY)
        
        # Winner prediction
        winner_prediction = self.winner_predictor.predict_winner(
            home_analysis, away_analysis, match_analysis, home_xg, away_xg
        )
        
        # Confidence
        distance = abs(final_prob - 0.5)
        if distance > 0.25:
            confidence = "VERY HIGH"
        elif distance > 0.15:
            confidence = "HIGH"
        elif distance > 0.08:
            confidence = "MEDIUM"
        elif distance > 0.05:
            confidence = "LOW"
        else:
            confidence = "VERY LOW"
        
        # Direction
        direction = "OVER" if final_prob > 0.5 else "UNDER"
        
        # Rationale
        rationale = f"**{match_analysis['match_type']}**: {match_analysis['explanation']}"
        if correction_rationale:
            for reason in correction_rationale:
                rationale += f" **{reason}**"
        if abs(correction) > 0.01:
            rationale += f" Base model adjusted by {correction*100:+.1f}%."
        rationale += f" **Confidence**: {confidence}"
        
        # Add winner rationale
        winner_rationale = self._generate_winner_rationale(match_analysis, winner_prediction)
        
        return {
            'final_probability': final_prob,
            'direction': direction,
            'match_type': match_analysis['match_type'],
            'confidence': confidence,
            'correction': correction,
            'correction_type': correction_type,
            'rationale': rationale,
            'base_probability': base_prob,
            'expected_goals': {
                'home': home_xg,
                'away': away_xg,
                'total': total_xg
            },
            'winner_prediction': winner_prediction,
            'winner_rationale': winner_rationale,
            'home_analysis': home_analysis,
            'away_analysis': away_analysis,
            'match_analysis': match_analysis
        }
    
    def _generate_winner_rationale(self, match_analysis, winner_prediction):
        net_diff = match_analysis['net_dominance_difference']
        home_net = match_analysis['home_net_dominance']
        away_net = match_analysis['away_net_dominance']
        
        if winner_prediction['predicted_winner'] == "HOME":
            return f"Home team has net dominance advantage of {net_diff:.2f}σ (Home: {home_net:.2f}σ vs Away: {away_net:.2f}σ)"
        elif winner_prediction['predicted_winner'] == "AWAY":
            return f"Away team has net dominance advantage of {abs(net_diff):.2f}σ (Away: {away_net:.2f}σ vs Home: {home_net:.2f}σ)"
        else:
            return f"Balanced net dominance (Difference: {abs(net_diff):.2f}σ) suggests draw likely"

# ========== STREAMLIT UI ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie"]
    selected_league = st.selectbox("Select League", leagues)
    
    df = load_league_data(selected_league)
    
    if df is not None:
        league_baselines = calculate_league_baselines(df)
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        home_teams = sorted(home_stats_df.index.unique())
        away_teams = sorted(away_stats_df.index.unique())
        common_teams = sorted(list(set(home_teams) & set(away_teams)))
        
        if len(common_teams) == 0:
            st.error("No teams with complete data")
            st.stop()
        
        home_team = st.selectbox("Home Team", common_teams)
        away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
        
        st.divider()
        
        if st.button("🎯 Generate Prediction", type="primary", use_container_width=True):
            calculate_btn = True
        else:
            calculate_btn = False

if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Generate Prediction'")
    st.stop()

try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"Team data error: {e}")
    st.stop()

# Generate prediction
st.header(f"🎯 {home_team} vs {away_team}")

engine = FootballIntelligenceEngine(league_baselines)
prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)

# ========== DISPLAY RESULTS ==========

# Main prediction card
final_prob = prediction['final_probability']
direction = prediction['direction']
confidence = prediction['confidence']
match_type = prediction['match_type']

if direction == "OVER":
    card_color = "#14532D" if confidence == "VERY HIGH" else "#166534" if confidence == "HIGH" else "#365314"
    text_color = "#22C55E" if confidence == "VERY HIGH" else "#4ADE80" if confidence == "HIGH" else "#84CC16"
else:
    card_color = "#7F1D1D" if confidence == "VERY HIGH" else "#991B1B" if confidence == "HIGH" else "#78350F"
    text_color = "#EF4444" if confidence == "VERY HIGH" else "#F87171" if confidence == "HIGH" else "#F59E0B"

st.markdown(f"""
<div style="background-color: {card_color}; padding: 25px; border-radius: 15px; text-align: center; margin: 20px 0;">
    <h1 style="color: {text_color}; margin: 0;">{direction} 2.5</h1>
    <div style="font-size: 48px; font-weight: bold; color: white; margin: 10px 0;">
        {final_prob*100:.1f}%
    </div>
    <div style="font-size: 18px; color: white;">
        Confidence: {confidence} | Match Type: {match_type}
    </div>
</div>
""", unsafe_allow_html=True)

# Winner prediction section
winner_pred = prediction['winner_prediction']
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        f"🏠 {home_team} Win",
        f"{winner_pred['home_win_probability']*100:.1f}%",
        delta="Favorite" if winner_pred['predicted_winner'] == "HOME" else None,
        delta_color="normal"
    )

with col2:
    st.metric(
        "🤝 Draw",
        f"{winner_pred['draw_probability']*100:.1f}%",
        delta="Most Likely" if winner_pred['predicted_winner'] == "DRAW" else None,
        delta_color="off"
    )

with col3:
    st.metric(
        f"✈️ {away_team} Win",
        f"{winner_pred['away_win_probability']*100:.1f}%",
        delta="Favorite" if winner_pred['predicted_winner'] == "AWAY" else None,
        delta_color="normal"
    )

# Predicted winner
predicted_winner = winner_pred['predicted_winner']
if predicted_winner == "HOME":
    winner_display = f"🏠 {home_team}"
    winner_color = "#22C55E"
elif predicted_winner == "AWAY":
    winner_display = f"✈️ {away_team}"
    winner_color = "#22C55E"
else:
    winner_display = "🤝 DRAW"
    winner_color = "#F59E0B"

st.markdown(f"""
<div style="background-color: #1E293B; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;">
    <h3 style="color: white; margin: 0;">Predicted Winner</h3>
    <div style="font-size: 32px; font-weight: bold; color: {winner_color}; margin: 5px 0;">
        {winner_display}
    </div>
    <div style="font-size: 16px; color: #94A3B8;">
        Confidence: {winner_pred['winner_confidence']} | Most Likely Score: {winner_pred['most_likely_score']}
    </div>
</div>
""", unsafe_allow_html=True)

# Rationale
st.info(f"**Rationale**: {prediction['rationale']}")
st.success(f"**Winner Analysis**: {prediction['winner_rationale']}")

# Detailed analysis
with st.expander("🔍 Detailed Analysis"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team}")
        home_analysis = prediction['home_analysis']
        st.write(f"**Attack**: {home_analysis['attack_tier']} ({home_analysis['attack_score']:.2f}σ)")
        st.write(f"**Defense**: {home_analysis['defense_tier']} ({home_analysis['defense_score']:.2f}σ)")
        st.write(f"**Net Dominance**: {home_analysis['net_dominance']:.2f}σ")
        st.write(f"**xG/match**: {home_analysis['xg_per_match']:.2f}")
        st.write(f"**xGA/match**: {home_analysis['xga_per_match']:.2f}")
    
    with col2:
        st.subheader(f"✈️ {away_team}")
        away_analysis = prediction['away_analysis']
        st.write(f"**Attack**: {away_analysis['attack_tier']} ({away_analysis['attack_score']:.2f}σ)")
        st.write(f"**Defense**: {away_analysis['defense_tier']} ({away_analysis['defense_score']:.2f}σ)")
        st.write(f"**Net Dominance**: {away_analysis['net_dominance']:.2f}σ")
        st.write(f"**xG/match**: {away_analysis['xg_per_match']:.2f}")
        st.write(f"**xGA/match**: {away_analysis['xga_per_match']:.2f}")
    
    st.subheader("📊 Model Details")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Base Probability", f"{prediction['base_probability']*100:.1f}%")
    
    with col4:
        correction = prediction['correction']
        if correction > 0:
            st.success(f"Correction: +{correction*100:.1f}%")
        elif correction < 0:
            st.info(f"Correction: {correction*100:.1f}%")
        else:
            st.metric("Correction", "0.0%")
    
    with col5:
        st.metric("Final", f"{final_prob*100:.1f}%", delta=direction)
    
    st.subheader("🎯 Expected Goals")
    col6, col7, col8 = st.columns(3)
    
    with col6:
        st.metric(f"{home_team} xG", f"{prediction['expected_goals']['home']:.2f}")
    
    with col7:
        st.metric(f"{away_team} xG", f"{prediction['expected_goals']['away']:.2f}")
    
    with col8:
        total_xg = prediction['expected_goals']['total']
        st.metric("Total xG", f"{total_xg:.2f}")
        
        if match_type == "DOMINANCE":
            st.info(f"Dominance match: {prediction['match_analysis']['dominant_team']} supplies {prediction['match_analysis']['dominant_share']:.0%} of expected goals")
        elif match_type == "DEFENSIVE_TACTICAL":
            st.warning("Genuine defensive context - low scoring likely")
        elif match_type == "FALSE_DEFENSIVE":
            st.info("Strong defense irrelevant against weak attack")
    
    st.subheader("📈 Net Dominance Analysis")
    col9, col10, col11 = st.columns(3)
    
    with col9:
        st.metric("Home Net Dominance", f"{prediction['match_analysis']['home_net_dominance']:.2f}σ")
    
    with col10:
        st.metric("Away Net Dominance", f"{prediction['match_analysis']['away_net_dominance']:.2f}σ")
    
    with col11:
        diff = prediction['match_analysis']['net_dominance_difference']
        st.metric("Dominance Difference", f"{diff:.2f}σ", 
                  delta="Home Advantage" if diff > 0 else "Away Advantage" if diff < 0 else "Balanced",
                  delta_color="normal" if diff > 0.2 else "off" if abs(diff) < 0.2 else "inverse")

# Export
st.divider()
st.subheader("📤 Export Prediction")

report = f"""
⚽ FOOTBALL INTELLIGENCE PREDICTION
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 GOALS PREDICTION
{direction} 2.5 Goals: {final_prob*100:.1f}%
Confidence: {confidence}
Match Type: {match_type}

🏆 WINNER PREDICTION
Predicted Winner: {winner_display}
Most Likely Score: {winner_pred['most_likely_score']}
Winner Confidence: {winner_pred['winner_confidence']}
Home Win: {winner_pred['home_win_probability']*100:.1f}%
Draw: {winner_pred['draw_probability']*100:.1f}%
Away Win: {winner_pred['away_win_probability']*100:.1f}%

🧠 ANALYSIS
{prediction['rationale']}
{prediction['winner_rationale']}

📊 EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

📈 DOMINANCE METRICS
Home Net Dominance: {prediction['match_analysis']['home_net_dominance']:.2f}σ
Away Net Dominance: {prediction['match_analysis']['away_net_dominance']:.2f}σ
Net Difference: {prediction['match_analysis']['net_dominance_difference']:.2f}σ

🔄 MODEL DETAILS
Base Probability: {prediction['base_probability']*100:.1f}%
Correction Applied: {prediction['correction']*100:+.1f}%

---
Generated by Football Intelligence Engine
Winners are determined by net dominance in attack-defense balance.
"""

st.code(report, language="text")

st.download_button(
    label="📥 Download Report",
    data=report,
    file_name=f"prediction_{home_team}_vs_{away_team}.txt",
    mime="text/plain"
)
