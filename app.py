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

# ========== CORRECTED CONSTANTS ==========
MAX_GOALS = 8
MAX_REGRESSION = 0.3
MIN_PROBABILITY = 0.1
MAX_PROBABILITY = 0.9

# Attack tiers (higher = better)
ATTACK_ELITE_PLUS = 1.5
ATTACK_ELITE = 1.0
ATTACK_ABOVE_AVG = 0.5
ATTACK_AVERAGE = 0.0
ATTACK_BELOW_AVG = -0.5

# Defense WEAKNESS tiers (higher = worse defense)
DEFENSE_VERY_WEAK = 1.5
DEFENSE_WEAK = 1.0
DEFENSE_AVERAGE = 0.5
DEFENSE_GOOD = 0.0
DEFENSE_STRONG = -0.5
DEFENSE_ELITE = -1.0

DOMINANCE_THRESHOLD = 0.65
HIGH_XG_THRESHOLD = 2.4

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
        
        # Attack score: higher = better attack
        attack_score = (xg_per_match - league_baselines['avg_xg']) / league_baselines['std_xg']
        
        # Defense WEAKNESS score: higher = worse defense (more conceding)
        # CRITICAL FIX: We track how WEAK the defense is, not how good
        defense_weakness = (xga_per_match - league_baselines['avg_xga']) / league_baselines['std_xga']
        
        regression_factor = min(max(goals_vs_xg_per_match, -MAX_REGRESSION), MAX_REGRESSION)
        
        attack_tier = TeamIntelligence._classify_attack(attack_score)
        defense_tier = TeamIntelligence._classify_defense_weakness(defense_weakness)
        
        return {
            'venue': venue,
            'attack_score': attack_score,
            'defense_weakness': defense_weakness,  # Changed from defense_score
            'attack_tier': attack_tier,
            'defense_tier': defense_tier,  # Now represents WEAKNESS tier
            'regression_factor': regression_factor,
            'xg_per_match': xg_per_match,
            'xga_per_match': xga_per_match,
            'goals_vs_xg_per_match': goals_vs_xg_per_match
        }
    
    @staticmethod
    def _classify_attack(score):
        # Higher attack score = better attack
        if score >= ATTACK_ELITE_PLUS: return "ELITE_PLUS"
        elif score >= ATTACK_ELITE: return "ELITE"
        elif score >= ATTACK_ABOVE_AVG: return "ABOVE_AVG"
        elif score >= ATTACK_AVERAGE: return "AVERAGE"
        elif score >= ATTACK_BELOW_AVG: return "BELOW_AVG"
        else: return "VERY_POOR"
    
    @staticmethod
    def _classify_defense_weakness(score):
        # Higher defense weakness score = WORSE defense
        if score >= DEFENSE_VERY_WEAK: return "VERY_WEAK"
        elif score >= DEFENSE_WEAK: return "WEAK"
        elif score >= DEFENSE_AVERAGE: return "AVERAGE"
        elif score >= DEFENSE_GOOD: return "GOOD"
        elif score >= DEFENSE_STRONG: return "STRONG"
        else: return "ELITE"

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
        
        # CORRECTED NET DOMINANCE CALCULATION
        # Net Dominance = Team Attack + Opponent Defense Weakness
        # Higher = more likely to score and win
        home_net_dominance = home_analysis['attack_score'] + away_analysis['defense_weakness']
        away_net_dominance = away_analysis['attack_score'] + home_analysis['defense_weakness']
        
        # Match type classification
        home_def = home_analysis['defense_tier']  # Now represents weakness
        away_def = away_analysis['defense_tier']  # Now represents weakness
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
                'net_dominance_difference': home_net_dominance - away_net_dominance
            }
        
        # 2. ATTACK DOMINANCE (Good attack vs Weak defense)
        elif ((home_att in ["ELITE", "ELITE_PLUS", "ABOVE_AVG"] and away_def in ["WEAK", "VERY_WEAK", "AVERAGE"]) or
              (away_att in ["ELITE", "ELITE_PLUS", "ABOVE_AVG"] and home_def in ["WEAK", "VERY_WEAK", "AVERAGE"])):
            return {
                'match_type': "ATTACK_DOMINANCE",
                'explanation': "Capable attack exploits vulnerable defense",
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
                'net_dominance_difference': home_net_dominance - away_net_dominance
            }
        
        # 3. GENUINE DEFENSIVE TACTICAL (Strong defense faces capable attack)
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
                'net_dominance_difference': home_net_dominance - away_net_dominance
            }
        
        # 4. FALSE DEFENSIVE (Strong defense but irrelevant due to weak attack)
        elif ((home_def in ["ELITE", "STRONG", "GOOD"] and away_att in ["BELOW_AVG", "VERY_POOR"]) or
              (away_def in ["ELITE", "STRONG", "GOOD"] and home_att in ["BELOW_AVG", "VERY_POOR"])):
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
                'net_dominance_difference': home_net_dominance - away_net_dominance
            }
        
        # 5. STANDARD (Balanced matchup)
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
                'net_dominance_difference': home_net_dominance - away_net_dominance
            }

class WinnerPrediction:
    @staticmethod
    def predict_winner(home_analysis, away_analysis, match_analysis, home_xg, away_xg):
        """
        CORRECTED winner prediction based on net dominance
        """
        home_net = match_analysis['home_net_dominance']
        away_net = match_analysis['away_net_dominance']
        
        # Home advantage boost
        home_advantage = 0.15
        
        # Adjusted net dominances with home advantage
        home_adjusted = home_net + home_advantage
        away_adjusted = away_net
        
        # Calculate win probabilities from net dominance difference
        diff = home_adjusted - away_adjusted
        
        # Use logistic function to convert to probabilities
        def logistic(x):
            return 1 / (1 + math.exp(-x))
        
        home_win_base = logistic(diff * 1.5)  # Steeper curve for clearer favorites
        away_win_base = 1 - home_win_base
        
        # Draw probability based on closeness
        draw_prob = max(0.15, min(0.35, 0.3 - abs(diff) * 0.4))
        
        # Match type adjustments
        match_type = match_analysis['match_type']
        
        if match_type == "DOMINANCE":
            if match_analysis['dominant_team'] == "HOME":
                home_win_base = min(1.0, home_win_base + 0.15)
            else:
                away_win_base = min(1.0, away_win_base + 0.15)
            draw_prob *= 0.7
        
        elif match_type == "DEFENSIVE_TACTICAL":
            draw_prob = min(0.4, draw_prob + 0.1)
        
        # Normalize to account for draw probability
        total_win_prob = home_win_base + away_win_base
        home_win_prob = home_win_base / total_win_prob * (1 - draw_prob)
        away_win_prob = away_win_base / total_win_prob * (1 - draw_prob)
        
        # Determine predicted winner
        max_prob = max(home_win_prob, away_win_prob, draw_prob)
        
        if home_win_prob == max_prob:
            predicted_winner = "HOME"
            confidence = "HIGH" if home_win_prob > 0.55 else "MEDIUM" if home_win_prob > 0.45 else "LOW"
        elif away_win_prob == max_prob:
            predicted_winner = "AWAY"
            confidence = "HIGH" if away_win_prob > 0.55 else "MEDIUM" if away_win_prob > 0.45 else "LOW"
        else:
            predicted_winner = "DRAW"
            confidence = "HIGH" if draw_prob > 0.4 else "MEDIUM" if draw_prob > 0.35 else "LOW"
        
        # Most likely score
        most_likely_score = WinnerPrediction._predict_most_likely_score(home_xg, away_xg)
        
        return {
            'home_win_probability': home_win_prob,
            'away_win_probability': away_win_prob,
            'draw_probability': draw_prob,
            'predicted_winner': predicted_winner,
            'winner_confidence': confidence,
            'most_likely_score': most_likely_score,
            'home_net_dominance': home_net,
            'away_net_dominance': away_net,
            'net_dominance_difference': home_net - away_net,
            'adjusted_difference': diff
        }
    
    @staticmethod
    def _predict_most_likely_score(home_xg, away_xg):
        # Simple Poisson-based prediction
        max_prob = 0
        best_score = "1-1"
        
        # Check most likely scores up to 3 goals each
        for i in range(4):
            for j in range(4):
                prob = poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg)
                if prob > max_prob:
                    max_prob = prob
                    best_score = f"{i}-{j}"
        
        return best_score

class PredictionEngine:
    def __init__(self, league_baselines):
        self.league_baselines = league_baselines
    
    def predict_expected_goals(self, home_analysis, away_analysis):
        """
        Predict expected goals based on attack and opponent defense weakness
        """
        home_attack = home_analysis['xg_per_match']
        away_defense_weakness_factor = 1 + (away_analysis['defense_weakness'] * 0.3)
        
        away_attack = away_analysis['xg_per_match']
        home_defense_weakness_factor = 1 + (home_analysis['defense_weakness'] * 0.3)
        
        # Base expected goals
        home_expected = home_attack * away_defense_weakness_factor
        away_expected = away_attack * home_defense_weakness_factor
        
        # Apply regression (over/underperformance vs xG)
        home_final = home_expected * (1 + home_analysis['regression_factor'])
        away_final = away_expected * (1 + away_analysis['regression_factor'])
        
        # Cap reasonable values
        home_final = max(min(home_final, 4.0), 0.3)
        away_final = max(min(away_final, 4.0), 0.3)
        
        return home_final, away_final
    
    def calculate_base_probability(self, total_xg):
        # Probability of over 2.5 goals using Poisson distribution
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
        
        # Corrections based on match type
        corrections = {
            "DOMINANCE": 0.10,           # Dominant team more likely over
            "ATTACK_DOMINANCE": 0.15,    # Attack dominance = more goals
            "DEFENSIVE_TACTICAL": -0.20, # Strong defense = fewer goals
            "FALSE_DEFENSIVE": -0.05,    # Weak attacks = fewer goals
            "STANDARD": 0.0
        }
        
        if match_type in corrections:
            correction = corrections[match_type]
            rationale.append(f"Match type: {match_type}")
        
        # Limit extreme corrections
        if base_prob > 0.70 and correction < 0:
            correction = max(correction, -0.15)
            rationale.append("Limited negative correction due to high base probability")
        elif base_prob < 0.30 and correction > 0:
            correction = min(correction, 0.15)
            rationale.append("Limited positive correction due to low base probability")
        
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
        
        # Base probability for over 2.5 goals
        base_prob = self.prediction_engine.calculate_base_probability(total_xg)
        
        # Match intelligence
        match_analysis = self.match_intel.analyze_match(home_analysis, away_analysis, home_xg, away_xg)
        
        # Correction based on match type
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
        
        # Confidence for over/under
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
        
        # Winner rationale
        winner_rationale = self._generate_winner_rationale(home_analysis, away_analysis, match_analysis, winner_prediction)
        
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
    
    def _generate_winner_rationale(self, home_analysis, away_analysis, match_analysis, winner_prediction):
        home_net = match_analysis['home_net_dominance']
        away_net = match_analysis['away_net_dominance']
        
        if winner_prediction['predicted_winner'] == "HOME":
            return f"Home attack ({home_analysis['attack_tier']}) benefits from away defense weakness ({away_analysis['defense_tier']}). Net dominance: {home_net:.2f}σ vs {away_net:.2f}σ"
        elif winner_prediction['predicted_winner'] == "AWAY":
            return f"Away attack ({away_analysis['attack_tier']}) benefits from home defense weakness ({home_analysis['defense_tier']}). Net dominance: {away_net:.2f}σ vs {home_net:.2f}σ"
        else:
            return f"Balanced matchup with similar net dominance. Home: {home_net:.2f}σ, Away: {away_net:.2f}σ"

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
        st.write(f"**Attack Tier**: {home_analysis['attack_tier']} (Score: {home_analysis['attack_score']:.2f}σ)")
        st.write(f"**Defense Weakness Tier**: {home_analysis['defense_tier']} (Score: {home_analysis['defense_weakness']:.2f}σ)")
        st.write(f"**xG/match**: {home_analysis['xg_per_match']:.2f}")
        st.write(f"**xGA/match**: {home_analysis['xga_per_match']:.2f}")
        st.write(f"**Goals vs xG**: {home_analysis['goals_vs_xg_per_match']:+.2f}")
    
    with col2:
        st.subheader(f"✈️ {away_team}")
        away_analysis = prediction['away_analysis']
        st.write(f"**Attack Tier**: {away_analysis['attack_tier']} (Score: {away_analysis['attack_score']:.2f}σ)")
        st.write(f"**Defense Weakness Tier**: {away_analysis['defense_tier']} (Score: {away_analysis['defense_weakness']:.2f}σ)")
        st.write(f"**xG/match**: {away_analysis['xg_per_match']:.2f}")
        st.write(f"**xGA/match**: {away_analysis['xga_per_match']:.2f}")
        st.write(f"**Goals vs xG**: {away_analysis['goals_vs_xg_per_match']:+.2f}")
    
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
    
    st.subheader("📈 Net Dominance Analysis")
    col9, col10, col11 = st.columns(3)
    
    with col9:
        home_net = prediction['match_analysis']['home_net_dominance']
        st.metric("Home Net Dominance", f"{home_net:.2f}σ",
                 delta="Strong" if home_net > 1.0 else "Weak" if home_net < -1.0 else "Average")
    
    with col10:
        away_net = prediction['match_analysis']['away_net_dominance']
        st.metric("Away Net Dominance", f"{away_net:.2f}σ",
                 delta="Strong" if away_net > 1.0 else "Weak" if away_net < -1.0 else "Average")
    
    with col11:
        net_diff = prediction['match_analysis']['net_dominance_difference']
        st.metric("Net Difference", f"{net_diff:.2f}σ",
                 delta="Home Advantage" if net_diff > 0.2 else "Away Advantage" if net_diff < -0.2 else "Balanced")

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
