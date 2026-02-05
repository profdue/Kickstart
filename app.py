import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine v2.0",
    page_icon="🧠",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine v2.0")
st.markdown("""
    **Hybrid Predictive Model: Performance-Based xG + Poisson Probabilities**
    *Step-by-step expected goals with mathematical consistency*
""")

# ========== CONSTANTS ==========
MAX_GOALS_CALC = 8  # Maximum goals to calculate in Poisson
MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99

# League average goals (will be calculated from data)
LEAGUE_AVG_GOALS = {
    "Premier League": 2.8,
    "Bundesliga": 3.2,
    "Serie A": 2.6,
    "La Liga": 2.5,
    "Ligue 1": 2.7,
    "Eredivisie": 3.1
}

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
    """Poisson probability mass function"""
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
        
        # Required columns check
        required = ['team', 'venue', 'matches', 'goals_for', 'goals_against', 
                   'goals_vs_xg', 'goals_allowed_vs_xga', 'xg', 'xga', 'points']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.warning(f"Missing columns: {missing}. Some features may be limited.")
            
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare home and away data with per-match averages"""
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    # Calculate per-match averages
    for df_part in [home_data, away_data]:
        df_part['goals_for_pm'] = df_part['goals_for'] / df_part['matches']
        df_part['goals_against_pm'] = df_part['goals_against'] / df_part['matches']
        df_part['goals_vs_xg_pm'] = df_part['goals_vs_xg'] / df_part['matches']
        df_part['goals_allowed_vs_xga_pm'] = df_part['goals_allowed_vs_xga'] / df_part['matches']
        df_part['xg_pm'] = df_part['xg'] / df_part['matches']
        df_part['xga_pm'] = df_part['xga'] / df_part['matches']
        df_part['points_pm'] = df_part['points'] / df_part['matches']
    
    return home_data.set_index('team'), away_data.set_index('team')

def calculate_league_metrics(df):
    """Calculate league-wide metrics including average goals"""
    if df is None or len(df) == 0:
        return {}
    
    # Calculate total goals per match in the league
    total_matches = df['matches'].sum() / 2  # Each match counted twice (home & away)
    total_goals = df['goals_for'].sum()
    
    avg_goals_per_match = total_goals / total_matches if total_matches > 0 else 2.5
    
    # Calculate home/away points averages
    home_data = df[df['venue'] == 'home']
    away_data = df[df['venue'] == 'away']
    
    home_pts_avg = home_data['points_pm'].mean() if 'points_pm' in home_data.columns else 1.5
    away_pts_avg = away_data['points_pm'].mean() if 'points_pm' in away_data.columns else 1.0
    
    return {
        'avg_goals_per_match': avg_goals_per_match,
        'home_pts_avg': home_pts_avg,
        'away_pts_avg': away_pts_avg
    }

class ExpectedGoalsPredictor:
    """Implements the step-by-step expected goals formula"""
    
    def __init__(self, league_metrics):
        self.league_metrics = league_metrics
        self.league_avg_goals = league_metrics.get('avg_goals_per_match', 2.5)
    
    def predict_expected_goals(self, home_stats, away_stats):
        """
        Step-by-step expected goals calculation using the refined logic
        
        Parameters:
        -----------
        home_stats: Series with home team statistics (per match averages)
        away_stats: Series with away team statistics (per match averages)
        
        Returns:
        --------
        tuple: (home_expected_goals, away_expected_goals)
        """
        
        # Step 1: Adjusted Goals (Weighted xG Adjustment)
        home_adjGF = home_stats['goals_for_pm'] + 0.6 * home_stats['goals_vs_xg_pm']
        home_adjGA = home_stats['goals_against_pm'] + 0.6 * home_stats['goals_allowed_vs_xga_pm']
        
        away_adjGF = away_stats['goals_for_pm'] + 0.6 * away_stats['goals_vs_xg_pm']
        away_adjGA = away_stats['goals_against_pm'] + 0.6 * away_stats['goals_allowed_vs_xga_pm']
        
        # Step 2: Dynamic Venue Factor
        venue_factor_home = 1 + 0.05 * (home_stats['points_pm'] - away_stats['points_pm']) / 3
        venue_factor_away = 1 + 0.05 * (away_stats['points_pm'] - home_stats['points_pm']) / 3
        
        # Clamp venue factors to reasonable range
        venue_factor_home = max(0.8, min(1.2, venue_factor_home))
        venue_factor_away = max(0.8, min(1.2, venue_factor_away))
        
        # Step 3: Expected Goals Calculation
        home_xg = (home_adjGF + away_adjGA) / 2 * venue_factor_home
        away_xg = (away_adjGF + home_adjGA) / 2 * venue_factor_away
        
        # Step 4: League Average Normalization
        normalization_factor = self.league_avg_goals / 2.5
        home_xg *= normalization_factor
        away_xg *= normalization_factor
        
        # Step 5: Apply reasonable bounds
        home_xg = max(0.2, min(5.0, home_xg))
        away_xg = max(0.2, min(5.0, away_xg))
        
        return home_xg, away_xg, {
            'home_adjGF': home_adjGF,
            'home_adjGA': home_adjGA,
            'away_adjGF': away_adjGF,
            'away_adjGA': away_adjGA,
            'venue_factor_home': venue_factor_home,
            'venue_factor_away': venue_factor_away,
            'normalization_factor': normalization_factor
        }

class PoissonProbabilityEngine:
    """Mathematically consistent probability engine using Poisson distributions"""
    
    @staticmethod
    def calculate_all_probabilities(home_xg, away_xg):
        """
        Calculate ALL probabilities from single Poisson distribution
        Returns mathematically consistent probabilities
        """
        # Calculate all score probabilities
        score_probabilities = []
        max_goals = min(MAX_GOALS_CALC, int(home_xg + away_xg) + 4)
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (poisson_pmf(home_goals, home_xg) * 
                       poisson_pmf(away_goals, away_xg))
                if prob > 0.0001:  # Ignore very small probabilities
                    score_probabilities.append({
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'probability': prob
                    })
        
        # Find most likely score
        most_likely = max(score_probabilities, key=lambda x: x['probability'])
        most_likely_score = f"{most_likely['home_goals']}-{most_likely['away_goals']}"
        
        # Calculate win/draw/loss probabilities
        home_win_prob = sum(p['probability'] for p in score_probabilities 
                           if p['home_goals'] > p['away_goals'])
        draw_prob = sum(p['probability'] for p in score_probabilities 
                       if p['home_goals'] == p['away_goals'])
        away_win_prob = sum(p['probability'] for p in score_probabilities 
                           if p['home_goals'] < p['away_goals'])
        
        # Calculate over/under probabilities
        over_2_5_prob = sum(p['probability'] for p in score_probabilities 
                           if p['home_goals'] + p['away_goals'] > 2.5)
        under_2_5_prob = sum(p['probability'] for p in score_probabilities 
                            if p['home_goals'] + p['away_goals'] < 2.5)
        
        # Calculate both teams to score
        btts_prob = sum(p['probability'] for p in score_probabilities 
                       if p['home_goals'] > 0 and p['away_goals'] > 0)
        
        # Get top 5 most likely scores
        top_scores = sorted(score_probabilities, key=lambda x: x['probability'], reverse=True)[:5]
        top_scores_formatted = [(f"{s['home_goals']}-{s['away_goals']}", s['probability']) 
                               for s in top_scores]
        
        return {
            'home_win_probability': home_win_prob,
            'draw_probability': draw_prob,
            'away_win_probability': away_win_prob,
            'over_2_5_probability': over_2_5_prob,
            'under_2_5_probability': under_2_5_prob,
            'btts_probability': btts_prob,
            'most_likely_score': most_likely_score,
            'top_scores': top_scores_formatted,
            'expected_home_goals': home_xg,
            'expected_away_goals': away_xg,
            'total_expected_goals': home_xg + away_xg,
            'score_probabilities': score_probabilities
        }

class MatchAnalyzer:
    """Analyze match dynamics and generate insights"""
    
    @staticmethod
    def analyze_match_dynamics(home_xg, away_xg, home_stats, away_stats):
        """Analyze match dynamics using expected goals"""
        total_goals = home_xg + away_xg
        delta = home_xg - away_xg
        
        # Winner prediction logic
        if delta > 1.5:
            predicted_winner = "HOME"
            winner_strength = "STRONG"
        elif delta > 0.5:
            predicted_winner = "HOME"
            winner_strength = "MODERATE"
        elif delta < -1.5:
            predicted_winner = "AWAY"
            winner_strength = "STRONG"
        elif delta < -0.5:
            predicted_winner = "AWAY"
            winner_strength = "MODERATE"
        else:
            predicted_winner = "DRAW"
            winner_strength = "CLOSE"
        
        # Total goals prediction with variance
        variance = abs(delta) * 0.4
        if total_goals - variance > 2.7:
            total_prediction = "OVER"
            total_confidence = "HIGH"
        elif total_goals + variance < 2.3:
            total_prediction = "UNDER"
            total_confidence = "HIGH"
        else:
            # Lean based on attacking strength
            avg_attack = (home_stats['goals_for_pm'] + away_stats['goals_for_pm']) / 2
            if avg_attack > 1.8:
                total_prediction = "OVER"
                total_confidence = "LEAN"
            else:
                total_prediction = "UNDER"
                total_confidence = "LEAN"
        
        # Calculate confidence scores
        winner_confidence = min(100, max(50, (abs(delta) / max(home_xg, away_xg, 0.1) * 100 + 
                                            (abs(total_goals - 2.5) / 2.5) * 30)))
        
        total_confidence_score = min(100, abs(total_goals - 2.5) / 2.5 * 100)
        
        return {
            'predicted_winner': predicted_winner,
            'winner_strength': winner_strength,
            'winner_confidence': winner_confidence,
            'total_prediction': total_prediction,
            'total_confidence': total_confidence,
            'total_confidence_score': total_confidence_score,
            'delta': delta,
            'total_goals': total_goals,
            'variance': variance
        }
    
    @staticmethod
    def generate_insights(home_xg, away_xg, home_stats, away_stats, calc_details):
        """Generate detailed insights about the match"""
        
        insights = []
        
        # Venue insight
        venue_factor = calc_details['venue_factor_home']
        if venue_factor > 1.1:
            insights.append(f"Strong home advantage (+{((venue_factor-1)*100):.0f}% venue factor)")
        elif venue_factor < 0.95:
            insights.append(f"Home disadvantage ({((venue_factor-1)*100):+.0f}% venue factor)")
        
        # Attack vs Defense insight
        home_attack_quality = home_stats['goals_for_pm'] + home_stats['goals_vs_xg_pm']
        away_defense_weakness = away_stats['goals_against_pm'] + away_stats['goals_allowed_vs_xga_pm']
        
        if home_attack_quality > 2.0 and away_defense_weakness > 1.5:
            insights.append(f"{home_stats.name} strong attack vs {away_stats.name} vulnerable defense")
        
        # Goal expectation insight
        if home_xg > 2.0 and away_xg > 1.5:
            insights.append("High-scoring affair expected from both sides")
        elif home_xg < 1.0 and away_xg < 1.0:
            insights.append("Low-scoring defensive match expected")
        
        # Performance consistency insight
        home_consistency = abs(home_stats['goals_vs_xg_pm'])
        away_consistency = abs(away_stats['goals_vs_xg_pm'])
        
        if home_consistency > 0.3:
            insights.append(f"{home_stats.name} shows inconsistent finishing")
        if away_consistency > 0.3:
            insights.append(f"{away_stats.name} shows inconsistent finishing")
        
        return insights

class FootballIntelligenceEngineV2:
    """Main engine combining performance-based xG with Poisson probabilities"""
    
    def __init__(self, league_metrics):
        self.league_metrics = league_metrics
        self.xg_predictor = ExpectedGoalsPredictor(league_metrics)
        self.probability_engine = PoissonProbabilityEngine()
        self.match_analyzer = MatchAnalyzer()
    
    def predict_match(self, home_team, away_team, home_stats, away_stats):
        """
        Generate complete match prediction
        
        Returns:
        --------
        dict: Complete prediction with probabilities and insights
        """
        
        # Step 1: Calculate expected goals using performance-based formula
        home_xg, away_xg, calc_details = self.xg_predictor.predict_expected_goals(
            home_stats, away_stats
        )
        
        # Step 2: Calculate all probabilities using Poisson distribution
        probabilities = self.probability_engine.calculate_all_probabilities(
            home_xg, away_xg
        )
        
        # Step 3: Analyze match dynamics
        dynamics = self.match_analyzer.analyze_match_dynamics(
            home_xg, away_xg, home_stats, away_stats
        )
        
        # Step 4: Generate insights
        insights = self.match_analyzer.generate_insights(
            home_xg, away_xg, home_stats, away_stats, calc_details
        )
        
        # Step 5: Determine final predictions
        # Total goals prediction
        total_prob = probabilities['over_2_5_probability']
        direction = "OVER" if total_prob > 0.5 else "UNDER"
        final_prob = total_prob if direction == "OVER" else probabilities['under_2_5_probability']
        
        # Winner prediction
        winner_pred = dynamics['predicted_winner']
        if winner_pred == "HOME":
            winner_display = home_team
            winner_prob = probabilities['home_win_probability']
        elif winner_pred == "AWAY":
            winner_display = away_team
            winner_prob = probabilities['away_win_probability']
        else:
            winner_display = "DRAW"
            winner_prob = probabilities['draw_probability']
        
        # Confidence categorization
        total_confidence = self._categorize_confidence(dynamics['total_confidence_score'])
        winner_confidence = self._categorize_confidence(dynamics['winner_confidence'])
        
        return {
            # Total goals prediction
            'total_goals': {
                'direction': direction,
                'probability': final_prob,
                'confidence': total_confidence,
                'confidence_score': dynamics['total_confidence_score'],
                'expected_total': home_xg + away_xg
            },
            
            # Winner prediction
            'winner': {
                'team': winner_display,
                'type': dynamics['predicted_winner'],
                'probability': winner_prob,
                'confidence': winner_confidence,
                'confidence_score': dynamics['winner_confidence'],
                'strength': dynamics['winner_strength'],
                'most_likely_score': probabilities['most_likely_score']
            },
            
            # All probabilities
            'probabilities': probabilities,
            
            # Expected goals
            'expected_goals': {
                'home': home_xg,
                'away': away_xg,
                'total': home_xg + away_xg
            },
            
            # Calculation details
            'calculation_details': calc_details,
            
            # Insights
            'insights': insights,
            
            # Match dynamics
            'dynamics': dynamics,
            
            # Team statistics
            'team_stats': {
                'home': home_stats.to_dict(),
                'away': away_stats.to_dict()
            }
        }
    
    def _categorize_confidence(self, score):
        """Categorize confidence score"""
        if score >= 80:
            return "VERY HIGH"
        elif score >= 70:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 50:
            return "LOW"
        else:
            return "VERY LOW"

# ========== STREAMLIT UI ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie"]
    selected_league = st.selectbox("Select League", leagues)
    
    df = load_league_data(selected_league)
    
    if df is not None:
        league_metrics = calculate_league_metrics(df)
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
        
        st.markdown("### 🎯 Prediction Settings")
        show_calculations = st.checkbox("Show Detailed Calculations", value=True)
        show_poisson = st.checkbox("Show Poisson Probabilities", value=True)
        
        if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
            calculate_btn = True
        else:
            calculate_btn = False

if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Generate Prediction'")
    
    # Show league statistics
    if league_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("League Avg Goals", f"{league_metrics['avg_goals_per_match']:.2f}")
        with col2:
            st.metric("Avg Home Points", f"{league_metrics['home_pts_avg']:.2f}")
        with col3:
            st.metric("Avg Away Points", f"{league_metrics['away_pts_avg']:.2f}")
    
    st.stop()

try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"Team data error: {e}")
    st.stop()

# Generate prediction
st.header(f"🎯 {home_team} vs {away_team}")
st.caption(f"League: {selected_league} | Avg Goals: {league_metrics['avg_goals_per_match']:.2f}")

engine = FootballIntelligenceEngineV2(league_metrics)
prediction = engine.predict_match(home_team, away_team, home_stats, away_stats)

# ========== DISPLAY RESULTS ==========

# Main prediction cards
col1, col2 = st.columns(2)

with col1:
    # Total goals prediction
    total_pred = prediction['total_goals']
    direction = total_pred['direction']
    confidence = total_pred['confidence']
    
    if direction == "OVER":
        card_color = "#14532D" if confidence == "VERY HIGH" else "#166534" if confidence == "HIGH" else "#365314"
        text_color = "#22C55E" if confidence == "VERY HIGH" else "#4ADE80" if confidence == "HIGH" else "#84CC16"
    else:
        card_color = "#7F1D1D" if confidence == "VERY HIGH" else "#991B1B" if confidence == "HIGH" else "#78350F"
        text_color = "#EF4444" if confidence == "VERY HIGH" else "#F87171" if confidence == "HIGH" else "#F59E0B"
    
    st.markdown(f"""
    <div style="background-color: {card_color}; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">TOTAL GOALS</h3>
        <div style="font-size: 36px; font-weight: bold; color: {text_color}; margin: 10px 0;">
            {direction} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {total_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: white;">
            Confidence: {confidence} ({total_pred['confidence_score']:.0f}/100)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Winner prediction
    winner_pred = prediction['winner']
    winner_confidence = winner_pred['confidence']
    
    if winner_pred['type'] == "HOME":
        winner_color = "#22C55E"
        icon = "🏠"
    elif winner_pred['type'] == "AWAY":
        winner_color = "#22C55E"
        icon = "✈️"
    else:
        winner_color = "#F59E0B"
        icon = "🤝"
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">PREDICTED WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: {winner_color}; margin: 10px 0;">
            {icon} {winner_pred['team']}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_pred['probability']*100:.1f}%
        </div>
        <div style="font-size: 16px; color: #94A3B8;">
            Strength: {winner_pred['strength']} | Confidence: {winner_confidence}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Insights
if prediction['insights']:
    st.subheader("🧠 Key Insights")
    for insight in prediction['insights']:
        st.write(f"• {insight}")

# Detailed Probabilities
st.subheader("📊 Detailed Probabilities")
col1, col2, col3, col4 = st.columns(4)

with col1:
    probs = prediction['probabilities']
    st.metric(f"🏠 {home_team} Win", f"{probs['home_win_probability']*100:.1f}%")

with col2:
    st.metric("🤝 Draw", f"{probs['draw_probability']*100:.1f}%")

with col3:
    st.metric(f"✈️ {away_team} Win", f"{probs['away_win_probability']*100:.1f}%")

with col4:
    st.metric("Both Teams Score", f"{probs['btts_probability']*100:.1f}%")

# Most Likely Scores
st.subheader("🎯 Most Likely Scores")
scores_cols = st.columns(5)
for idx, (score, prob) in enumerate(prediction['probabilities']['top_scores'][:5]):
    with scores_cols[idx]:
        st.metric(f"{score}", f"{prob*100:.1f}%")

# Expected Goals
st.subheader("⚽ Expected Goals")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(f"{home_team} xG", f"{prediction['expected_goals']['home']:.2f}")

with col2:
    st.metric(f"{away_team} xG", f"{prediction['expected_goals']['away']:.2f}")

with col3:
    total_xg = prediction['expected_goals']['total']
    st.metric("Total xG", f"{total_xg:.2f}", 
             delta=f"{'OVER' if total_xg > 2.5 else 'UNDER'} 2.5")

# Detailed Calculations (Expandable)
if show_calculations:
    with st.expander("🔧 Step-by-Step Calculation Details", expanded=False):
        calc_details = prediction['calculation_details']
        
        st.write("### Step 1: Adjusted Goals (60% weight on goals vs xG)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{home_team}:**")
            st.write(f"- Goals/Game: {home_stats['goals_for_pm']:.2f}")
            st.write(f"- Goals vs xG/Game: {home_stats['goals_vs_xg_pm']:.2f}")
            st.write(f"- Adjusted GF: {calc_details['home_adjGF']:.2f}")
            st.write(f"- Adjusted GA: {calc_details['home_adjGA']:.2f}")
        
        with col2:
            st.write(f"**{away_team}:**")
            st.write(f"- Goals/Game: {away_stats['goals_for_pm']:.2f}")
            st.write(f"- Goals vs xG/Game: {away_stats['goals_vs_xg_pm']:.2f}")
            st.write(f"- Adjusted GF: {calc_details['away_adjGF']:.2f}")
            st.write(f"- Adjusted GA: {calc_details['away_adjGA']:.2f}")
        
        st.write("### Step 2: Dynamic Venue Factor")
        st.write(f"**Home Venue Factor:** {calc_details['venue_factor_home']:.3f}")
        st.write(f"**Away Venue Factor:** {calc_details['venue_factor_away']:.3f}")
        st.write(f"*Based on home points/game: {home_stats['points_pm']:.2f} vs away: {away_stats['points_pm']:.2f}*")
        
        st.write("### Step 3: Expected Goals Calculation")
        st.write(f"**Home xG:** ({calc_details['home_adjGF']:.2f} + {calc_details['away_adjGA']:.2f}) / 2 × {calc_details['venue_factor_home']:.3f}")
        st.write(f"**Away xG:** ({calc_details['away_adjGF']:.2f} + {calc_details['home_adjGA']:.2f}) / 2 × {calc_details['venue_factor_away']:.3f}")
        
        st.write("### Step 4: League Normalization")
        st.write(f"**League Avg Goals:** {league_metrics['avg_goals_per_match']:.2f}")
        st.write(f"**Normalization Factor:** {calc_details['normalization_factor']:.3f}")
        st.write(f"*Adjusts to league scoring environment*")

# Poisson Distribution Details (Expandable)
if show_poisson:
    with st.expander("📈 Poisson Distribution Details", expanded=False):
        st.write("### Probability Mass Function")
        st.write(f"P(Home = k, Away = l) = (λ₁ᵏ e⁻λ₁ / k!) × (λ₂ˡ e⁻λ₂ / l!)")
        st.write(f"Where λ₁ = {prediction['expected_goals']['home']:.2f}, λ₂ = {prediction['expected_goals']['away']:.2f}")
        
        # Show probability matrix
        st.write("### Score Probability Matrix")
        max_show = 4
        matrix_data = []
        
        for home in range(max_show + 1):
            row = []
            for away in range(max_show + 1):
                prob = poisson_pmf(home, prediction['expected_goals']['home']) * \
                       poisson_pmf(away, prediction['expected_goals']['away'])
                row.append(f"{prob*100:.1f}%")
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(
            matrix_data,
            index=[f"Home {i}" for i in range(max_show + 1)],
            columns=[f"Away {i}" for i in range(max_show + 1)]
        )
        
        st.dataframe(matrix_df, use_container_width=True)

# Team Statistics Comparison
with st.expander("📋 Team Statistics Comparison", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏠 {home_team} (Home)")
        stats_data = {
            'Statistic': ['Goals/Game', 'Conceded/Game', 'Goals vs xG/Game', 
                         'Conceded vs xGA/Game', 'xG/Game', 'xGA/Game', 'Points/Game'],
            'Value': [home_stats['goals_for_pm'], home_stats['goals_against_pm'],
                     home_stats['goals_vs_xg_pm'], home_stats['goals_allowed_vs_xga_pm'],
                     home_stats['xg_pm'], home_stats['xga_pm'], home_stats['points_pm']]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.2f}")
        st.table(stats_df)
    
    with col2:
        st.subheader(f"✈️ {away_team} (Away)")
        stats_data = {
            'Statistic': ['Goals/Game', 'Conceded/Game', 'Goals vs xG/Game', 
                         'Conceded vs xGA/Game', 'xG/Game', 'xGA/Game', 'Points/Game'],
            'Value': [away_stats['goals_for_pm'], away_stats['goals_against_pm'],
                     away_stats['goals_vs_xg_pm'], away_stats['goals_allowed_vs_xga_pm'],
                     away_stats['xg_pm'], away_stats['xga_pm'], away_stats['points_pm']]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.2f}")
        st.table(stats_df)

# Export Report
st.divider()
st.subheader("📤 Export Prediction Report")

report = f"""
⚽ PERFORMANCE-BASED FOOTBALL PREDICTION ENGINE v2.0
Match: {home_team} vs {away_team}
League: {selected_league}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 TOTAL GOALS PREDICTION
{prediction['total_goals']['direction']} 2.5 Goals: {prediction['total_goals']['probability']*100:.1f}%
Confidence: {prediction['total_goals']['confidence']} ({prediction['total_goals']['confidence_score']:.0f}/100)
Expected Total Goals: {prediction['expected_goals']['total']:.2f}

🏆 WINNER PREDICTION
Predicted Winner: {prediction['winner']['team']}
Probability: {prediction['winner']['probability']*100:.1f}%
Strength: {prediction['winner']['strength']}
Confidence: {prediction['winner']['confidence']}
Most Likely Score: {prediction['winner']['most_likely_score']}

📊 DETAILED PROBABILITIES
Home Win: {prediction['probabilities']['home_win_probability']*100:.1f}%
Draw: {prediction['probabilities']['draw_probability']*100:.1f}%
Away Win: {prediction['probabilities']['away_win_probability']*100:.1f}%
Both Teams to Score: {prediction['probabilities']['btts_probability']*100:.1f}%

⚽ EXPECTED GOALS
{home_team}: {prediction['expected_goals']['home']:.2f} xG
{away_team}: {prediction['expected_goals']['away']:.2f} xG
Total: {prediction['expected_goals']['total']:.2f} xG

🎯 MOST LIKELY SCORES
"""
for score, prob in prediction['probabilities']['top_scores'][:5]:
    report += f"{score}: {prob*100:.1f}%\n"

report += f"""
🧠 KEY INSIGHTS
"""
for insight in prediction['insights']:
    report += f"• {insight}\n"

report += f"""
🔧 CALCULATION METHODOLOGY
1. Adjusted Goals with 60% weight on goals_vs_xG
2. Dynamic venue factor based on points performance
3. League average normalization
4. Poisson distribution for consistent probabilities
5. Variance-adjusted total goals prediction

---
Generated by Hybrid Football Intelligence Engine v2.0
Performance-Based xG + Mathematical Consistency
"""

st.code(report, language="text")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"prediction_{home_team}_vs_{away_team}.txt",
        mime="text/plain",
        use_container_width=True
    )

with col2:
    if st.button("📊 Add to History", use_container_width=True):
        st.session_state.prediction_history.append({
            'timestamp': datetime.now(),
            'home_team': home_team,
            'away_team': away_team,
            'league': selected_league,
            'prediction': prediction
        })
        st.success("Added to prediction history!")

# Show prediction history
if st.session_state.prediction_history:
    with st.expander("📚 Prediction History", expanded=False):
        for i, hist in enumerate(reversed(st.session_state.prediction_history[-5:])):
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{hist['home_team']} vs {hist['away_team']}**")
                    st.caption(f"{hist['timestamp'].strftime('%Y-%m-%d %H:%M')} | {hist['league']}")
                with col2:
                    st.write(f"📈 {hist['prediction']['total_goals']['direction']} 2.5")
                with col3:
                    st.write(f"🏆 {hist['prediction']['winner']['team']}")
                st.divider()
