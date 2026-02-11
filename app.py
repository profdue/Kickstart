import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine - ACTUAL STATISTICAL FINDINGS",
    page_icon="🎯",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine - ACTUAL STATISTICAL FINDINGS")
st.markdown("""
    **IMPLEMENTING YOUR REAL STATISTICAL ANALYSIS** - Home advantage + League-specific sweet spots
""")

# ========== YOUR ACTUAL STATISTICAL MODEL ==========

class YourStatisticalModel:
    """YOUR ACTUAL FINDINGS FROM DATA ANALYSIS"""
    
    def __init__(self, league_name):
        self.league_name = league_name
        self.model_type = "YOUR_ACTUAL_FINDINGS"
        self.version = "v1.0_real_analysis"
        
        # FROM YOUR 1049-MATCH ANALYSIS:
        self.home_advantage_goals = 3.36  # Home teams score +3.36 more goals
        self.home_win_rate = 0.4452       # 44.52% home wins
        self.away_win_rate = 0.2993       # 29.93% away wins
        self.draw_rate = 0.2555           # 25.55% draws
        self.per_match_home_advantage = 3.36 / 38  # +0.088 goals per match
        
        # FROM YOUR LEAGUE-SPECIFIC ANALYSIS (192 entries):
        self.league_sweet_spots = {
            "Premier League": {"under": 2.5, "over": 3.0, "notes": "Standard pattern"},
            "Serie A": {"under": 2.8, "over": 3.5, "notes": "More defensive, higher thresholds"},
            "Ligue 1": {"under": 2.5, "over": 3.0, "notes": "Similar to Premier League"},
            "Bundesliga": {"under": 2.6, "over": 3.2, "notes": "Estimated"},
            "La Liga": {"under": 2.6, "over": 3.2, "notes": "Estimated"},
            "Eredivisie": {"under": 2.7, "over": 3.3, "notes": "Estimated high-scoring"},
            "RFPL": {"under": 2.5, "over": 3.0, "notes": "Estimated"}
        }
        
        # Get this league's sweet spots
        self.sweet_spots = self.league_sweet_spots.get(
            league_name, 
            {"under": 2.5, "over": 3.0, "notes": "Default"}
        )
        
        # Performance baselines
        self.baseline_home_accuracy = 44.52  # "Always bet home" accuracy
        self.current_model_accuracy = 22.2   # Your current model's accuracy
    
    def predict_winner(self, home_stats, away_stats):
        """
        YOUR LOGIC: Intelligent home advantage application
        Home teams get +25% effectiveness but CAN'T overcome large quality gaps
        """
        # Extract REAL statistics
        home_scoring = home_stats.get('goals_scored', 1.5)
        home_conceding = home_stats.get('goals_conceded', 1.3)
        away_scoring = away_stats.get('goals_scored', 1.2)
        away_conceding = away_stats.get('goals_conceded', 1.5)
        
        # Calculate base strength WITHOUT home advantage
        home_base_strength = (home_scoring + away_conceding) / 2
        away_base_strength = (away_scoring + home_conceding) / 2
        
        # APPLY YOUR HOME ADVANTAGE FINDINGS:
        # 1. Home teams are 25% more effective
        home_adjusted = home_base_strength * 1.25
        
        # 2. Away teams are 20% less effective
        away_adjusted = away_base_strength * 0.80
        
        # 3. Add absolute home advantage (+0.088 goals per match)
        home_adjusted += self.per_match_home_advantage
        away_adjusted -= self.per_match_home_advantage
        
        # Calculate strength difference
        strength_diff = home_adjusted - away_adjusted
        strength_ratio = away_adjusted / home_adjusted if home_adjusted > 0 else 999
        
        # YOUR INTELLIGENT DECISION LOGIC:
        # Rule 1: If away team is MUCH stronger (>40%), home advantage can't overcome
        if strength_ratio > 1.4:
            confidence = 60 + min(20, (strength_ratio - 1.4) * 10)
            return "AWAY", confidence, f"AWAY_MUCH_STRONGER ({strength_ratio:.1f}x)"
        
        # Rule 2: If home team is MUCH stronger (>40%)
        elif strength_ratio < 0.71:  # 1/1.4 = 0.71
            confidence = 60 + min(20, (1/strength_ratio - 1.4) * 10)
            return "HOME", confidence, f"HOME_MUCH_STRONGER ({1/strength_ratio:.1f}x)"
        
        # Rule 3: Moderate away advantage (20-40%)
        elif strength_ratio > 1.2:
            confidence = 55 + min(10, (strength_ratio - 1.2) * 10)
            return "AWAY", confidence, f"AWAY_MODERATELY_STRONGER ({strength_ratio:.1f}x)"
        
        # Rule 4: Moderate home advantage (20-40%)
        elif strength_ratio < 0.83:  # 1/1.2 = 0.83
            confidence = 55 + min(10, (1/strength_ratio - 1.2) * 10)
            return "HOME", confidence, f"HOME_MODERATELY_STRONGER ({1/strength_ratio:.1f}x)"
        
        # Rule 5: Close match - use YOUR 44.52% home win baseline
        else:
            if strength_diff > 0:
                return "HOME", 52, "CLOSE_MATCH_HOME_EDGE"
            else:
                return "AWAY", 48, "CLOSE_MATCH_AWAY_EDGE"
    
    def predict_totals(self, sum_xg):
        """
        YOUR LOGIC: League-specific sweet spots for Over/Under
        From your analysis of 192 league entries
        """
        under_threshold = self.sweet_spots["under"]
        over_threshold = self.sweet_spots["over"]
        
        # APPLY YOUR SWEET SPOTS:
        # 1. If in UNDER sweet spot
        if sum_xg < under_threshold:
            # Calculate confidence based on distance from threshold
            distance = (under_threshold - sum_xg) / under_threshold
            if distance > 0.3:
                confidence = 75  # Well below threshold
            elif distance > 0.15:
                confidence = 65  # Comfortably below
            else:
                confidence = 60  # Just below
            
            return "UNDER", confidence, f"{self.league_name} UNDER sweet spot: {sum_xg:.1f} < {under_threshold}"
        
        # 2. If in OVER sweet spot
        elif sum_xg > over_threshold:
            # Calculate confidence based on distance from threshold
            if over_threshold > 0:
                distance = (sum_xg - over_threshold) / over_threshold
            else:
                distance = 0.2
            
            if distance > 0.3:
                confidence = 75  # Well above threshold
            elif distance > 0.15:
                confidence = 65  # Comfortably above
            else:
                confidence = 60  # Just above
            
            return "OVER", confidence, f"{self.league_name} OVER sweet spot: {sum_xg:.1f} > {over_threshold}"
        
        # 3. Intermediate range (uncertainty from your analysis)
        else:
            # Determine if closer to under or over threshold
            range_midpoint = (under_threshold + over_threshold) / 2
            
            if sum_xg < range_midpoint:
                confidence = 55
                return "UNDER", confidence, f"Intermediate range: {sum_xg:.1f} closer to UNDER ({under_threshold})"
            else:
                confidence = 55
                return "OVER", confidence, f"Intermediate range: {sum_xg:.1f} closer to OVER ({over_threshold})"
    
    def get_statistical_findings(self):
        """Display YOUR actual findings"""
        return {
            "home_advantage": f"+{self.home_advantage_goals} goals over season (+0.088 per match)",
            "home_win_rate": f"{self.home_win_rate*100:.1f}% of matches",
            "away_win_rate": f"{self.away_win_rate*100:.1f}% of matches",
            "draw_rate": f"{self.draw_rate*100:.1f}% of matches",
            "league_sweet_spots": f"UNDER < {self.sweet_spots['under']}, OVER > {self.sweet_spots['over']}",
            "baseline_accuracy": f"{self.baseline_home_accuracy}% (always bet home)",
            "current_model_accuracy": f"{self.current_model_accuracy}% (your current model)",
            "improvement_potential": f"+{self.baseline_home_accuracy - self.current_model_accuracy:.1f}% immediate gain"
        }

# ========== DATA LOADING FUNCTIONS ==========

@st.cache_data
def load_league_data(league_name):
    """Load your actual CSV files"""
    try:
        file_map = {
            "Premier League": "premier_league.csv",
            "Bundesliga": "bundesliga.csv", 
            "Serie A": "serie_a.csv",
            "La Liga": "laliga.csv",
            "Ligue 1": "ligue_1.csv",
            "Eredivisie": "eredivisie.csv",
            "RFPL": "rfpl.csv"
        }
        
        filename = file_map.get(league_name)
        if not filename:
            return None
            
        df = pd.read_csv(f"leagues/{filename}")
        
        # Calculate metrics needed for YOUR analysis
        df['goals_per_match'] = df['gf'] / df['matches']
        df['goals_allowed_per_match'] = df['ga'] / df['matches']
        df['xg_per_match'] = df['xg'] / df['matches']
        df['points_per_match'] = df['pts'] / df['matches']
        df['win_rate'] = df['wins'] / df['matches']
        
        return df
    except Exception as e:
        st.error(f"Error loading {league_name}: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare home and away data separately"""
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    return home_data.set_index('team'), away_data.set_index('team')

# ========== PROBABILITY CALCULATOR ==========

class ProbabilityCalculator:
    """Calculate match probabilities (for comparison)"""
    
    @staticmethod
    def poisson_probability(k, lam):
        """Poisson probability mass function"""
        return (math.exp(-lam) * (lam ** k)) / math.factorial(k)
    
    @staticmethod
    def calculate_match_probabilities(home_xg, away_xg):
        """Calculate all score probabilities"""
        scores = []
        
        # Check for realistic values
        home_xg = max(0.1, min(5.0, home_xg))
        away_xg = max(0.1, min(5.0, away_xg))
        
        # Calculate probabilities for reasonable scorelines
        max_goals = int(home_xg + away_xg) + 3
        
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (ProbabilityCalculator.poisson_probability(home_goals, home_xg) * 
                       ProbabilityCalculator.poisson_probability(away_goals, away_xg))
                
                if prob > 0.001:  # Only keep meaningful probabilities
                    scores.append({
                        'score': f"{home_goals}-{away_goals}",
                        'probability': prob,
                        'home_goals': home_goals,
                        'away_goals': away_goals
                    })
        
        # Calculate aggregated probabilities
        home_win_prob = sum(s['probability'] for s in scores if s['home_goals'] > s['away_goals'])
        draw_prob = sum(s['probability'] for s in scores if s['home_goals'] == s['away_goals'])
        away_win_prob = sum(s['probability'] for s in scores if s['home_goals'] < s['away_goals'])
        
        over_25_prob = sum(s['probability'] for s in scores if s['home_goals'] + s['away_goals'] > 2.5)
        under_25_prob = sum(s['probability'] for s in scores if s['home_goals'] + s['away_goals'] < 2.5)
        
        # Get most likely scores
        top_scores = sorted(scores, key=lambda x: x['probability'], reverse=True)[:5]
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'over_25': over_25_prob,
            'under_25': under_25_prob,
            'expected_home_goals': home_xg,
            'expected_away_goals': away_xg,
            'expected_total': home_xg + away_xg,
            'top_scores': [(s['score'], s['probability']) for s in top_scores]
        }

# ========== STREAMLIT UI ==========

# Sidebar
with st.sidebar:
    st.header("⚙️ YOUR STATISTICAL MODEL")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie", "RFPL"]
    selected_league = st.selectbox("Select League", leagues)
    
    # Create model
    model = YourStatisticalModel(selected_league)
    
    # Show YOUR findings
    st.divider()
    st.header("📊 YOUR ACTUAL FINDINGS")
    
    findings = model.get_statistical_findings()
    for key, value in findings.items():
        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Load data
    df = load_league_data(selected_league)
    
    if df is not None:
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        if len(home_stats_df) > 0 and len(away_stats_df) > 0:
            home_teams = sorted(home_stats_df.index.unique())
            away_teams = sorted(away_stats_df.index.unique())
            common_teams = sorted(list(set(home_teams) & set(away_teams)))
            
            if len(common_teams) > 0:
                home_team = st.selectbox("Home Team", common_teams)
                away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
                
                st.divider()
                
                if st.button("🚀 Generate Prediction Using YOUR Findings", type="primary", use_container_width=True):
                    calculate_btn = True
                else:
                    calculate_btn = False
            else:
                st.error("No common teams found between home and away data")
                st.stop()
        else:
            st.error("Could not prepare team data")
            st.stop()

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# Header
st.markdown("""
<div style="background-color: #0C4A6E; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h3 style="color: white; text-align: center; margin: 0;">
        🔬 IMPLEMENTING YOUR ACTUAL STATISTICAL FINDINGS
    </h3>
    <p style="color: #E0F2FE; text-align: center; margin: 5px 0 0 0;">
        1,049 matches analyzed • League-specific sweet spots • Intelligent home advantage
    </p>
</div>
""", unsafe_allow_html=True)

# Generate prediction
if 'calculate_btn' in locals() and calculate_btn:
    try:
        # Get team data
        home_stats_raw = home_stats_df.loc[home_team]
        away_stats_raw = away_stats_df.loc[away_team]
        
        # Prepare stats for YOUR model
        home_stats = {
            'goals_scored': home_stats_raw['goals_per_match'],
            'goals_conceded': home_stats_raw['goals_allowed_per_match'],
            'xg_per_match': home_stats_raw['xg_per_match'],
            'win_rate': home_stats_raw['win_rate']
        }
        
        away_stats = {
            'goals_scored': away_stats_raw['goals_per_match'],
            'goals_conceded': away_stats_raw['goals_allowed_per_match'],
            'xg_per_match': away_stats_raw['xg_per_match'],
            'win_rate': away_stats_raw['win_rate']
        }
        
        # Calculate sum_xg (YOUR key metric)
        sum_xg = home_stats['xg_per_match'] + away_stats['xg_per_match']
        
        # Get predictions using YOUR model
        predicted_winner, winner_confidence, winner_logic = model.predict_winner(home_stats, away_stats)
        predicted_totals, totals_confidence, totals_logic = model.predict_totals(sum_xg)
        
        # Calculate probabilities for comparison
        probabilities = ProbabilityCalculator.calculate_match_probabilities(
            home_stats['xg_per_match'],
            away_stats['xg_per_match']
        )
        
        # Store data
        prediction_data = {
            'league': selected_league,
            'home_team': home_team,
            'away_team': away_team,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'sum_xg': sum_xg,
            'predicted_winner': predicted_winner,
            'winner_confidence': winner_confidence,
            'winner_logic': winner_logic,
            'predicted_totals': predicted_totals,
            'totals_confidence': totals_confidence,
            'totals_logic': totals_logic,
            'probabilities': probabilities
        }
        
        st.session_state.prediction_data = prediction_data
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Display prediction or initial state
if 'prediction_data' not in st.session_state:
    st.info("👈 Select teams and click 'Generate Prediction Using YOUR Findings'")
    
    # Show example of YOUR logic
    with st.expander("See how YOUR findings are applied"):
        st.write("""
        **YOUR WINNER PREDICTION LOGIC:**
        
        1. Calculate base team strength
        2. Apply home advantage: Home ×1.25, Away ×0.80
        3. Add +0.088 goals to home team
        4. Check if home advantage can overcome quality gap
        5. If away team is 40%+ stronger → Predict AWAY
        6. Otherwise → Use 44.52% home win baseline
        
        **YOUR OVER/UNDER LOGIC:**
        
        1. Calculate sum_xg = home_xg_per_match + away_xg_per_match
        2. Apply league-specific sweet spots:
           - Premier League: UNDER < 2.5, OVER > 3.0
           - Serie A: UNDER < 2.8, OVER > 3.5
           - Ligue 1: UNDER < 2.5, OVER > 3.0
        3. Intermediate range = lower confidence
        """)
    st.stop()

# Display prediction
prediction_data = st.session_state.prediction_data

st.header(f"🎯 {prediction_data['home_team']} vs {prediction_data['away_team']}")
st.caption(f"League: {selected_league} • Using YOUR actual statistical findings")

# Show team statistics
with st.expander("📊 TEAM STATISTICS"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{prediction_data['home_team']} (Home):**")
        st.write(f"Goals scored: {prediction_data['home_stats']['goals_scored']:.2f}/match")
        st.write(f"Goals conceded: {prediction_data['home_stats']['goals_conceded']:.2f}/match")
        st.write(f"xG per match: {prediction_data['home_stats']['xg_per_match']:.2f}")
        st.write(f"Win rate: {prediction_data['home_stats']['win_rate']*100:.1f}%")
    
    with col2:
        st.write(f"**{prediction_data['away_team']} (Away):**")
        st.write(f"Goals scored: {prediction_data['away_stats']['goals_scored']:.2f}/match")
        st.write(f"Goals conceded: {prediction_data['away_stats']['goals_conceded']:.2f}/match")
        st.write(f"xG per match: {prediction_data['away_stats']['xg_per_match']:.2f}")
        st.write(f"Win rate: {prediction_data['away_stats']['win_rate']*100:.1f}%")
    
    st.write(f"**sum_xg (YOUR key metric):** {prediction_data['sum_xg']:.2f}")

# Prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction_data['predicted_winner']
    winner_conf = prediction_data['winner_confidence']
    winner_logic = prediction_data['winner_logic']
    
    # Get actual probability for comparison
    prob = prediction_data['probabilities']
    actual_prob = prob['home_win'] if winner_pred == 'HOME' else prob['away_win'] if winner_pred == 'AWAY' else prob['draw']
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">MATCH WINNER</h3>
        <div style="font-size: 36px; font-weight: bold; color: #60A5FA; margin: 10px 0;">
            {'🏠' if winner_pred == 'HOME' else '✈️' if winner_pred == 'AWAY' else '🤝'} {winner_pred}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_conf:.0f}%
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            {winner_logic}
        </div>
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            Poisson probability: {actual_prob*100:.1f}% • Baseline: {model.baseline_home_accuracy}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    totals_pred = prediction_data['predicted_totals']
    totals_conf = prediction_data['totals_confidence']
    totals_logic = prediction_data['totals_logic']
    sum_xg = prediction_data['sum_xg']
    
    # Get actual probability
    actual_prob = prob['over_25'] if totals_pred == 'OVER' else prob['under_25']
    
    # Color based on sweet spot
    under_thresh = model.sweet_spots['under']
    over_thresh = model.sweet_spots['over']
    
    if sum_xg < under_thresh:
        color = "#3B82F6"  # Blue for under sweet spot
    elif sum_xg > over_thresh:
        color = "#EF4444"  # Red for over sweet spot
    else:
        color = "#F59E0B"  # Yellow for intermediate
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">OVER/UNDER 2.5</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {'📈' if totals_pred == 'OVER' else '📉'} {totals_pred}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {totals_conf:.0f}%
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            {totals_logic}
        </div>
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            sum_xg: {sum_xg:.2f} • Poisson: {actual_prob*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sweet spot analysis
st.divider()
st.subheader("🎯 YOUR SWEET SPOT ANALYSIS")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Position Analysis:**")
    
    sum_xg = prediction_data['sum_xg']
    under = model.sweet_spots['under']
    over = model.sweet_spots['over']
    
    if sum_xg < under:
        st.success(f"✅ In UNDER sweet spot")
        st.write(f"{sum_xg:.2f} < {under}")
        distance = (under - sum_xg) / under
        st.write(f"Distance below threshold: {distance*100:.1f}%")
    elif sum_xg > over:
        st.success(f"✅ In OVER sweet spot")
        st.write(f"{sum_xg:.2f} > {over}")
        distance = (sum_xg - over) / over
        st.write(f"Distance above threshold: {distance*100:.1f}%")
    else:
        st.warning(f"⚠️ In intermediate range")
        st.write(f"{under} < {sum_xg:.2f} < {over}")
        range_width = over - under
        position = (sum_xg - under) / range_width
        st.write(f"Position in range: {position*100:.1f}%")

with col2:
    st.write("**League Comparison:**")
    
    # Compare to other leagues
    for other_league in ["Premier League", "Serie A", "Ligue 1"]:
        if other_league != selected_league:
            other_model = YourStatisticalModel(other_league)
            other_under = other_model.sweet_spots['under']
            other_over = other_model.sweet_spots['over']
            
            st.write(f"**{other_league}:**")
            st.write(f"UNDER < {other_under}, OVER > {other_over}")
            
            # Show what prediction would be in other league
            if sum_xg < other_under:
                other_pred = "UNDER"
            elif sum_xg > other_over:
                other_pred = "OVER"
            else:
                other_pred = "Intermediate"
            
            if other_pred != prediction_data['predicted_totals']:
                st.caption(f"(Would predict {other_pred} in {other_league})")

with col3:
    st.write("**Confidence Breakdown:**")
    
    st.write(f"Winner confidence: {prediction_data['winner_confidence']:.0f}%")
    st.write(f"Totals confidence: {prediction_data['totals_confidence']:.0f}%")
    
    # Show baseline comparison
    baseline_improvement = prediction_data['winner_confidence'] - model.baseline_home_accuracy
    st.write(f"vs Baseline ({model.baseline_home_accuracy}%): {baseline_improvement:+.1f}%")
    
    # Show vs current model
    current_improvement = prediction_data['winner_confidence'] - model.current_model_accuracy
    st.write(f"vs Current model ({model.current_model_accuracy}%): {current_improvement:+.1f}%")

# Probabilities table
st.divider()
st.subheader("📈 MATCH PROBABILITIES")

prob = prediction_data['probabilities']

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Home Win", f"{prob['home_win']*100:.1f}%")
    st.metric("Draw", f"{prob['draw']*100:.1f}%")
    st.metric("Away Win", f"{prob['away_win']*100:.1f}%")

with col2:
    st.metric("Over 2.5", f"{prob['over_25']*100:.1f}%")
    st.metric("Under 2.5", f"{prob['under_25']*100:.1f}%")

with col3:
    st.write("**Most Likely Scores:**")
    for score, prob_val in prob['top_scores']:
        st.write(f"{score}: {prob_val*100:.1f}%")
    
    st.write("**Expected Goals:**")
    st.write(f"Home: {prob['expected_home_goals']:.2f}")
    st.write(f"Away: {prob['expected_away_goals']:.2f}")
    st.write(f"Total: {prob['expected_total']:.2f}")

# Data collection
st.divider()
st.subheader("📝 TRACK PREDICTION ACCURACY")

col1, col2 = st.columns([2, 1])

with col1:
    score = st.text_input("Actual Final Score", placeholder="e.g., 2-1")
    
    with st.expander("View Prediction Logic"):
        st.write("**Winner Prediction Logic:**")
        st.write(prediction_data['winner_logic'])
        
        st.write("**Totals Prediction Logic:**")
        st.write(prediction_data['totals_logic'])
        
        st.write("**Statistical Basis:**")
        findings = model.get_statistical_findings()
        for key, value in findings.items():
            if key in ['home_advantage', 'league_sweet_spots', 'baseline_accuracy']:
                st.write(f"- {key.replace('_', ' ').title()}: {value}")

with col2:
    if st.button("💾 Save & Check Accuracy", type="primary", use_container_width=True):
        if score and '-' in score:
            try:
                home_goals, away_goals = map(int, score.split('-'))
                total_goals = home_goals + away_goals
                
                # Determine actual outcomes
                actual_winner = 'HOME' if home_goals > away_goals else 'AWAY' if away_goals > home_goals else 'DRAW'
                actual_over_under = 'OVER' if total_goals > 2.5 else 'UNDER'
                
                # Check accuracy
                winner_correct = actual_winner == prediction_data['predicted_winner']
                totals_correct = actual_over_under == prediction_data['predicted_totals']
                
                # Save data
                import json
                from datetime import datetime
                
                save_data = {
                    'timestamp': datetime.now().isoformat(),
                    'league': selected_league,
                    'home_team': prediction_data['home_team'],
                    'away_team': prediction_data['away_team'],
                    'score': score,
                    'predicted_winner': prediction_data['predicted_winner'],
                    'actual_winner': actual_winner,
                    'winner_correct': winner_correct,
                    'predicted_totals': prediction_data['predicted_totals'],
                    'actual_over_under': actual_over_under,
                    'totals_correct': totals_correct,
                    'sum_xg': float(prediction_data['sum_xg']),
                    'confidence': {
                        'winner': float(prediction_data['winner_confidence']),
                        'totals': float(prediction_data['totals_confidence'])
                    }
                }
                
                # Save to file
                with open("your_findings_predictions.json", "a") as f:
                    f.write(json.dumps(save_data) + "\n")
                
                # Show results
                if winner_correct and totals_correct:
                    st.balloons()
                    st.success(f"""
                    🎯 PERFECT PREDICTION!
                    
                    Winner: {prediction_data['predicted_winner']} ✓
                    Over/Under: {prediction_data['predicted_totals']} ✓
                    
                    Score: {score} ({actual_over_under})
                    """)
                elif winner_correct:
                    st.success(f"""
                    ✅ WINNER CORRECT!
                    
                    Winner: {prediction_data['predicted_winner']} ✓
                    Over/Under: {prediction_data['predicted_totals']} ✗ ({actual_over_under})
                    
                    Score: {score}
                    """)
                elif totals_correct:
                    st.success(f"""
                    ✅ TOTALS CORRECT!
                    
                    Winner: {prediction_data['predicted_winner']} ✗ ({actual_winner})
                    Over/Under: {prediction_data['predicted_totals']} ✓
                    
                    Score: {score}
                    """)
                else:
                    st.error(f"""
                    ❌ BOTH WRONG
                    
                    Winner: {prediction_data['predicted_winner']} ✗ ({actual_winner})
                    Over/Under: {prediction_data['predicted_totals']} ✗ ({actual_over_under})
                    
                    Score: {score}
                    """)
                
            except ValueError:
                st.error("Enter score like '2-1'")
        else:
            st.error("Enter valid score")

# Performance summary
st.divider()
st.subheader("📊 YOUR MODEL PERFORMANCE")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Statistical Foundation:**")
    st.write(f"• Home advantage: +{model.home_advantage_goals} goals")
    st.write(f"• Home wins: {model.home_win_rate*100:.1f}%")
    st.write(f"• League sweet spots applied")
    st.write(f"• Data: 1,049 matches analyzed")

with col2:
    st.write("**Current Status:**")
    st.write(f"• Baseline: {model.baseline_home_accuracy}%")
    st.write(f"• Current model: {model.current_model_accuracy}%")
    st.write(f"• Expected: 57.5%+")
    st.write(f"• Immediate gain: +{model.baseline_home_accuracy - model.current_model_accuracy:.1f}%")

with col3:
    st.write("**Key Insights Applied:**")
    st.write("1. Home advantage is REAL")
    st.write("2. But NOT absolute")
    st.write("3. League-specific thresholds")
    st.write("4. Simple > Complex")

# Footer
st.divider()
st.caption(f"🎯 YOUR ACTUAL STATISTICAL FINDINGS IMPLEMENTED | Baseline: {model.baseline_home_accuracy}% | League: {selected_league} | Sweet spots: UNDER<{model.sweet_spots['under']}, OVER>{model.sweet_spots['over']}")
