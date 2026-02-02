import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Football xG Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚽ Football Match Predictor")
st.markdown("""
    Predict match outcomes using Expected Goals (xG) regression analysis.
    This model adjusts for team over/underperformance and calculates probabilities using Poisson distribution.
""")

# Constants
MAX_GOALS = 8  # Maximum goals to calculate in Poisson distribution
REG_BASE_FACTOR = 0.75  # Base regression factor
REG_MATCH_THRESHOLD = 5  # Minimum matches for regression

# Cache factorial calculations for performance
@st.cache_data
def factorial_cache(n):
    return math.factorial(n)

# Manual Poisson PMF function
def poisson_pmf(k, lam):
    """Calculate Poisson probability manually without scipy"""
    if lam <= 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

@st.cache_data(ttl=3600)
def load_league_data(league_name):
    """Load league data from CSV with caching"""
    try:
        file_path = f"leagues/{league_name}.csv"
        df = pd.read_csv(file_path)
        
        # Basic validation
        required_cols = ['Team', 'Home_xG', 'Home_xGA', 'Away_xG', 'Away_xGA', 
                        'Home_Goals_vs_xG', 'Away_Goals_vs_xG', 'Home_Matches', 'Away_Matches']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV missing required columns. Found: {df.columns.tolist()}")
            return None
            
        return df
    except FileNotFoundError:
        st.error(f"League file not found: leagues/{league_name}.csv")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_regression_factors(home_team_stats, away_team_stats, regression_factor):
    """Calculate attack regression factors"""
    # Note: CSV uses negative for overperformance, positive for underperformance
    home_matches = home_team_stats['Home_Matches']
    away_matches = away_team_stats['Away_Matches']
    
    # Home team's attack regression
    if home_matches >= REG_MATCH_THRESHOLD:
        home_attack_reg = (home_team_stats['Home_Goals_vs_xG'] / home_matches) * regression_factor
    else:
        home_attack_reg = 0
    
    # Away team's attack regression
    if away_matches >= REG_MATCH_THRESHOLD:
        away_attack_reg = (away_team_stats['Away_Goals_vs_xG'] / away_matches) * regression_factor
    else:
        away_attack_reg = 0
    
    return home_attack_reg, away_attack_reg

def calculate_expected_goals(home_stats, away_stats, home_attack_reg, away_attack_reg):
    """Calculate expected goals for both teams"""
    # Home team's expected goals = Away team's xGA per match adjusted by home attack regression
    away_xga_per_match = away_stats['Away_xGA'] / away_stats['Away_Matches']
    home_expected = away_xga_per_match * (1 + home_attack_reg)
    
    # Away team's expected goals = Home team's xGA per match adjusted by away attack regression
    home_xga_per_match = home_stats['Home_xGA'] / home_stats['Home_Matches']
    away_expected = home_xga_per_match * (1 + away_attack_reg)
    
    # Apply floor to prevent negative or very low expected goals
    home_expected = max(home_expected, 0.1)
    away_expected = max(away_expected, 0.1)
    
    return home_expected, away_expected

def create_probability_matrix(home_lam, away_lam, max_goals=MAX_GOALS):
    """Create probability matrix for all score combinations"""
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_home = poisson_pmf(i, home_lam)
            prob_away = poisson_pmf(j, away_lam)
            prob_matrix[i, j] = prob_home * prob_away
    
    return prob_matrix

def calculate_outcome_probabilities(prob_matrix):
    """Calculate home win, draw, and away win probabilities"""
    home_win = np.sum(np.triu(prob_matrix, k=1))  # Sum where home > away (excluding diagonal)
    draw = np.sum(np.diag(prob_matrix))  # Sum where home = away
    away_win = np.sum(np.tril(prob_matrix, k=-1))  # Sum where home < away
    
    # Normalize to ensure sum = 1 (accounting for floating point errors)
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    
    return home_win, draw, away_win

def calculate_betting_markets(prob_matrix):
    """Calculate betting market probabilities"""
    # Over/Under 2.5 goals
    over_25 = 0
    under_25 = 0
    
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            total_goals = i + j
            prob = prob_matrix[i, j]
            
            if total_goals > 2.5:
                over_25 += prob
            else:
                under_25 += prob
    
    # Both Teams to Score (BTTS)
    btts_yes = 0
    btts_no = 0
    
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            prob = prob_matrix[i, j]
            
            if i >= 1 and j >= 1:
                btts_yes += prob
            else:
                btts_no += prob
    
    return over_25, under_25, btts_yes, btts_no

def get_risk_flags(home_stats, away_stats, home_xg, away_xg):
    """Generate risk flags and warnings"""
    flags = []
    
    # Over/underperformance warnings
    home_perf = home_stats['Home_Goals_vs_xG'] / home_stats['Home_Matches']
    away_perf = away_stats['Away_Goals_vs_xG'] / away_stats['Away_Matches']
    
    if abs(home_perf) > 0.3:
        flags.append(f"⚠️ Home team {'over' if home_perf < 0 else 'under'}performing by {abs(home_perf):.2f} goals/match")
    
    if abs(away_perf) > 0.3:
        flags.append(f"⚠️ Away team {'over' if away_perf < 0 else 'under'}performing by {abs(away_perf):.2f} goals/match")
    
    # Home/away form disparity
    home_home_ppg = home_stats['Home_Points'] / home_stats['Home_Matches'] if 'Home_Points' in home_stats else 0
    away_away_ppg = away_stats['Away_Points'] / away_stats['Away_Matches'] if 'Away_Points' in away_stats else 0
    
    if abs(home_home_ppg - away_away_ppg) > 1.0:
        flags.append(f"⚠️ Significant home/away form disparity: {home_home_ppg:.1f} vs {away_away_ppg:.1f} PPG")
    
    # High/low scoring match flags
    total_xg = home_xg + away_xg
    if total_xg > 3.0:
        flags.append("⚡ High-scoring match expected (Total xG > 3.0)")
    elif total_xg < 2.0:
        flags.append("🛡️ Low-scoring match expected (Total xG < 2.0)")
    
    # Match sample size warnings
    if home_stats['Home_Matches'] < 5:
        flags.append("📊 Small sample size for home team home stats")
    if away_stats['Away_Matches'] < 5:
        flags.append("📊 Small sample size for away team away stats")
    
    return flags

def get_betting_suggestions(home_win_prob, draw_prob, away_win_prob, over_25_prob, under_25_prob, btts_yes_prob):
    """Generate betting suggestions based on probabilities"""
    suggestions = []
    threshold = 0.55  # 55% probability threshold for value
    
    # Moneyline suggestions
    if home_win_prob > threshold:
        suggestions.append(f"✅ Home Win ({(home_win_prob*100):.1f}%)")
    if away_win_prob > threshold:
        suggestions.append(f"✅ Away Win ({(away_win_prob*100):.1f}%)")
    if draw_prob > threshold:
        suggestions.append(f"✅ Draw ({(draw_prob*100):.1f}%)")
    
    # Double chance
    home_draw_prob = home_win_prob + draw_prob
    away_draw_prob = away_win_prob + draw_prob
    if home_draw_prob > threshold:
        suggestions.append(f"✅ Home Win or Draw ({(home_draw_prob*100):.1f}%)")
    if away_draw_prob > threshold:
        suggestions.append(f"✅ Away Win or Draw ({(away_draw_prob*100):.1f}%)")
    
    # Over/Under
    if over_25_prob > threshold:
        suggestions.append(f"✅ Over 2.5 Goals ({(over_25_prob*100):.1f}%)")
    if under_25_prob > threshold:
        suggestions.append(f"✅ Under 2.5 Goals ({(under_25_prob*100):.1f}%)")
    
    # BTTS
    if btts_yes_prob > threshold:
        suggestions.append(f"✅ Both Teams to Score ({(btts_yes_prob*100):.1f}%)")
    elif btts_yes_prob < (1 - threshold):
        suggestions.append(f"❌ Both Teams NOT to Score ({((1-btts_yes_prob)*100):.1f}%)")
    
    return suggestions

# ========== SIDEBAR CONTROLS ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    # League selection
    leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Eredivisie"]
    selected_league = st.selectbox("Select League", leagues)
    
    # Load league data
    df = load_league_data(selected_league.lower().replace(" ", "_"))
    
    if df is not None:
        # Team selection
        teams = sorted(df['Team'].unique())
        home_team = st.selectbox("Home Team", teams)
        away_team = st.selectbox("Away Team", [t for t in teams if t != home_team])
        
        # Regression factor slider
        regression_factor = st.slider(
            "Regression Factor",
            min_value=0.0,
            max_value=2.0,
            value=REG_BASE_FACTOR,
            step=0.05,
            help="Adjust how much to regress team performance to mean (higher = more regression)"
        )
        
        # Calculate button
        calculate_btn = st.button("🎯 Calculate Predictions", type="primary", use_container_width=True)
        
        st.divider()
        
        # Display settings
        st.subheader("📊 Display Options")
        show_details = st.checkbox("Show Detailed Probabilities", value=True)
        show_matrix = st.checkbox("Show Score Probability Matrix", value=False)
        
        st.divider()
        
        # Export options
        st.subheader("📤 Export")
        if st.button("Generate Summary"):
            st.session_state.generate_summary = True

# ========== MAIN CONTENT ==========
if df is None:
    st.warning("Please select a valid league to continue")
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Calculate Predictions' to start")
    st.stop()

# Extract team stats
home_stats = df[df['Team'] == home_team].iloc[0]
away_stats = df[df['Team'] == away_team].iloc[0]

# ========== PHASE 1-3: Data Processing ==========
st.header(f"📊 {home_team} vs {away_team}")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader(f"🏠 {home_team}")
    st.metric("Home Matches", int(home_stats['Home_Matches']))
    st.metric("Home xG/match", f"{home_stats['Home_xG']/home_stats['Home_Matches']:.2f}")
    st.metric("Home xGA/match", f"{home_stats['Home_xGA']/home_stats['Home_Matches']:.2f}")
    if 'Home_Points' in home_stats:
        st.metric("Home PPG", f"{home_stats['Home_Points']/home_stats['Home_Matches']:.2f}")

with col2:
    st.subheader(f"✈️ {away_team}")
    st.metric("Away Matches", int(away_stats['Away_Matches']))
    st.metric("Away xG/match", f"{away_stats['Away_xG']/away_stats['Away_Matches']:.2f}")
    st.metric("Away xGA/match", f"{away_stats['Away_xGA']/away_stats['Away_Matches']:.2f}")
    if 'Away_Points' in away_stats:
        st.metric("Away PPG", f"{away_stats['Away_Points']/away_stats['Away_Matches']:.2f}")

with col3:
    # Calculate regression factors
    home_attack_reg, away_attack_reg = calculate_regression_factors(
        home_stats, away_stats, regression_factor
    )
    
    # Calculate expected goals
    home_xg, away_xg = calculate_expected_goals(
        home_stats, away_stats, home_attack_reg, away_attack_reg
    )
    
    st.subheader("🎯 Expected Goals")
    
    # Create a visual for expected goals
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(['Home xG', 'Away xG'], [home_xg, away_xg], 
                  color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax.set_ylabel('Expected Goals')
    ax.set_title('Match Expected Goals Distribution')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    total_xg = home_xg + away_xg
    st.metric("Total Expected Goals", f"{total_xg:.2f}")
    
    # Bias indicators
    if total_xg > 2.6:
        st.success(f"📈 Over bias: Total xG = {total_xg:.2f} > 2.6")
    elif total_xg < 2.3:
        st.info(f"📉 Under bias: Total xG = {total_xg:.2f} < 2.3")

# ========== PHASE 4: Poisson Probabilities ==========
st.divider()
st.header("📈 Probability Calculations")

# Create probability matrix
prob_matrix = create_probability_matrix(home_xg, away_xg)

# Calculate outcome probabilities
home_win_prob, draw_prob, away_win_prob = calculate_outcome_probabilities(prob_matrix)

# Calculate betting markets
over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob = calculate_betting_markets(prob_matrix)

# ========== LAYER 3: Score Probabilities ==========
with st.expander("🎯 Most Likely Scores", expanded=True):
    # Get top 5 most likely scores
    score_probs = []
    for i in range(min(6, prob_matrix.shape[0])):
        for j in range(min(6, prob_matrix.shape[1])):
            prob = prob_matrix[i, j]
            if prob > 0.001:  # Only include meaningful probabilities
                score_probs.append(((i, j), prob))
    
    # Sort by probability
    score_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Display top scores
    cols = st.columns(5)
    for idx, ((home_goals, away_goals), prob) in enumerate(score_probs[:5]):
        with cols[idx]:
            st.metric(
                label=f"{home_goals}-{away_goals}",
                value=f"{prob*100:.1f}%",
                delta="Most Likely" if idx == 0 else None
            )
    
    # Show most likely score prominently
    if score_probs:
        most_likely_score, most_likely_prob = score_probs[0]
        st.success(f"**Most Likely Score:** {most_likely_score[0]}-{most_likely_score[1]} ({(most_likely_prob*100):.1f}%)")

# ========== LAYER 4: Outcome Probabilities ==========
with st.expander("📊 Match Outcome Probabilities", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Home Win", f"{home_win_prob*100:.1f}%")
        st.progress(home_win_prob)
    
    with col2:
        st.metric("Draw", f"{draw_prob*100:.1f}%")
        st.progress(draw_prob)
    
    with col3:
        st.metric("Away Win", f"{away_win_prob*100:.1f}%")
        st.progress(away_win_prob)
    
    # Bar chart visualization
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    outcomes = ['Home Win', 'Draw', 'Away Win']
    probs = [home_win_prob, draw_prob, away_win_prob]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    bars = ax2.bar(outcomes, probs, color=colors, alpha=0.7)
    ax2.set_ylabel('Probability')
    ax2.set_title('Match Outcome Probabilities')
    ax2.set_ylim([0, 1])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height*100:.1f}%', ha='center', va='bottom')
    
    st.pyplot(fig2)

# ========== LAYER 5: Betting Markets ==========
with st.expander("💰 Betting Markets", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Over/Under 2.5 Goals")
        st.metric("Over 2.5", f"{over_25_prob*100:.1f}%")
        st.progress(over_25_prob)
        st.metric("Under 2.5", f"{under_25_prob*100:.1f}%")
        st.progress(under_25_prob)
    
    with col2:
        st.subheader("Both Teams to Score")
        st.metric("Yes", f"{btts_yes_prob*100:.1f}%")
        st.progress(btts_yes_prob)
        st.metric("No", f"{btts_no_prob*100:.1f}%")
        st.progress(btts_no_prob)
    
    # Moneyline odds
    st.subheader("Implied Odds")
    col_odds1, col_odds2, col_odds3 = st.columns(3)
    with col_odds1:
        if home_win_prob > 0:
            odds = 1 / home_win_prob
            st.metric("Home Win Odds", f"{odds:.2f}")
    
    with col_odds2:
        if draw_prob > 0:
            odds = 1 / draw_prob
            st.metric("Draw Odds", f"{odds:.2f}")
    
    with col_odds3:
        if away_win_prob > 0:
            odds = 1 / away_win_prob
            st.metric("Away Win Odds", f"{odds:.2f}")

# ========== LAYER 5: Betting Suggestions ==========
with st.expander("💡 Betting Suggestions", expanded=True):
    suggestions = get_betting_suggestions(
        home_win_prob, draw_prob, away_win_prob,
        over_25_prob, under_25_prob, btts_yes_prob
    )
    
    if suggestions:
        st.success("**Value Bets Found:**")
        for suggestion in suggestions:
            st.write(suggestion)
    else:
        st.info("No strong value bets identified (all probabilities < 55%)")
    
    # Double chance
    st.subheader("Double Chance")
    col_dc1, col_dc2 = st.columns(2)
    with col_dc1:
        home_draw_prob = home_win_prob + draw_prob
        st.metric("Home Win or Draw", f"{home_draw_prob*100:.1f}%")
    with col_dc2:
        away_draw_prob = away_win_prob + draw_prob
        st.metric("Away Win or Draw", f"{away_draw_prob*100:.1f}%")

# ========== LAYER 6: Risk Flags ==========
with st.expander("⚠️ Risk Flags & Warnings", expanded=False):
    flags = get_risk_flags(home_stats, away_stats, home_xg, away_xg)
    
    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.success("No significant risk flags identified")

# ========== OUTPUT FORMATS ==========
st.divider()
st.header("📤 Export & Share")

col_export1, col_export2 = st.columns(2)

with col_export1:
    # Generate summary
    if st.button("📋 Generate Summary Text"):
        summary = f"""
        ⚽ PREDICTION SUMMARY: {home_team} vs {away_team}
        
        📊 Expected Goals:
        • Home xG: {home_xg:.2f}
        • Away xG: {away_xg:.2f}
        • Total xG: {home_xg + away_xg:.2f}
        
        🎯 Most Likely Score: {score_probs[0][0][0] if score_probs else 'N/A'}-{score_probs[0][0][1] if score_probs else 'N/A'} ({(score_probs[0][1]*100 if score_probs else 0):.1f}%)
        
        📈 Outcome Probabilities:
        • Home Win: {home_win_prob*100:.1f}%
        • Draw: {draw_prob*100:.1f}%
        • Away Win: {away_win_prob*100:.1f}%
        
        💰 Betting Markets:
        • Over 2.5: {over_25_prob*100:.1f}%
        • Under 2.5: {under_25_prob*100:.1f}%
        • BTTS Yes: {btts_yes_prob*100:.1f}%
        """
        
        st.code(summary, language="text")
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name=f"prediction_{home_team}_vs_{away_team}.txt",
            mime="text/plain"
        )

with col_export2:
    # Export data as CSV
    if st.button("📊 Export Prediction Data"):
        export_data = {
            'Metric': [
                'Home Team', 'Away Team', 'Home xG', 'Away xG', 'Total xG',
                'Home Win %', 'Draw %', 'Away Win %',
                'Over 2.5 %', 'Under 2.5 %', 'BTTS Yes %', 'BTTS No %'
            ],
            'Value': [
                home_team, away_team, f"{home_xg:.2f}", f"{away_xg:.2f}", f"{home_xg+away_xg:.2f}",
                f"{home_win_prob*100:.1f}", f"{draw_prob*100:.1f}", f"{away_win_prob*100:.1f}",
                f"{over_25_prob*100:.1f}", f"{under_25_prob*100:.1f}",
                f"{btts_yes_prob*100:.1f}", f"{btts_no_prob*100:.1f}"
            ]
        }
        
        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"prediction_data_{home_team}_vs_{away_team}.csv",
            mime="text/csv"
        )

# ========== DETAILED PROBABILITY MATRIX ==========
if show_matrix:
    with st.expander("🔢 Detailed Probability Matrix", expanded=False):
        # Display matrix for scores 0-5
        matrix_df = pd.DataFrame(
            prob_matrix[:6, :6] * 100,
            columns=[f'Away {i}' for i in range(6)],
            index=[f'Home {i}' for i in range(6)]
        )
        
        # Format as percentages
        matrix_df = matrix_df.round(1)
        
        # Apply heatmap styling
        st.dataframe(
            matrix_df.style.background_gradient(cmap='Blues', axis=None).format('{:.1f}%'),
            use_container_width=True
        )

# ========== FOOTER ==========
st.divider()
st.caption(f"⚡ Predictions calculated using xG regression model | Regression factor: {regression_factor} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
