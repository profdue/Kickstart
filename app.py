import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Set page config
st.set_page_config(
    page_title="xG Prediction App",
    page_icon="⚽",
    layout="wide"
)

# Title
st.title("⚽ xG Football Match Predictor")
st.markdown("Using Understat xG regression method for match predictions")

# Sidebar for controls
with st.sidebar:
    st.header("📊 Settings")
    
    # League selection
    leagues = {
        'Premier League': 'premier_league.csv',
        'La Liga': 'laliga.csv',
        'Bundesliga': 'bundesliga.csv',
        'Serie A': 'seriea.csv',
        'Ligue 1': 'ligue_1.csv'
    }
    
    selected_league = st.selectbox(
        "Select League",
        list(leagues.keys())
    )
    
    # Load data
    @st.cache_data
    def load_league_data(league_file):
        try:
            df = pd.read_csv(f'leagues/{league_file}')
            return df
        except:
            st.error(f"Could not load {league_file}")
            return None
    
    df = load_league_data(leagues[selected_league])
    
    if df is not None:
        # Get unique teams
        home_teams = df[df['venue'] == 'home']['team'].unique()
        away_teams = df[df['venue'] == 'away']['team'].unique()
        
        # Team selection
        home_team = st.selectbox("Home Team", sorted(home_teams))
        away_team = st.selectbox("Away Team", sorted(away_teams))
        
        st.divider()
        st.markdown("### ⚙️ Prediction Settings")
        
        # Regression factor adjustment
        regression_factor = st.slider(
            "Regression Factor Strength",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            help="How strongly to apply regression to mean (0.1 = weak, 1.0 = strong)"
        )
        
        # Poisson max goals
        max_goals = st.slider(
            "Max Goals for Probability Matrix",
            min_value=5,
            max_value=10,
            value=7,
            help="Maximum number of goals to calculate probabilities for"
        )
        
        calculate_btn = st.button("🚀 Calculate Prediction", type="primary")

# Main content area
if df is not None and 'calculate_btn' in locals() and calculate_btn:
    if home_team == away_team:
        st.error("Home and Away teams cannot be the same!")
    else:
        # Get team stats
        home_stats = df[(df['team'] == home_team) & (df['venue'] == 'home')].iloc[0]
        away_stats = df[(df['team'] == away_team) & (df['venue'] == 'away')].iloc[0]
        
        # Create columns for layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.subheader(f"🏠 {home_team} vs {away_team} 🏃‍♂️")
            st.caption(f"{selected_league} • Home/Away Form")
        
        # Step 1: Display basic stats
        with st.expander("📈 Team Statistics", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{home_team} (Home)", 
                         f"{home_stats['wins']}-{home_stats['draws']}-{home_stats['losses']}",
                         f"xG: {home_stats['xg']:.2f}, xGA: {home_stats['xga']:.2f}")
                st.progress(home_stats['wins'] / home_stats['matches'])
                
            with col2:
                st.metric(f"{away_team} (Away)", 
                         f"{away_stats['wins']}-{away_stats['draws']}-{away_stats['losses']}",
                         f"xG: {away_stats['xg']:.2f}, xGA: {away_stats['xga']:.2f}")
                st.progress(away_stats['wins'] / away_stats['matches'])
        
        # Step 2: Calculate per-match averages
        home_xg_per_match = home_stats['xg'] / home_stats['matches']
        home_xga_per_match = home_stats['xga'] / home_stats['matches']
        away_xg_per_match = away_stats['xg'] / away_stats['matches']
        away_xga_per_match = away_stats['xga'] / away_stats['matches']
        
        # Step 3: Identify over/underperformance
        home_attack_regression = home_stats['goals_vs_xg'] / home_stats['matches'] * regression_factor
        home_defense_regression = home_stats['goals_allowed_vs_xga'] / home_stats['matches'] * regression_factor
        away_attack_regression = away_stats['goals_vs_xg'] / away_stats['matches'] * regression_factor
        away_defense_regression = away_stats['goals_allowed_vs_xga'] / away_stats['matches'] * regression_factor
        
        # Step 4: Adjust for home/away context
        home_expected = away_xga_per_match * (1 + home_attack_regression)
        away_expected = home_xga_per_match * (1 + away_attack_regression)
        
        # Step 5: Generate expected match goals
        home_final_expected = max(0.1, home_expected)  # Prevent negative
        away_final_expected = max(0.1, away_expected)
        
        # Display expected goals
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home_team} Expected Goals", f"{home_final_expected:.2f}")
        with col2:
            total_goals = home_final_expected + away_final_expected
            st.metric("Total Expected Goals", f"{total_goals:.2f}")
        with col3:
            st.metric(f"{away_team} Expected Goals", f"{away_final_expected:.2f}")
        
        # Step 6: Generate score probabilities using Poisson
        st.subheader("🎯 Score Probabilities")
        
        # Create probability matrix
        goals_range = range(max_goals + 1)
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in goals_range:
            for j in goals_range:
                prob = poisson.pmf(i, home_final_expected) * poisson.pmf(j, away_final_expected)
                prob_matrix[i, j] = prob
        
        # Get top 5 most likely scores
        scores = []
        for i in goals_range:
            for j in goals_range:
                scores.append({
                    'score': f"{i}-{j}",
                    'probability': prob_matrix[i, j] * 100,
                    'home_goals': i,
                    'away_goals': j
                })
        
        top_scores = sorted(scores, key=lambda x: x['probability'], reverse=True)[:5]
        
        # Display top scores
        cols = st.columns(5)
        for idx, score in enumerate(top_scores):
            with cols[idx]:
                st.metric(
                    score['score'],
                    f"{score['probability']:.1f}%",
                    delta="Most likely" if idx == 0 else None
                )
        
        # Step 7: Calculate match outcome probabilities
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for i in goals_range:
            for j in goals_range:
                prob = prob_matrix[i, j]
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # Display outcome probabilities
        st.subheader("🏆 Match Outcome Probabilities")
        
        # Visualize with bar chart using Streamlit's native chart
        outcome_data = pd.DataFrame({
            'Outcome': [f'{home_team} Win', 'Draw', f'{away_team} Win'],
            'Probability': [home_win_prob * 100, draw_prob * 100, away_win_prob * 100]
        })
        
        st.bar_chart(outcome_data.set_index('Outcome'))
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home_team} Win", f"{home_win_prob*100:.1f}%")
        with col2:
            st.metric("Draw", f"{draw_prob*100:.1f}%")
        with col3:
            st.metric(f"{away_team} Win", f"{away_win_prob*100:.1f}%")
        
        # Step 8: Betting suggestions
        st.subheader("💰 Betting Suggestions")
        
        # Calculate over/under probabilities
        under_25_prob = 0
        for i in goals_range:
            for j in goals_range:
                if i + j < 2.5:
                    under_25_prob += prob_matrix[i, j]
        
        over_25_prob = 1 - under_25_prob
        
        # BTTS probability
        btts_prob = 0
        for i in range(1, max_goals + 1):
            for j in range(1, max_goals + 1):
                btts_prob += prob_matrix[i, j]
        
        # Generate suggestions
        suggestions = []
        
        # 1x2 suggestions
        if home_win_prob > 0.55:
            decimal_odds = 1 / home_win_prob
            suggestions.append(f"**{home_team} to win** (if odds > {decimal_odds:.2f})")
        elif away_win_prob > 0.55:
            decimal_odds = 1 / away_win_prob
            suggestions.append(f"**{away_team} to win** (if odds > {decimal_odds:.2f})")
        
        # BTTS suggestion
        if btts_prob > 0.55:
            suggestions.append(f"**BTTS: Yes** ({btts_prob*100:.1f}% probability)")
        elif btts_prob < 0.45:
            suggestions.append(f"**BTTS: No** ({(1-btts_prob)*100:.1f}% probability)")
        
        # Over/Under 2.5
        if total_goals > 2.6:
            suggestions.append(f"**Over 2.5 goals** ({over_25_prob*100:.1f}% probability)")
        elif total_goals < 2.3:
            suggestions.append(f"**Under 2.5 goals** ({under_25_prob*100:.1f}% probability)")
        else:
            if over_25_prob > 0.52:
                suggestions.append(f"**Lean Over 2.5** ({over_25_prob*100:.1f}% probability)")
            elif under_25_prob > 0.52:
                suggestions.append(f"**Lean Under 2.5** ({under_25_prob*100:.1f}% probability)")
        
        # Double chance if close
        if 0.45 <= home_win_prob <= 0.55:
            suggestions.append(f"**{home_team} Double Chance** (Home win or draw: {(home_win_prob+draw_prob)*100:.1f}%)")
        
        # Display suggestions
        if suggestions:
            for suggestion in suggestions:
                st.write(f"✅ {suggestion}")
        else:
            st.info("No strong betting suggestions - match too close to call")
        
        # Step 9: Risk flags
        st.subheader("⚠️ Risk Flags")
        
        flags = []
        
        # High regression risk
        if abs(home_stats['goals_vs_xg'] / home_stats['matches']) > 0.3:
            flags.append(f"{home_team} massively {'over' if home_stats['goals_vs_xg'] > 0 else 'under'}performing xG")
        
        if abs(away_stats['goals_vs_xg'] / away_stats['matches']) > 0.3:
            flags.append(f"{away_team} massively {'over' if away_stats['goals_vs_xg'] > 0 else 'under'}performing xG")
        
        # Home/away form disparity
        home_win_rate = home_stats['wins'] / home_stats['matches']
        away_win_rate = away_stats['wins'] / away_stats['matches']
        
        if abs(home_win_rate - away_win_rate) > 0.4:
            flags.append("Large disparity between home and away form")
        
        # Low expected goals
        if total_goals < 2.0:
            flags.append("Low-scoring match expected")
        elif total_goals > 3.5:
            flags.append("High-scoring match expected")
        
        # Display flags
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.success("No major risk flags detected")
        
        # Step 10: Detailed probability table
        with st.expander("📊 View Detailed Probability Matrix"):
            # Create a smaller matrix for display
            display_goals = 5
            display_matrix = prob_matrix[:display_goals+1, :display_goals+1] * 100
            
            display_df = pd.DataFrame(
                display_matrix,
                columns=[f"Away {i}" for i in range(display_goals+1)],
                index=[f"Home {i}" for i in range(display_goals+1)]
            )
            
            st.dataframe(
                display_df.style.format("{:.1f}%").background_gradient(cmap='Blues'),
                use_container_width=True
            )
        
        # Export results
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Copy Prediction Summary"):
                summary = f"""
                {home_team} vs {away_team}
                Expected Goals: {home_team} {home_final_expected:.1f} - {away_final_expected:.1f} {away_team}
                Total Expected: {total_goals:.1f}
                
                Most Likely Score: {top_scores[0]['score']} ({top_scores[0]['probability']:.1f}%)
                
                Win Probabilities:
                {home_team}: {home_win_prob*100:.1f}%
                Draw: {draw_prob*100:.1f}%
                {away_team}: {away_win_prob*100:.1f}%
                
                Over 2.5: {over_25_prob*100:.1f}%
                Under 2.5: {under_25_prob*100:.1f}%
                BTTS: {btts_prob*100:.1f}%
                """
                st.code(summary)
        
        with col2:
            st.download_button(
                label="📥 Download Prediction Data",
                data=pd.DataFrame({
                    'Metric': ['Home xG', 'Away xG', 'Total xG', 'Home Win %', 'Draw %', 'Away Win %', 'Over 2.5 %', 'BTTS %'],
                    'Value': [home_final_expected, away_final_expected, total_goals, 
                             home_win_prob*100, draw_prob*100, away_win_prob*100,
                             over_25_prob*100, btts_prob*100]
                }).to_csv(index=False),
                file_name=f"prediction_{home_team}_vs_{away_team}.csv",
                mime="text/csv"
            )

else:
    # Initial state or no data
    st.info("👈 Select a league and teams from the sidebar, then click 'Calculate Prediction'")
    
    # Display sample prediction
    st.subheader("📋 How it works:")
    st.markdown("""
    1. **Select league** and teams from sidebar
    2. **xG regression** adjusts for over/underperformance
    3. **Poisson distribution** calculates score probabilities
    4. **Betting suggestions** based on value thresholds
    
    ### 🔍 Key Metrics:
    - **xG (Expected Goals)**: Quality of chances created
    - **xGA (Expected Goals Against)**: Quality of chances conceded
    - **xPTS (Expected Points)**: Points deserved based on xG
    
    ### ⚡ Regression Logic:
    - Teams overperforming xG → expect regression (fewer goals)
    - Teams underperforming xG → expect improvement (more goals)
    """)
    
    # Show data structure
    if df is not None:
        with st.expander("View Data Structure"):
            st.dataframe(df.head())
            st.caption(f"Total records: {len(df)} (20 teams × 2 venues = 40)")

# Footer
st.divider()
st.caption("📊 Data Source: Understat • ⚽ xG Regression Prediction Model • Made with Streamlit")
