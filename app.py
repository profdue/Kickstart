import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="⚽ LEAGUE-SPECIFIC SWEET SPOT MODEL",
    page_icon="🎯",
    layout="wide"
)

st.title("⚽ LEAGUE-SPECIFIC SWEET SPOT MODEL")
st.markdown("""
    **USING YOUR ACTUAL LEAGUE-SPECIFIC FINDINGS** - Premier League, Serie A, Ligue 1 sweet spots
""")

# ========== YOUR ACTUAL LEAGUE-SPECIFIC FINDINGS ==========

class LeagueSweetSpotModel:
    """ACTUALLY uses your league-specific sweet spot findings"""
    
    LEAGUE_SWEET_SPOTS = {
        # YOUR ACTUAL FINDINGS:
        "Premier League": {
            "under_sweet_spot": 2.5,  # sum_xg < 2.5 = HIGH probability UNDER
            "over_sweet_spot": 3.0,   # sum_xg > 3.0 = HIGH probability OVER
            "intermediate_range": (2.5, 3.0),  # Uncertainty zone
            "notes": "Similar to Ligue 1 pattern"
        },
        "Serie A": {
            "under_sweet_spot": 2.8,  # sum_xg < 2.8 = HIGH probability UNDER
            "over_sweet_spot": 3.5,   # sum_xg > 3.5 = HIGH probability OVER
            "intermediate_range": (2.8, 3.5),  # Uncertainty zone
            "notes": "More defensive, higher thresholds needed"
        },
        "Ligue 1": {
            "under_sweet_spot": 2.5,  # sum_xg < 2.5 = HIGH probability UNDER
            "over_sweet_spot": 3.0,   # sum_xg > 3.0 = HIGH probability OVER
            "intermediate_range": (2.5, 3.0),  # Uncertainty zone
            "notes": "Similar to Premier League pattern"
        },
        # For leagues you haven't analyzed yet, use conservative defaults
        "Bundesliga": {
            "under_sweet_spot": 2.6,  # Conservative estimate
            "over_sweet_spot": 3.2,
            "intermediate_range": (2.6, 3.2),
            "notes": "Estimated - needs analysis"
        },
        "La Liga": {
            "under_sweet_spot": 2.6,
            "over_sweet_spot": 3.2,
            "intermediate_range": (2.6, 3.2),
            "notes": "Estimated - needs analysis"
        },
        "Eredivisie": {
            "under_sweet_spot": 2.7,  # Higher scoring league estimate
            "over_sweet_spot": 3.3,
            "intermediate_range": (2.7, 3.3),
            "notes": "Estimated high-scoring league"
        },
        "RFPL": {
            "under_sweet_spot": 2.5,
            "over_sweet_spot": 3.0,
            "intermediate_range": (2.5, 3.0),
            "notes": "Estimated"
        }
    }
    
    def __init__(self, league_name):
        self.league_name = league_name
        self.sweet_spots = self.LEAGUE_SWEET_SPOTS.get(league_name, self.LEAGUE_SWEET_SPOTS["Premier League"])
        self.model_type = "LEAGUE_SPECIFIC_SWEET_SPOTS"
        self.version = "v6.0_actual_findings"
        
    def predict_totals(self, sum_xg):
        """
        YOUR ACTUAL METHOD: Apply league-specific sweet spots
        """
        under_threshold = self.sweet_spots["under_sweet_spot"]
        over_threshold = self.sweet_spots["over_sweet_spot"]
        
        # Apply YOUR league-specific sweet spots
        if sum_xg < under_threshold:
            # In UNDER sweet spot for this league
            confidence = self._calculate_league_confidence(sum_xg, "under")
            return "UNDER", confidence, f"{self.league_name} UNDER sweet spot: {sum_xg:.1f}<{under_threshold}"
        
        elif sum_xg > over_threshold:
            # In OVER sweet spot for this league
            confidence = self._calculate_league_confidence(sum_xg, "over")
            return "OVER", confidence, f"{self.league_name} OVER sweet spot: {sum_xg:.1f}>{over_threshold}"
        
        else:
            # Intermediate range - use distribution
            range_mid = (under_threshold + over_threshold) / 2
            
            if sum_xg < range_mid:
                # Closer to under threshold
                confidence = 55
                return "UNDER", confidence, f"Intermediate range: {sum_xg:.1f} closer to under ({under_threshold})"
            else:
                # Closer to over threshold
                confidence = 55
                return "OVER", confidence, f"Intermediate range: {sum_xg:.1f} closer to over ({over_threshold})"
    
    def _calculate_league_confidence(self, sum_xg, prediction_type):
        """Calculate confidence based on distance from threshold"""
        if prediction_type == "under":
            threshold = self.sweet_spots["under_sweet_spot"]
            # How far below the threshold are we?
            distance_below = (threshold - sum_xg) / threshold
            
            if distance_below > 0.3:  # Well below threshold
                return 75
            elif distance_below > 0.15:
                return 65
            else:
                return 60
        
        else:  # "over"
            threshold = self.sweet_spots["over_sweet_spot"]
            # How far above the threshold are we?
            if threshold == 0:
                return 65
            distance_above = (sum_xg - threshold) / threshold
            
            if distance_above > 0.3:  # Well above threshold
                return 75
            elif distance_above > 0.15:
                return 65
            else:
                return 60
    
    def get_league_analysis(self):
        """Display YOUR league-specific findings"""
        return {
            "league": self.league_name,
            "under_sweet_spot": self.sweet_spots["under_sweet_spot"],
            "over_sweet_spot": self.sweet_spots["over_sweet_spot"],
            "intermediate_range": f"{self.sweet_spots['intermediate_range'][0]}-{self.sweet_spots['intermediate_range'][1]}",
            "notes": self.sweet_spots["notes"],
            "analysis_basis": "YOUR actual statistical findings from data analysis"
        }

# ========== LEAGUE COMPARISON VISUALIZER ==========

def create_league_comparison_chart():
    """Visualize YOUR league-specific sweet spots"""
    leagues = ["Premier League", "Serie A", "Ligue 1", "Bundesliga", "La Liga"]
    data = []
    
    for league in leagues:
        model = LeagueSweetSpotModel(league)
        spots = model.sweet_spots
        
        data.append({
            'League': league,
            'Under Threshold': spots['under_sweet_spot'],
            'Over Threshold': spots['over_sweet_spot'],
            'Range Width': spots['over_sweet_spot'] - spots['under_sweet_spot']
        })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot thresholds
    x = range(len(leagues))
    ax.bar(x, df['Over Threshold'], color='red', alpha=0.6, label='Over Threshold')
    ax.bar(x, df['Under Threshold'], color='blue', alpha=0.6, label='Under Threshold')
    
    # Add range lines
    for i, row in df.iterrows():
        ax.plot([i, i], [row['Under Threshold'], row['Over Threshold']], 
                'k-', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('League')
    ax.set_ylabel('sum_of_avg_xg Threshold')
    ax.set_title('YOUR LEAGUE-SPECIFIC SWEET SPOTS')
    ax.set_xticks(x)
    ax.set_xticklabels(leagues, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    for i, row in df.iterrows():
        ax.text(i, row['Over Threshold'] + 0.05, f"{row['Over Threshold']}", 
                ha='center', fontsize=9)
        ax.text(i, row['Under Threshold'] - 0.1, f"{row['Under Threshold']}", 
                ha='center', fontsize=9)
    
    return fig

# ========== DATA LOADING ==========

@st.cache_data
def load_league_data(league_name):
    """Load league data and calculate xG metrics"""
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
        
        # Calculate YOUR metrics
        df['xg_per_match'] = df['xg'] / df['matches']
        df['xga_per_match'] = df['xga'] / df['matches']
        df['goals_per_match'] = df['gf'] / df['matches']
        
        return df
    except Exception as e:
        st.error(f"Error loading {league_name}: {str(e)}")
        return None

def prepare_team_data(df):
    """Separate home and away data"""
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    return home_data.set_index('team'), away_data.set_index('team')

# ========== STREAMLIT UI ==========

with st.sidebar:
    st.header("⚙️ LEAGUE-SPECIFIC SWEET SPOTS")
    
    leagues = ["Premier League", "Serie A", "Ligue 1", "Bundesliga", "La Liga", "Eredivisie", "RFPL"]
    selected_league = st.selectbox("Select League", leagues)
    
    # Create model for selected league
    model = LeagueSweetSpotModel(selected_league)
    
    # Show league-specific findings
    st.divider()
    st.header("📊 YOUR LEAGUE-SPECIFIC FINDINGS")
    
    analysis = model.get_league_analysis()
    for key, value in analysis.items():
        if key != "analysis_basis":
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Load data
    df = load_league_data(selected_league)
    
    if df is not None:
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        if len(home_stats_df) > 0 and len(away_stats_df) > 0:
            home_teams = sorted(home_stats_df.index.unique())
            away_teams = sorted(away_stats_df.index.unique())
            common_teams = sorted(list(set(home_teams) & set(away_teams)))
            
            home_team = st.selectbox("Home Team", common_teams)
            away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
            
            st.divider()
            
            if st.button("🚀 Apply League-Specific Sweet Spots", type="primary"):
                calculate_btn = True
            else:
                calculate_btn = False

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# Show league comparison
st.markdown("""
<div style="background-color: #0C4A6E; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h3 style="color: white; text-align: center; margin: 0;">
        🔬 YOUR LEAGUE-SPECIFIC SWEET SPOT FINDINGS
    </h3>
    <p style="color: #E0F2FE; text-align: center; margin: 5px 0 0 0;">
        Premier League, Serie A, Ligue 1 analyzed • Distinct thresholds identified • League-specific strategies needed
    </p>
</div>
""", unsafe_allow_html=True)

# Show league comparison chart
st.subheader("📊 LEAGUE-SPECIFIC THRESHOLD COMPARISON")
fig = create_league_comparison_chart()
st.pyplot(fig)

st.caption("""
**YOUR KEY FINDINGS:** 
- **Serie A** requires higher sum_xg thresholds (more defensive)  
- **Premier League & Ligue 1** have similar patterns
- Each league has its own "sweet spots" for over/under predictions
""")

# Generate prediction
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Calculate YOUR key metric
        home_xg = home_stats['xg_per_match']
        away_xg = away_stats['xg_per_match']
        sum_xg = home_xg + away_xg
        
        # Get prediction using league-specific sweet spots
        predicted_totals, confidence, logic = model.predict_totals(sum_xg)
        
        # Simple winner prediction
        xg_diff = home_xg - away_xg
        if xg_diff > 0.3:
            predicted_winner = "HOME"
            winner_confidence = 60
            winner_logic = f"Home xG advantage: +{xg_diff:.2f}"
        elif xg_diff < -0.3:
            predicted_winner = "AWAY"
            winner_confidence = 60
            winner_logic = f"Away xG advantage: +{abs(xg_diff):.2f}"
        else:
            predicted_winner = "DRAW"
            winner_confidence = 50
            winner_logic = "Close xG match"
        
        # Store data
        prediction_data = {
            'league': selected_league,
            'home_team': home_team,
            'away_team': away_team,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'sum_xg': sum_xg,
            'predicted_winner': predicted_winner,
            'winner_confidence': winner_confidence,
            'winner_logic': winner_logic,
            'predicted_totals': predicted_totals,
            'totals_confidence': confidence,
            'totals_logic': logic,
            'league_thresholds': {
                'under': model.sweet_spots['under_sweet_spot'],
                'over': model.sweet_spots['over_sweet_spot']
            }
        }
        
        st.session_state.prediction_data = prediction_data
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Display prediction
if 'prediction_data' not in st.session_state:
    st.info("👈 Select teams to apply YOUR league-specific sweet spots")
    st.stop()

prediction_data = st.session_state.prediction_data

st.divider()
st.header(f"🎯 {prediction_data['home_team']} vs {prediction_data['away_team']}")
st.caption(f"League: {selected_league} • Applying YOUR league-specific sweet spots")

# Show league-specific application
with st.expander("🔬 HOW YOUR LEAGUE-SPECIFIC FINDINGS ARE APPLIED"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Your League Analysis:**")
        st.write(f"League: {selected_league}")
        st.write(f"Under threshold: {prediction_data['league_thresholds']['under']}")
        st.write(f"Over threshold: {prediction_data['league_thresholds']['over']}")
        st.write(f"Intermediate range: {prediction_data['league_thresholds']['under']}-{prediction_data['league_thresholds']['over']}")
        
        # Show which other leagues have similar patterns
        similar_leagues = []
        for league in ["Premier League", "Serie A", "Ligue 1"]:
            if league != selected_league:
                other_model = LeagueSweetSpotModel(league)
                if (other_model.sweet_spots['under_sweet_spot'] == prediction_data['league_thresholds']['under'] and
                    other_model.sweet_spots['over_sweet_spot'] == prediction_data['league_thresholds']['over']):
                    similar_leagues.append(league)
        
        if similar_leagues:
            st.write(f"Similar pattern to: {', '.join(similar_leagues)}")
    
    with col2:
        st.write("**This Match's Metrics:**")
        st.write(f"Home xG/match: {prediction_data['home_xg']:.2f}")
        st.write(f"Away xG/match: {prediction_data['away_xg']:.2f}")
        st.write(f"**sum_of_avg_xg: {prediction_data['sum_xg']:.2f}**")
        
        # Show position relative to thresholds
        under_thresh = prediction_data['league_thresholds']['under']
        over_thresh = prediction_data['league_thresholds']['over']
        
        if prediction_data['sum_xg'] < under_thresh:
            st.success(f"✅ Below UNDER threshold ({prediction_data['sum_xg']:.1f} < {under_thresh})")
            st.write("**Your finding:** HIGH probability UNDER")
        elif prediction_data['sum_xg'] > over_thresh:
            st.success(f"✅ Above OVER threshold ({prediction_data['sum_xg']:.1f} > {over_thresh})")
            st.write("**Your finding:** HIGH probability OVER")
        else:
            st.warning(f"⚠️ In intermediate range")
            st.write(f"({under_thresh} < {prediction_data['sum_xg']:.1f} < {over_thresh})")
            st.write("**Your finding:** Less certain prediction")

# Prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction_data['predicted_winner']
    winner_conf = prediction_data['winner_confidence']
    winner_logic = prediction_data['winner_logic']
    
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
    </div>
    """, unsafe_allow_html=True)

with col2:
    totals_pred = prediction_data['predicted_totals']
    totals_conf = prediction_data['totals_confidence']
    totals_logic = prediction_data['totals_logic']
    sum_xg = prediction_data['sum_xg']
    under_thresh = prediction_data['league_thresholds']['under']
    over_thresh = prediction_data['league_thresholds']['over']
    
    # Determine color based on position
    if sum_xg < under_thresh:
        color = "#3B82F6"  # Blue for under sweet spot
        spot_type = "UNDER SWEET SPOT"
    elif sum_xg > over_thresh:
        color = "#EF4444"  # Red for over sweet spot
        spot_type = "OVER SWEET SPOT"
    else:
        color = "#F59E0B"  # Yellow for intermediate
        spot_type = "INTERMEDIATE RANGE"
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">LEAGUE-SPECIFIC PREDICTION</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {'📉' if totals_pred == 'UNDER' else '📈'} {totals_pred} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {totals_conf:.0f}%
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            {spot_type}
        </div>
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            {totals_logic}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sweet spot analysis
st.divider()
st.subheader("🎯 LEAGUE-SPECIFIC SWEET SPOT ANALYSIS")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Threshold Comparison:**")
    
    # Compare to other leagues
    comparison_data = []
    for league in ["Premier League", "Serie A", "Ligue 1"]:
        if league != selected_league:
            comp_model = LeagueSweetSpotModel(league)
            comparison_data.append({
                'League': league,
                'Under': comp_model.sweet_spots['under_sweet_spot'],
                'Over': comp_model.sweet_spots['over_sweet_spot']
            })
    
    for comp in comparison_data:
        under_diff = prediction_data['league_thresholds']['under'] - comp['Under']
        over_diff = prediction_data['league_thresholds']['over'] - comp['Over']
        
        st.write(f"vs {comp['League']}:")
        st.write(f"  Under: {under_diff:+.1f}")
        st.write(f"  Over: {over_diff:+.1f}")

with col2:
    # Visual threshold position
    sum_xg = prediction_data['sum_xg']
    under = prediction_data['league_thresholds']['under']
    over = prediction_data['league_thresholds']['over']
    
    # Create a simple bar
    total_range = over - under
    if total_range > 0:
        position = (sum_xg - under) / total_range
        position = max(0, min(1, position))
    else:
        position = 0.5
    
    st.write("**Position in League Range:**")
    
    # Create visual bar
    bar_length = 20
    filled = int(position * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    st.write(f"Under ({under}) ┤{bar}┤ Over ({over})")
    st.write(f"Current: **{sum_xg:.2f}**")
    
    if position < 0.33:
        st.success("Near UNDER sweet spot")
    elif position > 0.66:
        st.success("Near OVER sweet spot")
    else:
        st.warning("In intermediate uncertainty range")

with col3:
    st.write("**Expected Match Characteristics:**")
    
    sum_xg = prediction_data['sum_xg']
    under = prediction_data['league_thresholds']['under']
    over = prediction_data['league_thresholds']['over']
    
    if sum_xg < under - 0.3:
        st.write("• Very defensive match expected")
        st.write("• Likely 0-1 or 1-1 scoreline")
        st.write("• Low chance of many goals")
    elif sum_xg < under:
        st.write("• Defensive match")
        st.write("• Probably UNDER 2.5")
        st.write("• 1-2 goals likely")
    elif sum_xg > over + 0.3:
        st.write("• High-scoring match expected")
        st.write("• Likely 3+ goals")
        st.write("• Good chance for OVER 2.5")
    elif sum_xg > over:
        st.write("• Attacking match")
        st.write("• Probably OVER 2.5")
        st.write("• 3 goals possible")
    else:
        st.write("• Balanced match")
        st.write("• Could go either way")
        st.write("• 2-3 goals range")

# Data collection
st.divider()
st.subheader("📝 TRACK LEAGUE-SPECIFIC ACCURACY")

col1, col2 = st.columns([2, 1])

with col1:
    score = st.text_input("Actual Final Score", key="score_input")
    
    with st.expander("View League-Specific Logic"):
        st.write("**Your League-Specific Findings Applied:**")
        st.write(f"League: {selected_league}")
        st.write(f"Under threshold: {under}")
        st.write(f"Over threshold: {over}")
        st.write(f"This match's sum_xg: {sum_xg:.2f}")
        
        if sum_xg < under:
            st.write(f"**Decision:** sum_xg ({sum_xg:.2f}) < {under} → Predict UNDER")
        elif sum_xg > over:
            st.write(f"**Decision:** sum_xg ({sum_xg:.2f}) > {over} → Predict OVER")
        else:
            st.write(f"**Decision:** {under} < {sum_xg:.2f} < {over} → Use distribution")

with col2:
    if st.button("💾 Save & Track Accuracy", type="primary"):
        if score:
            try:
                home_goals, away_goals = map(int, score.split('-'))
                total_goals = home_goals + away_goals
                actual_over_under = "OVER" if total_goals > 2.5 else "UNDER"
                
                correct = actual_over_under == prediction_data['predicted_totals']
                
                # Save data
                save_data = {
                    'timestamp': datetime.now().isoformat(),
                    'league': selected_league,
                    'home_team': prediction_data['home_team'],
                    'away_team': prediction_data['away_team'],
                    'score': score,
                    'total_goals': total_goals,
                    'actual_over_under': actual_over_under,
                    'predicted_over_under': prediction_data['predicted_totals'],
                    'sum_xg': float(prediction_data['sum_xg']),
                    'league_under_threshold': float(under),
                    'league_over_threshold': float(over),
                    'confidence': float(prediction_data['totals_confidence']),
                    'correct': correct,
                    'sweet_spot_position': 'UNDER' if sum_xg < under else 'OVER' if sum_xg > over else 'INTERMEDIATE'
                }
                
                # Save to file
                with open("league_sweet_spot_predictions.json", "a") as f:
                    f.write(json.dumps(save_data) + "\n")
                
                # Show result
                if correct:
                    st.balloons()
                    st.success(f"""
                    ✅ CORRECT! 
                    
                    Predicted: {prediction_data['predicted_totals']}
                    Actual: {actual_over_under} ({score})
                    
                    Your {selected_league} sweet spot worked!
                    """)
                else:
                    st.error(f"""
                    ❌ INCORRECT
                    
                    Predicted: {prediction_data['predicted_totals']}
                    Actual: {actual_over_under} ({score})
                    
                    sum_xg: {sum_xg:.2f}
                    League thresholds: UNDER<{under}, OVER>{over}
                    """)
                
            except:
                st.error("Enter score like '2-1'")
        else:
            st.error("Enter actual score")

# League strategy recommendations
st.divider()
st.subheader("📈 YOUR LEAGUE-SPECIFIC BETTING STRATEGIES")

col1, col2 = st.columns(2)

with col1:
    st.write("**Based on YOUR Findings:**")
    
    if selected_league == "Serie A":
        st.write("""
        **Serie A Strategy:**
        • Higher thresholds needed (2.8/3.5)
        • More defensive league
        • Wait for clearer signals
        • Be patient with OVER bets
        """)
    elif selected_league in ["Premier League", "Ligue 1"]:
        st.write("""
        **Premier League/Ligue 1 Strategy:**
        • Lower thresholds (2.5/3.0)
        • More balanced league
        • Clearer sweet spots
        • Can be more aggressive
        """)
    else:
        st.write("""
        **General Strategy:**
        • Use league-specific thresholds
        • Track accuracy by league
        • Adjust based on results
        """)

with col2:
    st.write("**Key Takeaways from Your Analysis:**")
    st.write("1. **League-specific thresholds matter**")
    st.write("2. **Serie A is more defensive**")
    st.write("3. **Premier League & Ligue 1 are similar**")
    st.write("4. **Intermediate ranges are uncertain**")
    st.write("5. **Track performance by league**")

# Footer
st.divider()
st.caption(f"🎯 YOUR LEAGUE-SPECIFIC SWEET SPOT MODEL | {selected_league}: UNDER<{under}, OVER>{over} | Based on YOUR actual statistical findings")
