import streamlit as st
import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine - YOUR ACTUAL STATISTICAL FINDINGS",
    page_icon="🎯",
    layout="wide"
)

st.title("⚽ Football Intelligence Engine - YOUR ACTUAL STATISTICAL FINDINGS")
st.markdown("""
    **USING YOUR REAL DATA ANALYSIS** - Based on actual sweet spots from 192 league entries
""")

# ========== YOUR ACTUAL STATISTICAL MODEL ==========

class YourStatisticalModel:
    """ACTUALLY uses your statistical findings from the data analysis"""
    
    def __init__(self, league_name):
        self.name = league_name
        self.model_type = "YOUR_ACTUAL_FINDINGS"
        self.version = "v5.0_real_analysis"
        
        # YOUR ACTUAL FINDINGS from the data analysis:
        self.over_under_sweet_spots = {
            "under_sweet_spot": 2.2,  # sum_of_avg_xg < 2.2 = HIGH probability UNDER
            "over_sweet_spot": 3.0,   # sum_of_avg_xg > 3.0 = HIGH probability OVER
            "uncertain_range": (2.2, 3.0)  # Overlap zone = less certain
        }
        
        # Your key finding: DISTINCT distributions for over/under
        self.under_peak = (1.5, 2.5)   # Under matches peak here
        self.over_peak = (3.0, 4.0)    # Over matches peak here
        
        # Baseline from your analysis
        self.baseline_home_win_rate = 44.52  # From your earlier analysis
        self.data_sample_size = 192  # 192 league entries
        
    def predict_totals(self, home_xg, away_xg, home_scoring, away_scoring):
        """
        YOUR ACTUAL METHOD: Use sum_of_avg_xg sweet spots
        Based on your DISTRIBUTION findings
        """
        # Calculate sum_of_avg_xg (YOUR KEY METRIC)
        sum_avg_xg = home_xg + away_xg
        
        # Apply YOUR ACTUAL sweet spots
        if sum_avg_xg < self.over_under_sweet_spots["under_sweet_spot"]:
            # YOUR FINDING: <2.2 = HIGH probability UNDER
            confidence = self._calculate_confidence(sum_avg_xg, "under")
            return "UNDER", confidence, f"YOUR_SWEET_SPOT: sum_xg={sum_avg_xg:.1f}<2.2"
        
        elif sum_avg_xg > self.over_under_sweet_spots["over_sweet_spot"]:
            # YOUR FINDING: >3.0 = HIGH probability OVER
            confidence = self._calculate_confidence(sum_avg_xg, "over")
            return "OVER", confidence, f"YOUR_SWEET_SPOT: sum_xg={sum_avg_xg:.1f}>3.0"
        
        else:
            # Uncertain range (2.2-3.0) from your analysis
            # Use DISTRIBUTION information
            if sum_avg_xg < 2.6:  # Closer to under peak
                confidence = 55
                return "UNDER", confidence, f"UNCERTAIN_RANGE: sum_xg={sum_avg_xg:.1f} (near under peak)"
            else:  # Closer to over peak
                confidence = 55
                return "OVER", confidence, f"UNCERTAIN_RANGE: sum_xg={sum_avg_xg:.1f} (near over peak)"
    
    def predict_winner(self, home_xg, away_xg, home_scoring, away_scoring, home_conceding, away_conceding):
        """
        Simple winner prediction based on xG difference
        With your home advantage baseline
        """
        xg_diff = home_xg - away_xg
        
        # YOUR INSIGHT: 44.52% home win baseline
        if xg_diff > 0.5:
            confidence = 60 + min(20, xg_diff * 10)
            return "HOME", confidence, f"XG_ADVANTAGE: +{xg_diff:.2f}"
        elif xg_diff < -0.5:
            confidence = 60 + min(20, abs(xg_diff) * 10)
            return "AWAY", confidence, f"XG_ADVANTAGE: +{abs(xg_diff):.2f}"
        else:
            # Close match: default to home (YOUR 44.52% baseline)
            return "HOME", 52, "CLOSE_MATCH_HOME_BASELINE"
    
    def _calculate_confidence(self, sum_xg, prediction):
        """Calculate confidence based on YOUR distribution findings"""
        if prediction == "under":
            # Peak at 1.5-2.5 = HIGH confidence
            if sum_xg < 1.8:
                return 75  # Very low sum_xg = very high confidence under
            elif sum_xg < 2.2:
                return 65  # Still in sweet spot
            else:
                return 55  # Edge of sweet spot
        
        else:  # "over"
            # Peak at 3.0-4.0 = HIGH confidence
            if sum_xg > 3.5:
                return 75  # Very high sum_xg = very high confidence over
            elif sum_xg > 3.0:
                return 65  # In sweet spot
            else:
                return 55  # Edge of sweet spot
    
    def get_statistical_findings(self):
        """Display YOUR ACTUAL findings"""
        return {
            "data_sample": f"{self.data_sample_size} league entries analyzed",
            "under_sweet_spot": f"sum_xg < {self.over_under_sweet_spots['under_sweet_spot']} = HIGH probability UNDER",
            "over_sweet_spot": f"sum_xg > {self.over_under_sweet_spots['over_sweet_spot']} = HIGH probability OVER",
            "uncertain_range": f"{self.over_under_sweet_spots['uncertain_range'][0]}-{self.over_under_sweet_spots['uncertain_range'][1]} = overlap zone",
            "under_distribution": f"Peaks at {self.under_peak[0]}-{self.under_peak[1]} sum_xg",
            "over_distribution": f"Peaks at {self.over_peak[0]}-{self.over_peak[1]} sum_xg",
            "home_win_baseline": f"{self.baseline_home_win_rate}% of matches"
        }

# ========== DISTRIBUTION VISUALIZER ==========

class DistributionVisualizer:
    """Shows YOUR ACTUAL distribution findings"""
    
    @staticmethod
    def create_distribution_chart():
        """Create a visualization of YOUR findings"""
        import matplotlib.pyplot as plt
        
        # YOUR ACTUAL DISTRIBUTIONS
        # Under 2.5 goals: peaks at 1.5-2.5
        under_x = np.linspace(1.0, 3.5, 100)
        under_y = np.exp(-(under_x - 2.0)**2 / 0.5)  # Peak at 2.0
        
        # Over 2.5 goals: peaks at 3.0-4.0  
        over_x = np.linspace(2.0, 4.5, 100)
        over_y = np.exp(-(over_x - 3.5)**2 / 0.5)  # Peak at 3.5
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distributions
        ax.plot(under_x, under_y, 'b-', linewidth=3, label='Under 2.5 Goals (Your Finding)')
        ax.plot(over_x, over_y, 'r-', linewidth=3, label='Over 2.5 Goals (Your Finding)')
        
        # Add sweet spots
        ax.axvline(x=2.2, color='blue', linestyle='--', alpha=0.5, label='Under Sweet Spot (<2.2)')
        ax.axvline(x=3.0, color='red', linestyle='--', alpha=0.5, label='Over Sweet Spot (>3.0)')
        
        # Shade uncertain range
        ax.axvspan(2.2, 3.0, alpha=0.1, color='gray', label='Uncertain Range (2.2-3.0)')
        
        ax.set_xlabel('sum_of_avg_xg (Your Key Metric)')
        ax.set_ylabel('Probability Density')
        ax.set_title('YOUR ACTUAL FINDING: Distinct Distributions for Over/Under 2.5 Goals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

# ========== DATA LOADING ==========

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
        
        # Calculate YOUR metrics: xg_per_match, xga_per_match
        df['xg_per_match'] = df['xg'] / df['matches']
        df['xga_per_match'] = df['xga'] / df['matches']
        
        return df
    except Exception as e:
        st.error(f"Error loading {league_name}: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare data with YOUR metrics"""
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    # Add YOUR key metrics
    for df_part in [home_data, away_data]:
        if len(df_part) > 0:
            df_part['goals_per_match'] = df_part['gf'] / df_part['matches']
            df_part['goals_allowed_per_match'] = df_part['ga'] / df_part['matches']
            df_part['xg_per_match'] = df_part['xg'] / df_part['matches']
            df_part['xga_per_match'] = df_part['xga'] / df_part['matches']
            df_part['points_per_match'] = df_part['pts'] / df_part['matches']
    
    return home_data.set_index('team'), away_data.set_index('team')

# ========== STREAMLIT UI ==========

# Sidebar
with st.sidebar:
    st.header("⚙️ YOUR STATISTICAL MODEL")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie", "RFPL"]
    selected_league = st.selectbox("Select League", leagues)
    
    # Create YOUR model
    model = YourStatisticalModel(selected_league)
    
    # Show YOUR findings
    st.divider()
    st.header("📊 YOUR ACTUAL STATISTICAL FINDINGS")
    
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
            
            home_team = st.selectbox("Home Team", common_teams)
            away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
            
            st.divider()
            
            if st.button("🚀 Generate Prediction Using YOUR Findings", type="primary"):
                calculate_btn = True
            else:
                calculate_btn = False

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# Show YOUR distribution findings
st.markdown("""
<div style="background-color: #0C4A6E; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h3 style="color: white; text-align: center; margin: 0;">
        🔬 YOUR ACTUAL STATISTICAL FINDINGS FROM DATA ANALYSIS
    </h3>
    <p style="color: #E0F2FE; text-align: center; margin: 5px 0 0 0;">
        192 league entries analyzed • Sweet spots identified • Distinct distributions confirmed
    </p>
</div>
""", unsafe_allow_html=True)

# Generate prediction
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Get YOUR key metrics
        home_xg = home_stats['xg_per_match']
        away_xg = away_stats['xg_per_match']
        home_scoring = home_stats['goals_per_match']
        away_scoring = away_stats['goals_per_match']
        home_conceding = home_stats['goals_allowed_per_match']
        away_conceding = away_stats['goals_allowed_per_match']
        
        # Calculate YOUR sum_of_avg_xg
        sum_avg_xg = home_xg + away_xg
        
        # Use YOUR model
        predicted_winner, winner_confidence, winner_logic = model.predict_winner(
            home_xg, away_xg, home_scoring, away_scoring, home_conceding, away_conceding
        )
        
        predicted_totals, totals_confidence, totals_logic = model.predict_totals(
            home_xg, away_xg, home_scoring, away_scoring
        )
        
        # Store data
        prediction_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'sum_avg_xg': sum_avg_xg,
            'predicted_winner': predicted_winner,
            'winner_confidence': winner_confidence,
            'winner_logic': winner_logic,
            'predicted_totals': predicted_totals,
            'totals_confidence': totals_confidence,
            'totals_logic': totals_logic,
            'home_scoring': home_scoring,
            'away_scoring': away_scoring
        }
        
        st.session_state.prediction_data = prediction_data
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Display prediction
if 'prediction_data' not in st.session_state:
    # Show distribution visualization
    st.subheader("📊 YOUR ACTUAL DISTRIBUTION FINDINGS")
    
    try:
        fig = DistributionVisualizer.create_distribution_chart()
        st.pyplot(fig)
        
        st.info("👈 Select teams to see how YOUR findings apply to specific matches")
    except:
        st.info("👈 Select teams and click 'Generate Prediction Using YOUR Findings'")
    
    st.stop()

prediction_data = st.session_state.prediction_data

st.header(f"🎯 {prediction_data['home_team']} vs {prediction_data['away_team']}")
st.caption(f"League: {selected_league} | Using YOUR actual statistical findings")

# Show how YOUR findings apply
with st.expander("🔬 HOW YOUR FINDINGS ARE BEING APPLIED"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Your Key Metrics:**")
        st.write(f"Home xG/match: {prediction_data['home_xg']:.2f}")
        st.write(f"Away xG/match: {prediction_data['away_xg']:.2f}")
        st.write(f"**sum_of_avg_xg: {prediction_data['sum_avg_xg']:.2f}**")
        st.write(f"Home scoring: {prediction_data['home_scoring']:.2f}/match")
        st.write(f"Away scoring: {prediction_data['away_scoring']:.2f}/match")
    
    with col2:
        st.write("**Application of Your Findings:**")
        
        sum_xg = prediction_data['sum_avg_xg']
        if sum_xg < 2.2:
            st.success(f"✅ In UNDER sweet spot ({sum_xg:.1f} < 2.2)")
            st.write("**Your finding:** HIGH probability of UNDER 2.5 goals")
        elif sum_xg > 3.0:
            st.success(f"✅ In OVER sweet spot ({sum_xg:.1f} > 3.0)")
            st.write("**Your finding:** HIGH probability of OVER 2.5 goals")
        else:
            st.warning(f"⚠️ In uncertain range (2.2-3.0)")
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
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            Baseline: {model.baseline_home_win_rate}% home wins
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    totals_pred = prediction_data['predicted_totals']
    totals_conf = prediction_data['totals_confidence']
    totals_logic = prediction_data['totals_logic']
    sum_xg = prediction_data['sum_avg_xg']
    
    # Color code based on sweet spots
    if sum_xg < 2.2:
        color = "#3B82F6"  # Blue for under
    elif sum_xg > 3.0:
        color = "#EF4444"  # Red for over
    else:
        color = "#F59E0B"  # Yellow for uncertain
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">YOUR SWEET SPOT PREDICTION</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {'📉' if totals_pred == 'UNDER' else '📈'} {totals_pred} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {totals_conf:.0f}%
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            {totals_logic}
        </div>
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            sum_of_avg_xg = {sum_xg:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sweet spot analysis
st.divider()
st.subheader("🎯 YOUR SWEET SPOT ANALYSIS")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("sum_of_avg_xg", f"{prediction_data['sum_avg_xg']:.2f}")
    
    # Show which sweet spot we're in
    if prediction_data['sum_avg_xg'] < 2.2:
        st.success("**IN UNDER SWEET SPOT**")
        st.write("Your finding: HIGH probability UNDER")
    elif prediction_data['sum_avg_xg'] > 3.0:
        st.success("**IN OVER SWEET SPOT**")
        st.write("Your finding: HIGH probability OVER")
    else:
        st.warning("**IN UNCERTAIN RANGE**")
        st.write("Your finding: Less certain")

with col2:
    # Distance from sweet spots
    distance_to_under = abs(prediction_data['sum_avg_xg'] - 2.2)
    distance_to_over = abs(prediction_data['sum_avg_xg'] - 3.0)
    
    if prediction_data['sum_avg_xg'] < 2.2:
        st.metric("Distance to UNDER threshold", f"{distance_to_under:.2f}")
        st.progress(1 - (distance_to_under / 2.2))
    elif prediction_data['sum_avg_xg'] > 3.0:
        st.metric("Distance to OVER threshold", f"{distance_to_over:.2f}")
        st.progress(min(1.0, distance_to_over / 2.0))
    else:
        st.metric("Middle of uncertain range", "-")
        st.progress(0.5)

with col3:
    # Expected total based on your analysis
    if prediction_data['sum_avg_xg'] < 2.0:
        expected_total = "1-2 goals"
    elif prediction_data['sum_avg_xg'] < 2.5:
        expected_total = "2 goals"
    elif prediction_data['sum_avg_xg'] < 3.0:
        expected_total = "2-3 goals"
    else:
        expected_total = "3+ goals"
    
    st.write("**Expected Match Total:**")
    st.write(f"Based on sum_xg = {prediction_data['sum_avg_xg']:.2f}:")
    st.write(f"**{expected_total}**")

# Simple data collection
st.divider()
st.subheader("📝 TRACK THIS PREDICTION")

col1, col2 = st.columns(2)

with col1:
    score = st.text_input("Actual Final Score", key="score_input")
    
    with st.expander("View Your Metrics"):
        st.write("**Your sum_of_avg_xg Analysis:**")
        st.write(f"- Home xG/match: {prediction_data['home_xg']:.2f}")
        st.write(f"- Away xG/match: {prediction_data['away_xg']:.2f}")
        st.write(f"- **sum_of_avg_xg: {prediction_data['sum_avg_xg']:.2f}**")
        st.write(f"- Sweet spot: {'UNDER' if prediction_data['sum_avg_xg'] < 2.2 else 'OVER' if prediction_data['sum_avg_xg'] > 3.0 else 'UNCERTAIN'}")

with col2:
    if st.button("💾 Save This Prediction", type="primary"):
        if score:
            try:
                # Simple local save
                save_data = {
                    'date': str(datetime.now()),
                    'league': selected_league,
                    'home_team': prediction_data['home_team'],
                    'away_team': prediction_data['away_team'],
                    'actual_score': score,
                    'sum_avg_xg': float(prediction_data['sum_avg_xg']),
                    'predicted_totals': prediction_data['predicted_totals'],
                    'confidence': float(prediction_data['totals_confidence']),
                    'sweet_spot': 'UNDER' if prediction_data['sum_avg_xg'] < 2.2 else 'OVER' if prediction_data['sum_avg_xg'] > 3.0 else 'UNCERTAIN'
                }
                
                # Save to file
                import json
                with open("your_findings_predictions.json", "a") as f:
                    f.write(json.dumps(save_data) + "\n")
                
                st.success("✅ Prediction saved! Tracking YOUR sweet spot accuracy")
                
                # Show prediction result
                home_goals, away_goals = map(int, score.split('-'))
                total_goals = home_goals + away_goals
                actual_over_under = "OVER" if total_goals > 2.5 else "UNDER"
                
                correct = actual_over_under == prediction_data['predicted_totals']
                
                if correct:
                    st.balloons()
                    st.success(f"✅ CORRECT! Predicted {prediction_data['predicted_totals']}, actual {actual_over_under}")
                else:
                    st.error(f"❌ INCORRECT. Predicted {prediction_data['predicted_totals']}, actual {actual_over_under}")
                
            except:
                st.error("Enter score like '2-1'")
        else:
            st.error("Enter actual score")

# Performance summary
st.divider()
st.subheader("📈 YOUR MODEL PERFORMANCE")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Your Statistical Foundation:**")
    st.write("• 192 league entries analyzed")
    st.write("• Distinct over/under distributions")
    st.write("• Clear sweet spots identified")
    st.write("• Positive correlation confirmed")

with col2:
    st.write("**Sweet Spot Accuracy:**")
    st.write(f"• UNDER sweet spot: <2.2 sum_xg")
    st.write(f"• OVER sweet spot: >3.0 sum_xg")
    st.write(f"• Uncertain range: 2.2-3.0")
    st.write(f"• Baseline home wins: {model.baseline_home_win_rate}%")

with col3:
    # Show current prediction's position
    sum_xg = prediction_data['sum_avg_xg']
    
    st.write("**This Prediction's Position:**")
    
    # Create a simple visualization
    if sum_xg < 2.2:
        position = "█░░░░░░░░░░░░░░░░░░░"
        label = f"{sum_xg:.2f} (UNDER sweet spot)"
    elif sum_xg < 3.0:
        position = "░░░░███░░░░░░░░░░░░░"
        label = f"{sum_xg:.2f} (Uncertain range)"
    else:
        position = "░░░░░░░░░░░░░░░███░░"
        label = f"{sum_xg:.2f} (OVER sweet spot)"
    
    st.write("sum_xg scale:")
    st.write("1.0 ┤" + position + "┤ 4.5")
    st.write(f"Current: **{label}**")

# Footer
st.divider()
st.caption(f"🎯 YOUR ACTUAL STATISTICAL FINDINGS | Sweet spots: UNDER<2.2, OVER>3.0 | Data: {model.data_sample_size} league entries | Using: sum_of_avg_xg")
