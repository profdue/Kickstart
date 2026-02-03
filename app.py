import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Football xG Predictor Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚽ Football Match Predictor Pro")
st.markdown("""
    Advanced match outcome prediction using xG regression with match-type classification.
    This model identifies game behavior patterns before calculating probabilities.
""")

# Constants
MAX_GOALS = 8
REG_BASE_FACTOR = 0.75
REG_MATCH_THRESHOLD = 5
MAX_REGRESSION = 0.3

# Match type thresholds
SUPPRESSION_THRESHOLD = 3
VOLATILITY_THRESHOLD = 3

# Initialize session state
if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

def factorial_cache(n):
    """Cache factorial calculations for performance"""
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    """Calculate Poisson probability manually"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

@st.cache_data(ttl=3600)
def load_league_data(league_name):
    """Load league data from CSV with caching"""
    try:
        # Map display names to actual file names
        file_map = {
            "premier_league": "premier_league.csv",
            "bundesliga": "bundesliga.csv",
            "serie_a": "serie_a.csv",
            "laliga": "laliga.csv",
            "ligue_1": "ligue_1.csv",
            "eredivisie": "eredivisie.csv"
        }
        
        actual_filename = file_map.get(league_name, f"{league_name}.csv")
        file_path = f"leagues/{actual_filename}"
        
        df = pd.read_csv(file_path)
        
        # Basic validation
        required_cols = ['team', 'venue', 'matches', 'xg', 'xga', 'goals_vs_xg']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
            return None
            
        return df
    except FileNotFoundError:
        st.error(f"⚠️ League file not found: leagues/{actual_filename}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare home and away stats from the data"""
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    home_stats = home_data.set_index('team')
    away_stats = away_data.set_index('team')
    
    return home_stats, away_stats

def calculate_regression_factors(home_team_stats, away_team_stats, regression_factor):
    """Calculate attack regression factors with intelligent capping"""
    home_matches = home_team_stats['matches']
    away_matches = away_team_stats['matches']
    
    # Calculate base regression
    if home_matches >= REG_MATCH_THRESHOLD:
        home_base_reg = (home_team_stats['goals_vs_xg'] / home_matches) * regression_factor
    else:
        home_base_reg = 0
    
    if away_matches >= REG_MATCH_THRESHOLD:
        away_base_reg = (away_team_stats['goals_vs_xg'] / away_matches) * regression_factor
    else:
        away_base_reg = 0
    
    # Apply asymmetric capping based on team strength
    # Strong teams (high xPts) get full regression, weak teams get capped
    
    # Estimate team strength from available data
    home_strength = home_team_stats.get('xpts', 0) / max(home_matches, 1)
    away_strength = away_team_stats.get('xpts', 0) / max(away_matches, 1)
    
    # Strong teams (top tier): full regression
    # Weak teams (bottom tier): capped at 50%
    # Middle teams: 75%
    
    # For now, use a simple heuristic based on wins
    home_wins = home_team_stats.get('wins', 0)
    away_wins = away_team_stats.get('wins', 0)
    
    home_win_rate = home_wins / max(home_matches, 1)
    away_win_rate = away_wins / max(away_matches, 1)
    
    # Apply asymmetric capping
    if home_win_rate > 0.6:  # Strong team
        home_attack_reg = max(min(home_base_reg, MAX_REGRESSION), -MAX_REGRESSION)
    elif home_win_rate < 0.3:  # Weak team
        home_attack_reg = max(min(home_base_reg, MAX_REGRESSION * 0.5), -MAX_REGRESSION * 0.5)
    else:  # Middle team
        home_attack_reg = max(min(home_base_reg, MAX_REGRESSION * 0.75), -MAX_REGRESSION * 0.75)
    
    if away_win_rate > 0.6:  # Strong team
        away_attack_reg = max(min(away_base_reg, MAX_REGRESSION), -MAX_REGRESSION)
    elif away_win_rate < 0.3:  # Weak team
        away_attack_reg = max(min(away_base_reg, MAX_REGRESSION * 0.5), -MAX_REGRESSION * 0.5)
    else:  # Middle team
        away_attack_reg = max(min(away_base_reg, MAX_REGRESSION * 0.75), -MAX_REGRESSION * 0.75)
    
    return home_attack_reg, away_attack_reg

def calculate_match_type_scores(home_stats, away_stats, home_xg_per_match, away_xg_per_match, 
                               home_xga_per_match, away_xga_per_match, home_attack_reg, away_attack_reg,
                               league_name, home_win_prob=None):
    """Calculate suppression and volatility scores for match classification"""
    suppression_score = 0
    volatility_score = 0
    
    # Get win rates for form disparity calculation
    home_wins = home_stats.get('wins', 0)
    away_wins = away_stats.get('wins', 0)
    home_matches = home_stats['matches']
    away_matches = away_stats['matches']
    
    home_win_rate = home_wins / max(home_matches, 1)
    away_win_rate = away_wins / max(away_matches, 1)
    form_disparity = abs(home_win_rate - away_win_rate)
    
    # --- SUPPRESSION SCORE CALCULATION ---
    
    # 1. Tactical league
    if league_name in ["serie_a", "laliga"]:
        suppression_score += 1
    elif league_name == "ligue_1":
        suppression_score += 0.5
    
    # 2. Away favorite >55%
    if home_win_prob and (1 - home_win_prob) > 0.55:  # Away win prob >55%
        suppression_score += 1
    
    # 3. Home team weak attack (xG < 1.2)
    if home_xg_per_match < 1.2:
        suppression_score += 1
    
    # 4. Form disparity >30%
    if form_disparity > 0.3:
        suppression_score += 1
    
    # 5. Both teams underperforming (positive regression)
    if home_attack_reg > 0.15 and away_attack_reg > 0.15:
        suppression_score += 1
    
    # 6. Recent defensive form (simplified - low goals against)
    if 'ga' in home_stats and home_stats['ga'] / max(home_matches, 1) < 1.0:
        suppression_score += 0.5
    if 'ga' in away_stats and away_stats['ga'] / max(away_matches, 1) < 1.0:
        suppression_score += 0.5
    
    # --- VOLATILITY SCORE CALCULATION ---
    
    # 1. High total expected goals
    total_xg = home_xg_per_match + away_xg_per_match
    if total_xg > 3.0:
        volatility_score += 1
    
    # 2. Weak defenses (high xGA)
    if home_xga_per_match > 1.8 or away_xga_per_match > 1.8:
        volatility_score += 1
    
    # 3. High regression magnitude
    if abs(home_attack_reg) > 0.25 or abs(away_attack_reg) > 0.25:
        volatility_score += 1
    
    # 4. Strong attack vs weak defense mismatch
    if (home_xg_per_match > 1.5 and away_xga_per_match > 1.5) or (away_xg_per_match > 1.5 and home_xga_per_match > 1.5):
        volatility_score += 1
    
    # 5. Both teams have decent attack AND poor defense
    if home_xg_per_match > 1.3 and home_xga_per_match > 1.5 and away_xg_per_match > 1.3 and away_xga_per_match > 1.5:
        volatility_score += 1
    
    # 6. Extreme form disparity (can lead to blowouts)
    if form_disparity > 0.4:
        volatility_score += 1
    
    return suppression_score, volatility_score

def classify_match_type(suppression_score, volatility_score):
    """Classify match based on suppression and volatility scores"""
    if suppression_score >= SUPPRESSION_THRESHOLD:
        return "controlled", suppression_score, volatility_score
    elif volatility_score >= VOLATILITY_THRESHOLD:
        return "open", suppression_score, volatility_score
    else:
        return "balanced", suppression_score, volatility_score

def adjust_probabilities_for_match_type(prob_matrix, match_type, home_xg, away_xg):
    """Adjust probability matrix based on match type classification"""
    adjusted_matrix = prob_matrix.copy()
    
    if match_type == "controlled":
        # For controlled matches: boost probabilities for low-scoring outcomes
        # and reduce probabilities for high-scoring outcomes
        
        # Boost 0-0, 1-0, 0-1, 1-1
        boost_factors = {
            (0, 0): 1.3,   # 30% boost for 0-0
            (1, 0): 1.2,   # 20% boost for 1-0
            (0, 1): 1.2,   # 20% boost for 0-1
            (1, 1): 1.15,  # 15% boost for 1-1
        }
        
        # Reduce high-scoring outcomes
        reduce_factor = 0.7  # 30% reduction
        
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                if (i, j) in boost_factors:
                    adjusted_matrix[i, j] *= boost_factors[(i, j)]
                elif i + j > 2:  # Total goals > 2
                    adjusted_matrix[i, j] *= reduce_factor
        
    elif match_type == "open":
        # For open matches: widen the distribution tails
        # Increase variance by boosting extreme outcomes
        
        # Boost blowout outcomes (3+ goal difference)
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                goal_diff = abs(i - j)
                if goal_diff >= 3:
                    adjusted_matrix[i, j] *= 1.4  # 40% boost for blowouts
                elif goal_diff == 2:
                    adjusted_matrix[i, j] *= 1.2  # 20% boost for 2-goal wins
        
        # Slightly reduce 0-0 and 1-1
        adjusted_matrix[0, 0] *= 0.6
        adjusted_matrix[1, 1] *= 0.8
    
    # For balanced matches, no adjustment
    
    # Normalize the matrix
    total = adjusted_matrix.sum()
    if total > 0:
        adjusted_matrix /= total
    
    return adjusted_matrix

def adjust_market_probabilities(over_25_prob, btts_yes_prob, match_type, league_name):
    """Adjust market probabilities based on match type and league"""
    adjusted_over_25 = over_25_prob
    adjusted_btts_yes = btts_yes_prob
    
    # League-specific adjustments
    league_factors = {
        "serie_a": {"over": 0.8, "btts": 0.8},
        "laliga": {"over": 0.85, "btts": 0.85},
        "ligue_1": {"over": 0.9, "btts": 0.9},
        "bundesliga": {"over": 1.1, "btts": 1.05},
        "eredivisie": {"over": 1.15, "btts": 1.1},
        "premier_league": {"over": 1.0, "btts": 1.0}  # Neutral
    }
    
    league_factor = league_factors.get(league_name, {"over": 1.0, "btts": 1.0})
    
    # Match type adjustments
    if match_type == "controlled":
        adjusted_over_25 *= 0.7  # 30% reduction
        adjusted_btts_yes *= 0.7  # 30% reduction
    elif match_type == "open":
        adjusted_over_25 *= 1.15  # 15% increase
        adjusted_btts_yes *= 1.1  # 10% increase
    
    # Apply league factors
    adjusted_over_25 *= league_factor["over"]
    adjusted_btts_yes *= league_factor["btts"]
    
    # Ensure probabilities stay within bounds
    adjusted_over_25 = max(0.05, min(0.95, adjusted_over_25))
    adjusted_btts_yes = max(0.05, min(0.95, adjusted_btts_yes))
    
    adjusted_under_25 = 1 - adjusted_over_25
    adjusted_btts_no = 1 - adjusted_btts_yes
    
    return adjusted_over_25, adjusted_under_25, adjusted_btts_yes, adjusted_btts_no

def calculate_expected_goals(home_stats, away_stats, home_attack_reg, away_attack_reg):
    """Calculate expected goals for both teams using geometric mean"""
    
    home_xg_per_match = home_stats['xg'] / max(home_stats['matches'], 1)
    away_xga_per_match = away_stats['xga'] / max(away_stats['matches'], 1)
    
    away_xg_per_match = away_stats['xg'] / max(away_stats['matches'], 1)
    home_xga_per_match = home_stats['xga'] / max(home_stats['matches'], 1)
    
    # Geometric mean gives balanced estimate
    home_expected = np.sqrt(home_xg_per_match * away_xga_per_match) * (1 + home_attack_reg)
    away_expected = np.sqrt(away_xg_per_match * home_xga_per_match) * (1 + away_attack_reg)
    
    # Apply bounds
    home_expected = max(home_expected, 0.3)
    away_expected = max(away_expected, 0.3)
    
    home_expected = min(home_expected, 4.0)
    away_expected = min(away_expected, 4.0)
    
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
    home_win = np.sum(np.triu(prob_matrix, k=1))
    draw = np.sum(np.diag(prob_matrix))
    away_win = np.sum(np.tril(prob_matrix, k=-1))
    
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    
    return home_win, draw, away_win

def calculate_betting_markets(prob_matrix):
    """Calculate betting market probabilities"""
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

def get_match_type_insights(match_type, suppression_score, volatility_score):
    """Generate insights based on match type classification"""
    insights = []
    
    if match_type == "controlled":
        insights.append("🎯 **Match Type: Controlled / Tactical**")
        insights.append("• Expect a tactical, low-scoring match")
        insights.append("• High probability of 0-0, 1-0, 0-1, or 1-1 scorelines")
        insights.append("• Both Teams to Score markets are less reliable")
        insights.append("• Under 2.5 goals is more likely than usual")
        
    elif match_type == "open":
        insights.append("🎯 **Match Type: Open / Volatile**")
        insights.append("• Expect an open, high-scoring match")
        insights.append("• Higher chance of blowouts and comebacks")
        insights.append("• Both Teams to Score and Over 2.5 markets are stronger")
        insights.append("• Scorelines may be more extreme than typical")
        
    else:  # balanced
        insights.append("🎯 **Match Type: Balanced / Honest**")
        insights.append("• Standard Poisson distribution applies well")
        insights.append("• Match should follow statistical expectations")
        insights.append("• All markets are reasonably reliable")
    
    insights.append(f"📊 Suppression Score: {suppression_score:.1f}/5.0")
    insights.append(f"📈 Volatility Score: {volatility_score:.1f}/5.0")
    
    return insights

def get_risk_flags(home_stats, away_stats, home_xg, away_xg, home_reg, away_reg, match_type):
    """Generate risk flags and warnings"""
    flags = []
    
    home_perf = home_stats['goals_vs_xg'] / max(home_stats['matches'], 1)
    away_perf = away_stats['goals_vs_xg'] / max(away_stats['matches'], 1)
    
    # Match type specific warnings
    if match_type == "controlled":
        flags.append("🛡️ **Tactical Match Alert**: Expect defensive, low-scoring football")
        flags.append("⚠️ Avoid BTTS and Over markets in these conditions")
    
    elif match_type == "open":
        flags.append("⚡ **Volatile Match Alert**: High potential for blowouts/extreme scores")
        flags.append("📈 Consider BTTS and Over markets more seriously")
    
    # Extreme regression warnings
    if abs(home_reg) > 0.2:
        flags.append(f"⚠️ High home team regression: {home_reg:.3f} ({'over' if home_reg < 0 else 'under'}performance)")
    
    if abs(away_reg) > 0.2:
        flags.append(f"⚠️ High away team regression: {away_reg:.3f} ({'over' if away_reg < 0 else 'under'}performance)")
    
    # Extreme over/underperformance
    if abs(home_perf) > 0.5:
        flags.append(f"🚨 Extreme home team {'over' if home_perf < 0 else 'under'}performance: {abs(home_perf):.2f} goals/match")
    
    if abs(away_perf) > 0.5:
        flags.append(f"🚨 Extreme away team {'over' if away_perf < 0 else 'under'}performance: {abs(away_perf):.2f} goals/match")
    
    # Form disparity
    if 'wins' in home_stats and 'wins' in away_stats:
        home_win_rate = home_stats['wins'] / max(home_stats['matches'], 1)
        away_win_rate = away_stats['wins'] / max(away_stats['matches'], 1)
        
        if abs(home_win_rate - away_win_rate) > 0.3:
            flags.append(f"⚠️ Significant form disparity: {home_win_rate:.0%} vs {away_win_rate:.0%} win rate")
    
    # High/low scoring match flags
    total_xg = home_xg + away_xg
    if total_xg > 3.5:
        flags.append("⚡ Very high-scoring match expected (Total xG > 3.5)")
    elif total_xg > 3.0:
        flags.append("📈 High-scoring match expected (Total xG > 3.0)")
    elif total_xg < 2.0:
        flags.append("🛡️ Low-scoring match expected (Total xG < 2.0)")
    
    # Sample size warnings
    if home_stats['matches'] < 8:
        flags.append("📊 Limited sample size for home team home stats")
    if away_stats['matches'] < 8:
        flags.append("📊 Limited sample size for away team away stats")
    
    return flags

def get_betting_suggestions(home_win_prob, draw_prob, away_win_prob, over_25_prob, under_25_prob, btts_yes_prob, match_type):
    """Generate betting suggestions based on probabilities and match type"""
    suggestions = []
    
    # Adjust thresholds based on match type
    if match_type == "controlled":
        threshold = 0.6  # Higher threshold for controlled matches
        btts_threshold = 0.65
    elif match_type == "open":
        threshold = 0.52  # Lower threshold for open matches
        btts_threshold = 0.5
    else:
        threshold = 0.55  # Standard threshold for balanced matches
        btts_threshold = 0.55
    
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
    
    if match_type == "controlled":
        # In controlled matches, double chance is often safer
        if home_draw_prob > 0.7:
            suggestions.append(f"🛡️ Home Win or Draw ({(home_draw_prob*100):.1f}%) - Safer in tactical match")
        if away_draw_prob > 0.7:
            suggestions.append(f"🛡️ Away Win or Draw ({(away_draw_prob*100):.1f}%) - Safer in tactical match")
    else:
        if home_draw_prob > threshold:
            suggestions.append(f"✅ Home Win or Draw ({(home_draw_prob*100):.1f}%)")
        if away_draw_prob > threshold:
            suggestions.append(f"✅ Away Win or Draw ({(away_draw_prob*100):.1f}%)")
    
    # Over/Under with match type context
    if match_type == "controlled":
        if under_25_prob > 0.6:
            suggestions.append(f"🛡️ Under 2.5 Goals ({(under_25_prob*100):.1f}%) - Strong in tactical matches")
    elif match_type == "open":
        if over_25_prob > 0.6:
            suggestions.append(f"⚡ Over 2.5 Goals ({(over_25_prob*100):.1f}%) - Strong in open matches")
    else:
        if over_25_prob > threshold:
            suggestions.append(f"✅ Over 2.5 Goals ({(over_25_prob*100):.1f}%)")
        if under_25_prob > threshold:
            suggestions.append(f"✅ Under 2.5 Goals ({(under_25_prob*100):.1f}%)")
    
    # BTTS with match type context
    if match_type == "controlled":
        if btts_yes_prob < 0.4:
            suggestions.append(f"🛡️ Both Teams NOT to Score ({((1-btts_yes_prob)*100):.1f}%) - Likely in tactical matches")
    elif match_type == "open":
        if btts_yes_prob > 0.6:
            suggestions.append(f"⚡ Both Teams to Score ({(btts_yes_prob*100):.1f}%) - Likely in open matches")
    else:
        if btts_yes_prob > btts_threshold:
            suggestions.append(f"✅ Both Teams to Score ({(btts_yes_prob*100):.1f}%)")
        elif btts_yes_prob < (1 - btts_threshold):
            suggestions.append(f"❌ Both Teams NOT to Score ({((1-btts_yes_prob)*100):.1f}%)")
    
    # Match type specific guidance
    if match_type == "controlled" and not any("🛡️" in s for s in suggestions):
        suggestions.append("📝 Note: Controlled matches favor defensive outcomes")
    
    if match_type == "open" and not any("⚡" in s for s in suggestions):
        suggestions.append("📝 Note: Open matches favor attacking outcomes")
    
    return suggestions

# ========== SIDEBAR CONTROLS ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie"]
    selected_league = st.selectbox("Select League", leagues)
    
    # Map display name to filename
    league_to_file = {
        "Premier League": "premier_league",
        "Bundesliga": "bundesliga",
        "Serie A": "serie_a",
        "La Liga": "laliga",
        "Ligue 1": "ligue_1",
        "Eredivisie": "eredivisie"
    }
    
    league_key = league_to_file[selected_league]
    df = load_league_data(league_key)
    
    if df is not None:
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        available_home_teams = sorted(home_stats_df.index.unique())
        available_away_teams = sorted(away_stats_df.index.unique())
        common_teams = sorted(list(set(available_home_teams) & set(available_away_teams)))
        
        if len(common_teams) == 0:
            st.error("❌ No teams with both home and away data available")
        else:
            home_team = st.selectbox("Home Team", common_teams)
            away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
            
            regression_factor = st.slider(
                "Regression Factor",
                min_value=0.0,
                max_value=2.0,
                value=REG_BASE_FACTOR,
                step=0.05,
                help="Adjust how much to regress team performance to mean (higher = more regression)"
            )
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                enable_classification = st.checkbox("Enable Match Classification", value=True,
                    help="Classify matches as Controlled/Balanced/Open for better predictions")
                show_classification_details = st.checkbox("Show Classification Details", value=True)
                adjust_markets = st.checkbox("Adjust Markets by Match Type", value=True)
            
            calculate_btn = st.button("🎯 Calculate Predictions", type="primary", use_container_width=True)
            
            st.divider()
            st.subheader("📊 Display Options")
            show_matrix = st.checkbox("Show Score Probability Matrix", value=False)
            show_calculation = st.checkbox("Show Calculation Details", value=False)

# ========== MAIN CONTENT ==========
if df is None:
    st.warning("📁 Please add league CSV files to the 'leagues' folder")
    st.info("""
    **Required CSV files in 'leagues' folder:**
    - `premier_league.csv` ✓
    - `bundesliga.csv` ✓
    - `serie a.csv` ✓
    - `laliga.csv` ✓
    - `ligue_1.csv` ✓
    - `eredivisie.csv` (optional)
    """)
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Calculate Predictions' to start")
    
    with st.expander("📋 Preview of Loaded Data"):
        st.dataframe(df.head(10))
    st.stop()

try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"❌ Team data not found: {e}")
    st.stop()

# ========== PHASE 1: DATA PROCESSING ==========
st.header(f"📊 {home_team} vs {away_team}")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader(f"🏠 {home_team} (Home)")
    st.metric("Matches", int(home_stats['matches']))
    if 'wins' in home_stats:
        st.metric("Wins", int(home_stats['wins']))
    home_xg_per_match = home_stats['xg'] / max(home_stats['matches'], 1)
    home_xga_per_match = home_stats['xga'] / max(home_stats['matches'], 1)
    st.metric("xG/match", f"{home_xg_per_match:.2f}")
    st.metric("xGA/match", f"{home_xga_per_match:.2f}")
    if 'gf' in home_stats and 'ga' in home_stats:
        st.metric("GF-GA", f"{home_stats['gf']}-{home_stats['ga']}")

with col2:
    st.subheader(f"✈️ {away_team} (Away)")
    st.metric("Matches", int(away_stats['matches']))
    if 'wins' in away_stats:
        st.metric("Wins", int(away_stats['wins']))
    away_xg_per_match = away_stats['xg'] / max(away_stats['matches'], 1)
    away_xga_per_match = away_stats['xga'] / max(away_stats['matches'], 1)
    st.metric("xG/match", f"{away_xg_per_match:.2f}")
    st.metric("xGA/match", f"{away_xga_per_match:.2f}")
    if 'gf' in away_stats and 'ga' in away_stats:
        st.metric("GF-GA", f"{away_stats['gf']}-{away_stats['ga']}")

with col3:
    # Calculate regression factors with asymmetric capping
    home_attack_reg, away_attack_reg = calculate_regression_factors(
        home_stats, away_stats, regression_factor
    )
    
    # Calculate expected goals
    home_xg, away_xg = calculate_expected_goals(
        home_stats, away_stats, home_attack_reg, away_attack_reg
    )
    
    st.subheader("🎯 Expected Goals")
    
    xg_data = pd.DataFrame({
        'Team': [home_team, away_team],
        'Expected Goals': [home_xg, away_xg]
    })
    
    st.bar_chart(xg_data.set_index('Team'))
    
    col_xg1, col_xg2, col_xg3 = st.columns(3)
    with col_xg1:
        st.metric("Home xG", f"{home_xg:.2f}")
    with col_xg2:
        st.metric("Away xG", f"{away_xg:.2f}")
    with col_xg3:
        total_xg = home_xg + away_xg
        st.metric("Total xG", f"{total_xg:.2f}")
    
    # Bias indicators
    if total_xg > 2.6:
        st.success(f"📈 Over bias: Total xG = {total_xg:.2f} > 2.6")
    elif total_xg < 2.3:
        st.info(f"📉 Under bias: Total xG = {total_xg:.2f} < 2.3")
    
    st.caption(f"Regression factors: Home {home_attack_reg:.3f}, Away {away_attack_reg:.3f}")

# ========== PHASE 2: MATCH CLASSIFICATION ==========
if enable_classification:
    st.divider()
    st.header("🎯 Match Classification Analysis")
    
    # First calculate base probabilities for classification
    base_prob_matrix = create_probability_matrix(home_xg, away_xg)
    base_home_win_prob, base_draw_prob, base_away_win_prob = calculate_outcome_probabilities(base_prob_matrix)
    
    # Calculate match type scores
    suppression_score, volatility_score = calculate_match_type_scores(
        home_stats, away_stats, home_xg_per_match, away_xg_per_match,
        home_xga_per_match, away_xga_per_match, home_attack_reg, away_attack_reg,
        league_key, base_home_win_prob
    )
    
    # Classify match
    match_type, final_suppression_score, final_volatility_score = classify_match_type(
        suppression_score, volatility_score
    )
    
    # Display classification results
    col_type1, col_type2, col_type3 = st.columns(3)
    
    with col_type1:
        if match_type == "controlled":
            st.info(f"**Match Type:** 🛡️ Controlled")
            st.metric("Suppression Score", f"{final_suppression_score:.1f}/5")
        elif match_type == "open":
            st.success(f"**Match Type:** ⚡ Open")
            st.metric("Volatility Score", f"{final_volatility_score:.1f}/5")
        else:
            st.warning(f"**Match Type:** ⚖️ Balanced")
    
    with col_type2:
        st.metric("Suppression", f"{final_suppression_score:.1f}/5.0")
    
    with col_type3:
        st.metric("Volatility", f"{final_volatility_score:.1f}/5.0")
    
    # Show classification insights
    insights = get_match_type_insights(match_type, final_suppression_score, final_volatility_score)
    for insight in insights:
        st.write(insight)
    
    if show_classification_details:
        with st.expander("📋 Classification Details"):
            st.write(f"**Suppression Factors:**")
            st.write(f"- League: {selected_league} ({'tactical' if league_key in ['serie_a', 'laliga'] else 'neutral'})")
            st.write(f"- Home xG/match: {home_xg_per_match:.2f} {'< 1.2' if home_xg_per_match < 1.2 else ''}")
            st.write(f"- Form disparity: {abs((home_stats.get('wins', 0)/max(home_stats['matches'], 1)) - (away_stats.get('wins', 0)/max(away_stats['matches'], 1))):.1%}")
            st.write(f"- Both teams underperforming: {home_attack_reg > 0.15 and away_attack_reg > 0.15}")
            
            st.write(f"\n**Volatility Factors:**")
            st.write(f"- Total xG: {home_xg_per_match + away_xg_per_match:.2f} {'> 3.0' if (home_xg_per_match + away_xg_per_match) > 3.0 else ''}")
            st.write(f"- Weak defenses: Home xGA {home_xga_per_match:.2f}, Away xGA {away_xga_per_match:.2f}")
            st.write(f"- High regression: Home {home_attack_reg:.3f}, Away {away_attack_reg:.3f}")

# ========== PHASE 3: PROBABILITY CALCULATIONS ==========
st.divider()
st.header("📈 Probability Calculations")

# Create base probability matrix
prob_matrix = create_probability_matrix(home_xg, away_xg)

# Apply match type adjustments if enabled
if enable_classification:
    adjusted_matrix = adjust_probabilities_for_match_type(prob_matrix, match_type, home_xg, away_xg)
    # Calculate outcome probabilities from adjusted matrix
    home_win_prob, draw_prob, away_win_prob = calculate_outcome_probabilities(adjusted_matrix)
    
    # Calculate market probabilities from adjusted matrix
    base_over_25, base_under_25, base_btts_yes, base_btts_no = calculate_betting_markets(adjusted_matrix)
else:
    # Use base probabilities
    home_win_prob, draw_prob, away_win_prob = calculate_outcome_probabilities(prob_matrix)
    base_over_25, base_under_25, base_btts_yes, base_btts_no = calculate_betting_markets(prob_matrix)

# Adjust market probabilities based on match type and league
if enable_classification and adjust_markets:
    over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob = adjust_market_probabilities(
        base_over_25, base_btts_yes, match_type, league_key
    )
else:
    over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob = base_over_25, base_under_25, base_btts_yes, base_btts_no

# ========== LAYER 3: SCORE PROBABILITIES ==========
with st.expander("🎯 Most Likely Scores", expanded=True):
    # Get top 5 most likely scores
    if enable_classification:
        display_matrix = adjusted_matrix
    else:
        display_matrix = prob_matrix
    
    score_probs = []
    for i in range(min(6, display_matrix.shape[0])):
        for j in range(min(6, display_matrix.shape[1])):
            prob = display_matrix[i, j]
            if prob > 0.001:
                score_probs.append(((i, j), prob))
    
    score_probs.sort(key=lambda x: x[1], reverse=True)
    
    cols = st.columns(5)
    for idx, ((home_goals, away_goals), prob) in enumerate(score_probs[:5]):
        with cols[idx]:
            st.metric(
                label=f"{home_goals}-{away_goals}",
                value=f"{prob*100:.1f}%",
                delta="Most Likely" if idx == 0 else None
            )
    
    if score_probs:
        most_likely_score, most_likely_prob = score_probs[0]
        st.success(f"**Most Likely Score:** {most_likely_score[0]}-{most_likely_score[1]} ({(most_likely_prob*100):.1f}%)")

# ========== LAYER 4: OUTCOME PROBABILITIES ==========
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
    
    outcome_data = pd.DataFrame({
        'Outcome': ['Home Win', 'Draw', 'Away Win'],
        'Probability': [home_win_prob, draw_prob, away_win_prob]
    })
    
    st.bar_chart(outcome_data.set_index('Outcome'))

# ========== LAYER 5: BETTING MARKETS ==========
with st.expander("💰 Betting Markets", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Over/Under 2.5 Goals")
        st.metric("Over 2.5", f"{over_25_prob*100:.1f}%")
        st.progress(over_25_prob)
        st.metric("Under 2.5", f"{under_25_prob*100:.1f}%")
        st.progress(under_25_prob)
        
        if enable_classification and adjust_markets:
            if match_type == "controlled":
                st.caption("📉 Adjusted downward for tactical match")
            elif match_type == "open":
                st.caption("📈 Adjusted upward for open match")
    
    with col2:
        st.subheader("Both Teams to Score")
        st.metric("Yes", f"{btts_yes_prob*100:.1f}%")
        st.progress(btts_yes_prob)
        st.metric("No", f"{btts_no_prob*100:.1f}%")
        st.progress(btts_no_prob)
        
        if enable_classification and adjust_markets:
            if match_type == "controlled":
                st.caption("📉 Adjusted downward for tactical match")
            elif match_type == "open":
                st.caption("📈 Adjusted upward for open match")
    
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

# ========== LAYER 6: BETTING SUGGESTIONS ==========
with st.expander("💡 Betting Suggestions", expanded=True):
    if enable_classification:
        suggestions = get_betting_suggestions(
            home_win_prob, draw_prob, away_win_prob,
            over_25_prob, under_25_prob, btts_yes_prob, match_type
        )
    else:
        suggestions = get_betting_suggestions(
            home_win_prob, draw_prob, away_win_prob,
            over_25_prob, under_25_prob, btts_yes_prob, "balanced"
        )
    
    if suggestions:
        st.success("**Value Bets Found:**")
        for suggestion in suggestions:
            st.write(suggestion)
    else:
        st.info("No strong value bets identified")
    
    st.subheader("Double Chance")
    col_dc1, col_dc2 = st.columns(2)
    with col_dc1:
        home_draw_prob = home_win_prob + draw_prob
        st.metric("Home Win or Draw", f"{home_draw_prob*100:.1f}%")
    with col_dc2:
        away_draw_prob = away_win_prob + draw_prob
        st.metric("Away Win or Draw", f"{away_draw_prob*100:.1f}%")

# ========== LAYER 7: RISK FLAGS ==========
with st.expander("⚠️ Risk Flags & Warnings", expanded=False):
    if enable_classification:
        flags = get_risk_flags(home_stats, away_stats, home_xg, away_xg, home_attack_reg, away_attack_reg, match_type)
    else:
        flags = get_risk_flags(home_stats, away_stats, home_xg, away_xg, home_attack_reg, away_attack_reg, "balanced")
    
    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.success("No significant risk flags identified")

# ========== OUTPUT FORMATS ==========
st.divider()
st.header("📤 Export & Share")

summary = f"""
⚽ PREDICTION SUMMARY: {home_team} vs {away_team}
League: {selected_league}

📊 Team Statistics:
• {home_team} (Home): {home_stats['matches']} matches, {home_stats.get('wins', 'N/A')} wins
• {away_team} (Away): {away_stats['matches']} matches, {away_stats.get('wins', 'N/A')} wins

🎯 Match Classification: {match_type.upper() if enable_classification else 'N/A'}
• Suppression Score: {final_suppression_score:.1f}/5.0
• Volatility Score: {final_volatility_score:.1f}/5.0

🎯 Expected Goals:
• {home_team} xG: {home_xg:.2f}
• {away_team} xG: {away_xg:.2f}
• Total xG: {home_xg + away_xg:.2f}

📈 Most Likely Score: {score_probs[0][0][0] if score_probs else 'N/A'}-{score_probs[0][0][1] if score_probs else 'N/A'} ({(score_probs[0][1]*100 if score_probs else 0):.1f}%)

🏆 Outcome Probabilities:
• {home_team} Win: {home_win_prob*100:.1f}%
• Draw: {draw_prob*100:.1f}%
• {away_team} Win: {away_win_prob*100:.1f}%

💰 Betting Markets:
• Over 2.5 Goals: {over_25_prob*100:.1f}%
• Under 2.5 Goals: {under_25_prob*100:.1f}%
• Both Teams to Score: {btts_yes_prob*100:.1f}%

📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Regression Factor: {regression_factor}
Match Classification: {enable_classification}
Market Adjustment: {adjust_markets}
"""

st.code(summary, language="text")

col_export1, col_export2 = st.columns(2)

with col_export1:
    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name=f"prediction_{home_team}_vs_{away_team}.txt",
        mime="text/plain"
    )

with col_export2:
    export_data = {
        'Metric': [
            'Home Team', 'Away Team', 'League', 'Match Type', 'Suppression Score', 'Volatility Score',
            'Home xG', 'Away xG', 'Total xG',
            'Home Win %', 'Draw %', 'Away Win %',
            'Over 2.5 %', 'Under 2.5 %', 'BTTS Yes %', 'BTTS No %',
            'Most Likely Score', 'Regression Factor', 'Classification Enabled', 'Market Adjustment'
        ],
        'Value': [
            home_team, away_team, selected_league,
            match_type if enable_classification else 'N/A',
            f"{final_suppression_score:.1f}" if enable_classification else 'N/A',
            f"{final_volatility_score:.1f}" if enable_classification else 'N/A',
            f"{home_xg:.2f}", f"{away_xg:.2f}", f"{home_xg+away_xg:.2f}",
            f"{home_win_prob*100:.1f}", f"{draw_prob*100:.1f}", f"{away_win_prob*100:.1f}",
            f"{over_25_prob*100:.1f}", f"{under_25_prob*100:.1f}",
            f"{btts_yes_prob*100:.1f}", f"{btts_no_prob*100:.1f}",
            f"{score_probs[0][0][0] if score_probs else 'N/A'}-{score_probs[0][0][1] if score_probs else 'N/A'}",
            f"{regression_factor}",
            str(enable_classification),
            str(adjust_markets)
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
        if enable_classification:
            display_matrix = adjusted_matrix
        else:
            display_matrix = prob_matrix
        
        matrix_data = []
        for i in range(6):
            row = []
            for j in range(6):
                row.append(f"{display_matrix[i, j]*100:.2f}%")
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(
            matrix_data,
            columns=[f'Away {i}' for i in range(6)],
            index=[f'Home {i}' for i in range(6)]
        )
        
        st.dataframe(matrix_df, use_container_width=True)

# ========== CALCULATION DETAILS ==========
if show_calculation:
    with st.expander("🧮 Calculation Details", expanded=False):
        st.write("**Expected Goals Calculation:**")
        st.write(f"{home_team} xG = √({home_xg_per_match:.2f} × {away_xga_per_match:.2f}) × (1 + {home_attack_reg:.3f})")
        st.write(f"= √({home_xg_per_match * away_xga_per_match:.2f}) × {1 + home_attack_reg:.3f}")
        st.write(f"= {np.sqrt(home_xg_per_match * away_xga_per_match):.2f} × {1 + home_attack_reg:.3f}")
        st.write(f"= **{home_xg:.2f}**")
        
        st.write(f"\n{away_team} xG = √({away_xg_per_match:.2f} × {home_xga_per_match:.2f}) × (1 + {away_attack_reg:.3f})")
        st.write(f"= √({away_xg_per_match * home_xga_per_match:.2f}) × {1 + away_attack_reg:.3f}")
        st.write(f"= {np.sqrt(away_xg_per_match * home_xga_per_match):.2f} × {1 + away_attack_reg:.3f}")
        st.write(f"= **{away_xg:.2f}**")

# ========== FOOTER ==========
st.divider()
st.caption(f"⚡ Predictions calculated using advanced xG regression with match classification | Regression factor: {regression_factor} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")