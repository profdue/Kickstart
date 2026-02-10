import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
import json
import os
from supabase import create_client, Client
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚽ Football Intelligence Engine - TRUTHFUL STATISTICAL MODELS",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Football Intelligence Engine - TRUTHFUL STATISTICAL MODELS")
st.markdown("""
    **REALITY-BASED PREDICTIONS** - Using statistical insights INTELLIGENTLY, not blindly
""")

# ========== SUPABASE INITIALIZATION ==========
def init_supabase():
    """Initialize Supabase client"""
    try:
        supabase_url = st.secrets.get("SUPABASE_URL")
        supabase_key = st.secrets.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            st.warning("Supabase credentials not found in secrets. Using local storage only.")
            return None
            
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Error initializing Supabase: {str(e)}")
        return None

# Initialize
supabase = init_supabase()

# ========== TRUTHFUL STATISTICAL ENGINE ==========

class TruthfulStatisticalEngine:
    """APPLIES YOUR STATISTICAL INSIGHTS INTELLIGENTLY"""
    
    def __init__(self, league_name):
        self.name = league_name
        self.model_type = "TRUTHFUL_STATISTICAL"
        self.version = "v4.0_intelligent"
        
        # YOUR PROVEN STATISTICAL INSIGHTS:
        self.home_advantage_goals = 3.36  # +3.36 goals over season
        self.home_win_rate = 0.4452       # 44.52% home wins
        self.away_win_rate = 0.2993       # 29.93% away wins
        self.draw_rate = 0.2555           # 25.55% draws
        
        # INTELLIGENT APPLICATION:
        self.per_match_home_boost = 0.088  # +0.088 goals per match
        self.home_scoring_multiplier = 1.25  # Home teams score 25% more
        self.away_scoring_penalty = 0.80    # Away teams score 20% less
        
        # BUT: These apply RELATIVE to team strength
        self.baseline_accuracy = 44.52     # "Always bet home" baseline
        self.current_model_accuracy = 22.2  # Your current model's accuracy
        
    def predict_winner(self, home_stats, away_stats, home_xg=None, away_xg=None):
        """
        INTELLIGENT prediction using your statistical insights
        CONSIDERS: Team strength, home advantage, and REALITY
        """
        # Get REAL performance data
        home_scoring = home_stats.get('avg_home_goals_scored', 1.5)
        home_conceding = home_stats.get('avg_home_goals_conceded', 1.3)
        away_scoring = away_stats.get('avg_away_goals_scored', 1.2)
        away_conceding = away_stats.get('avg_away_goals_conceded', 1.5)
        
        # 1. Calculate BASE team strengths WITHOUT home advantage
        home_base_strength = (home_scoring + away_conceding) / 2
        away_base_strength = (away_scoring + home_conceding) / 2
        
        # 2. Apply your statistical insights INTELLIGENTLY
        # Home advantage: +25% effectiveness for home, -20% for away
        home_adjusted = home_base_strength * self.home_scoring_multiplier
        away_adjusted = away_base_strength * self.away_scoring_penalty
        
        # 3. Add absolute home advantage (+0.088 goals)
        home_adjusted += self.per_match_home_boost
        away_adjusted -= self.per_match_home_boost
        
        # 4. Calculate REAL strength difference
        strength_diff = home_adjusted - away_adjusted
        strength_ratio = away_adjusted / home_adjusted if home_adjusted > 0 else 999
        
        # 5. INTELLIGENT DECISION MAKING:
        # If away team is MUCH stronger (>40% stronger), home advantage can't overcome it
        if strength_ratio > 1.4:  # Away team is 40%+ stronger
            confidence = self._calculate_reality_based_confidence(strength_ratio, is_home=False)
            return "AWAY", confidence, f"AWAY_MUCH_STRONGER ({strength_ratio:.1f}x)"
        
        # If home team is MUCH stronger (>40% stronger)
        elif strength_ratio < 0.71:  # Home team is 40%+ stronger (1/0.71 = 1.4)
            confidence = self._calculate_reality_based_confidence(1/strength_ratio, is_home=True)
            return "HOME", confidence, f"HOME_MUCH_STRONGER ({1/strength_ratio:.1f}x)"
        
        # If away team is moderately stronger (20-40% stronger)
        elif strength_ratio > 1.2:  # Away 20-40% stronger
            # Slight edge to away despite home advantage
            confidence = 55 + min(10, (strength_ratio - 1.2) * 20)
            return "AWAY", confidence, f"AWAY_MODERATELY_STRONGER ({strength_ratio:.1f}x)"
        
        # If home team is moderately stronger (20-40% stronger)
        elif strength_ratio < 0.83:  # Home 20-40% stronger (1/0.83 = 1.2)
            confidence = 60 + min(15, (1/strength_ratio - 1.2) * 20)
            return "HOME", confidence, f"HOME_MODERATELY_STRONGER ({1/strength_ratio:.1f}x)"
        
        # Close match: Use your 44.52% home win baseline
        else:
            # Strength difference < 20%, home gets baseline advantage
            if strength_diff > 0:
                confidence = 52  # Slight edge for home
                return "HOME", confidence, "CLOSE_MATCH_HOME_EDGE"
            else:
                confidence = 48  # Slight edge for away
                return "AWAY", confidence, "CLOSE_MATCH_AWAY_EDGE"
    
    def _calculate_reality_based_confidence(self, strength_ratio, is_home=True):
        """Calculate confidence based on ACTUAL strength differences"""
        if is_home:
            # Home team is stronger
            base = 60  # Start at 60% for home advantage
            boost = (strength_ratio - 1) * 30  # Add 10% per 0.33x strength advantage
        else:
            # Away team is stronger
            base = 40  # Start lower for away team
            boost = (strength_ratio - 1) * 25  # Add boost for strength
        
        confidence = base + boost
        return min(85, max(45, confidence))  # Keep within reasonable bounds
    
    def predict_totals(self, home_stats, away_stats, total_xg=None):
        """Intelligent totals prediction based on REAL scoring patterns"""
        
        # Get REAL scoring rates
        home_scoring = home_stats.get('avg_home_goals_scored', 1.5)
        away_scoring = away_stats.get('avg_away_goals_scored', 1.2)
        home_conceding = home_stats.get('avg_home_goals_conceded', 1.3)
        away_conceding = away_stats.get('avg_away_goals_conceded', 1.5)
        
        # EXPECTED TOTAL calculation:
        # 1. Average of teams' scoring rates
        avg_scoring = (home_scoring + away_scoring) / 2
        
        # 2. Apply your home advantage insight (+25% for home scoring)
        home_boosted_scoring = home_scoring * self.home_scoring_multiplier
        
        # 3. Calculate expected total
        expected_total = (home_boosted_scoring + away_scoring) / 2
        
        # 4. Adjust for defensive strength
        defensive_factor = (home_conceding + away_conceding) / 2.6
        expected_total *= (1.1 - (defensive_factor - 1.4) * 0.3)
        
        # INTELLIGENT THRESHOLDS:
        if expected_total > 3.0:
            confidence = 70
            return "OVER", confidence, f"VERY_HIGH_SCORING: {expected_total:.1f} expected"
        elif expected_total > 2.7:
            confidence = 65
            return "OVER", confidence, f"HIGH_SCORING: {expected_total:.1f} expected"
        elif expected_total < 2.0:
            confidence = 70
            return "UNDER", confidence, f"VERY_LOW_SCORING: {expected_total:.1f} expected"
        elif expected_total < 2.3:
            confidence = 60
            return "UNDER", confidence, f"LOW_SCORING: {expected_total:.1f} expected"
        else:
            # 2.3-2.7 range: Use historical distribution
            # From your analysis, more matches end under 2.5
            confidence = 55
            return "UNDER", confidence, f"NEUTRAL_TO_UNDER: {expected_total:.1f} expected"
    
    def get_statistical_insights(self):
        """Display how your insights are being applied INTELLIGENTLY"""
        return {
            "home_advantage": f"+{self.home_advantage_goals} goals (applied RELATIVE to team strength)",
            "home_win_rate": f"{self.home_win_rate*100:.1f}% (used as BASELINE, not absolute)",
            "intelligent_application": "Home advantage CANNOT overcome large quality gaps",
            "scoring_multiplier": f"Home: ×{self.home_scoring_multiplier}, Away: ×{self.away_scoring_penalty}",
            "current_vs_baseline": f"Current model: {self.current_model_accuracy}% vs Baseline: {self.baseline_accuracy}%",
            "goal": f"Target: 57.5%+ accuracy by applying insights INTELLIGENTLY"
        }

# ========== REALITY CHECK ENGINE ==========

class RealityCheckEngine:
    """
    CHECKS if predictions make REAL-WORLD sense
    Based on your ACTUAL statistical findings
    """
    
    def __init__(self, league_name):
        self.name = league_name
        self.model_type = "REALITY_CHECK"
        self.version = "v4.1_reality"
        
        # Your statistical constants
        self.home_win_baseline = 0.4452
        
        # Reality thresholds
        self.min_sensible_confidence = 40
        self.max_sensible_confidence = 85
        self.strength_gap_threshold = 1.5  # 50% stronger
        
    def check_prediction(self, home_stats, away_stats, predicted_winner, confidence):
        """
        VERIFIES if prediction makes REAL-WORLD sense
        Returns: (is_sensible, reason, adjusted_prediction, adjusted_confidence)
        """
        
        # Get key metrics
        home_scoring = home_stats.get('avg_home_goals_scored', 1.5)
        away_scoring = away_stats.get('avg_away_goals_scored', 1.2)
        home_win_rate = home_stats.get('home_wins_rate', 0.4)
        away_win_rate = away_stats.get('away_wins_rate', 0.3)
        
        # Calculate REALITY metrics
        scoring_ratio = away_scoring / home_scoring if home_scoring > 0 else 999
        win_rate_ratio = away_win_rate / home_win_rate if home_win_rate > 0 else 999
        
        # CHECKS:
        
        # 1. Check if confidence is within sensible bounds
        if confidence < self.min_sensible_confidence or confidence > self.max_sensible_confidence:
            return False, f"Confidence {confidence}% outside sensible range", None, None
        
        # 2. Check if prediction matches strength reality
        if predicted_winner == "HOME":
            # If away team scores MUCH more, home win prediction is suspect
            if scoring_ratio > self.strength_gap_threshold:
                # Away team scores 50%+ more goals
                adjusted_conf = max(40, confidence - 30)  # Reduce confidence
                return True, f"Away team scores {scoring_ratio:.1f}x more goals", "AWAY", adjusted_conf
        elif predicted_winner == "AWAY":
            # If home team scores MUCH more, away win prediction is suspect
            if 1/scoring_ratio > self.strength_gap_threshold:
                # Home team scores 50%+ more goals
                adjusted_conf = max(40, confidence - 30)
                return True, f"Home team scores {1/scoring_ratio:.1f}x more goals", "HOME", adjusted_conf
        
        # 3. Check win rate reality
        if predicted_winner == "HOME" and away_win_rate > home_win_rate * 2:
            # Away team wins twice as often
            return True, f"Away team wins {away_win_rate*100:.1f}% vs Home {home_win_rate*100:.1f}%", "AWAY", 55
        
        # Prediction passes reality check
        return True, "Prediction matches team strength reality", predicted_winner, confidence

# ========== ENGINE FACTORY ==========

class IntelligentEngineFactory:
    """Factory for INTELLIGENT statistical engines"""
    
    @staticmethod
    def create_engine(league_name):
        """Create truthful statistical engine"""
        return TruthfulStatisticalEngine(league_name)
    
    @staticmethod
    def create_reality_checker(league_name):
        """Create reality check engine"""
        return RealityCheckEngine(league_name)
    
    @staticmethod
    def get_statistical_facts():
        """Your ACTUAL statistical findings"""
        return {
            "home_goals": "16.69 goals (25% more than away)",
            "away_goals": "13.33 goals", 
            "home_wins": "44.52% of matches",
            "away_wins": "29.93% of matches",
            "draws": "25.55% of matches",
            "home_advantage": "+3.36 goals over season (+0.088 per match)",
            "current_accuracy": "22.2% (your current model)",
            "baseline": "44.52% (always bet home)",
            "target": "57.5%+ (intelligent application)"
        }

# ========== DATA FUNCTIONS ==========

def save_match_prediction(prediction_data, actual_score, league_name, engine, reality_check=None):
    """Save match prediction with reality check"""
    try:
        home_goals, away_goals = map(int, actual_score.split('-'))
        total_goals = home_goals + away_goals
        
        actual_winner = 'HOME' if home_goals > away_goals else 'AWAY' if away_goals > home_goals else 'DRAW'
        actual_over_under = 'OVER' if total_goals > 2.5 else 'UNDER'
        
        # Apply reality check if available
        if reality_check:
            predicted_winner = reality_check[2] if reality_check[0] else prediction_data['predicted_winner']
            winner_confidence = reality_check[3] if reality_check[0] else prediction_data['winner_confidence']
            reality_note = reality_check[1] if reality_check[0] else "No reality check applied"
        else:
            predicted_winner = prediction_data['predicted_winner']
            winner_confidence = prediction_data['winner_confidence']
            reality_note = "No reality check"
        
        match_data = {
            "match_date": datetime.now().date().isoformat(),
            "league": league_name,
            "home_team": prediction_data.get('home_team', 'Unknown'),
            "away_team": prediction_data.get('away_team', 'Unknown'),
            
            # Reality-checked predictions
            "predicted_winner": predicted_winner,
            "winner_confidence": float(winner_confidence),
            "winner_logic": prediction_data.get('winner_logic', 'UNKNOWN'),
            "reality_check_note": reality_note,
            
            "predicted_totals_direction": prediction_data.get('predicted_totals', 'UNKNOWN'),
            "totals_confidence": float(prediction_data.get('totals_confidence', 50)),
            "totals_logic": prediction_data.get('totals_logic', 'UNKNOWN'),
            
            # Actual results
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_total_goals": total_goals,
            "actual_winner": actual_winner,
            "actual_over_under": actual_over_under,
            
            # Model info
            "model_version": getattr(engine, 'version', 'unknown'),
            "model_type": getattr(engine, 'model_type', 'unknown'),
            "notes": "Intelligent statistical application"
        }
        
        if supabase:
            response = supabase.table("match_predictions").insert(match_data).execute()
            if hasattr(response, 'data') and response.data:
                return True, "✅ Match saved with reality check"
            else:
                return False, "❌ Failed to save"
        else:
            with open("match_predictions_backup.json", "a") as f:
                f.write(json.dumps(match_data) + "\n")
            return True, "⚠️ Saved locally"
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def get_match_stats():
    """Get accuracy statistics"""
    try:
        if supabase:
            response = supabase.table("match_predictions").select("id", count="exact").execute()
            total_matches = response.count or 0
            
            # Get accuracy
            acc_response = supabase.table("match_predictions").select("predicted_winner", "actual_winner").execute()
            
            correct = 0
            total = 0
            if acc_response.data:
                for match in acc_response.data:
                    if match['predicted_winner'] and match['actual_winner']:
                        total += 1
                        if match['predicted_winner'] == match['actual_winner']:
                            correct += 1
            
            accuracy = (correct / total * 100) if total > 0 else 0
            
            return {
                'total_matches': total_matches,
                'prediction_accuracy': accuracy,
                'correct_predictions': correct,
                'total_predictions': total
            }
        else:
            if os.path.exists("match_predictions_backup.json"):
                with open("match_predictions_backup.json", "r") as f:
                    lines = f.readlines()
                
                correct = 0
                total = 0
                for line in lines:
                    try:
                        data = json.loads(line)
                        if data.get('predicted_winner') and data.get('actual_winner'):
                            total += 1
                            if data['predicted_winner'] == data['actual_winner']:
                                correct += 1
                    except:
                        continue
                
                accuracy = (correct / total * 100) if total > 0 else 0
                
                return {
                    'total_matches': len(lines),
                    'prediction_accuracy': accuracy,
                    'correct_predictions': correct,
                    'total_predictions': total
                }
    except:
        pass
    
    return {
        'total_matches': 0,
        'prediction_accuracy': 0,
        'correct_predictions': 0,
        'total_predictions': 0
    }

# ========== EXPECTED GOALS CALCULATOR ==========

class RealisticExpectedGoals:
    """Realistic xG calculation"""
    
    def __init__(self, league_metrics):
        self.league_avg = league_metrics.get('avg_goals_per_match', 2.5)
        
    def calculate(self, home_stats, away_stats):
        """Calculate realistic expected goals"""
        home_scoring = home_stats['goals_for_pm']
        away_scoring = away_stats['goals_for_pm']
        home_conceding = home_stats['goals_against_pm']
        away_conceding = away_stats['goals_against_pm']
        
        # Simple realistic calculation
        home_xg = (home_scoring + away_conceding) / 2
        away_xg = (away_scoring + home_conceding) / 2
        
        # Apply realistic adjustments
        home_xg *= 1.15  # Home advantage
        away_xg *= 0.85  # Away disadvantage
        
        # Ensure realism
        home_xg = max(0.3, min(4.0, home_xg))
        away_xg = max(0.3, min(4.0, away_xg))
        
        return home_xg, away_xg

# Poisson probability functions
def poisson_pmf(k, lam):
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

class ProbabilityCalculator:
    @staticmethod
    def calculate_probabilities(home_xg, away_xg):
        probs = []
        for h in range(0, 7):
            for a in range(0, 7):
                prob = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)
                if prob > 0.001:
                    probs.append({'home': h, 'away': a, 'prob': prob})
        
        home_win = sum(p['prob'] for p in probs if p['home'] > p['away'])
        draw = sum(p['prob'] for p in probs if p['home'] == p['away'])
        away_win = sum(p['prob'] for p in probs if p['home'] < p['away'])
        
        over_25 = sum(p['prob'] for p in probs if p['home'] + p['away'] > 2.5)
        under_25 = sum(p['prob'] for p in probs if p['home'] + p['away'] < 2.5)
        
        top_scores = sorted(probs, key=lambda x: x['prob'], reverse=True)[:3]
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'over_25': over_25,
            'under_25': under_25,
            'expected_home': home_xg,
            'expected_away': away_xg,
            'expected_total': home_xg + away_xg,
            'top_scores': [(f"{s['home']}-{s['away']}", s['prob']) for s in top_scores]
        }

# ========== DATA LOADING ==========

@st.cache_data
def load_league_data(league_name):
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
        
        filename = file_map.get(league_name, f"{league_name.lower().replace(' ', '_')}.csv")
        df = pd.read_csv(f"leagues/{filename}")
        return df
    except:
        return None

def prepare_team_data(df):
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    for df_part in [home_data, away_data]:
        if len(df_part) > 0:
            df_part['goals_for_pm'] = df_part['gf'] / df_part['matches']
            df_part['goals_against_pm'] = df_part['ga'] / df_part['matches']
            df_part['win_rate'] = df_part['wins'] / df_part['matches']
            df_part['goals_vs_xg_pm'] = df_part['goals_vs_xg'] / df_part['matches']
    
    return home_data.set_index('team'), away_data.set_index('team')

def calculate_league_metrics(df):
    if df is None or len(df) == 0:
        return {}
    
    total_matches = df['matches'].sum() / 2
    total_goals = df['gf'].sum()
    avg_goals = total_goals / total_matches if total_matches > 0 else 2.5
    
    return {'avg_goals_per_match': avg_goals}

# ========== STREAMLIT UI ==========

with st.sidebar:
    st.header("⚙️ TRUTHFUL STATISTICAL ENGINE")
    
    leagues = ["Premier League", "Bundesliga", "Serie A", "La Liga", "Ligue 1", "Eredivisie", "RFPL"]
    selected_league = st.selectbox("Select League", leagues)
    
    # Enable reality checking
    enable_reality_check = st.checkbox("🔍 Enable Reality Check", value=True,
                                      help="Checks if predictions make real-world sense")
    
    # Create engines
    engine = IntelligentEngineFactory.create_engine(selected_league)
    reality_checker = IntelligentEngineFactory.create_reality_checker(selected_league) if enable_reality_check else None
    
    # Display statistical facts
    st.divider()
    st.header("📊 STATISTICAL FACTS (YOUR ANALYSIS)")
    
    facts = IntelligentEngineFactory.get_statistical_facts()
    for key, value in facts.items():
        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Load data
    df = load_league_data(selected_league)
    
    if df is not None:
        league_metrics = calculate_league_metrics(df)
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        if len(home_stats_df) > 0 and len(away_stats_df) > 0:
            home_teams = sorted(home_stats_df.index.unique())
            away_teams = sorted(away_stats_df.index.unique())
            common_teams = sorted(list(set(home_teams) & set(away_teams)))
            
            home_team = st.selectbox("Home Team", common_teams)
            away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
            
            st.divider()
            
            if st.button("🚀 Generate TRUTHFUL Prediction", type="primary", use_container_width=True):
                calculate_btn = True
            else:
                calculate_btn = False
    
    # Performance tracking
    st.divider()
    st.header("📈 PERFORMANCE TRACKING")
    
    stats = get_match_stats()
    st.metric("Total Matches", stats['total_matches'])
    
    if stats['total_predictions'] > 0:
        st.metric("Accuracy", f"{stats['prediction_accuracy']:.1f}%")
        st.metric("Correct", f"{stats['correct_predictions']}/{stats['total_predictions']}")

# Main content
if df is None:
    st.error("Please add CSV files to the 'leagues' folder")
    st.stop()

# Truth banner
st.markdown("""
<div style="background-color: #0C4A6E; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h3 style="color: white; text-align: center; margin: 0;">
        🔍 TRUTHFUL STATISTICAL PREDICTIONS
    </h3>
    <p style="color: #E0F2FE; text-align: center; margin: 5px 0 0 0;">
        Home advantage: +3.36 goals • Applied INTELLIGENTLY, not blindly • Reality checks enabled
    </p>
</div>
""", unsafe_allow_html=True)

# Generate prediction
if 'calculate_btn' in locals() and calculate_btn:
    try:
        home_stats = home_stats_df.loc[home_team]
        away_stats = away_stats_df.loc[away_team]
        
        # Prepare stats for engine
        home_real_stats = {
            'avg_home_goals_scored': home_stats['goals_for_pm'],
            'avg_home_goals_conceded': home_stats['goals_against_pm'],
            'home_wins_rate': home_stats['win_rate']
        }
        
        away_real_stats = {
            'avg_away_goals_scored': away_stats['goals_for_pm'],
            'avg_away_goals_conceded': away_stats['goals_against_pm'],
            'away_wins_rate': away_stats['win_rate']
        }
        
        # Calculate realistic xG
        xg_calculator = RealisticExpectedGoals(league_metrics)
        home_xg, away_xg = xg_calculator.calculate(home_stats, away_stats)
        
        # Get predictions
        predicted_winner, winner_confidence, winner_logic = engine.predict_winner(
            home_real_stats, away_real_stats, home_xg, away_xg
        )
        
        predicted_totals, totals_confidence, totals_logic = engine.predict_totals(
            home_real_stats, away_real_stats, home_xg + away_xg
        )
        
        # Apply reality check
        reality_result = None
        if reality_checker:
            reality_result = reality_checker.check_prediction(
                home_real_stats, away_real_stats, predicted_winner, winner_confidence
            )
            
            if reality_result[0]:  # If sensible
                predicted_winner = reality_result[2]
                winner_confidence = reality_result[3]
                winner_logic = f"{winner_logic} | {reality_result[1]}"
        
        # Calculate probabilities
        probs = ProbabilityCalculator.calculate_probabilities(home_xg, away_xg)
        
        # Store data
        prediction_data = {
            'home_team': home_team,
            'away_team': away_team,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'predicted_winner': predicted_winner,
            'winner_confidence': winner_confidence,
            'winner_logic': winner_logic,
            'predicted_totals': predicted_totals,
            'totals_confidence': totals_confidence,
            'totals_logic': totals_logic,
            'probabilities': probs,
            'reality_check': reality_result,
            'home_stats': home_real_stats,
            'away_stats': away_real_stats
        }
        
        st.session_state.prediction_data = prediction_data
        st.session_state.selected_teams = (home_team, away_team)
        st.session_state.engine = engine
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Display prediction
if 'prediction_data' not in st.session_state:
    st.info("👈 Select teams and click 'Generate TRUTHFUL Prediction'")
    st.stop()

prediction_data = st.session_state.prediction_data
home_team, away_team = st.session_state.selected_teams
engine = st.session_state.engine

st.header(f"🎯 {home_team} vs {away_team}")
st.caption(f"League: {selected_league} | Engine: {engine.model_type} | Intelligent statistical application")

# Show how insights are applied
with st.expander("🔬 HOW YOUR INSIGHTS ARE APPLIED INTELLIGENTLY"):
    insights = engine.get_statistical_insights()
    for key, value in insights.items():
        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Show team comparison
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{home_team} (Home):**")
        st.write(f"Goals scored: {prediction_data['home_stats']['avg_home_goals_scored']:.2f}/match")
        st.write(f"Goals conceded: {prediction_data['home_stats']['avg_home_goals_conceded']:.2f}/match")
        st.write(f"Win rate: {prediction_data['home_stats']['home_wins_rate']*100:.1f}%")
    
    with col2:
        st.write(f"**{away_team} (Away):**")
        st.write(f"Goals scored: {prediction_data['away_stats']['avg_away_goals_scored']:.2f}/match")
        st.write(f"Goals conceded: {prediction_data['away_stats']['avg_away_goals_conceded']:.2f}/match")
        st.write(f"Win rate: {prediction_data['away_stats']['away_wins_rate']*100:.1f}%")

# Prediction cards
col1, col2 = st.columns(2)

with col1:
    winner_pred = prediction_data['predicted_winner']
    winner_conf = prediction_data['winner_confidence']
    winner_logic = prediction_data['winner_logic']
    
    # Get actual probability
    prob = prediction_data['probabilities']
    actual_prob = prob['home_win'] if winner_pred == 'HOME' else prob['away_win'] if winner_pred == 'AWAY' else prob['draw']
    
    # Color based on confidence
    if winner_conf > 65:
        color = "#10B981"  # Green
    elif winner_conf > 55:
        color = "#F59E0B"  # Yellow
    else:
        color = "#EF4444"  # Red
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">WINNER (INTELLIGENT)</h3>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin: 10px 0;">
            {'🏠' if winner_pred == 'HOME' else '✈️' if winner_pred == 'AWAY' else '🤝'} {winner_pred}
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {winner_conf:.0f}%
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            {winner_logic}
        </div>
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            Poisson probability: {actual_prob*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show reality check result if available
    if prediction_data.get('reality_check'):
        reality = prediction_data['reality_check']
        if reality[0]:
            st.success(f"✅ Reality check passed: {reality[1]}")
        else:
            st.warning(f"⚠️ {reality[1]}")

with col2:
    totals_pred = prediction_data['predicted_totals']
    totals_conf = prediction_data['totals_confidence']
    totals_logic = prediction_data['totals_logic']
    
    actual_prob = prob['over_25'] if totals_pred == 'OVER' else prob['under_25']
    
    st.markdown(f"""
    <div style="background-color: #1E293B; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">TOTAL GOALS</h3>
        <div style="font-size: 36px; font-weight: bold; color: #60A5FA; margin: 10px 0;">
            {'📈' if totals_pred == 'OVER' else '📉'} {totals_pred} 2.5
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin: 10px 0;">
            {totals_conf:.0f}%
        </div>
        <div style="font-size: 14px; color: #D1D5DB; margin-top: 10px;">
            {totals_logic}
        </div>
        <div style="font-size: 12px; color: #9CA3AF; margin-top: 5px;">
            Poisson probability: {actual_prob*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# Strength comparison
st.divider()
st.subheader("📊 REALITY-BASED COMPARISON")

home_stats = prediction_data['home_stats']
away_stats = prediction_data['away_stats']

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Team Strength Analysis:**")
    
    home_attack = home_stats['avg_home_goals_scored']
    away_defense = away_stats['avg_away_goals_conceded']
    home_expected_goals = (home_attack + away_defense) / 2 * 1.25
    
    away_attack = away_stats['avg_away_goals_scored']
    home_defense = home_stats['avg_home_goals_conceded']
    away_expected_goals = (away_attack + home_defense) / 2 * 0.8
    
    st.write(f"Home expected goals: {home_expected_goals:.2f}")
    st.write(f"Away expected goals: {away_expected_goals:.2f}")
    st.write(f"Strength ratio: {away_expected_goals/home_expected_goals:.2f}x")

with col2:
    st.write("**Your Statistical Insights Applied:**")
    st.write(f"Home advantage: +{engine.per_match_home_boost:.3f} goals")
    st.write(f"Home scoring boost: ×{engine.home_scoring_multiplier}")
    st.write(f"Away scoring penalty: ×{engine.away_scoring_penalty}")
    st.write(f"Baseline home win rate: {engine.home_win_rate*100:.1f}%")

with col3:
    st.write("**Expected Match Outcome:**")
    st.write(f"Expected goals: Home {prediction_data['home_xg']:.2f} - Away {prediction_data['away_xg']:.2f}")
    st.write(f"Total expected: {prediction_data['home_xg'] + prediction_data['away_xg']:.2f}")
    
    # Show what the old model would have predicted
    old_home_strength = (home_attack + away_defense) / 2
    old_pred = "HOME" if old_home_strength > 1.4 else "DRAW" if old_home_strength > 1.0 else "AWAY"
    st.write(f"Old model would predict: {old_pred}")

# Data collection
st.divider()
st.subheader("📝 COLLECT MATCH DATA")

col1, col2 = st.columns([2, 1])

with col1:
    score = st.text_input("Actual Final Score", key="score_input")
    
    with st.expander("View Detailed Analysis"):
        st.write("**Expected Goals Calculation:**")
        st.write(f"- Home xG: {prediction_data['home_xg']:.2f}")
        st.write(f"- Away xG: {prediction_data['away_xg']:.2f}")
        st.write(f"- Total xG: {prediction_data['home_xg'] + prediction_data['away_xg']:.2f}")
        
        st.write("**Statistical Application:**")
        st.write(f"- Home advantage applied: +{engine.per_match_home_boost:.3f} goals")
        st.write(f"- Home scoring multiplier: ×{engine.home_scoring_multiplier}")
        st.write(f"- Away scoring penalty: ×{engine.away_scoring_penalty}")

with col2:
    if st.button("💾 Save Match", type="primary", use_container_width=True):
        if not score or '-' not in score:
            st.error("Enter valid score like '2-1'")
        else:
            try:
                success, message = save_match_prediction(
                    prediction_data, score, selected_league, engine, 
                    prediction_data.get('reality_check')
                )
                
                if success:
                    st.success(f"""
                    {message}
                    
                    **Model:** {engine.model_type}
                    **Reality check:** {prediction_data.get('reality_check', ['Not applied', ''])[1]}
                    """)
                    
                    # Reset
                    for key in ['prediction_data', 'selected_teams', 'engine']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                else:
                    st.error(message)
            except:
                st.error("Enter valid score")

# Probabilities
st.divider()
st.subheader("🎲 MATCH PROBABILITIES")

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
    
    st.write(f"**Expected Goals:**")
    st.write(f"Home: {prob['expected_home']:.2f}")
    st.write(f"Away: {prob['expected_away']:.2f}")
    st.write(f"Total: {prob['expected_total']:.2f}")

# Performance
st.divider()
st.subheader("📈 TRUTHFUL PERFORMANCE TRACKING")

stats = get_match_stats()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Matches Collected", stats['total_matches'])
    if stats['total_predictions'] > 0:
        st.metric("Accuracy", f"{stats['prediction_accuracy']:.1f}%")

with col2:
    baseline = 44.52
    current = stats['prediction_accuracy']
    improvement = current - baseline
    
    st.metric("Baseline", f"{baseline}%")
    st.metric("Improvement", f"{improvement:+.1f}%")

with col3:
    st.write("**Key Principles:**")
    st.write("• Home advantage is REAL")
    st.write("• But NOT absolute")
    st.write("• Team quality MATTERS")
    st.write("• Reality checks are CRITICAL")

# Footer
st.divider()
st.caption(f"📊 TRUTHFUL STATISTICAL MODELS | Current Accuracy: {stats['prediction_accuracy']:.1f}% | Target: 57.5%+ | Reality checks: {'ON' if enable_reality_check else 'OFF'}")
