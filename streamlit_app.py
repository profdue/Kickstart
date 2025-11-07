"""
Hybrid Precision Engine - COMPLETE WORKING VERSION
Enhanced with correlation awareness and style-based probability adjustments
"""

from math import exp, factorial
import numpy as np

class HybridPrecisionEngine:
    def __init__(self, constraints=None):
        # Default constraint set (tunable)
        self.constraints = {
            'max_transitions_per_game': 12,        
            'possession_zero_sum': True,           
            'game_state_damping': 0.7,             
            'style_interaction_caps': {
                'HIGH_PRESS_VS_COUNTER': 1.25,
                'POSSESSION_VS_DEEP_DEFENSE': 0.85
            },
            'max_total_xg': 5.0,                   
            'uncertainty_compression_threshold': 0.55,
            'uncertainty_compression_factor': 0.80,
            'draw_correlation_threshold': 0.7,     # When to boost draw probability
            'draw_boost_amount': 0.08,             # How much to boost draws
            'style_win_advantages': {              # Style-based win probability adjustments
                'COUNTER_VS_HIGH_PRESS': 0.10,
                'POSSESSION_VS_DEEP_DEFENSE': 0.08,
                'HIGH_PRESS_VS_POSSESSION': 0.06
            }
        }
        if constraints:
            self.constraints.update(constraints)

    # ------------------------- Utility math -------------------------
    def _poisson_prob(self, lam, k):
        """Compute Poisson probability P(X=k) for mean lam"""
        return (lam ** k) * exp(-lam) / factorial(k)

    # ------------------------- Base expectations -------------------------
    def _calculate_base_expected_goals(self, home_profile, away_profile, context=None):
        """Compute naive base xG for home & away."""
        home_xg = home_profile.get('xg_per_game', 1.0)
        away_xg = away_profile.get('xg_per_game', 1.0)

        # Home advantage
        home_adv_multiplier = 1.0 + 0.14 * (1 if home_profile.get('is_home', True) else 0)
        base_home = home_xg * home_adv_multiplier
        base_away = away_xg

        return {'home': base_home, 'away': base_away}

    # ------------------------- Constraint layer -------------------------
    def _apply_football_constraints(self, base_xg, home_profile, away_profile):
        constrained = base_xg.copy()

        # CONSTRAINT 1: possession zero-sum
        if self.constraints['possession_zero_sum']:
            home_pos = home_profile.get('possession', None)
            away_pos = away_profile.get('possession', None)
            if home_pos is not None and away_pos is not None and (home_pos + away_pos) > 100:
                scale = 100.0 / (home_pos + away_pos)
                constrained['home'] *= scale
                constrained['away'] *= scale

        # CONSTRAINT 2: style interaction caps
        home_style = home_profile.get('style', '').upper()
        away_style = away_profile.get('style', '').upper()

        # High press (away) vs Counter (home)
        if home_style == 'COUNTER' and away_style == 'HIGH_PRESS':
            cap = self.constraints['style_interaction_caps'].get('HIGH_PRESS_VS_COUNTER', 1.25)
            style_boost = min(cap, 1.0 + 0.15 * (base_xg['home'] / max(0.5, base_xg['home'])))
            constrained['home'] *= style_boost

        # CONSTRAINT 3: total xg cap
        total_xg = constrained['home'] + constrained['away']
        if total_xg > self.constraints['max_total_xg']:
            excess = total_xg - self.constraints['max_total_xg']
            damping = 1.0 / (1.0 + excess * 0.3)
            constrained['home'] *= damping
            constrained['away'] *= damping

        # CONSTRAINT 4: open play transition cap
        open_styles = {'HIGH_PRESS', 'ATTACKING', 'POSSESSION', 'OPEN'}
        open_play = 0
        if home_style in open_styles:
            open_play += 1
        if away_style in open_styles:
            open_play += 1
            
        if open_play == 2:
            constrained_total = constrained['home'] + constrained['away']
            max_reasonable = 4.0
            if constrained_total > max_reasonable:
                constrained['home'] *= (max_reasonable / constrained_total)
                constrained['away'] *= (max_reasonable / constrained_total)

        # Ensure non-negative
        constrained['home'] = max(0.01, constrained['home'])
        constrained['away'] = max(0.01, constrained['away'])

        return constrained

    # ------------------------- Game flow simulation -------------------------
    def _simulate_game_flow(self, xg_estimates, home_profile, away_profile):
        """Simulate plausible game-state scenarios"""
        scenarios = []

        home_style = home_profile.get('style', '').upper()
        away_style = away_profile.get('style', '').upper()

        # Base scenario probabilities
        base_home_first = 0.34
        base_away_first = 0.33
        base_gfh = 0.33

        # Style-based adjustments
        if home_style == 'COUNTER' and away_style == 'HIGH_PRESS':
            base_home_first += 0.06
            base_gfh -= 0.03
            base_away_first -= 0.03

        # Normalize
        total = base_home_first + base_away_first + base_gfh
        base_home_first /= total
        base_away_first /= total
        base_gfh /= total

        # Scenario 1: Home scores first
        scenarios.append({
            'prob': base_home_first,
            'home_xg': xg_estimates['home'] * 0.80,
            'away_xg': xg_estimates['away'] * 1.10
        })

        # Scenario 2: Away scores first
        scenarios.append({
            'prob': base_away_first,
            'home_xg': xg_estimates['home'] * 1.20,
            'away_xg': xg_estimates['away'] * 0.90
        })

        # Scenario 3: Goalless first half
        scenarios.append({
            'prob': base_gfh,
            'home_xg': xg_estimates['home'] * 1.10,
            'away_xg': xg_estimates['away'] * 1.10
        })

        final_home = sum(s['prob'] * s['home_xg'] for s in scenarios)
        final_away = sum(s['prob'] * s['away_xg'] for s in scenarios)

        # Apply final damping if needed
        total = final_home + final_away
        if total > 4.0:
            excess = total - 4.0
            damping = 1.0 / (1.0 + excess * 0.25)
            final_home *= damping
            final_away *= damping

        return {'home': final_home, 'away': final_away}

    # ------------------------- Enhanced Probability Conversion -------------------------
    def _xg_to_match_outcome_probs(self, home_xg, away_xg, home_profile, away_profile, max_goals=6):
        """Convert xG to probabilities with correlation awareness and style adjustments"""
        
        # Step 1: Independent Poisson (traditional approach)
        home_probs = [self._poisson_prob(home_xg, k) for k in range(0, max_goals + 1)]
        away_probs = [self._poisson_prob(away_xg, k) for k in range(0, max_goals + 1)]

        # Handle tail mass
        home_tail = 1.0 - sum(home_probs)
        away_tail = 1.0 - sum(away_probs)
        home_probs[-1] += home_tail
        away_probs[-1] += away_tail

        # Compute independent probabilities
        home_win_ind = 0.0
        draw_ind = 0.0
        away_win_ind = 0.0

        for i, ph in enumerate(home_probs):
            for j, pa in enumerate(away_probs):
                p = ph * pa
                if i > j:
                    home_win_ind += p
                elif i == j:
                    draw_ind += p
                else:
                    away_win_ind += p

        # Step 2: Dixon-Coles style correlation adjustment
        xg_ratio = min(home_xg, away_xg) / max(home_xg, away_xg)
        if xg_ratio > self.constraints['draw_correlation_threshold']:
            draw_boost = self.constraints['draw_boost_amount']
            # Reduce both win probabilities to boost draw
            reduction_factor = 1 - (draw_boost / (1 - draw_ind))
            home_win_adj = home_win_ind * reduction_factor
            away_win_adj = away_win_ind * reduction_factor
            draw_adj = draw_ind + draw_boost
        else:
            home_win_adj, draw_adj, away_win_adj = home_win_ind, draw_ind, away_win_ind

        # Step 3: Style-based win probability adjustments
        home_style = home_profile.get('style', '').upper()
        away_style = away_profile.get('style', '').upper()
        
        style_advantage = 0.0
        style_key = f"{home_style}_VS_{away_style}"
        
        if style_key in self.constraints['style_win_advantages']:
            style_advantage = self.constraints['style_win_advantages'][style_key]
            home_win_adj += style_advantage
            # Reduce opponent's win probability more than draw
            away_win_adj -= style_advantage * 0.7
            draw_adj -= style_advantage * 0.3

        # Ensure probabilities are valid
        home_win_adj = max(0.0, home_win_adj)
        away_win_adj = max(0.0, away_win_adj)
        draw_adj = max(0.0, draw_adj)

        # Normalize
        total = home_win_adj + draw_adj + away_win_adj
        if total > 0:
            home_win_adj /= total
            draw_adj /= total
            away_win_adj /= total

        return {
            'home_win': home_win_adj,
            'draw': draw_adj,
            'away_win': away_win_adj
        }

    # ------------------------- Uncertainty calibration -------------------------
    def _apply_uncertainty_calibration(self, probs):
        max_prob = max(probs.values())
        threshold = self.constraints['uncertainty_compression_threshold']
        factor = self.constraints['uncertainty_compression_factor']
        
        if max_prob > threshold:
            compressed = {}
            largest = max(probs, key=probs.get)
            for k, v in probs.items():
                if k == largest:
                    compressed[k] = v * factor
                else:
                    # Distribute the reduction proportionally
                    compressed[k] = v + (v / (1 - probs[largest] + 1e-12)) * (v * (1 - factor))
            
            # Normalize
            total = sum(compressed.values())
            for k in compressed:
                compressed[k] /= total
            return compressed
        
        return probs

    # ------------------------- Additional Market Calculations -------------------------
    def _calculate_additional_markets(self, home_xg, away_xg, max_goals=6):
        """Calculate Over/Under and BTTS probabilities"""
        over_2_5 = 0.0
        btts_yes = 0.0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = self._poisson_prob(home_xg, i) * self._poisson_prob(away_xg, j)
                # Over 2.5
                if i + j > 2.5:
                    over_2_5 += prob
                # Both Teams to Score
                if i > 0 and j > 0:
                    btts_yes += prob
        
        return {
            'over_2.5': over_2_5,
            'under_2.5': 1 - over_2_5,
            'btts_yes': btts_yes,
            'btts_no': 1 - btts_yes
        }

    # ------------------------- Public API -------------------------
    def predict_match(self, home_profile, away_profile, context=None):
        """Main prediction method - returns comprehensive match analysis"""
        
        # 1. Base expected goals
        base = self._calculate_base_expected_goals(home_profile, away_profile, context)

        # 2. Apply football constraints
        constrained = self._apply_football_constraints(base, home_profile, away_profile)

        # 3. Simulate game flow dynamics
        dynamic = self._simulate_game_flow(constrained, home_profile, away_profile)

        # 4. Convert to outcome probabilities with enhanced model
        raw_probs = self._xg_to_match_outcome_probs(
            dynamic['home'], dynamic['away'], home_profile, away_profile
        )

        # 5. Calibrate uncertainty
        calibrated_probs = self._apply_uncertainty_calibration(raw_probs)

        # 6. Calculate additional markets
        additional_markets = self._calculate_additional_markets(dynamic['home'], dynamic['away'])

        # 7. Calculate confidence
        confidence = self._calculate_confidence(calibrated_probs, dynamic)

        # Build comprehensive output
        output = {
            'expected_goals': {
                'base': base,
                'constrained': constrained,
                'final': dynamic
            },
            'probabilities': {
                'raw': raw_probs,
                'calibrated': calibrated_probs
            },
            'additional_markets': additional_markets,
            'confidence': confidence,
            'key_insights': self._generate_insights(home_profile, away_profile, calibrated_probs, dynamic)
        }
        
        return output

    def _calculate_confidence(self, probs, xg_data):
        """Calculate prediction confidence score"""
        max_prob = max(probs.values())
        xg_diff = abs(xg_data['home'] - xg_data['away'])
        total_xg = xg_data['home'] + xg_data['away']
        
        # Confidence factors
        clarity_score = max_prob  # Higher max probability = more clarity
        xg_balance_score = 1.0 - (xg_diff / max(total_xg, 1.0))  # More balanced = less confident
        
        # Combined confidence (weighted)
        confidence_score = (clarity_score * 0.7) + (xg_balance_score * 0.3)
        
        if confidence_score >= 0.7:
            return "HIGH"
        elif confidence_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_insights(self, home_profile, away_profile, probs, xg_data):
        """Generate tactical insights about the match"""
        insights = []
        
        home_style = home_profile.get('style', '').upper()
        away_style = away_profile.get('style', '').upper()
        
        # Style matchup insights
        if home_style == 'COUNTER' and away_style == 'HIGH_PRESS':
            insights.append("Tottenham's counter-attacking style perfectly exploits United's high press")
        
        if xg_data['home'] + xg_data['away'] > 3.5:
            insights.append("High expected goal total suggests an open, entertaining match")
        
        if probs['draw'] > 0.25:
            insights.append("Close match expected with significant draw probability")
        
        # Confidence insights
        max_prob = max(probs.values())
        if max_prob > 0.45:
            winning_team = "Tottenham" if probs['home_win'] == max_prob else "Man United"
            insights.append(f"{winning_team} has a clear advantage in this matchup")
        
        return insights

# ------------------------- Demo Execution -------------------------
def run_demo():
    """Run the Tottenham vs Man United demo"""
    print("⚽ HYBRID PRECISION ENGINE - COMPLETE DEMO")
    print("=" * 50)
    
    engine = HybridPrecisionEngine()

    # Tottenham (Home) - using stats from our conversation
    tottenham = {
        'name': 'Tottenham',
        'xg_per_game': 1.01,     # 10.10 xG / 10 games
        'possession': 53,
        'style': 'COUNTER',
        'is_home': True
    }

    # Man United (Away)
    manunited = {
        'name': 'Man United', 
        'xg_per_game': 1.78,     # 17.80 xG / 10 games
        'possession': 51,
        'style': 'HIGH_PRESS',
        'is_home': False
    }

    print("🏠 HOME: Tottenham")
    print(f"   xG/game: {tottenham['xg_per_game']}, Possession: {tottenham['possession']}%, Style: {tottenham['style']}")
    
    print("✈️ AWAY: Man United")
    print(f"   xG/game: {manunited['xg_per_game']}, Possession: {manunited['possession']}%, Style: {manunited['style']}")
    
    print("\n" + "=" * 50)
    print("🎯 GENERATING PREDICTION...")
    print("=" * 50)

    result = engine.predict_match(tottenham, manunited)

    # Display results
    print("\n📊 EXPECTED GOALS PROGRESSION:")
    print(f"   Base:        Tottenham {result['expected_goals']['base']['home']:.2f} - {result['expected_goals']['base']['away']:.2f} Man United")
    print(f"   Constrained: Tottenham {result['expected_goals']['constrained']['home']:.2f} - {result['expected_goals']['constrained']['away']:.2f} Man United")
    print(f"   Final:       Tottenham {result['expected_goals']['final']['home']:.2f} - {result['expected_goals']['final']['away']:.2f} Man United")
    print(f"   Total xG:    {result['expected_goals']['final']['home'] + result['expected_goals']['final']['away']:.2f}")

    print(f"\n🏆 MATCH OUTCOME PROBABILITIES ({result['confidence']} CONFIDENCE):")
    print(f"   Tottenham Win: {result['probabilities']['calibrated']['home_win']:.1%}")
    print(f"   Draw:          {result['probabilities']['calibrated']['draw']:.1%}") 
    print(f"   Man United Win: {result['probabilities']['calibrated']['away_win']:.1%}")

    print(f"\n📈 ADDITIONAL MARKETS:")
    print(f"   Over 2.5 Goals:  {result['additional_markets']['over_2.5']:.1%}")
    print(f"   Under 2.5 Goals: {result['additional_markets']['under_2.5']:.1%}")
    print(f"   BTTS Yes:        {result['additional_markets']['btts_yes']:.1%}")
    print(f"   BTTS No:         {result['additional_markets']['btts_no']:.1%}")

    print(f"\n💡 KEY INSIGHTS:")
    for insight in result['key_insights']:
        print(f"   • {insight}")

    print(f"\n🎪 COMPARISON WITH ORIGINAL APP:")
    print(f"   Our Total xG: {result['expected_goals']['final']['home'] + result['expected_goals']['final']['away']:.1f} vs App's 5.2")
    print(f"   Our Tottenham Win: {result['probabilities']['calibrated']['home_win']:.1%} vs App's 52.7%")
    print(f"   Our Over 2.5: {result['additional_markets']['over_2.5']:.1%} vs App's 87.6%")

if __name__ == '__main__':
    run_demo()
