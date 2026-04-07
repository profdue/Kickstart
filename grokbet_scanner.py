#!/usr/bin/env python3
"""
GrokBet Sniper System - Automated Football Betting Scanner
Runs on GitHub Actions, sends Telegram alerts when locks found
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging for GitHub Actions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (from GitHub Secrets)
# ============================================================================

FOOTBALL_API_KEY = os.environ.get('FOOTBALL_API_KEY', '')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# Validate configuration
if not FOOTBALL_API_KEY:
    logger.error("❌ FOOTBALL_API_KEY not set in secrets")
    exit(1)

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("⚠️ Telegram credentials missing - will run in test mode")

# ============================================================================
# RATE-LIMITED API PROVIDER
# ============================================================================

class RateLimitedFootballProvider:
    def __init__(self, api_key: str, requests_per_minute: int = 8):
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': api_key}
        self.request_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.cache = {}
        self.request_count = 0
        self.start_time = datetime.now()
    
    def wait_if_needed(self):
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            wait_time = self.request_interval - time_since_last
            logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def make_request(self, url: str, params: dict = None, use_cache: bool = True):
        import time
        cache_key = f"{url}_{json.dumps(params, sort_keys=True) if params else ''}"
        
        if use_cache and cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < 3600:
                return cache_data
        
        self.wait_if_needed()
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if use_cache:
                    self.cache[cache_key] = (datetime.now(), data)
                return data
            elif response.status_code == 429:
                logger.warning("Rate limit hit, waiting 10s...")
                time.sleep(10)
                return self.make_request(url, params, use_cache)
            else:
                logger.error(f"API Error {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def get_upcoming_matches(self, league_code: str = 'PL', days_ahead: int = 3):
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/competitions/{league_code}/matches"
        params = {'status': 'SCHEDULED', 'dateFrom': start_date, 'dateTo': end_date, 'limit': 30}
        
        data = self.make_request(url, params)
        if data and 'matches' in data:
            matches = [m for m in data['matches'] if m.get('utcDate', '')[:10] <= end_date]
            logger.info(f"Found {len(matches)} matches in {league_code}")
            return matches
        return []
    
    def get_team_stats(self, team_id: int, competition: str = 'PL'):
        cache_key = f"stats_{team_id}_{competition}"
        url = f"{self.base_url}/teams/{team_id}/matches"
        params = {'competition': competition, 'limit': 10, 'status': 'FINISHED'}
        
        data = self.make_request(url, params, use_cache=True)
        if not data or 'matches' not in data or not data['matches']:
            return {'scored_avg': 1.5, 'conceded_avg': 1.5, 'form': 50, 'conv_rate': 12}
        
        matches = data['matches']
        total_goals_for = 0
        total_goals_against = 0
        wins = draws = 0
        
        for match in matches:
            if match['homeTeam']['id'] == team_id:
                goals_for = match['score']['fullTime']['home'] or 0
                goals_against = match['score']['fullTime']['away'] or 0
            else:
                goals_for = match['score']['fullTime']['away'] or 0
                goals_against = match['score']['fullTime']['home'] or 0
            
            total_goals_for += goals_for
            total_goals_against += goals_against
            if goals_for > goals_against:
                wins += 1
            elif goals_for == goals_against:
                draws += 1
        
        played = len(matches)
        scored_avg = total_goals_for / played
        conceded_avg = total_goals_against / played
        form = self._calculate_form(matches[-5:], team_id)
        conv_rate = min(25, max(8, (scored_avg / 1.5) * 12))
        
        return {
            'scored_avg': round(scored_avg, 2),
            'conceded_avg': round(conceded_avg, 2),
            'form': round(form, 1),
            'conv_rate': round(conv_rate, 1)
        }
    
    def _calculate_form(self, matches: list, team_id: int) -> float:
        if not matches:
            return 50
        points = 0
        for match in matches:
            if match['homeTeam']['id'] == team_id:
                if match['score']['fullTime']['home'] > match['score']['fullTime']['away']:
                    points += 3
                elif match['score']['fullTime']['home'] == match['score']['fullTime']['away']:
                    points += 1
            else:
                if match['score']['fullTime']['away'] > match['score']['fullTime']['home']:
                    points += 3
                elif match['score']['fullTime']['away'] == match['score']['fullTime']['home']:
                    points += 1
        return (points / (len(matches) * 3)) * 100
    
    def get_h2h(self, team1_id: int, team2_id: int, limit: int = 5) -> Tuple[int, int, int]:
        cache_key = f"h2h_{team1_id}_{team2_id}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < 86400:
                return cache_data
        
        url = f"{self.base_url}/teams/{team1_id}/matches"
        params = {'limit': 30, 'status': 'FINISHED'}
        data = self.make_request(url, params)
        
        if not data or 'matches' not in data:
            return (1, 2, 1)
        
        team1_wins = team2_wins = draws = count = 0
        for match in data['matches']:
            if count >= limit:
                break
            opponent_id = match['awayTeam']['id'] if match['homeTeam']['id'] == team1_id else match['homeTeam']['id']
            if opponent_id == team2_id:
                if match['homeTeam']['id'] == team1_id:
                    home_score = match['score']['fullTime']['home'] or 0
                    away_score = match['score']['fullTime']['away'] or 0
                else:
                    home_score = match['score']['fullTime']['away'] or 0
                    away_score = match['score']['fullTime']['home'] or 0
                
                if home_score > away_score:
                    team1_wins += 1
                elif away_score > home_score:
                    team2_wins += 1
                else:
                    draws += 1
                count += 1
        
        result = (team1_wins, draws, team2_wins) if count > 0 else (1, 2, 1)
        self.cache[cache_key] = (datetime.now(), result)
        return result
    
    def print_stats(self):
        elapsed = (datetime.now() - self.start_time).seconds
        logger.info(f"📊 API Stats: {self.request_count} requests, {len(self.cache)} cached, {self.request_count / (elapsed/60):.1f}/min")


# ============================================================================
# GROKBET LOCK DETECTOR
# ============================================================================

class GrokBetLockDetector:
    @staticmethod
    def calculate_efficiency(scored_avg: float, conceded_avg: float, form_pct: float, conv_pct: float) -> float:
        conv_decimal = conv_pct / 100.0
        form_decimal = form_pct / 100.0
        weakness_multiplier = 1.0 - form_decimal
        attack_score = scored_avg * conv_decimal
        defense_penalty = conceded_avg * weakness_multiplier
        return attack_score - defense_penalty
    
    @staticmethod
    def check_locks(home_team: str, away_team: str,
                    home_scored: float, home_conceded: float,
                    away_scored: float, away_conceded: float,
                    home_form: float, away_form: float,
                    home_conv: float, away_conv: float,
                    h2h_home: int, h2h_away: int,
                    total_xg: float, efficiency_gap: float) -> List[Dict]:
        
        locks = []
        
        # Lock thresholds
        WEAK_CONV, WEAK_SCORED = 10, 1.2
        ELITE_ATTACK_SCORED, LOW_XG_THRESHOLD = 1.5, 2.5
        HIGH_XG_THRESHOLD, GOOD_CONV = 3.0, 11
        ELITE_DEFENSE_WARNING = 0.8
        HOME_FORM_THRESHOLD, AWAY_FORM_THRESHOLD = 60, 60
        H2H_HOME_MIN, H2H_AWAY_MIN = 2, 2
        HOME_SCORED_MIN, AWAY_SCORED_MIN = 1.5, 1.5
        SMALL_GAP_MAX, BALANCED_FORM_MIN, BALANCED_FORM_MAX = 0.3, 40, 60
        DRAW_XG_MAX = 2.5
        
        # LOCK #1: BTTS NO
        weak_attack_home = home_conv <= WEAK_CONV and home_scored <= WEAK_SCORED
        weak_attack_away = away_conv <= WEAK_CONV and away_scored <= WEAK_SCORED
        low_xg = total_xg <= LOW_XG_THRESHOLD
        
        if weak_attack_home and away_scored <= ELITE_ATTACK_SCORED and low_xg:
            locks.append({'type': 'BTTS No', 'bet': 'BTTS No', 'confidence': 'HIGH',
                         'reason': f"LOCK #1: {home_team} weak attack ({home_conv}%, {home_scored:.2f}) + low xG ({total_xg:.2f})"})
        elif weak_attack_away and home_scored <= ELITE_ATTACK_SCORED and low_xg:
            locks.append({'type': 'BTTS No', 'bet': 'BTTS No', 'confidence': 'HIGH',
                         'reason': f"LOCK #1: {away_team} weak attack ({away_conv}%, {away_scored:.2f}) + low xG ({total_xg:.2f})"})
        
        # LOCK #2: High-Scoring
        if total_xg >= HIGH_XG_THRESHOLD and home_conv >= GOOD_CONV and away_conv >= GOOD_CONV:
            if home_conceded >= ELITE_DEFENSE_WARNING and away_conceded >= ELITE_DEFENSE_WARNING:
                locks.append({'type': 'High-Scoring', 'bet': 'BTTS Yes & Over 2.5', 'confidence': 'HIGH',
                             'reason': f"LOCK #2: High xG ({total_xg:.2f}) + good conversion ({home_conv}%/{away_conv}%)"})
        
        # LOCK #3: Winner
        if efficiency_gap > 0 and home_form >= HOME_FORM_THRESHOLD:
            if h2h_home >= H2H_HOME_MIN or home_scored >= HOME_SCORED_MIN:
                locks.append({'type': 'Winner', 'bet': f'{home_team} to Win', 'confidence': 'HIGH',
                             'reason': f"LOCK #3: Positive gap ({efficiency_gap:+.3f}) + home form {home_form:.0f}%"})
        
        if efficiency_gap < 0 and away_form >= AWAY_FORM_THRESHOLD:
            if h2h_away >= H2H_AWAY_MIN or away_scored >= AWAY_SCORED_MIN:
                locks.append({'type': 'Winner', 'bet': f'{away_team} to Win', 'confidence': 'HIGH',
                             'reason': f"LOCK #3: Negative gap ({efficiency_gap:+.3f}) + away form {away_form:.0f}%"})
        
        # LOCK #4: Draw
        if abs(efficiency_gap) <= SMALL_GAP_MAX:
            if (BALANCED_FORM_MIN <= home_form <= BALANCED_FORM_MAX and 
                BALANCED_FORM_MIN <= away_form <= BALANCED_FORM_MAX):
                if total_xg <= DRAW_XG_MAX:
                    locks.append({'type': 'Draw', 'bet': 'Draw', 'confidence': 'MEDIUM',
                                 'reason': f"LOCK #4: Small gap ({abs(efficiency_gap):.3f}) + balanced form ({home_form:.0f}%/{away_form:.0f}%)"})
        
        return locks


# ============================================================================
# TELEGRAM NOTIFIER
# ============================================================================

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str) -> bool:
        if not self.bot_token or not self.chat_id:
            return False
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def send_lock_alert(self, match_info: Dict, lock: Dict) -> bool:
        message = f"""
🔫 <b>SNIPER SYSTEM - LOCK DETECTED!</b>

<b>🎯 Match:</b> {match_info['home_team']} vs {match_info['away_team']}
<b>🏆 League:</b> {match_info['league']}
<b>📅 Date:</b> {match_info['date']}
<b>🔒 Lock Type:</b> {lock['type']}
<b>✅ Best Bet:</b> {lock['bet']}
<b>🎯 Confidence:</b> {lock['confidence']}

<b>📊 Stats:</b>
• Home Form: {match_info['home_form']:.0f}%
• Away Form: {match_info['away_form']:.0f}%
• Total xG: {match_info['total_xg']:.2f}
• Efficiency Gap: {match_info['efficiency_gap']:+.3f}
• H2H: {match_info['h2h_home']}-{match_info['h2h_draws']}-{match_info['h2h_away']}

<b>💡 Reason:</b>
{lock['reason']}

<b>💰 Suggested Stake:</b> 1.0% (LOCK bet)

<i>#SniperSystem #{lock['type'].replace('-', '')} #GrokBet</i>
"""
        return self.send_message(message)


# ============================================================================
# ALERT MANAGER (Prevents duplicates)
# ============================================================================

class AlertManager:
    def __init__(self, filename: str = 'sent_alerts.json'):
        self.filename = filename
        self.sent_alerts = self._load()
    
    def _load(self) -> Dict:
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.sent_alerts, f, indent=2)
    
    def already_sent(self, match_id: str, lock_type: str) -> bool:
        key = f"{match_id}_{lock_type}"
        if key in self.sent_alerts:
            last_sent = datetime.fromisoformat(self.sent_alerts[key])
            if (datetime.now() - last_sent).total_seconds() < 86400:  # 24 hours
                return True
        return False
    
    def mark_sent(self, match_id: str, lock_type: str):
        key = f"{match_id}_{lock_type}"
        self.sent_alerts[key] = datetime.now().isoformat()
        self._save()


# ============================================================================
# MAIN SCANNER
# ============================================================================

class GrokBetScanner:
    def __init__(self):
        self.provider = RateLimitedFootballProvider(FOOTBALL_API_KEY, requests_per_minute=8)
        self.detector = GrokBetLockDetector()
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.alert_manager = AlertManager()
        self.results = []
        
        # Leagues to scan (add more as needed)
        self.leagues = {
            'PL': 'Premier League',
            'PD': 'La Liga',
            'BL1': 'Bundesliga',
            'SA': 'Serie A',
            'FL1': 'Ligue 1',
        }
    
    def scan_match(self, match: Dict, league_code: str, league_name: str):
        """Scan a single match for locks"""
        try:
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            match_id = str(match['id'])
            match_date = match['utcDate'][:10]
            
            logger.info(f"🔍 {home_team} vs {away_team}")
            
            home_id = match['homeTeam']['id']
            away_id = match['awayTeam']['id']
            
            home_stats = self.provider.get_team_stats(home_id, league_code)
            away_stats = self.provider.get_team_stats(away_id, league_code)
            h2h_home, h2h_draws, h2h_away = self.provider.get_h2h(home_id, away_id)
            
            # Calculate metrics
            home_xg = (home_stats['scored_avg'] + away_stats['conceded_avg']) / 2
            away_xg = (away_stats['scored_avg'] + home_stats['conceded_avg']) / 2
            total_xg = home_xg + away_xg
            
            home_efficiency = self.detector.calculate_efficiency(
                home_stats['scored_avg'], home_stats['conceded_avg'],
                home_stats['form'], home_stats['conv_rate']
            )
            away_efficiency = self.detector.calculate_efficiency(
                away_stats['scored_avg'], away_stats['conceded_avg'],
                away_stats['form'], away_stats['conv_rate']
            )
            efficiency_gap = home_efficiency - away_efficiency
            
            locks = self.detector.check_locks(
                home_team, away_team,
                home_stats['scored_avg'], home_stats['conceded_avg'],
                away_stats['scored_avg'], away_stats['conceded_avg'],
                home_stats['form'], away_stats['form'],
                home_stats['conv_rate'], away_stats['conv_rate'],
                h2h_home, h2h_away, total_xg, efficiency_gap
            )
            
            if locks:
                match_info = {
                    'home_team': home_team, 'away_team': away_team,
                    'league': league_name, 'date': match_date,
                    'home_form': home_stats['form'], 'away_form': away_stats['form'],
                    'total_xg': total_xg, 'efficiency_gap': efficiency_gap,
                    'h2h_home': h2h_home, 'h2h_draws': h2h_draws, 'h2h_away': h2h_away
                }
                
                for lock in locks:
                    if not self.alert_manager.already_sent(match_id, lock['type']):
                        logger.info(f"  🔒 {lock['type']} LOCK - {lock['bet']}")
                        
                        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                            self.telegram.send_lock_alert(match_info, lock)
                        
                        self.alert_manager.mark_sent(match_id, lock['type'])
                        
                        self.results.append({
                            'match': f"{home_team} vs {away_team}",
                            'date': match_date, 'lock_type': lock['type'],
                            'bet': lock['bet'], 'confidence': lock['confidence']
                        })
                    else:
                        logger.info(f"  ⏭️ {lock['type']} already alerted (24h)")
            
            return locks
            
        except Exception as e:
            logger.error(f"Error scanning {match.get('homeTeam', {}).get('name', 'unknown')}: {e}")
            return []
    
    def scan_all(self, days_ahead: int = 3, max_matches: int = 10):
        """Scan all leagues for locks"""
        logger.info("="*60)
        logger.info(f"🎯 GrokBet Sniper Scan - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Scanning next {days_ahead} days, max {max_matches} matches/league")
        logger.info("="*60)
        
        total_matches = 0
        
        for league_code, league_name in self.leagues.items():
            logger.info(f"\n📊 {league_name} ({league_code})")
            matches = self.provider.get_upcoming_matches(league_code, days_ahead)
            matches = matches[:max_matches]
            total_matches += len(matches)
            
            for i, match in enumerate(matches, 1):
                logger.info(f"[{i}/{len(matches)}] ", end="")
                self.scan_match(match, league_code, league_name)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("📊 SCAN SUMMARY")
        logger.info("="*60)
        logger.info(f"Leagues scanned: {len(self.leagues)}")
        logger.info(f"Matches analyzed: {total_matches}")
        logger.info(f"New locks found: {len(self.results)}")
        
        self.provider.print_stats()
        
        if self.results:
            logger.info("\n🎯 LOCKS FOUND:")
            for r in self.results:
                logger.info(f"  • {r['match']}: {r['lock_type']} - {r['bet']}")
        
        return self.results


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("🔫 GrokBet Sniper System Starting...")
    
    scanner = GrokBetScanner()
    results = scanner.scan_all(days_ahead=3, max_matches=10)
    
    # Save results for GitHub artifact
    with open('scan_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Send summary to Telegram if there are results
    if results and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        summary = f"""
🔫 <b>GROKBET SNIPER - SCAN COMPLETE</b>

📊 Found <b>{len(results)}</b> new locks:

"""
        for r in results[:5]:
            summary += f"• {r['match']}: <b>{r['lock_type']}</b> → {r['bet']}\n"
        
        if len(results) > 5:
            summary += f"\n+ {len(results) - 5} more locks"
        
        TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).send_message(summary)
    
    logger.info(f"✅ Scan complete. Found {len(results)} locks")
    return results


if __name__ == "__main__":
    main()
