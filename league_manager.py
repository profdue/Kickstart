LEAGUE_CONFIGS = {
    "EPL": {
        "teams": [
            "Arsenal", "Man City", "Liverpool", "Chelsea", "Tottenham", "Man United",
            "Newcastle", "Brighton", "West Ham", "Crystal Palace", "Wolves", "Fulham",
            "Bournemouth", "Aston Villa", "Brentford", "Everton", "Nott'ham Forest",
            "Luton", "Burnley", "Sheffield Utd"
        ],
        "baselines": {
            "avg_home_xG": 1.65,
            "avg_away_xG": 1.35,
            "avg_goals": 2.75,
            "home_advantage": 1.15,
            "avg_btts_prob": 0.52
        },
        "characteristics": ["high_pace", "physical", "transition_heavy"]
    },
    "La Liga": {
        "teams": [
            "Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Bilbao", "Real Sociedad",
            "Villarreal", "Betis", "Sevilla", "Valencia", "Osasuna", "Getafe", "Girona",
            "Mallorca", "Celta Vigo", "Rayo Vallecano", "Alaves", "Cadiz", "Granada",
            "Almeria", "Las Palmas"
        ],
        "baselines": {
            "avg_home_xG": 1.55,
            "avg_away_xG": 1.25,
            "avg_goals": 2.55,
            "home_advantage": 1.20,
            "avg_btts_prob": 0.48
        },
        "characteristics": ["technical", "possession_based", "slower_pace"]
    },
    "Bundesliga": {
        "teams": [
            "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
            "Union Berlin", "Freiburg", "Wolfsburg", "Mainz", "Monchengladbach",
            "Eintracht Frankfurt", "Koln", "Werder Bremen", "Bochum", "Augsburg",
            "Stuttgart", "Heidenheim", "Darmstadt"
        ],
        "baselines": {
            "avg_home_xG": 1.75,
            "avg_away_xG": 1.45,
            "avg_goals": 3.10,
            "home_advantage": 1.12,
            "avg_btts_prob": 0.58
        },
        "characteristics": ["high_scoring", "pressing", "youth_development"]
    },
    "Serie A": {
        "teams": [
            "Inter", "Juventus", "AC Milan", "Napoli", "Atalanta", "Roma", "Lazio",
            "Fiorentina", "Bologna", "Torino", "Monza", "Udinese", "Sassuolo",
            "Empoli", "Salernitana", "Lecce", "Frosinone", "Genoa", "Verona", "Cagliari"
        ],
        "baselines": {
            "avg_home_xG": 1.50,
            "avg_away_xG": 1.20,
            "avg_goals": 2.55,
            "home_advantage": 1.18,
            "avg_btts_prob": 0.46
        },
        "characteristics": ["tactical", "defensive", "structured"]
    },
    "Ligue 1": {
        "teams": [
            "PSG", "Lens", "Marseille", "Monaco", "Rennes", "Lille", "Nice",
            "Lorient", "Reims", "Lyon", "Montpellier", "Toulouse", "Clermont",
            "Strasbourg", "Nantes", "Brest", "Le Havre", "Metz"
        ],
        "baselines": {
            "avg_home_xG": 1.45,
            "avg_away_xG": 1.15,
            "avg_goals": 2.45,
            "home_advantage": 1.16,
            "avg_btts_prob": 0.44
        },
        "characteristics": ["physical", "transition", "psg_dominated"]
    },
    "RFPL": {
        "teams": [
            "Zenit", "CSKA Moscow", "Spartak Moscow", "Dynamo Moscow", "Krasnodar",
            "Lokomotiv Moscow", "Rostov", "Akhmat", "Sochi", "Orenburg", "Krylya Sovetov",
            "Fakel", "Ural", "NN", "Baltika", "Rubin"
        ],
        "baselines": {
            "avg_home_xG": 1.40,
            "avg_away_xG": 1.10,
            "avg_goals": 2.35,
            "home_advantage": 1.22,
            "avg_btts_prob": 0.42
        },
        "characteristics": ["physical", "defensive", "strong_home_advantage"]
    }
}

def get_league_baselines(league):
    """Get baseline statistics for a specific league."""
    return LEAGUE_CONFIGS.get(league, {}).get("baselines", {})

def get_league_teams(league):
    """Get teams for a specific league."""
    return LEAGUE_CONFIGS.get(league, {}).get("teams", [])