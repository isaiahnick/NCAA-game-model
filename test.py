import pandas as pd
import numpy as np

# -----------------------------------------
# 1. Load Excel Data
# -----------------------------------------
def load_data(filepath):
    """Load and prepare data from Excel file."""
    xls = pd.ExcelFile(filepath, engine="openpyxl")
    
    # Parse all sheets
    Auburn_df = xls.parse("Auburn Player Stats")
    Florida_df = xls.parse("Florida Player Stats")
    Auburn_team_df = xls.parse("Auburn Team Stats")
    Florida_team_df = xls.parse("Florida Team Stats")
    
    # Strip column names for consistency
    Auburn_df.columns = Auburn_df.columns.str.strip()
    Florida_df.columns = Florida_df.columns.str.strip()
    
    # Filter out players marked for exclusion
    if "Disclude?" in Auburn_df.columns:
        Auburn_df = Auburn_df[~(Auburn_df["Disclude?"] == "x")]
    
    if "Disclude?" in Florida_df.columns:
        Florida_df = Florida_df[~(Florida_df["Disclude?"] == "x")]
    
    return Auburn_df, Florida_df, Auburn_team_df, Florida_team_df

# -----------------------------------------
# 2. Helper Functions
# -----------------------------------------
def get_team_stat(team_df, stat_name):
    """Extract a specific team stat from the team stats dataframe."""
    for i, row in enumerate(team_df.values):
        if isinstance(row[0], str) and row[0].strip() == stat_name:
            # Convert to float and handle potential string values
            try:
                off_val = float(row[1]) if row[1] is not None else None
                def_val = float(row[2]) if row[2] is not None else None
                return off_val, def_val  # Offensive, Defensive values
            except (ValueError, TypeError):
                return row[1], row[2]  # If conversion fails, return as is
    return None, None

def split_made_attempted(series):
    """Split strings like '115-194' into made and attempted values."""
    return series.astype(str).str.extract(r'(?P<made>\d+)-(?P<att>\d+)').astype(float).fillna(0)

def calculate_position_factor(height_in, weight_lb):
    """Estimate a player's position based on height and weight.
    1 = point guard, 5 = center
    """
    # Handle non-numeric inputs
    try:
        height_in = float(height_in)
        weight_lb = float(weight_lb)
    except (ValueError, TypeError):
        return 3.0  # Default to middle position if conversion fails
    
    # Normalize height and weight to 0-1 scale based on typical ranges
    height_factor = (height_in - 68) / (84 - 68) if height_in >= 68 else 0  # 5'8" to 7'0"
    weight_factor = (weight_lb - 160) / (260 - 160) if weight_lb >= 160 else 0  # 160 to 260 lbs
    
    # Clip to 0-1 range
    height_factor = max(0, min(1, height_factor))
    weight_factor = max(0, min(1, weight_factor))
    
    # Combine factors (weight more important for determining bigs)
    position_factor = 0.4 * height_factor + 0.6 * weight_factor
    
    # Scale to 1-5 range
    return 1 + position_factor * 4

def get_matchup_advantage(player_df, opp_df):
    """Calculate defensive matchup advantages for each player."""
    # Make copies to avoid modifying originals
    player_df = player_df.copy()
    opp_df = opp_df.copy()
    
    # Calculate position factors for all players
    player_df['Position_Factor'] = player_df.apply(
        lambda x: calculate_position_factor(x['Ht'], x['Wt']), axis=1)
    opp_df['Position_Factor'] = opp_df.apply(
        lambda x: calculate_position_factor(x['Ht'], x['Wt']), axis=1)
    
    # Calculate defensive metrics for opponent players
    opp_df['Blk%'] = pd.to_numeric(opp_df['Blk%'], errors='coerce').fillna(0)
    opp_df['Stl%'] = pd.to_numeric(opp_df['Stl%'], errors='coerce').fillna(0)
    opp_df['DR%'] = pd.to_numeric(opp_df['DR%'], errors='coerce').fillna(0)
    
    # Create a defensive rating composite
    opp_df['Def_Rating'] = (
        opp_df['Blk%'] + 
        opp_df['Stl%'] * 2 + 
        opp_df['DR%'] / 2
    ) / 3
    
    # Normalize to 0-1 scale
    min_def = opp_df['Def_Rating'].min()
    max_def = opp_df['Def_Rating'].max()
    range_def = max_def - min_def if max_def > min_def else 1
    opp_df['Def_Rating_Norm'] = (opp_df['Def_Rating'] - min_def) / range_def
    
    # Create matchup matrix
    player_advantages = {}
    
    for _, player in player_df.iterrows():
        # Find closest position matches among opponents
        opp_df['Pos_Diff'] = abs(opp_df['Position_Factor'] - player['Position_Factor'])
        
        # Convert %Min to numeric
        opp_df['%Min'] = pd.to_numeric(opp_df['%Min'], errors='coerce').fillna(0)
        
        # Weight by minutes played (more minutes = more likely to defend this player)
        opp_df['Matchup_Likelihood'] = (1 / (1 + opp_df['Pos_Diff'])) * (opp_df['%Min'] / 100)
        
        # Normalize likelihoods to sum to 1
        total_likelihood = opp_df['Matchup_Likelihood'].sum()
        if total_likelihood > 0:
            opp_df['Matchup_Likelihood'] = opp_df['Matchup_Likelihood'] / total_likelihood
        
        # Calculate weighted average defensive impact
        def_impact = (opp_df['Matchup_Likelihood'] * opp_df['Def_Rating_Norm']).sum()
        
        # Convert to adjustment factor (0.85 - 1.15 range)
        # Better defenders result in lower offensive efficiency
        adjustment = 1.15 - (0.3 * def_impact)
        
        player_advantages[player['Player']] = adjustment
    
    return player_advantages

def calculate_assists(player_df, team_expected_fg_made, matchup_advantages):
    """
    Calculate expected assists for each player on a team.
    """
    # Make a copy to avoid modifying the original dataframe
    df = player_df.copy()
    
    # Convert ARate to numeric, handling potential string values
    df['ARate'] = pd.to_numeric(df['ARate'], errors='coerce').fillna(0)
    
    # Calculate minutes adjustment factor
    df['Min_Factor'] = df['Adj_Min%'] / 100 if 'Adj_Min%' in df.columns else df['%Min'] / 100
    
    # Calculate each player's share of potential assists based on ARate and minutes
    df['Assist_Share'] = df['ARate'] * df['Min_Factor']
    
    # Normalize to sum to 1
    if df['Assist_Share'].sum() > 0:
        df['Assist_Share'] = df['Assist_Share'] / df['Assist_Share'].sum()
    else:
        # If all zeros, distribute evenly
        df['Assist_Share'] = 1 / len(df)
    
    # Apply matchup adjustments to assist share
    df['Matchup_Adj'] = df['Player'].map(
        {player: 1/adv for player, adv in matchup_advantages.items()}
    ).fillna(1.0)
    
    # Adjust assist share based on matchups
    df['Adj_Assist_Share'] = df['Assist_Share'] * df['Matchup_Adj']
    
    # Normalize again after adjustments
    if df['Adj_Assist_Share'].sum() > 0:
        df['Adj_Assist_Share'] = df['Adj_Assist_Share'] / df['Adj_Assist_Share'].sum()
    else:
        df['Adj_Assist_Share'] = 1 / len(df)
    
    # Calculate the expected total assists for the team
    assist_rate = 0.54  # 54% of field goals are assisted on average
    total_team_assists = team_expected_fg_made * assist_rate
    
    # Distribute the total assists according to each player's adjusted share
    return df['Adj_Assist_Share'] * total_team_assists

# -----------------------------------------
# 3. Model Function
# -----------------------------------------
def estimate_player_stats(team1_stats_df, team2_stats_df, team1_team_df, team2_team_df, home_team="team1"):
    """
    Model to estimate player stats using additive effects.
    """
    # --- Step 1: Extract base NCAA averages ---
    ncaa_avg_off_eff = 107.0  # NCAA average offensive efficiency
    ncaa_avg_def_eff = 107.0  # NCAA average defensive efficiency
    ncaa_avg_pace = 67.5      # NCAA average possessions per game
    
    # --- Step 2: Extract team metrics ---
    # Get offensive and defensive efficiency 
    team1_off_eff, team1_def_eff = get_team_stat(team1_team_df, "Adj. Efficiency")
    team2_off_eff, team2_def_eff = get_team_stat(team2_team_df, "Adj. Efficiency")
    
    # Default to NCAA average if not found
    team1_off_eff = team1_off_eff if team1_off_eff else ncaa_avg_off_eff
    team1_def_eff = team1_def_eff if team1_def_eff else ncaa_avg_def_eff
    team2_off_eff = team2_off_eff if team2_off_eff else ncaa_avg_off_eff
    team2_def_eff = team2_def_eff if team2_def_eff else ncaa_avg_def_eff
    
    # Get pace metrics
    team1_pace, _ = get_team_stat(team1_team_df, "Adj. Tempo")
    team2_pace, _ = get_team_stat(team2_team_df, "Adj. Tempo")
    
    # Default to NCAA average if not found
    team1_pace = team1_pace if team1_pace else ncaa_avg_pace
    team2_pace = team2_pace if team2_pace else ncaa_avg_pace
    
    # Calculate expected possessions using additive model
    expected_possessions = ncaa_avg_pace - (ncaa_avg_pace - team1_pace) - (ncaa_avg_pace - team2_pace)
    
    # Apply small home court advantage if applicable
    if home_team == "team1" or home_team == "team2":
        expected_possessions += 1.0  # Home teams get approximately 1 extra possession
    
    # --- Step 3: Calculate expected team offensive efficiency ---
    # Calculate adjusted offensive efficiency using the additive model
    team1_expected_off_eff = ncaa_avg_off_eff - (ncaa_avg_off_eff - team1_off_eff) - (ncaa_avg_off_eff - team2_def_eff)
    team2_expected_off_eff = ncaa_avg_off_eff - (ncaa_avg_off_eff - team2_off_eff) - (ncaa_avg_off_eff - team1_def_eff)
    
    # Calculate expected points
    team1_expected_points = expected_possessions * (team1_expected_off_eff / 100)
    team2_expected_points = expected_possessions * (team2_expected_off_eff / 100)
    
    # --- Step 4: Get additional team stats ---
    # Get turnover percentages
    team1_off_to_pct, team1_def_to_pct = get_team_stat(team1_team_df, "Turnover %:")
    team2_off_to_pct, team2_def_to_pct = get_team_stat(team2_team_df, "Turnover %:")
    
    # Convert to proper decimal form
    team1_off_to_pct = team1_off_to_pct / 100 if team1_off_to_pct else 0.17
    team1_def_to_pct = team1_def_to_pct / 100 if team1_def_to_pct else 0.17
    team2_off_to_pct = team2_off_to_pct / 100 if team2_off_to_pct else 0.17
    team2_def_to_pct = team2_def_to_pct / 100 if team2_def_to_pct else 0.17
    
    # Calculate expected turnover rates using additive model
    ncaa_avg_to = 0.172  # NCAA average turnover rate
    team1_expected_to = ncaa_avg_to - (ncaa_avg_to - team1_off_to_pct) - (ncaa_avg_to - team2_def_to_pct)
    team2_expected_to = ncaa_avg_to - (ncaa_avg_to - team2_off_to_pct) - (ncaa_avg_to - team1_def_to_pct)
    
    # Adjust possessions by turnover rates
    team1_adj_poss = expected_possessions * (1 - team1_expected_to)
    team2_adj_poss = expected_possessions * (1 - team2_expected_to)
    
    # Get 3PA/FGA rates
    team1_3pafga_off, team1_3pafga_def = get_team_stat(team1_team_df, "3PA/FGA:")
    team2_3pafga_off, team2_3pafga_def = get_team_stat(team2_team_df, "3PA/FGA:")
    
    # Convert to proper decimal form
    team1_3pafga_off = team1_3pafga_off / 100 if team1_3pafga_off else 0.39
    team1_3pafga_def = team1_3pafga_def / 100 if team1_3pafga_def else 0.39
    team2_3pafga_off = team2_3pafga_off / 100 if team2_3pafga_off else 0.39
    team2_3pafga_def = team2_3pafga_def / 100 if team2_3pafga_def else 0.39
    
    # Calculate expected 3PA rates using additive model
    ncaa_avg_3pa = 0.391  # NCAA average 3PA/FGA rate
    team1_expected_3pa_rate = ncaa_avg_3pa - (ncaa_avg_3pa - team1_3pafga_off) - (ncaa_avg_3pa - team2_3pafga_def)
    team2_expected_3pa_rate = ncaa_avg_3pa - (ncaa_avg_3pa - team2_3pafga_off) - (ncaa_avg_3pa - team1_3pafga_def)
    
    # Get effective field goal percentages
    team1_efg_off, team1_efg_def = get_team_stat(team1_team_df, "Effective FG%:")
    team2_efg_off, team2_efg_def = get_team_stat(team2_team_df, "Effective FG%:")
    
    # Convert to proper decimal form
    team1_efg_off = team1_efg_off / 100 if team1_efg_off else 0.51
    team1_efg_def = team1_efg_def / 100 if team1_efg_def else 0.51
    team2_efg_off = team2_efg_off / 100 if team2_efg_off else 0.51
    team2_efg_def = team2_efg_def / 100 if team2_efg_def else 0.51
    
    # Calculate expected effective FG% using additive model
    ncaa_avg_efg = 0.509  # NCAA average eFG%
    team1_expected_efg = ncaa_avg_efg - (ncaa_avg_efg - team1_efg_off) - (ncaa_avg_efg - team2_efg_def)
    team2_expected_efg = ncaa_avg_efg - (ncaa_avg_efg - team2_efg_off) - (ncaa_avg_efg - team1_efg_def)
    
    # Get offensive and defensive rebounding percentages
    team1_oreb_pct, _ = get_team_stat(team1_team_df, "Off. Reb. %:")
    _, team1_dreb_pct = get_team_stat(team1_team_df, "Off. Reb. %:")
    team2_oreb_pct, _ = get_team_stat(team2_team_df, "Off. Reb. %:")
    _, team2_dreb_pct = get_team_stat(team2_team_df, "Off. Reb. %:")
    
    # Convert to proper decimal form
    team1_oreb_pct = team1_oreb_pct / 100 if team1_oreb_pct else 0.30
    team1_dreb_pct = (100 - team1_dreb_pct) / 100 if team1_dreb_pct else 0.70
    team2_oreb_pct = team2_oreb_pct / 100 if team2_oreb_pct else 0.30
    team2_dreb_pct = (100 - team2_dreb_pct) / 100 if team2_dreb_pct else 0.70
    
    # Calculate expected rebounding rates using additive model
    ncaa_avg_oreb = 0.298  # NCAA average OREB%
    team1_expected_oreb = ncaa_avg_oreb - (ncaa_avg_oreb - team1_oreb_pct) - (ncaa_avg_oreb - (1 - team2_dreb_pct))
    team2_expected_oreb = ncaa_avg_oreb - (ncaa_avg_oreb - team2_oreb_pct) - (ncaa_avg_oreb - (1 - team1_dreb_pct))
    
    # --- Step 5: Calculate field goal attempts ---
    # Realistic FGA - teams typically take 55-65 FGA per game
    team1_fga = team1_adj_poss * 0.95  # About 65 FGA for a 69-possession game
    team2_fga = team2_adj_poss * 0.95
    
    # Estimate made and missed FGA based on expected eFG%
    team1_expected_fg_made = team1_fga * team1_expected_efg / 1.5  # Divisor adjusts for 3-point value
    team2_expected_fg_made = team2_fga * team2_expected_efg / 1.5
    
    team1_expected_fg_missed = team1_fga - team1_expected_fg_made
    team2_expected_fg_missed = team2_fga - team2_expected_fg_made
    
    # Calculate expected rebounds
    team1_oreb = team2_expected_fg_missed * team1_expected_oreb
    team1_dreb = team1_expected_fg_missed * (1 - team2_expected_oreb)
    team2_oreb = team1_expected_fg_missed * team2_expected_oreb
    team2_dreb = team2_expected_fg_missed * (1 - team1_expected_oreb)
    
    # --- Step 6: Calculate matchup-specific adjustments ---
    team1_advantages = get_matchup_advantage(team1_stats_df, team2_stats_df)
    team2_advantages = get_matchup_advantage(team2_stats_df, team1_stats_df)
    
    # --- Step 7: Process individual player stats ---
    def process_team(player_df, team_name, adj_poss, total_fga, expected_points, three_pa_rate, 
                     expected_efg, oreb, dreb, matchup_advantages):
        """Process player-level projections for a team."""
        player_df = player_df.copy()
        
        # Parse shooting data
        player_df[['3PM', '3PA']] = split_made_attempted(player_df['3PM-A'])
        player_df[['2PM', '2PA']] = split_made_attempted(player_df['2PM-A'])
        player_df[['FTM', 'FTA']] = split_made_attempted(player_df['FTM-A'])
        
        # Convert percentage columns to numeric
        pct_columns = ['Pct', '%Min', '%Poss', '%Shots', 'OR%', 'DR%', 'TORate', 'FC/40', 'Min']
        for col in pct_columns:
            if col in player_df.columns:
                player_df[col] = pd.to_numeric(player_df[col], errors='coerce').fillna(0)
        
        # Calculate shooting percentages
        player_df['3P%'] = player_df['3PM'] / player_df['3PA'].replace(0, 1)
        player_df['2P%'] = player_df['2PM'] / player_df['2PA'].replace(0, 1)
        player_df['FT%'] = player_df['FTM'] / player_df['FTA'].replace(0, 1)
        
        # Calculate shot ratios
        player_df['FGA'] = player_df['2PA'] + player_df['3PA']
        
        # Extract games played for per-game calculation
        player_df['Games'] = pd.to_numeric(player_df['G'], errors='coerce').fillna(0)
        player_df['Games'] = player_df['Games'].clip(1)  # Ensure at least 1 game
        
        # Calculate minutes per game and true minute percentage
        max_games = player_df['Games'].max()
        total_available_minutes = max_games * 40 * 5
        
        if 'Min' in player_df.columns and player_df['Min'].sum() > 0:
            # Use actual minutes data if available
            player_df['Min_Per_Game'] = player_df['Min'] / player_df['Games']
        else:
            # Derive minutes from %Min
            player_df['Derived_Total_Min'] = (player_df['%Min'] / 100) * total_available_minutes / 5
            player_df['Min_Per_Game'] = player_df['Derived_Total_Min'] / player_df['Games']
        
        # Calculate percentage of a full game (40 minutes in college)
        player_df['%Min_Per_Game'] = (player_df['Min_Per_Game'] / 40) * 100
        player_df['%Min_Per_Game'] = player_df['%Min_Per_Game'].clip(0, 100)
        
        # Calculate per-game FGA for each player
        player_df['FGA_Per_Game'] = player_df['FGA'] / player_df['Games']
        
        # Use per-game FGA to calculate shot share
        total_fga_per_game = player_df['FGA_Per_Game'].sum()
        player_df['FGA_Share'] = player_df['FGA_Per_Game'] / total_fga_per_game if total_fga_per_game > 0 else 0
        
        # Adjust shot share based on minutes
        player_df['Adj_FGA_Share'] = player_df['FGA_Share'] * (player_df['%Min_Per_Game'] / 100)
        
        # Normalize to sum to 1
        if player_df['Adj_FGA_Share'].sum() > 0:
            player_df['Adj_FGA_Share'] = player_df['Adj_FGA_Share'] / player_df['Adj_FGA_Share'].sum()
        
        # Calculate expected FGA per player
        player_df['Exp_FGA'] = player_df['Adj_FGA_Share'] * total_fga
        
        # Individual 3PA rates, blended with team expected rate
        player_df['3PA_Rate'] = player_df['3PA'] / player_df['FGA'].replace(0, 1)
        player_df['Adj_3PA_Rate'] = (player_df['3PA_Rate'] * 0.7) + (three_pa_rate * 0.3)
        
        # Calculate expected 3PA and 2PA
        player_df['Exp_3PA'] = player_df['Exp_FGA'] * player_df['Adj_3PA_Rate']
        player_df['Exp_2PA'] = player_df['Exp_FGA'] - player_df['Exp_3PA']
        
        # Apply matchup advantages to shooting percentages
        player_df['Matchup_Adj'] = player_df['Player'].map(matchup_advantages).fillna(1.0)
        player_df['Adj_3P%'] = player_df['3P%'] * player_df['Matchup_Adj']
        player_df['Adj_2P%'] = player_df['2P%'] * player_df['Matchup_Adj']
        
        # Calculate expected makes from adjusted shooting percentages
        player_df['Exp_3PM'] = player_df['Exp_3PA'] * player_df['Adj_3P%']
        player_df['Exp_2PM'] = player_df['Exp_2PA'] * player_df['Adj_2P%']
        
        # Calculate expected points from field goals
        player_df['Exp_Pts_2P'] = player_df['Exp_2PM'] * 2
        player_df['Exp_Pts_3P'] = player_df['Exp_3PM'] * 3
        
        # Calculate free throw attempts
        player_df['FTA_Rate'] = player_df['FTA'] / player_df['FGA'].replace(0, 1)
        
        # Normalize FTA rates to get realistic total
        ft_ratio = 0.38  # Target FTA/FGA ratio
        avg_fta_rate = player_df['FTA_Rate'].mean()
        if avg_fta_rate > 0:
            player_df['Adj_FTA_Rate'] = player_df['FTA_Rate'] * (ft_ratio / avg_fta_rate)
        else:
            player_df['Adj_FTA_Rate'] = ft_ratio / len(player_df)
            
        player_df['Exp_FTA'] = player_df['Exp_FGA'] * player_df['Adj_FTA_Rate']
        player_df['Exp_FTM'] = player_df['Exp_FTA'] * player_df['FT%']
        player_df['Exp_Pts_FT'] = player_df['Exp_FTM']
        
        # Calculate total expected points from all sources
        player_df['Raw_Exp_Pts'] = player_df['Exp_Pts_2P'] + player_df['Exp_Pts_3P'] + player_df['Exp_Pts_FT']
        total_raw_pts = player_df['Raw_Exp_Pts'].sum()
        
        # Scale to match the expected total points from team efficiency
        if total_raw_pts > 0:
            scaling_factor = expected_points / total_raw_pts
            player_df['Exp_Pts'] = player_df['Raw_Exp_Pts'] * scaling_factor
        else:
            # Fallback if raw points calculation fails
            player_df['Exp_Pts'] = expected_points * (player_df['%Min_Per_Game'] / player_df['%Min_Per_Game'].sum())
        
        # Calculate foul statistics
        player_df['Foul_Rate'] = player_df['FC/40'] / 40  # Fouls per minute
        player_df['Foul_Risk'] = player_df['Foul_Rate'] * player_df['%Min_Per_Game'] / 100 * 40
        
        # Set adjusted minutes directly to the per-game minutes percentage
        player_df['Adj_Min%'] = player_df['%Min_Per_Game']
        
        # Ensure minutes sum to 500 (5 players * 100%)
        if player_df['Adj_Min%'].sum() != 500:
            scale_factor = 500 / player_df['Adj_Min%'].sum()
            player_df['Adj_Min%'] = player_df['Adj_Min%'] * scale_factor
        
        # Set final points
        player_df['Final_Pts'] = player_df['Exp_Pts']
        
        # Final FG% calculation
        player_df['Final_FG%'] = ((player_df['Exp_2PM'] + player_df['Exp_3PM']) / 
                                 (player_df['Exp_2PA'] + player_df['Exp_3PA']).replace(0, 1))
        
        # Calculate rebound projections
        player_df['OR%'] = pd.to_numeric(player_df['OR%'], errors='coerce').fillna(0)
        player_df['DR%'] = pd.to_numeric(player_df['DR%'], errors='coerce').fillna(0)
        
        # Calculate rebounding shares based on OR%, DR% values and minutes played
        player_df['OREB_Share'] = player_df['OR%'] * (player_df['Adj_Min%'] / 100)
        player_df['DREB_Share'] = player_df['DR%'] * (player_df['Adj_Min%'] / 100)
        
        # Normalize shares to sum to 1
        if player_df['OREB_Share'].sum() > 0:
            player_df['OREB_Share'] = player_df['OREB_Share'] / player_df['OREB_Share'].sum()
        else:
            player_df['OREB_Share'] = 1 / len(player_df)
            
        if player_df['DREB_Share'].sum() > 0:
            player_df['DREB_Share'] = player_df['DREB_Share'] / player_df['DREB_Share'].sum()
        else:
            player_df['DREB_Share'] = 1 / len(player_df)
        
        player_df['Exp_OREB'] = player_df['OREB_Share'] * oreb
        player_df['Exp_DREB'] = player_df['DREB_Share'] * dreb
        player_df['Exp_REB'] = player_df['Exp_OREB'] + player_df['Exp_DREB']

        # Calculate total expected field goals made for assist calculation
        expected_fg_made = (player_df['Exp_2PM'] + player_df['Exp_3PM']).sum()
        
        # Calculate per-game assist rate
        player_df['ARate_Per_Game'] = pd.to_numeric(player_df['ARate'], errors='coerce').fillna(0) / player_df['Games'] * player_df['Games'].mean()
        
        # Calculate expected assists
        temp_df = player_df.copy()
        temp_df['ARate'] = temp_df['ARate_Per_Game']
        player_df['Exp_AST'] = calculate_assists(temp_df, expected_fg_made, matchup_advantages)
        
        # Prepare result dataframe
        result_df = pd.DataFrame()
        result_df['Player'] = player_df['Player']
        result_df['Team'] = team_name
        result_df['Min%'] = player_df['Adj_Min%']
        result_df['Pts'] = player_df['Final_Pts'].round(1)
        result_df['FG%'] = player_df['Final_FG%'].round(3)
        result_df['3PM'] = player_df['Exp_3PM'].round(1)
        result_df['REB'] = player_df['Exp_REB'].round(1)
        result_df['AST'] = player_df['Exp_AST'].round(1)
        
        return result_df
    
    # Process each team
    team1_results = process_team(
        team1_stats_df, "team1", team1_adj_poss, team1_fga, team1_expected_points,
        team1_expected_3pa_rate, team1_expected_efg, team1_oreb, team1_dreb, 
        team1_advantages
    )
    
    team2_results = process_team(
        team2_stats_df, "team2", team2_adj_poss, team2_fga, team2_expected_points,
        team2_expected_3pa_rate, team2_expected_efg, team2_oreb, team2_dreb, 
        team2_advantages
    )
    
    # Combine results
    combined_results = pd.concat([team1_results, team2_results], ignore_index=True)
    
    return combined_results

# -----------------------------------------
# 4. Print Results
# -----------------------------------------
def print_player_stats(summary_df, team1_name="team1", team2_name="team2"):
    """Print formatted player projections and team totals."""
    print("\n" + "="*60)
    print("PLAYER PROJECTIONS".center(60))
    print("="*60)
    
    teams = {"team1": team1_name, "team2": team2_name}
    
    for team_id, team_name in teams.items():
        team_df = summary_df[summary_df['Team'] == team_id]
        
        print(f"\n{team_name} PROJECTED STATS:")
        print("-"*60)
        print(f"{'Player':<25} {'Min%':<6} {'Pts':<6} {'FG%':<6} {'3PM':<6} {'REB':<6} {'AST':<6}")
        print("-"*60)
        
        for _, row in team_df.iterrows():
            print(f"{row['Player']:<25} {row['Min%']:<6.1f} {row['Pts']:<6.1f} "
                  f"{row['FG%']:<6.3f} {row['3PM']:<6.1f} {row['REB']:<6.1f} {row['AST']:<6.1f}")
        
        # Calculate team totals
        team_pts = team_df['Pts'].sum()
        team_3pm = team_df['3PM'].sum()
        team_reb = team_df['REB'].sum()
        team_ast = team_df['AST'].sum()
        
        print("-"*60)
        print(f"{'TEAM TOTALS':<25} {'500.0':<6} {team_pts:<6.1f} {'':6} {team_3pm:<6.1f} {team_reb:<6.1f} {team_ast:<6.1f}")
    
    # Print final score prediction
    team1_score = summary_df.loc[summary_df['Team'] == 'team1', 'Pts'].sum()
    team2_score = summary_df.loc[summary_df['Team'] == 'team2', 'Pts'].sum()
    
    print("\n" + "="*60)
    print("PREDICTED FINAL SCORE".center(60))
    print("="*60)
    print(f"{team1_name}: {team1_score:.1f}")
    print(f"{team2_name}: {team2_score:.1f}")
    print(f"Spread: {abs(team1_score - team2_score):.1f} {(team1_name if team1_score > team2_score else team2_name)}")
    print("="*60)

# -----------------------------------------
# 5. Main Function
# -----------------------------------------
def main(filepath, team1_name="team1", team2_name="team2", home_team="team1"):
    """Main function to run the model."""
    # Load data
    team1_df, team2_df, team1_team_df, team2_team_df = load_data(filepath)
    
    # Run the model
    results = estimate_player_stats(team1_df, team2_df, team1_team_df, team2_team_df, home_team)
    
    # Print results
    print_player_stats(results, team1_name, team2_name)
    
    return results

# -----------------------------------------
# 6. Run the Model
# -----------------------------------------
if __name__ == "__main__":
    filepath = "/Users/isaiahnick/Desktop/Florida Auburn Metrics.xlsx"  # Update path as needed
    results = main(filepath, "Auburn", "Florida", "team1")