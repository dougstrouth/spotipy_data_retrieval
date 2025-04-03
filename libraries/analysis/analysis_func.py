# analysis_func.py
"""
A library of functions to perform analysis on enriched Spotify saved tracks data,
including track info, artist genres/popularity, and album details.

Assumes input DataFrames contain columns generated by the multi-stage
data retrieval script (e.g., popularity, artist_popularity, artist_genres,
added_at, album_release_date, duration_ms, explicit, etc.), but EXCLUDES
detailed audio features (danceability, energy, valence, etc.).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any, Union

# Optional: Configure plotting style globally
# sns.set_theme(style="whitegrid")
# plt.rcParams['figure.figsize'] = (12, 6) # Example default figure size

# --- Helper Functions ---

def _explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to explode the 'artist_genres' column into individual rows per genre."""
    if 'artist_genres' not in df.columns:
        # If the column doesn't exist at all (e.g., artist enrichment failed)
        print("Warning: 'artist_genres' column not found. Returning empty DataFrame for genre analysis.")
        return pd.DataFrame(columns=df.columns.tolist() + ['genre'])
        # Or raise ValueError("DataFrame must contain 'artist_genres' column.")

    # Ensure NaN values are handled before splitting
    temp_df = df.dropna(subset=['artist_genres']).copy()
    if temp_df.empty:
        # Return DataFrame with same columns as input + 'genre' if empty after dropna
        print("Warning: No valid non-null 'artist_genres' found. Returning empty DataFrame for genre analysis.")
        return pd.DataFrame(columns=df.columns.tolist() + ['genre'])

    temp_df['genre_list'] = temp_df['artist_genres'].str.split(',')
    genre_exploded_df = temp_df.explode('genre_list')
    # Handle potential empty strings after split if genres were like "pop,"
    genre_exploded_df['genre'] = genre_exploded_df['genre_list'].str.strip()
    # Filter out empty strings that might result from splitting
    genre_exploded_df = genre_exploded_df[genre_exploded_df['genre'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    return genre_exploded_df.drop(columns=['genre_list'])

def _parse_release_date(df: pd.DataFrame, date_col: str = 'album_release_date') -> pd.DataFrame:
    """
    Helper to safely parse release date column into datetime objects and year.
    Uses existing 'release_datetime' and 'release_year' if available from spotify_func.py processing.
    """
    df_copy = df.copy()
    # Check if pre-parsed columns exist and are valid
    if 'release_datetime' in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy['release_datetime']):
        # Ensure 'release_year' also exists if datetime does
        if 'release_year' not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy['release_year']):
            df_copy['release_year'] = df_copy['release_datetime'].dt.year.astype('Int64')
        return df_copy # Use existing parsed columns
    elif date_col not in df_copy.columns:
         raise ValueError(f"DataFrame must contain '{date_col}' or pre-parsed 'release_datetime'.")
    else:
        print(f"Info: Parsing '{date_col}' as 'release_datetime'/'release_year' were not found or invalid.")
        # Fallback to parsing original column
        # Handle yyyy, yyyy-MM, yyyy-MM-DD formats robustly
        df_copy['temp_date_str'] = df_copy[date_col].astype(str)
        df_copy['release_datetime'] = pd.to_datetime(df_copy['temp_date_str'], errors='coerce')
        
        # Attempt to fix parsing for year/month precision if available
        if 'album_release_date_precision' in df_copy.columns:
            year_mask = (df_copy['album_release_date_precision'] == 'year') & df_copy['release_datetime'].isna()
            month_mask = (df_copy['album_release_date_precision'] == 'month') & df_copy['release_datetime'].isna()
            
            df_copy.loc[year_mask, 'release_datetime'] = pd.to_datetime(df_copy.loc[year_mask, 'temp_date_str'] + '-01-01', errors='coerce')
            df_copy.loc[month_mask, 'release_datetime'] = pd.to_datetime(df_copy.loc[month_mask, 'temp_date_str'] + '-01', errors='coerce')

        # Extract year only after successful parsing
        df_copy['release_year'] = df_copy['release_datetime'].dt.year.astype('Int64') # Use nullable integer
        df_copy = df_copy.drop(columns=['temp_date_str'], errors='ignore')
        return df_copy


def _get_top_genres(df: pd.DataFrame, n: int = 10) -> List[str]:
    """Helper function to get the most frequent genres from an already exploded DataFrame or one with 'artist_genres'."""
    if 'genre' not in df.columns and 'artist_genres' not in df.columns:
         raise ValueError("DataFrame needs 'genre' or 'artist_genres' column.")

    if 'genre' not in df.columns:
        exploded_df = _explode_genres(df)
    else:
        exploded_df = df # Assume already exploded if 'genre' exists

    if exploded_df.empty:
        return []
    top_genres = exploded_df['genre'].value_counts().nlargest(n).index.tolist()
    return top_genres

# --- Time-Based Analysis ---

def plot_tracks_added_distribution(df: pd.DataFrame, period: str = 'M') -> Optional[plt.Axes]:
    """
    Plots the distribution of tracks added to the library over time.

    Args:
        df: DataFrame containing 'added_at' column (datetime).
        period: Pandas offset string ('M' for month, 'Y' for year, 'D' for day).

    Returns:
        Matplotlib Axes object with the plot, or None if data is empty/missing.
    """
    if 'added_at' not in df.columns:
        print("Warning: 'added_at' column not found. Cannot plot added distribution.")
        return None
    # Ensure added_at is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['added_at']):
        df['added_at'] = pd.to_datetime(df['added_at'], errors='coerce', utc=True)

    if df.empty or df['added_at'].isnull().all():
        print("Warning: No valid 'added_at' data found.")
        return None

    counts = df.set_index('added_at').resample(period).size()
    ax = counts.plot(kind='line', figsize=(12, 6), title=f"Tracks Added Per {period}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Tracks Added")
    return ax

def plot_release_year_distribution(df: pd.DataFrame, bin_size: Optional[int] = 1) -> Optional[plt.Axes]:
    """
    Plots the distribution of album release years for saved tracks.

    Args:
        df: DataFrame containing 'album_release_date' or pre-parsed 'release_year'.
        bin_size: Size of bins for histogram (e.g., 1 for year, 10 for decade). If None, use auto bins.

    Returns:
        Matplotlib Axes object with the histogram, or None if no valid years found.
    """
    try:
        df_parsed = _parse_release_date(df)
    except ValueError as e:
        print(f"Warning: Cannot parse release date - {e}")
        return None

    years = df_parsed['release_year'].dropna()
    if years.empty:
        print("Warning: No valid 'release_year' data found.")
        return None

    plt.figure(figsize=(12, 6))
    ax = sns.histplot(years.astype(int), binwidth=bin_size, kde=False) # Convert to int for plotting if needed
    ax.set_title("Distribution of Release Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Tracks")
    return ax

def calculate_add_vs_release_lag(df: pd.DataFrame, time_unit: str = 'days') -> Optional[pd.Series]:
    """
    Calculates the time difference between track add and album release.

    Args:
        df: DataFrame with 'added_at' and 'album_release_date' (or pre-parsed 'release_datetime').
        time_unit: 'days' or 'years'.

    Returns:
        Pandas Series containing the lag, or None if data is missing/invalid.
    """
    if 'added_at' not in df.columns:
        print("Warning: 'added_at' column not found.")
        return None
    # Ensure added_at is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['added_at']):
        df['added_at'] = pd.to_datetime(df['added_at'], errors='coerce', utc=True)

    try:
        df_parsed = _parse_release_date(df)
    except ValueError as e:
        print(f"Warning: Cannot parse release date for lag calculation - {e}")
        return None

    # Check if necessary columns exist and are datetime
    # Inside calculate_add_vs_release_lag function

    # Check if necessary columns exist and are datetime
    if 'added_at' in df_parsed.columns and \
       'release_datetime' in df_parsed.columns and \
       pd.api.types.is_datetime64_any_dtype(df_parsed['added_at']) and \
       pd.api.types.is_datetime64_any_dtype(df_parsed['release_datetime']):

        # --- FIX: Make release_datetime timezone-aware (UTC) before subtracting ---
        # Using errors='ignore' or nonexistent/ambiguous='NaT' is safer
        release_dt_aware = df_parsed['release_datetime'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
        added_at_aware = df_parsed['added_at'] # Already aware

        # Perform subtraction with both columns being tz-aware (UTC)
        lag = (added_at_aware - release_dt_aware)
        # --- END FIX ---

        if time_unit == 'years':
            return lag.dt.days / 365.25
        else: # Default to days
            return lag.dt.days
    else:
        print("Warning: Missing or invalid 'added_at' or 'release_datetime' for lag calculation.")
        return None


def plot_add_vs_release_lag_distribution(df: pd.DataFrame, time_unit: str = 'days', bins: int = 50) -> Optional[plt.Axes]:
    """Calculates and plots the distribution of the add vs. release lag."""
    lag_series = calculate_add_vs_release_lag(df, time_unit=time_unit)
    if lag_series is None or lag_series.isnull().all():
        print("Warning: Cannot plot lag distribution, no valid lag data.")
        return None

    plt.figure(figsize=(12, 6))
    ax = sns.histplot(lag_series.dropna(), bins=bins)
    ax.set_title(f"Distribution of Lag Between Track Add and Release ({time_unit.capitalize()})")
    ax.set_xlabel(f"Lag ({time_unit.capitalize()})")
    ax.set_ylabel("Number of Tracks")
    # Optional: Adjust x-axis limits if lag is highly skewed
    # lower_bound, upper_bound = lag_series.quantile([0.01, 0.99])
    # ax.set_xlim(lower_bound, upper_bound)
    return ax

def plot_feature_over_release_years(df: pd.DataFrame, feature: str, agg_func: str = 'mean', window: Optional[int] = 5) -> Optional[plt.Axes]:
    """
    Plots the trend of a numeric feature (e.g., popularity) across release years.

    Args:
        df: DataFrame with 'album_release_date' (or 'release_year') and the specified feature.
        feature: The numeric column name to plot (e.g., 'popularity', 'artist_popularity').
        agg_func: Aggregation function ('mean', 'median').
        window: Size of rolling average window (e.g., 5 for 5-year average). If None, plot raw aggregate.

    Returns:
        Matplotlib Axes object, or None if data is missing/invalid.
    """
    if feature not in df.columns:
        print(f"Warning: Feature '{feature}' not found.")
        return None
    if not pd.api.types.is_numeric_dtype(df[feature]):
         df[feature] = pd.to_numeric(df[feature], errors='coerce') # Attempt conversion

    try:
        df_parsed = _parse_release_date(df)
    except ValueError as e:
        print(f"Warning: Cannot parse release date - {e}")
        return None

    if df_parsed['release_year'].isnull().all():
        print("Warning: No valid 'release_year' data.")
        return None

    # Use release_year and the specified feature
    analysis_df = df_parsed[['release_year', feature]].dropna()
    if analysis_df.empty:
        print(f"Warning: No valid data points for '{feature}' and 'release_year'.")
        return None

    agg_data = analysis_df.groupby('release_year')[feature].agg(agg_func)
    if agg_data.empty: return None

    plt.figure(figsize=(12, 6))
    if window:
        rolling_agg = agg_data.rolling(window=window, center=True, min_periods=1).mean()
        ax = rolling_agg.plot(label=f'{window}-Year Rolling {agg_func.capitalize()}')
        ax.set_ylabel(f'{window}-Year Rolling {agg_func.capitalize()} of {feature}')
    else:
        ax = agg_data.plot(label=f'{agg_func.capitalize()}')
        ax.set_ylabel(f'{agg_func.capitalize()} of {feature}')

    ax.set_title(f"Trend of {feature.replace('_', ' ').title()} Over Release Years")
    ax.set_xlabel("Release Year")
    ax.legend()
    return ax

# --- Popularity Analysis ---

def plot_popularity_distribution(df: pd.DataFrame, column: str = 'popularity') -> Optional[plt.Axes]:
    """Plots a histogram of popularity scores (track 'popularity' or 'artist_popularity')."""
    if column not in df.columns:
        print(f"Warning: Popularity column '{column}' not found.")
        return None
    if not pd.api.types.is_numeric_dtype(df[column]):
         df[column] = pd.to_numeric(df[column], errors='coerce')

    if df[column].isnull().all():
        print(f"Warning: No valid data in popularity column '{column}'.")
        return None

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df[column].dropna(), kde=True, bins=20) # Added bins
    ax.set_title(f"Distribution of {column.replace('_', ' ').title()}")
    ax.set_xlabel(column.replace('_', ' ').title())
    return ax

def plot_popularity_correlation(df: pd.DataFrame) -> Optional[plt.Axes]:
    """Creates a scatter plot comparing track popularity vs. artist popularity."""
    req_cols = ['popularity', 'artist_popularity']
    if not all(c in df.columns for c in req_cols):
        missing = [c for c in req_cols if c not in df.columns]
        print(f"Warning: Missing required columns for popularity correlation: {missing}")
        return None

    # Ensure numeric types
    plot_df = df[req_cols].copy()
    for col in req_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

    plot_df = plot_df.dropna()
    if plot_df.empty:
        print("Warning: No valid non-null data for popularity correlation.")
        return None

    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(data=plot_df, x='artist_popularity', y='popularity', alpha=0.5)
    ax.set_title("Track Popularity vs. Average Artist Popularity")
    ax.set_xlabel("Average Artist Popularity")
    ax.set_ylabel("Track Popularity")
    # Optional: Add correlation coefficient
    try:
        corr = plot_df['artist_popularity'].corr(plot_df['popularity'])
        ax.text(0.05, 0.95, f'Corr: {corr:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    except Exception as e:
        print(f"Could not calculate correlation: {e}")
    return ax

def get_top_n_items(df: pd.DataFrame, display_cols: List[str], sort_col: str, n: int = 20, ascending: bool = False) -> pd.DataFrame:
    """Gets the top N rows based on a specified numeric column, showing specific display columns."""
    if sort_col not in df.columns: raise ValueError(f"Missing sort column '{sort_col}'.")
    if not pd.api.types.is_numeric_dtype(df[sort_col]):
         df[sort_col] = pd.to_numeric(df[sort_col], errors='coerce') # Attempt conversion

    valid_display_cols = [col for col in display_cols if col in df.columns]
    if not valid_display_cols: raise ValueError("None of the specified display_cols exist.")

    return df.sort_values(by=sort_col, ascending=ascending, na_position='last').head(n)[valid_display_cols]


# --- Artist & Album Analysis ---

def get_top_n_artists(df: pd.DataFrame, n: int = 25) -> Optional[pd.DataFrame]:
    """Counts track occurrences per artist and returns the top N."""
    # Uses 'artist_names' for readability if available, falls back to IDs
    id_col = 'artist_ids'
    name_col = 'artist_names'

    if id_col not in df.columns:
        print(f"Warning: Missing '{id_col}' column for artist analysis.")
        return None
    if df[id_col].isnull().all():
        print(f"Warning: Column '{id_col}' contains no valid data.")
        return None

    # Explode both IDs and Names simultaneously if names are available
    has_names = name_col in df.columns and not df[name_col].isnull().all()
    cols_to_explode = [id_col]
    if has_names:
        cols_to_explode.append(name_col)

    temp_df = df[cols_to_explode].dropna(subset=[id_col]).copy()

    # --- FIX: Apply split and ensure object dtype for lists ---
    for col in cols_to_explode:
        # Apply split. Result is a Series containing lists or NaN
        split_col_series = temp_df[col].str.split(',')
        # Assign this Series back. Pandas will typically handle dtype adjustment,
        # or we can force object dtype if needed.
        temp_df[col] = split_col_series.astype('object') # Force object dtype to hold lists
    # --- END FIX ---

    # Now temp_df[col] contains lists (or NaN) and has object dtype

    # --- Use DataFrame.explode() directly ---
    try:
        # Explode columns which now contain lists
        exploded = temp_df.explode(cols_to_explode)
    except ValueError as e:
        # Handle potential errors if columns aren't list-like after split
        print(f"Error during DataFrame explode: {e}")
        print("Check if artist_ids/artist_names columns contain valid comma-separated strings.")
        return None
    # --- END Explode ---


    if exploded is not None and not exploded.empty:
        # Strip whitespace after exploding
        # Important: Handle potential non-string types after explode if source had NaN
        exploded[id_col] = exploded[id_col].astype(str).str.strip()
        if has_names:
             exploded[name_col] = exploded[name_col].astype(str).str.strip()

        # Filter out any potential empty strings after stripping
        # Also filter out 'nan' strings resulting from NaNs if necessary
        exploded = exploded[(exploded[id_col] != '') & (exploded[id_col].str.lower() != 'nan')]

        # Group by ID, get the first name associated (usually the same), count tracks
        grouped = exploded.groupby(id_col)

        if has_names:
            counts = grouped.agg(
                artist_name=(name_col, 'first'),
                track_count=(id_col, 'size')
            )
        else:
             counts = grouped.agg(
                track_count=(id_col, 'size')
            )
        
        top_artists = counts.sort_values('track_count', ascending=False).head(n)
        return top_artists.reset_index()

    else:
        print("Warning: No valid artists found after exploding.")
        return None


def get_top_n_albums(df: pd.DataFrame, n: int = 25) -> Optional[pd.DataFrame]:
    """Counts track occurrences per album and returns the top N."""
    if 'album_id' not in df.columns:
        print("Warning: Missing 'album_id'.")
        return None
    if 'album_name' not in df.columns:
        print("Warning: Missing 'album_name'.")
        # Still proceed but won't show album name
        use_name = False
    else:
        use_name = True

    if df['album_id'].isnull().all():
        print("Warning: No valid 'album_id' data.")
        return None

    group_cols = ['album_id']
    if use_name:
        group_cols.append('album_name')

    counts = df.groupby(group_cols).size().nlargest(n)
    return counts.reset_index(name='track_count')

# --- Genre Analysis ---

def plot_genre_distribution(df: pd.DataFrame, top_n: int = 20) -> Optional[plt.Axes]:
    """Plots the distribution (bar chart) of the top N genres."""
    try:
        exploded_df = _explode_genres(df)
    except ValueError as e:
        print(f"Warning: Cannot explode genres - {e}")
        return None

    if exploded_df.empty:
        print("Warning: No genre data to plot distribution.")
        return None

    counts = exploded_df['genre'].value_counts().nlargest(top_n)
    if counts.empty:
        print("Warning: No genres found after counting.")
        return None

    plt.figure(figsize=(12, max(6, top_n * 0.4))) # Adjust height slightly
    ax = sns.barplot(x=counts.values, y=counts.index, palette="viridis")
    ax.set_title(f"Top {top_n} Genres Distribution")
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Genre")
    return ax

def plot_feature_by_genre(df: pd.DataFrame, feature: str, top_n_genres: int = 10, plot_type: str = 'box') -> Optional[plt.Axes]:
    """
    Plots the distribution of a numeric feature (e.g., popularity) across top N genres.
    """
    if feature not in df.columns:
        print(f"Warning: Missing feature '{feature}'.")
        return None

    try:
        exploded_df = _explode_genres(df)
    except ValueError as e:
        print(f"Warning: Cannot explode genres - {e}")
        return None

    if exploded_df.empty:
        print("Warning: No genre data to analyze feature by genre.")
        return None

    top_genres = _get_top_genres(exploded_df, n=top_n_genres)
    if not top_genres:
         print("Warning: Could not determine top genres.")
         return None

    plot_df = exploded_df[exploded_df['genre'].isin(top_genres)].copy()

    # Ensure feature is numeric
    if not pd.api.types.is_numeric_dtype(plot_df[feature]):
        plot_df[feature] = pd.to_numeric(plot_df[feature], errors='coerce')

    plot_df.dropna(subset=[feature], inplace=True)

    if plot_df.empty:
        print(f"Warning: No valid data points for '{feature}' within the top {top_n_genres} genres.")
        return None

    plt.figure(figsize=(12, max(6, top_n_genres * 0.5))) # Adjust height
    order = top_genres # Plot in order of frequency

    if plot_type == 'box':
        ax = sns.boxplot(data=plot_df, y='genre', x=feature, order=order, palette="viridis")
    elif plot_type == 'violin':
        ax = sns.violinplot(data=plot_df, y='genre', x=feature, order=order, palette="viridis", inner='quartile')
    else:
        print("Warning: plot_type must be 'box' or 'violin'. Defaulting to 'box'.")
        ax = sns.boxplot(data=plot_df, y='genre', x=feature, order=order, palette="viridis")


    ax.set_title(f"Distribution of {feature.replace('_',' ').title()} by Top {top_n_genres} Genres")
    ax.set_xlabel(feature.replace('_',' ').title())
    ax.set_ylabel("Genre")
    return ax

def analyze_genre_explicitness(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculates the percentage of explicit tracks per genre."""
    if 'explicit' not in df.columns:
        print("Warning: Missing 'explicit' column.")
        return None
    # Ensure explicit is boolean or convertible
    if not pd.api.types.is_bool_dtype(df['explicit']) and not pd.api.types.is_numeric_dtype(df['explicit']):
         try:
             # Attempt conversion, handling potential string 'True'/'False'
             df['explicit'] = df['explicit'].map({'true': True, 'false': False, 1: True, 0: False, True: True, False: False}).astype(bool)
         except Exception:
              print("Warning: Could not convert 'explicit' column to boolean.")
              return None

    try:
        exploded_df = _explode_genres(df)
    except ValueError as e:
        print(f"Warning: Cannot explode genres - {e}")
        return None

    if exploded_df.empty:
        print("Warning: No genre data to analyze explicitness.")
        return None

    # Calculate mean of boolean/numeric explicit column (True=1, False=0)
    explicit_pct = exploded_df.groupby('genre')['explicit'].mean() * 100
    return explicit_pct.sort_values(ascending=False).reset_index(name='explicit_percentage')

def calculate_genre_diversity(df: pd.DataFrame, method: str = 'shannon') -> Optional[float]:
    """Calculates a diversity index (e.g., Shannon, Simpson) for saved genres."""
    try:
        exploded_df = _explode_genres(df)
    except ValueError as e:
        print(f"Warning: Cannot explode genres - {e}")
        return None

    if exploded_df.empty:
        print("Warning: No genre data to calculate diversity.")
        return None

    counts = exploded_df['genre'].value_counts()
    if counts.empty:
        print("Warning: No genres found after counting for diversity calculation.")
        return None

    total = counts.sum()
    if total == 0: return 0.0 # Avoid division by zero if counts is somehow empty

    proportions = counts / total

    # Filter out zero proportions to avoid log(0)
    proportions = proportions[proportions > 0]
    if proportions.empty: return 0.0

    if method == 'shannon':
        # Shannon index: - sum(p_i * log(p_i))
        return -np.sum(proportions * np.log(proportions))
    elif method == 'simpson':
        # Simpson index: 1 - sum(p_i^2)
        return 1 - np.sum(proportions**2)
    else:
        raise ValueError("method must be 'shannon' or 'simpson'")


# --- Audio Feature Analysis (REMOVED as features are not available) ---

# DEFAULT_AUDIO_FEATURES = [...] # REMOVED

# def plot_audio_feature_distributions(...): # REMOVED
# def plot_audio_feature_correlations(...): # REMOVED
# def plot_feature_vs_popularity(...): # Modified - see below

# --- Track Characteristics Analysis ---

def plot_duration_distribution(df: pd.DataFrame, time_unit: str = 'minutes') -> Optional[plt.Axes]:
    """Plots a histogram of track durations."""
    if 'duration_ms' not in df.columns:
        print("Warning: Missing 'duration_ms'.")
        return None
    if not pd.api.types.is_numeric_dtype(df['duration_ms']):
         df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')

    if df['duration_ms'].isnull().all():
        print("Warning: No valid 'duration_ms' data.")
        return None

    plt.figure(figsize=(10, 6))
    if time_unit == 'minutes':
        durations = df['duration_ms'].dropna() / 60000
        xlabel = "Duration (minutes)"
        # Sensible bins for minutes
        binwidth = 0.5 # 30-second bins
    elif time_unit == 'seconds':
        durations = df['duration_ms'].dropna() / 1000
        xlabel = "Duration (seconds)"
        binwidth = 15 # 15-second bins
    else: # milliseconds
        durations = df['duration_ms'].dropna()
        xlabel = "Duration (ms)"
        binwidth = None # Auto bins

    ax = sns.histplot(durations, kde=True, binwidth=binwidth)
    ax.set_title("Distribution of Track Durations")
    ax.set_xlabel(xlabel)
    # Optional: Limit x-axis if very long tracks skew it
    # upper_limit = durations.quantile(0.99) # e.g., exclude longest 1%
    # ax.set_xlim(0, upper_limit)
    return ax

def plot_feature_vs_popularity(df: pd.DataFrame, feature: str, kind: str = 'scatter') -> Optional[plt.Axes]:
    """Plots a numeric feature vs. track popularity (e.g., duration_ms vs popularity)."""
    if feature not in df.columns:
        print(f"Warning: Missing feature '{feature}'.")
        return None
    if 'popularity' not in df.columns:
        print("Warning: Missing 'popularity'.")
        return None

    # Ensure numeric types
    plot_df = df[[feature, 'popularity']].copy()
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors='coerce')
    plot_df['popularity'] = pd.to_numeric(plot_df['popularity'], errors='coerce')

    plot_df.dropna(inplace=True)
    if plot_df.empty:
        print(f"Warning: No valid data points for '{feature}' vs 'popularity'.")
        return None

    plt.figure(figsize=(10, 6))
    title = f"Track Popularity vs. {feature.replace('_',' ').title()}"
    xlabel = feature.replace('_',' ').title()

    # Convert duration to minutes if applicable for better axis labels
    if feature == 'duration_ms':
         plot_df[feature] = plot_df[feature] / 60000
         xlabel = "Duration (minutes)"
         title = f"Track Popularity vs. Duration (minutes)"

    if kind == 'scatter':
        ax = sns.scatterplot(data=plot_df, x=feature, y='popularity', alpha=0.3)
    elif kind == 'hexbin':
        # Requires matplotlib hexbin - might need adjustments for sns style
        try:
             ax = plot_df.plot.hexbin(x=feature, y='popularity', gridsize=30, cmap='viridis', sharex=False) # sharex=False for matplotlib>=3.3
        except Exception as e:
             print(f"Could not create hexbin plot (ensure matplotlib is up to date): {e}")
             # Fallback to scatter
             ax = sns.scatterplot(data=plot_df, x=feature, y='popularity', alpha=0.3)
    else:
        print("Warning: kind must be 'scatter' or 'hexbin'. Defaulting to 'scatter'.")
        ax = sns.scatterplot(data=plot_df, x=feature, y='popularity', alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Track Popularity")
    return ax

def analyze_explicit_tracks_summary(df: pd.DataFrame, numeric_cols: List[str] = ['popularity', 'artist_popularity', 'duration_ms']) -> Optional[pd.DataFrame]:
    """Provides summary statistics comparing explicit vs. non-explicit tracks using available numeric features."""
    if 'explicit' not in df.columns:
        print("Warning: Missing 'explicit'.")
        return None
    # Ensure explicit is boolean or convertible
    if not pd.api.types.is_bool_dtype(df['explicit']) and not pd.api.types.is_numeric_dtype(df['explicit']):
         try:
             # Attempt conversion
             df['explicit'] = df['explicit'].map({'true': True, 'false': False, 1: True, 0: False, True: True, False: False}).astype(bool)
         except Exception:
              print("Warning: Could not convert 'explicit' column to boolean.")
              return None

    actual_cols = [col for col in numeric_cols if col in df.columns]
    if not actual_cols:
        print("Warning: No available numeric columns found for explicit summary.")
        return None # Return None if no numeric columns to analyze found

    # Ensure numeric types for analysis columns
    summary_df = df[['explicit'] + actual_cols].copy()
    for col in actual_cols:
         summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

    summary_df = summary_df.dropna(subset=['explicit'] + actual_cols)
    if summary_df.empty:
        print("Warning: No valid data after handling NaNs for explicit summary.")
        return None

    summary = summary_df.groupby('explicit').agg(
        track_count=('explicit', 'size'),
        **{f'mean_{col}': (col, 'mean') for col in actual_cols},
        **{f'median_{col}': (col, 'median') for col in actual_cols} # Added median
    ).reset_index()
    return summary

# --- Advanced / Combined Analysis (Placeholders - REMOVED or kept commented) ---

# def plot_genre_network(df: pd.DataFrame, min_cooccurrence: int = 5): # Kept commented
#     """ Creates and potentially plots a network graph of genre co-occurrence."""
#     # Implementation requires networkx, calculating co-occurrence matrix, building graph
#     pass


# === Example Usage (Updated for available data) ===
# import pandas as pd
# import analysis_func as sal # Assuming saved as analysis_func.py
# import matplotlib.pyplot as plt
#
# # Load the FINAL enriched data (which excludes audio features)
# final_df = pd.read_parquet('path/to/saved_tracks_enriched_final.parquet')
#
# # --- Perform various analyses ---
#
# # Plot tracks added per year
# ax_added = sal.plot_tracks_added_distribution(final_df, period='Y')
# if ax_added: plt.show()
#
# # Plot release year distribution (decade bins)
# ax_release = sal.plot_release_year_distribution(final_df, bin_size=10)
# if ax_release: plt.show()
#
# # Plot track popularity distribution
# ax_track_pop = sal.plot_popularity_distribution(final_df, column='popularity')
# if ax_track_pop: plt.show()
#
# # Plot artist popularity distribution
# ax_artist_pop = sal.plot_popularity_distribution(final_df, column='artist_popularity')
# if ax_artist_pop: plt.show()
#
# # Plot track vs artist popularity correlation
# ax_pop_corr = sal.plot_popularity_correlation(final_df)
# if ax_pop_corr: plt.show()
#
# # Plot duration distribution in minutes
# ax_duration = sal.plot_duration_distribution(final_df, time_unit='minutes')
# if ax_duration: plt.show()
#
# # Plot popularity trend over release years
# ax_pop_trend = sal.plot_feature_over_release_years(final_df, feature='popularity', window=5)
# if ax_pop_trend: plt.show()
#
# # Plot top genres
# ax_genres = sal.plot_genre_distribution(final_df, top_n=25)
# if ax_genres: plt.tight_layout(); plt.show()
#
# # Plot track popularity distribution by top genres
# ax_pop_genre = sal.plot_feature_by_genre(final_df, feature='popularity', top_n_genres=15, plot_type='box')
# if ax_pop_genre: plt.show()
#
# # Get top artists
# top_artists = sal.get_top_n_artists(final_df, n=15)
# if top_artists is not None: print("\n--- Top 15 Artists (by track count) ---\n", top_artists)
#
# # Get top albums
# top_albums = sal.get_top_n_albums(final_df, n=15)
# if top_albums is not None: print("\n--- Top 15 Albums (by track count) ---\n", top_albums)
#
# # Calculate genre diversity
# diversity_shannon = sal.calculate_genre_diversity(final_df, method='shannon')
# diversity_simpson = sal.calculate_genre_diversity(final_df, method='simpson')
# if diversity_shannon is not None: print(f"\nGenre Diversity (Shannon Index): {diversity_shannon:.3f}")
# if diversity_simpson is not None: print(f"Genre Diversity (Simpson Index): {diversity_simpson:.3f}")
#
# # Analyze explicit tracks summary
# explicit_summary = sal.analyze_explicit_tracks_summary(final_df)
# if explicit_summary is not None: print("\n--- Explicit vs Non-Explicit Summary ---\n", explicit_summary)
#
# # Get top 10 most popular tracks
# top_tracks = sal.get_top_n_items(final_df,
#                                  display_cols=['track_name', 'artist_names', 'album_name', 'popularity'],
#                                  sort_col='popularity', n=10)
# print("\n--- Top 10 Most Popular Tracks ---\n", top_tracks)
#
# # Get top 10 least popular tracks (might be newly added/obscure)
# bottom_tracks = sal.get_top_n_items(final_df,
#                                     display_cols=['track_name', 'artist_names', 'album_name', 'popularity'],
#                                     sort_col='popularity', n=10, ascending=True)
# print("\n--- Top 10 Least Popular Tracks ---\n", bottom_tracks)