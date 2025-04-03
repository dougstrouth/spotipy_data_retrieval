import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import pandas as pd
from collections import Counter
import logging
from typing import Optional, List, Dict, Any
import time

# --- Configuration ---
# Consider making these paths relative or using environment variables for portability
DEFAULT_EXPORT_PATH = '/Users/dougstrouth/Library/Mobile Documents/com~apple~CloudDocs/Documents/Code/datasets/spotify'
DEFAULT_SAVED_TRACKS_PATH = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks.parquet')
DEFAULT_GENRE_TRACKS_PATH = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_genre.parquet')
DEFAULT_GENRE_ANALYSIS_PATH = os.path.join(DEFAULT_EXPORT_PATH, 'genre_analysis.parquet') # Specific name for genre analysis output
BATCH_SIZE = 50  # Spotify API's max for artists() - Note: Max is 50, not 100
TRACK_FETCH_LIMIT = 50  # Spotify API's track retrieval limit (Max is 50)
SKIP_SONG_COLLECTION = True  # Default: Skip collection, requires saved_tracks.parquet
INITIAL_DELAY = 1  # Initial delay in seconds for retry
MAX_RETRIES = 5  # Maximum number of retries

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Spotify Client ---
def create_spotify_client() -> Optional[spotipy.Spotify]:
    """Authenticates with Spotify and returns a Spotipy client."""
    try:
        # Adjust path if needed or load from project root
        env_path = "/Users/dougstrouth/Documents/.env"
        if not os.path.exists(env_path):
            logging.warning(f".env file not found at {env_path}. Attempting to load from current directory.")
            env_path = ".env" # Fallback to current directory

        load_dotenv(dotenv_path=env_path)

        client_id = os.getenv("spotify_client_id")
        client_secret = os.getenv("spotify_client_secret")
        redirect_uri = os.getenv("spotify_redirect_uri")

        if not all([client_id, client_secret, redirect_uri]):
            logging.error("Missing Spotify credentials (client_id, client_secret, redirect_uri) in environment variables.")
            return None

        sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-library-read", # Only need user-library-read for saved tracks
                open_browser=False # Prevents automatically opening browser for auth
            )
        )
        # Test connection
        sp.current_user()
        logging.info("Spotify client created and authenticated successfully.")
        return sp
    except Exception as e:
        logging.error(f"Error creating Spotify client: {e}")
        return None


# --- General API Call Function with Backoff-Retry ---
def call_spotify_api(api_function: callable, *args, **kwargs) -> Any:
    """General function to call Spotify API with backoff-retry."""
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            results = api_function(*args, **kwargs)  # Call the function directly
            return results
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:  # Rate Limit Exceeded
                retry_after = e.headers.get('Retry-After')
                wait_time = int(retry_after) if retry_after else delay
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                delay = min(delay * 2, 60) # Exponential backoff with a cap
            # Handle other specific Spotify errors if needed
            # elif e.http_status == 401: # Unauthorized
            #     logging.error("Spotify API Error 401: Unauthorized. Check credentials/token.")
            #     raise # Re-raise immediately, retrying won't help
            else:
                logging.error(f"Spotify API error: {e} (Status: {e.http_status}, Attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt == MAX_RETRIES - 1: # Raise on last attempt
                     raise
                time.sleep(delay) # Wait before retrying other errors too
                delay = min(delay * 2, 60)
        except Exception as e:
            logging.error(f"Unexpected error calling API function {api_function.__name__}: {e} (Attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt == MAX_RETRIES - 1:
                 raise
            time.sleep(delay) # Wait before retrying other errors too
            delay = min(delay * 2, 60)


    logging.error(f"Max retries ({MAX_RETRIES}) reached for {api_function.__name__}. API call failed.")
    raise Exception("Max retries reached")


# --- Data Retrieval ---
def fetch_artist_data(spotify_client: spotipy.Spotify, artist_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetches artist data from Spotify API in batches."""
    artist_data = {}
    fetched_count = 0
    total_ids = len(artist_ids)
    logging.info(f"Starting artist data fetch for {total_ids} unique IDs.")

    for i in range(0, total_ids, BATCH_SIZE):
        batch_ids = artist_ids[i:min(i + BATCH_SIZE, total_ids)]
        try:
            # *** CORRECTED API CALL ***
            # Use the positional argument 'artists' instead of keyword 'artist_ids'
            results = call_spotify_api(spotify_client.artists, artists=batch_ids)
            if results and results.get('artists'):
                for artist in results['artists']:
                    if artist: # Check if artist object is not None
                        artist_data[artist['id']] = {
                            'genres': artist.get('genres', []),
                            'popularity': artist.get('popularity'),
                            'name': artist.get('name') # Keep name for potential debugging
                        }
                        fetched_count += 1
            else:
                logging.warning(f"Received no artist data in response for batch starting at index {i}.")

            logging.info(f"Fetched data for artists {i+1}-{min(i + BATCH_SIZE, total_ids)} / {total_ids}")

        except Exception as e:
            # Error is logged within call_spotify_api, but we add context here
            logging.error(f"Failed to fetch artist data batch starting at index {i}. IDs: {batch_ids}. Error: {e}")
            # Optionally continue to next batch or break, depending on desired robustness
            # continue

    logging.info(f"Finished artist data fetch. Successfully retrieved data for {fetched_count} out of {len(artist_data)} unique IDs attempted in valid batches.")
    return artist_data


def get_and_process_saved_tracks(spotify_client: spotipy.Spotify, max_tracks: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Retrieves and processes saved tracks from the Spotify API."""
    offset = 0
    all_tracks_data = []
    total_tracks_retrieved = 0

    logging.info("Starting retrieval of saved tracks...")

    while True:
        try:
            # Fetch tracks using the generic API call wrapper
            results = call_spotify_api(spotify_client.current_user_saved_tracks, limit=TRACK_FETCH_LIMIT, offset=offset)
            items = results.get('items', []) if results else [] # Safely access 'items'

            if not items:
                logging.info(f"No more saved tracks found at offset {offset}.")
                break

            num_items = len(items)
            processed_count = 0
            for item in items:
                if item and item.get('track'):  # Check for None and 'track'
                    processed_data = process_track_item(item)
                    if processed_data: # Ensure process_track_item didn't return None
                        all_tracks_data.append(processed_data)
                        processed_count +=1

            total_tracks_retrieved += processed_count
            logging.info(f"Retrieved batch offset={offset}, limit={TRACK_FETCH_LIMIT}. Processed {processed_count} tracks. Total retrieved: {total_tracks_retrieved}")


            if max_tracks and total_tracks_retrieved >= max_tracks:
                logging.info(f"Reached max_tracks limit ({max_tracks}). Stopping retrieval.")
                # Trim excess tracks if necessary
                all_tracks_data = all_tracks_data[:max_tracks]
                break

            # Check if the number of items returned is less than the limit, indicating the last page
            total_in_response = results.get('total', float('inf')) # Get total if available
            if num_items < TRACK_FETCH_LIMIT or (offset + num_items) >= total_in_response:
                 logging.info(f"Reached end of saved tracks (retrieved {num_items}, limit {TRACK_FETCH_LIMIT} or reached total {total_in_response}).")
                 break


            offset += TRACK_FETCH_LIMIT # Prepare for next batch

        except Exception as e:
            # Error is logged within call_spotify_api, but we add context here
            logging.error(f"Stopping track retrieval due to error at offset {offset}: {e}")
            # Depending on the error, you might want to return partial data or None
            # If partial data is acceptable even with errors:
            if not all_tracks_data: return None # No data recovered
            break # Stop processing further batches
            # If no data should be returned on error:
            # return None

    if not all_tracks_data:
        logging.warning("No track data successfully retrieved or processed.")
        return None

    logging.info(f"Successfully retrieved and processed {len(all_tracks_data)} tracks.")

    df = pd.DataFrame(all_tracks_data)

    # Define expected dtypes
    expected_dtypes = {
        'added_at': 'datetime64[ns, UTC]', # Explicitly set UTC timezone if applicable
        'track_name': 'string',
        'album_name': 'string',
        'artist_names': 'string', # Comma-separated names
        'album_release_date': 'string', # Keep as string initially
        'duration_ms': 'int64', # Use int64 to avoid potential overflow with int32
        'explicit': 'boolean', # Use pandas nullable boolean type
        'popularity': 'int32',
        'track_id': 'string',
        'album_id': 'string',
        'artist_ids': 'string', # Comma-separated IDs
        'isrc': 'string',
    }

    # Convert types, handling potential missing columns gracefully
    for col, dtype in expected_dtypes.items():
        if col in df.columns:
            try:
                if dtype == 'datetime64[ns, UTC]':
                     df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                else:
                     df[col] = df[col].astype(dtype, errors='ignore') # Use 'ignore' for flexibility if casting fails for some rows
            except Exception as e:
                logging.warning(f"Could not convert column '{col}' to {dtype}: {e}")
        else:
            logging.warning(f"Column '{col}' not found in retrieved data. Skipping type conversion.")

    # Attempt to parse release date to datetime, coercing errors
    if 'album_release_date' in df.columns:
         df['album_release_date_dt'] = pd.to_datetime(df['album_release_date'], errors='coerce')
         # Handle different date formats (YYYY, YYYY-MM, YYYY-MM-DD) if needed during analysis

    logging.info("DataFrame created and types converted.")
    return df


def process_track_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Processes a single track item from the Spotify API response."""
    try:
        track = item.get('track')
        if not track:
            logging.warning(f"Skipping item due to missing 'track' data: {item.get('added_at')}")
            return None

        album = track.get('album', {})
        artists = track.get('artists', [])

        # Ensure essential IDs are present
        track_id = track.get('id')
        album_id = album.get('id')
        if not track_id:
             logging.warning(f"Skipping track due to missing 'track_id': {track.get('name')}")
             return None

        artist_ids = [a.get('id') for a in artists if a and a.get('id')]
        artist_names = [a.get('name') for a in artists if a and a.get('name')]

        # Handle cases where artist info might be missing/malformed
        valid_artist_ids = [aid for aid in artist_ids if aid]
        valid_artist_names = [aname for aname in artist_names if aname]


        return {
            'added_at': item.get('added_at'),
            'track_name': track.get('name'),
            'album_name': album.get('name'),
            'artist_names': ', '.join(valid_artist_names) if valid_artist_names else None,
            'album_release_date': album.get('release_date'),
            'duration_ms': track.get('duration_ms'),
            'explicit': track.get('explicit'),
            'popularity': track.get('popularity'),
            'track_id': track_id,
            'album_id': album_id,
            'artist_ids': ', '.join(valid_artist_ids) if valid_artist_ids else None,
            'isrc': track.get('external_ids', {}).get('isrc'),
        }
    except Exception as e:
        logging.error(f"Error processing track item: {e}. Item: {item}")
        return None

# --- Data Enrichment (Modified for 'album_artist_ids') ---
def enrich_track_dataframe(spotify_client: Optional[spotipy.Spotify], track_df: pd.DataFrame) -> pd.DataFrame:
    """Enriches the track DataFrame with artist data (genres, popularity) using the IDs found in the 'album_artist_ids' column.""" # Modified Docstring

    if track_df is None or track_df.empty:
        logging.warning("Track DataFrame is empty or None. Skipping artist enrichment.")
        return pd.DataFrame() # Return empty DataFrame

    enriched_df = track_df.copy() # Work on a copy

    # --- Collect Artist IDs ---
    all_artist_ids = set()
    # *** MODIFIED: Look for 'album_artist_ids' ***
    id_column_name = 'album_artist_ids'
    if id_column_name in enriched_df.columns:
        # Ensure the column exists and handle potential missing values (NaN) before splitting
        for ids_str in enriched_df[id_column_name].dropna():
            if isinstance(ids_str, str): # Check if it's a non-empty string
                 all_artist_ids.update(aid.strip() for aid in ids_str.split(',') if aid.strip()) # Split and strip whitespace
    else:
        # *** MODIFIED: Check for 'album_artist_ids' ***
        logging.warning(f"Column '{id_column_name}' not found in DataFrame. Cannot perform enrichment.")
        # Add empty columns so the rest of the script doesn't break downstream
        enriched_df['artist_genres'] = pd.Series(dtype='string')
        enriched_df['artist_popularity'] = pd.Series(dtype='float64')
        return enriched_df

    artist_ids_list = list(all_artist_ids)

    if not artist_ids_list:
        # *** MODIFIED: Reference 'album_artist_ids' ***
        logging.info(f"No valid artist IDs found in '{id_column_name}' column to enrich the track data.")
        enriched_df['artist_genres'] = pd.Series(dtype='string')
        enriched_df['artist_popularity'] = pd.Series(dtype='float64')
        return enriched_df

    # --- Fetch Artist Data ---
    artist_data = {}
    if spotify_client:  # Only fetch if spotify_client is available
        logging.info(f"Fetching data for {len(artist_ids_list)} unique artists (from {id_column_name})...")
        try:
            # Uses the corrected fetch_artist_data from the previous full script
            artist_data = fetch_artist_data(spotify_client, artist_ids_list)
            logging.info(f"Successfully fetched data for {len(artist_data)} artists.")
        except Exception as e:
             logging.error(f"Enrichment failed: Could not fetch artist data. Error: {e}")
             enriched_df['artist_genres'] = pd.Series(dtype='string')
             enriched_df['artist_popularity'] = pd.Series(dtype='float64')
             return enriched_df
    else:
        logging.warning("No Spotify client provided for enrichment. Cannot fetch artist data.")
        enriched_df['artist_genres'] = pd.Series(dtype='string')
        enriched_df['artist_popularity'] = pd.Series(dtype='float64')
        return enriched_df

    # --- Prepare and Add Data to DataFrame ---
    if not artist_data:
         logging.warning("No artist data was successfully fetched. Enrichment columns will be empty.")
         enriched_df['artist_genres'] = pd.Series(dtype='string')
         enriched_df['artist_popularity'] = pd.Series(dtype='float64')
         return enriched_df
    else:
         # Call prepare_artist_data, making sure it knows which column to use
         return prepare_artist_data(enriched_df, artist_data, id_column_name)


def prepare_artist_data(track_df: pd.DataFrame, artist_data: Dict[str, Dict[str, Any]], id_column: str) -> pd.DataFrame:
    """Prepares artist data (genres, avg popularity) and adds it to the track DataFrame, using the specified ID column.""" # Modified Docstring + id_column parameter

    logging.info(f"Preparing and adding artist data to DataFrame based on column '{id_column}'...")

    # --- Define helper functions for applying artist data ---
    def get_aggregated_genres(artist_ids_str):
        if pd.isna(artist_ids_str) or not isinstance(artist_ids_str, str):
            return None
        ids = [aid.strip() for aid in artist_ids_str.split(',') if aid.strip()]
        genres = set()
        for artist_id in ids:
            data = artist_data.get(artist_id)
            if data:
                genres.update(data.get('genres', []))
        return ', '.join(sorted(list(genres))) if genres else None

    def get_average_popularity(artist_ids_str):
        if pd.isna(artist_ids_str) or not isinstance(artist_ids_str, str):
            return None
        ids = [aid.strip() for aid in artist_ids_str.split(',') if aid.strip()]
        pops = []
        for artist_id in ids:
             data = artist_data.get(artist_id)
             if data and data.get('popularity') is not None:
                  pops.append(data['popularity'])
        return sum(pops) / len(pops) if pops else None

    # --- Apply the functions row-wise using .apply ---
    # *** MODIFIED: Apply based on the passed 'id_column' ***
    if id_column in track_df.columns:
        track_df['artist_genres'] = track_df[id_column].apply(get_aggregated_genres)
        track_df['artist_popularity'] = track_df[id_column].apply(get_average_popularity)
    else:
        logging.warning(f"Column '{id_column}' disappeared unexpectedly before applying enrichment data.")
        track_df['artist_genres'] = pd.Series(dtype='string')
        track_df['artist_popularity'] = pd.Series(dtype='float64')

    # --- Convert types for consistency ---
    track_df['artist_genres'] = track_df['artist_genres'].astype('string')
    track_df['artist_popularity'] = track_df['artist_popularity'].astype('Float64')

    logging.info("Finished adding artist enrichment data.")
    return track_df

# --- Data Analysis (No changes needed here directly, but relies on above) ---
# This function already reads 'artist_genres', 'artist_popularity', and 'popularity'
# which are either original columns or created by the enrichment functions above.
# Just ensure the input DataFrame (`enriched_df`) has these columns correctly populated.
def analyze_genres(track_df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes genres from the track DataFrame (using the enriched 'artist_genres')."""

    logging.info("Starting genre analysis...")

    # Check if the necessary columns exist and have data
    required_cols = ['artist_genres', 'artist_popularity', 'popularity'] # 'popularity' is track popularity
    if not all(col in track_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in track_df.columns]
        logging.warning(f"Required columns missing for genre analysis: {missing}. Returning empty analysis.")
        # If 'popularity' is missing, it's likely the base parquet file issue discussed before.
        if 'popularity' in missing:
             logging.warning("The base track 'popularity' column is missing. Consider regenerating the base parquet file.")
        return pd.DataFrame(columns=['Genre', 'Track Count', 'Average Artist Popularity', 'Average Song Popularity'])

    if track_df['artist_genres'].isnull().all():
        logging.warning("Column 'artist_genres' is empty or all null. Cannot analyze genres.")
        return pd.DataFrame(columns=['Genre', 'Track Count', 'Average Artist Popularity', 'Average Song Popularity'])

    # --- Explode Genres ---
    temp_df = track_df[['track_id', 'artist_genres', 'artist_popularity', 'popularity']].copy()
    temp_df.dropna(subset=['artist_genres'], inplace=True)

    temp_df['genre_list'] = temp_df['artist_genres'].str.split(',')
    genre_exploded_df = temp_df.explode('genre_list')

    genre_exploded_df['Genre'] = genre_exploded_df['genre_list'].str.strip()
    genre_exploded_df = genre_exploded_df[genre_exploded_df['Genre'] != '']

    if genre_exploded_df.empty:
         logging.warning("No valid genres found after exploding and cleaning. Cannot perform analysis.")
         return pd.DataFrame(columns=['Genre', 'Track Count', 'Average Artist Popularity', 'Average Song Popularity'])

    # --- Calculate Metrics ---
    genre_counts = genre_exploded_df['Genre'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Track Count']

    genre_exploded_df['artist_popularity'] = pd.to_numeric(genre_exploded_df['artist_popularity'], errors='coerce')
    genre_artist_popularity = genre_exploded_df.groupby('Genre')['artist_popularity'].mean().reset_index()
    genre_artist_popularity.rename(columns={'artist_popularity': 'Average Artist Popularity'}, inplace=True)

    genre_exploded_df['popularity'] = pd.to_numeric(genre_exploded_df['popularity'], errors='coerce')
    genre_song_popularity = genre_exploded_df.groupby('Genre')['popularity'].mean().reset_index()
    genre_song_popularity.rename(columns={'popularity': 'Average Song Popularity'}, inplace=True)

    # --- Merge Results ---
    analysis_df = pd.merge(genre_counts, genre_artist_popularity, on='Genre', how='left')
    analysis_df = pd.merge(analysis_df, genre_song_popularity, on='Genre', how='left')

    analysis_df = analysis_df.sort_values(by='Track Count', ascending=False).reset_index(drop=True)

    logging.info(f"Genre analysis complete. Found {len(analysis_df)} unique genres.")
    return analysis_df

# --- Data Export ---
def export_data(df: Optional[pd.DataFrame], file_path: str, export_format: str = "parquet") -> None:
    """Exports the DataFrame to a file (CSV or Parquet)."""
    if df is None or df.empty:
        logging.warning(f"DataFrame is empty or None. Skipping export to {file_path}")
        return

    # Ensure the directory exists
    export_dir = os.path.dirname(file_path)
    try:
        os.makedirs(export_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directory {export_dir}: {e}")
        return # Stop if directory can't be created

    # Validate format
    allowed_formats = ["parquet", "csv"]
    if export_format.lower() not in allowed_formats:
        logging.error(f"Invalid export_format: {export_format}. Must be one of {allowed_formats}. Defaulting to parquet.")
        export_format = "parquet" # Or raise ValueError

    # Adjust file extension if needed (e.g., if file_path had .csv but format is parquet)
    base, _ = os.path.splitext(file_path)
    full_path = f"{base}.{export_format.lower()}"

    try:
        logging.info(f"Attempting to export data to {full_path}...")
        if export_format.lower() == "csv":
            df.to_csv(full_path, index=False, encoding='utf-8-sig')
        elif export_format.lower() == "parquet":
            df.to_parquet(full_path, index=False, engine='pyarrow', compression='snappy') # PyArrow is generally recommended
        logging.info(f"Data successfully exported to {full_path}")
    except ImportError as e:
         if 'pyarrow' in str(e) and export_format.lower() == "parquet":
              logging.error("PyArrow library not found. Please install it (`pip install pyarrow`) to export to Parquet.")
         elif 'fastparquet' in str(e) and export_format.lower() == "parquet":
              logging.error("FastParquet library not found. Please install it (`pip install fastparquet`) or pyarrow to export to Parquet.")
         else:
              logging.error(f"ImportError during export to {full_path}: {e}")
    except Exception as e:
        logging.error(f"Error exporting data to {full_path}: {e}")


# --- Main Execution ---
def main(spotify_client: Optional[spotipy.Spotify] = None, max_tracks: Optional[int] = None, skip_collection: bool = False) -> None:
    """Main function to control data retrieval, enrichment, analysis, and export."""

    track_df = None

    # Step 1: Get Base Track Data (either fetch or load)
    if not skip_collection:
        logging.info("Attempting to fetch saved tracks from Spotify API...")
        if not spotify_client:
            logging.warning("Spotify client not provided to main(). Attempting to create one.")
            spotify_client = create_spotify_client()
            if not spotify_client:
                logging.error("Failed to create Spotify client. Aborting.")
                return # Exit if client creation fails

        track_df = get_and_process_saved_tracks(spotify_client, max_tracks)

        if track_df is None or track_df.empty:
            logging.error("Failed to retrieve or process any saved tracks. Aborting.")
            return

        # Export the raw retrieved data
        export_data(track_df, DEFAULT_SAVED_TRACKS_PATH, export_format="parquet")
    else:
        logging.info(f"Skipping song collection. Attempting to load base data from {DEFAULT_SAVED_TRACKS_PATH}...")
        try:
            track_df = pd.read_parquet(DEFAULT_SAVED_TRACKS_PATH)
            logging.info(f"Successfully loaded {len(track_df)} tracks from {DEFAULT_SAVED_TRACKS_PATH}")
        except FileNotFoundError:
            logging.error(f"Saved tracks file not found at: {DEFAULT_SAVED_TRACKS_PATH}. Cannot proceed with skip_collection=True.")
            logging.error("Please run the script once with skip_collection=False (or the appropriate flag) to generate the base file.")
            return # Exit if base file not found when skipping
        except Exception as e:
             logging.error(f"Error loading data from {DEFAULT_SAVED_TRACKS_PATH}: {e}")
             return # Exit on other loading errors

    # Step 2: Enrich Data
    logging.info("Attempting to enrich track data with artist information...")
    # Ensure client is available if enrichment is needed and wasn't created in step 1
    if not spotify_client and not skip_collection: # Should have client from above
         pass # Already handled
    elif not spotify_client and skip_collection:
         logging.warning("Spotify client not provided while skipping collection. Attempting to create one for enrichment.")
         spotify_client = create_spotify_client()
         if not spotify_client:
              logging.warning("Could not create Spotify client for enrichment. Proceeding without enrichment.")
              # Create an empty enriched_df based on track_df structure but without new columns
              enriched_df = track_df.copy()
              enriched_df['artist_genres'] = pd.Series(dtype='string')
              enriched_df['artist_popularity'] = pd.Series(dtype='float64')
         else:
              enriched_df = enrich_track_dataframe(spotify_client, track_df)
    else: # Client was provided or created earlier
         enriched_df = enrich_track_dataframe(spotify_client, track_df)

    if enriched_df is None or enriched_df.empty:
        logging.error("Enrichment process failed or resulted in empty data. Aborting analysis and final export.")
        return

    # Export enriched data
    export_data(enriched_df, DEFAULT_GENRE_TRACKS_PATH, export_format="parquet")

    # Step 3: Analyze Genres
    logging.info("Attempting to analyze genres...")
    genre_analysis_df = analyze_genres(enriched_df)

    # Export genre analysis
    export_data(genre_analysis_df, DEFAULT_GENRE_ANALYSIS_PATH, export_format="parquet")

    logging.info("--- Processing Complete ---")
    if not skip_collection:
         logging.info(f"Raw track data saved to: {DEFAULT_SAVED_TRACKS_PATH}")
    logging.info(f"Enriched track data saved to: {DEFAULT_GENRE_TRACKS_PATH}")
    if genre_analysis_df is not None and not genre_analysis_df.empty:
         logging.info(f"Genre analysis saved to: {DEFAULT_GENRE_ANALYSIS_PATH}")
    else:
         logging.info("Genre analysis was empty or failed, not saved.")


if __name__ == "__main__":
    # --- IMPORTANT ---
    # 1. Fix the `TypeError` in `Workspace_artist_data` first (done in the code above).
    # 2. If you encounter the `KeyError: 'Column not found: popularity'` or suspect
    #    your saved file is outdated, run the script ONCE with skip_collection=False
    #    to regenerate `saved_tracks.parquet`.
    # 3. After that, you can set skip_collection=True for faster runs that only
    #    perform enrichment and analysis based on the saved file.

    # Example: Run to generate base file and perform full process
    # print("Running full process to generate/update base files...")
    # client = create_spotify_client()
    # if client:
    #     main(client, skip_collection=False)
    # else:
    #     print("Could not create Spotify client. Exiting.")

    # Example: Run assuming base file exists and is up-to-date
    print("Running enrichment and analysis using existing base file...")
    client = create_spotify_client()
    if client:
        main(client, skip_collection=False) # Set to True to load from file
    else:
        print("Could not create Spotify client. Exiting.")