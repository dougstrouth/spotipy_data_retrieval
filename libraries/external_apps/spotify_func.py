# %%capture
# # Use this in Jupyter to suppress pip install output if needed
# # %pip install spotipy pandas python-dotenv pyarrow requests python-dateutil numpy

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from dotenv import load_dotenv
import pandas as pd
import numpy as np # Needed for specific type checks
import logging
from typing import Optional, List, Dict, Any, Set
import time
import re # For ID validation
import requests # For ConnectionError handling
# from dateutil.parser import isoparse # Using pd.to_datetime which is usually sufficient

# --- Configuration ---
# TODO: Update paths if needed
DEFAULT_EXPORT_PATH = '/Users/dougstrouth/Library/Mobile Documents/com~apple~CloudDocs/Documents/Code/datasets/spotify'
ENV_FILE_PATH = "/Users/dougstrouth/Documents/.env" # Assumed location of your .env file

# --- Constants ---
ARTIST_BATCH_SIZE = 50
TRACK_FETCH_LIMIT = 50
ALBUM_BATCH_SIZE = 20

INITIAL_DELAY = 1
MAX_RETRIES = 5

# File Paths (Derived from DEFAULT_EXPORT_PATH)
# Ensure the base export path exists
os.makedirs(DEFAULT_EXPORT_PATH, exist_ok=True)

PARQUET_BASE_TRACKS = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_base.parquet')
PARQUET_ARTIST_CACHE = os.path.join(DEFAULT_EXPORT_PATH, 'artist_details.parquet')
PARQUET_ALBUM_CACHE = os.path.join(DEFAULT_EXPORT_PATH, 'album_details.parquet')
PARQUET_ARTIST_ENRICHED = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_artist_enriched.parquet')
PARQUET_ALBUM_ENRICHED = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_album_enriched.parquet')
FINAL_DATA_PATH = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_enriched_final.parquet')
PARQUET_GENRE_ANALYSIS = os.path.join(DEFAULT_EXPORT_PATH, 'genre_analysis.parquet')

# --- Central Schema Definitions ---
# Defines expected columns and their target pandas dtypes (using nullable types)
# Schema for the base DataFrame after initial fetch and processing
BASE_SCHEMA = {
    'added_at': 'datetime64[ns, UTC]', 'track_id': 'string', 'track_name': 'string',
    'popularity': 'Int32', 'duration_ms': 'Int64', 'explicit': 'boolean', 'isrc': 'string',
    'track_uri': 'string', 'track_href': 'string', 'track_external_url_spotify': 'string',
    'track_preview_url': 'string', 'track_is_local': 'boolean', 'track_is_playable': 'boolean',
    'track_disc_number': 'Int32', 'track_number': 'Int32', 'track_type': 'string',
    'track_available_markets': 'string', 'artist_ids': 'string', 'artist_names': 'string',
    'track_artist_uris': 'string', 'album_id': 'string', 'album_name': 'string',
    'album_release_date': 'string', 'album_uri': 'string', 'album_href': 'string',
    'album_external_url_spotify': 'string', 'album_type': 'string',
    'album_release_date_precision': 'string', 'album_total_tracks': 'Int32',
    'album_available_markets': 'string', 'album_image_url_64': 'string',
    'album_image_url_300': 'string', 'album_image_url_640': 'string',
    'album_artist_ids': 'string', 'album_artist_names': 'string',
    # Columns added during initial processing:
    'release_datetime': 'datetime64[ns, UTC]', 'release_year': 'Int64'
}
# Schema for the artist cache file
ARTIST_CACHE_SCHEMA = {
    'artist_id': 'string', 'artist_fetched_genres_list': 'object',
    'artist_fetched_popularity': 'Int32', 'artist_fetched_name': 'string',
    'artist_last_fetched': 'datetime64[ns, UTC]'
}
# Schema for the new album cache file
ALBUM_CACHE_SCHEMA = {
    'album_id': 'string', 'album_fetched_genres_list': 'object',
    'album_fetched_popularity': 'Int32', 'album_fetched_label': 'string',
    'album_last_fetched': 'datetime64[ns, UTC]'
}
# Schema for columns added *during* enrichment steps
ENRICHMENT_SCHEMA = {
    'artist_genres': 'string', 'artist_popularity': 'Float64',
    'album_genres': 'string', 'album_label': 'string', 'album_popularity': 'Int32'
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING) # Quieter logs

# --- Spotify Client & API Wrapper ---
def create_spotify_client() -> Optional[spotipy.Spotify]:
    """Authenticates with Spotify and returns a Spotipy client."""
    try:
        if not os.path.exists(ENV_FILE_PATH):
            logging.warning(f".env file not found at {ENV_FILE_PATH}. Trying current dir.")
            env_path = ".env" # Fallback assumes .env in script directory
            if not os.path.exists(env_path):
                 logging.error(f".env file not found in current directory either. Cannot load credentials.")
                 return None
        else: env_path = ENV_FILE_PATH
        load_dotenv(dotenv_path=env_path)
        client_id = os.getenv("spotify_client_id")
        client_secret = os.getenv("spotify_client_secret")
        redirect_uri = os.getenv("spotify_redirect_uri") # e.g., http://127.0.0.1:8888/

        if not all([client_id, client_secret, redirect_uri]):
            logging.error("Missing Spotify credentials (client_id, client_secret, redirect_uri) in environment variables.")
            return None
        # --- IMPORTANT: Fix Redirect URI ---
        if 'localhost' in redirect_uri:
            logging.error("FATAL: Redirect URI uses 'localhost'. Please update .env and Spotify Dashboard to use 'http://127.0.0.1:PORT/' (e.g., http://127.0.0.1:8888/)")
            return None # Prevent running with deprecated URI

        scopes = "user-library-read"
        # Define cache path explicitly within the project or a known location
        script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
        cache_path = os.path.join(script_dir, ".spotify_cache") # Cache in script's dir

        sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri,
                scope=scopes, open_browser=False, cache_handler=spotipy.CacheFileHandler(cache_path=cache_path)
            )
        )
        user_info = sp.current_user()
        logging.info(f"Spotify client created and authenticated for user: {user_info.get('display_name','N/A')}")
        return sp
    except Exception as e:
        logging.error(f"Error creating Spotify client: {e}", exc_info=True)
        return None

def call_spotify_api(api_function: callable, *args, **kwargs) -> Any:
    """General function to call Spotify API with backoff-retry and improved logging."""
    # (Implementation from previous version - unchanged)
    delay = INITIAL_DELAY
    func_name = getattr(api_function, '__name__', 'unknown_api_function')
    logging.debug(f"Calling API: {func_name} Args: {args} Kwargs: {kwargs}")
    for attempt in range(MAX_RETRIES):
        try:
            results = api_function(*args, **kwargs); return results
        except spotipy.exceptions.SpotifyException as e:
            log_prefix = f"Spotify API error calling {func_name} (Attempt {attempt + 1}/{MAX_RETRIES})"
            # Avoid logging potentially large kwargs content directly in error message by default
            log_details = f"Status={e.http_status}, Code={e.code}, URL={e.url}, Msg='{e.msg}'"
            if e.http_status == 429:
                retry_after = int(e.headers.get('Retry-After', delay))
                logging.warning(f"{log_prefix}. Rate limit exceeded. Retrying in {retry_after}s...")
                time.sleep(retry_after); delay = min(delay * 2, 60)
            elif e.http_status in [401, 403]:
                 logging.error(f"{log_prefix}. Auth/Permission Error: {log_details}")
                 if attempt >= 1 or e.http_status == 403: raise
                 time.sleep(delay*2); delay = min(delay * 2, 60)
            else:
                logging.error(f"{log_prefix}: {log_details}", exc_info=True) # Log full traceback for others
                if attempt == MAX_RETRIES - 1: raise
                time.sleep(delay); delay = min(delay * 2, 60)
        except requests.exceptions.ConnectionError as e:
             logging.error(f"Connection error calling {func_name} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}", exc_info=True)
             if attempt == MAX_RETRIES - 1: raise
             time.sleep(delay); delay = min(delay * 2, 60)
        except Exception as e:
            logging.error(f"Unexpected error calling {func_name} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}", exc_info=True)
            if attempt == MAX_RETRIES - 1: raise
            time.sleep(delay); delay = min(delay * 2, 60)
    logging.error(f"Max retries ({MAX_RETRIES}) reached for {func_name}. API call failed.")
    raise Exception(f"Max retries reached for {func_name}")


# --- Data Processing & Type Application ---
# Refactored extraction logic into helpers
def _extract_track_level_data(track: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts track-specific fields."""
    return { 'track_id': track.get('id'), 'track_name': track.get('name'), 'popularity': track.get('popularity'), 'duration_ms': track.get('duration_ms'), 'explicit': track.get('explicit'), 'isrc': track.get('external_ids', {}).get('isrc'), 'track_uri': track.get('uri'), 'track_href': track.get('href'), 'track_external_url_spotify': track.get('external_urls', {}).get('spotify'), 'track_preview_url': track.get('preview_url'), 'track_is_local': track.get('is_local', False), 'track_is_playable': track.get('is_playable'), 'track_disc_number': track.get('disc_number'), 'track_number': track.get('track_number'), 'track_type': track.get('type'), 'track_available_markets': ','.join(track.get('available_markets', [])) or None }
def _extract_track_artist_data(track_artists: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts and aggregates track artist info."""
    valid_ids, valid_names, valid_uris = [], [], []
    for artist in track_artists:
        if artist and artist.get('id'): valid_ids.append(artist['id']); valid_names.append(artist.get('name')); valid_uris.append(artist.get('uri'))
    return { 'artist_ids': ', '.join(filter(None, valid_ids)) or None, 'artist_names': ', '.join(filter(None, valid_names)) or None, 'track_artist_uris': ', '.join(filter(None, valid_uris)) or None }
def _extract_album_level_data(album: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts album-specific fields."""
    images = album.get('images', [])
    return { 'album_id': album.get('id'), 'album_name': album.get('name'), 'album_release_date': album.get('release_date'), 'album_uri': album.get('uri'), 'album_href': album.get('href'), 'album_external_url_spotify': album.get('external_urls', {}).get('spotify'), 'album_type': album.get('album_type'), 'album_release_date_precision': album.get('release_date_precision'), 'album_total_tracks': album.get('total_tracks'), 'album_available_markets': ','.join(album.get('available_markets', [])) or None, 'album_image_url_64': next((img['url'] for img in images if img and img.get('height') == 64), None), 'album_image_url_300': next((img['url'] for img in images if img and img.get('height') == 300), None), 'album_image_url_640': next((img['url'] for img in images if img and img.get('height') == 640), None) }
def _extract_album_artist_data(album_artists: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extracts and aggregates album artist info."""
    valid_ids, valid_names = [], []
    for artist in album_artists:
        if artist and artist.get('id'): valid_ids.append(artist['id']); valid_names.append(artist.get('name'))
    return { 'album_artist_ids': ', '.join(filter(None, valid_ids)) or None, 'album_artist_names': ', '.join(filter(None, valid_names)) or None }

def process_track_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Processes a single saved track item using helper functions."""
    try:
        added_at = item.get('added_at'); track = item.get('track')
        if not track or not track.get('id'): return None
        album = track.get('album', {}); track_artists = track.get('artists', []); album_artists = album.get('artists', []) # Album artists from album object
        processed_data = {'added_at': added_at}
        processed_data.update(_extract_track_level_data(track)); processed_data.update(_extract_track_artist_data(track_artists))
        processed_data.update(_extract_album_level_data(album)); processed_data.update(_extract_album_artist_data(album_artists))
        if not processed_data.get('track_id'): return None
        return processed_data
    except Exception as e: logging.error(f"Error processing track item: {e}", exc_info=True); return None

def apply_data_types(df: pd.DataFrame, schema_map: Dict[str, str]) -> pd.DataFrame:
    """Applies specified pandas data types to DataFrame columns based on a schema map."""
    # (Implementation from previous version - unchanged)
    if df is None or df.empty: return df
    logging.info(f"Applying data types based on schema ({len(schema_map)} columns defined)...")
    df_out = df.copy(); converted_cols, skipped_cols, error_cols = 0, 0, 0
    processed_cols = set() # Keep track of columns processed
    for col, dtype_str in schema_map.items():
        if col not in df_out.columns: skipped_cols += 1; continue
        processed_cols.add(col)
        current_dtype = df_out[col].dtype
        try:
            target_dtype = pd.core.dtypes.common.pandas_dtype(dtype_str)
            if current_dtype == target_dtype and not pd.api.types.is_object_dtype(current_dtype): converted_cols +=1; continue
            # --- Conversion Logic ---
            if dtype_str == 'datetime64[ns, UTC]': df_out[col] = pd.to_datetime(df_out[col], errors='coerce', utc=True)
            elif dtype_str.startswith('Int'): df_out[col] = pd.to_numeric(df_out[col], errors='coerce').astype(target_dtype)
            elif dtype_str.startswith('Float'): df_out[col] = pd.to_numeric(df_out[col], errors='coerce').astype(target_dtype) # Match 'Float64'
            elif dtype_str == 'boolean':
                # Robust boolean conversion
                map_dict = {'true': True, 'false': False, '1': True, '0': False, 1: True, 0: False, 1.0: True, 0.0: False}
                temp_col = df_out[col]
                if pd.api.types.is_string_dtype(temp_col) or pd.api.types.is_object_dtype(temp_col):
                    temp_col = temp_col.astype(str).str.lower().map(map_dict)
                df_out[col] = temp_col.astype(target_dtype)
            else: # string types etc.
                 df_out[col] = df_out[col].astype(target_dtype)
            converted_cols += 1
        except Exception as e:
            logging.warning(f"Could not convert '{col}' to {dtype_str}. Current: {current_dtype}. Error: {e}")
            error_cols += 1
    untouched_cols = len(df_out.columns) - len(processed_cols)
    logging.info(f"Type conversion: Applied/Verified={converted_cols}, Skipped(Not in Schema)={untouched_cols}, Skipped(Not Found)={skipped_cols}, Errors={error_cols}")
    return df_out

def _parse_release_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parses album_release_date into datetime and year columns."""
    # (Implementation from previous version - unchanged)
    if 'album_release_date' in df.columns:
         logging.info("Parsing 'album_release_date' -> 'release_datetime', 'release_year'.")
         df_out = df.copy()
         # Handle different precisions before converting
         df_out['temp_date'] = df_out['album_release_date'].astype(str)
         df_out['release_datetime'] = pd.to_datetime(df_out['temp_date'], errors='coerce')
         df_out.loc[df_out['album_release_date_precision'] == 'year', 'release_datetime'] = pd.to_datetime(df_out['temp_date'] + '-01-01', errors='coerce')
         df_out.loc[df_out['album_release_date_precision'] == 'month', 'release_datetime'] = pd.to_datetime(df_out['temp_date'] + '-01', errors='coerce')
         # Extract year
         df_out['release_year'] = df_out['release_datetime'].dt.year.astype('Int64')
         df_out = df_out.drop(columns=['temp_date'])
         # Apply UTC timezone for consistency if desired, though release dates don't have inherent TZ
         # df_out['release_datetime'] = df_out['release_datetime'].dt.tz_localize('UTC')
         return df_out
    return df

# --- Data Retrieval ---
def get_and_process_saved_tracks(spotify_client: spotipy.Spotify, max_tracks: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Retrieves, processes, and type-converts saved tracks."""
    # (Implementation from previous version - unchanged)
    offset = 0; all_tracks_data = []; total_processed = 0; total_retrieved = 0
    logging.info("Starting retrieval of saved tracks...")
    while True:
        processed_batch = 0
        try:
            results = call_spotify_api(spotify_client.current_user_saved_tracks, limit=TRACK_FETCH_LIMIT, offset=offset)
            if results is None: break
            items = results.get('items', [])
            if not items: break
            num_items = len(items); total_retrieved = offset + num_items
            for item in items:
                processed = process_track_item(item)
                if processed: all_tracks_data.append(processed); processed_batch += 1
            total_processed += processed_batch
            logging.info(f"Batch offset={offset}: Processed {processed_batch}/{num_items}. Total OK: {total_processed}")
            if max_tracks and total_processed >= max_tracks:
                logging.info(f"Reached max_tracks limit ({max_tracks})."); all_tracks_data = all_tracks_data[:max_tracks]; break
            total_avail = results.get('total', float('inf'))
            if num_items < TRACK_FETCH_LIMIT or total_retrieved >= total_avail: logging.info("Reached end of saved tracks."); break
            offset += TRACK_FETCH_LIMIT
        except Exception as e: logging.error(f"Error retrieving batch offset {offset}: {e}", exc_info=True); break
    if not all_tracks_data: logging.warning("No track data processed."); return None
    logging.info(f"Processed {len(all_tracks_data)} tracks into dicts.")
    df = pd.DataFrame(all_tracks_data)
    logging.info(f"Created DataFrame shape: {df.shape}.")
    df = apply_data_types(df, BASE_SCHEMA)
    df = _parse_release_date_columns(df)
    logging.info("Base track DataFrame prepared.")
    return df


# --- Artist Data Fetching & Caching (Refactored) ---
def load_artist_cache(cache_path: str) -> pd.DataFrame:
    """Loads the artist cache DataFrame, applies schema types."""
    # (Implementation from previous version - unchanged)
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path); logging.info(f"Loaded {len(df)} artists from cache: {cache_path}")
            if 'artist_id' not in df.columns: logging.warning("Artist cache missing 'artist_id'. Ignoring."); return pd.DataFrame()
            if 'artist_last_fetched' in df.columns: df['artist_last_fetched'] = pd.to_datetime(df['artist_last_fetched'], errors='coerce', utc=True)
            df = apply_data_types(df, ARTIST_CACHE_SCHEMA); return df # Apply schema types
        except Exception as e: logging.error(f"Error loading artist cache: {e}. Ignoring.", exc_info=True)
    else: logging.info(f"Artist cache not found: {cache_path}.")
    return pd.DataFrame()

def fetch_artist_data(spotify_client: spotipy.Spotify, artist_ids: List[str]) -> Optional[pd.DataFrame]:
    """Fetches artist data, returns DataFrame with schema types applied."""
    # (Implementation from previous version - unchanged)
    unique_artist_ids = list(set(aid for aid in artist_ids if aid))
    artist_data_list = []; fetched_count = 0; total_ids = len(unique_artist_ids)
    if not total_ids: return None
    logging.info(f"Fetching details for {total_ids} unique artist IDs.")
    fetch_timestamp = pd.Timestamp.now(tz='UTC')
    for i in range(0, total_ids, ARTIST_BATCH_SIZE):
        batch_ids = unique_artist_ids[i:min(i + ARTIST_BATCH_SIZE, total_ids)]
        try:
            results = call_spotify_api(spotify_client.artists, artists=batch_ids)
            if results and results.get('artists'):
                for artist in results['artists']:
                    if artist and artist.get('id'):
                        artist_data_list.append({ 'artist_id': artist['id'], 'artist_fetched_genres_list': artist.get('genres', []), 'artist_fetched_popularity': artist.get('popularity'), 'artist_fetched_name': artist.get('name'), 'artist_last_fetched': fetch_timestamp }); fetched_count += 1
        except Exception as e: logging.error(f"Failed artist batch fetch index {i}: {e}", exc_info=True)
    if not artist_data_list: logging.warning("No artist data fetched."); return None
    logging.info(f"Fetched data for {fetched_count}/{total_ids} artists.")
    artist_df = pd.DataFrame(artist_data_list)
    artist_df = apply_data_types(artist_df, ARTIST_CACHE_SCHEMA); return artist_df

def update_artist_cache(spotify_client: Optional[spotipy.Spotify], required_artist_ids: Set[str],
                       cache_path: str, force_refresh: bool = False) -> pd.DataFrame:
    """Manages the artist cache and returns the final complete artist DataFrame."""
    # (Implementation from previous version - unchanged)
    existing_artist_df = load_artist_cache(cache_path)
    cached_ids = set(existing_artist_df['artist_id']) if 'artist_id' in existing_artist_df else set()
    ids_to_fetch = list(required_artist_ids - cached_ids) if not force_refresh else list(required_artist_ids)
    if force_refresh and not existing_artist_df.empty: logging.info("Forcing refresh of artist cache."); existing_artist_df = pd.DataFrame(); cached_ids = set()
    logging.info(f"Artist Cache: Required={len(required_artist_ids)}, Cached={len(cached_ids)}, To Fetch={len(ids_to_fetch)}")
    newly_fetched_df = None
    if ids_to_fetch and spotify_client: newly_fetched_df = fetch_artist_data(spotify_client, ids_to_fetch)
    elif not spotify_client and ids_to_fetch: logging.warning("Need new artists but no Spotify client.")
    final_df = existing_artist_df
    if newly_fetched_df is not None and not newly_fetched_df.empty:
        final_df = pd.concat([existing_artist_df, newly_fetched_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['artist_id'], keep='last').reset_index(drop=True)
        logging.info(f"Saving updated artist cache ({len(final_df)} total) to {cache_path}")
        export_data(final_df, cache_path)
    elif not existing_artist_df.empty : logging.info("Using only cached artist data.")
    else: logging.warning("No artist data available (cached or fetched)."); return pd.DataFrame() # Return empty df if none
    return final_df

# Replace the entire function definition in your script

def apply_artist_data_to_tracks(track_df: pd.DataFrame, artist_data_df: pd.DataFrame) -> pd.DataFrame:
    """Applies artist cache data to track DataFrame, with fix for empty arrays."""
    if artist_data_df is None or artist_data_df.empty:
        logging.warning("No artist data to apply.")
        track_df_out = track_df.copy()
        if 'artist_genres' not in track_df_out.columns: track_df_out['artist_genres'] = pd.Series(dtype='string')
        if 'artist_popularity' not in track_df_out.columns: track_df_out['artist_popularity'] = pd.Series(dtype='Float64')
        return track_df_out

    logging.info(f"Applying artist data ({len(artist_data_df)} artists) to {len(track_df)} tracks...")
    lookup_cols = ['artist_id', 'artist_fetched_genres_list', 'artist_fetched_popularity']
    if not all(col in artist_data_df.columns for col in lookup_cols):
        missing = [c for c in lookup_cols if c not in artist_data_df.columns]; logging.error(f"Artist data missing required columns: {missing}."); return track_df

    # Create map for efficient lookup
    artist_map = artist_data_df.set_index('artist_id')[lookup_cols[1:]].to_dict('index')

    # --- Nested Helper Function _get_genres Definition STARTS HERE ---
    def _get_genres(ids_str):
        # Check input string
        if pd.isna(ids_str) or not isinstance(ids_str, str): return None
        # Create the 'ids' list INSIDE this function
        ids = [aid.strip() for aid in ids_str.split(',') if aid.strip()]
        genres = set()

        # The loop belongs INSIDE _get_genres
        for aid in ids:
            data = artist_map.get(aid)
            if data:
                genre_data = data.get('artist_fetched_genres_list')

                # --- REFINED Checks - Order Matters! ---
                # 1. Check for None directly
                if genre_data is None:
                    continue

                # 2. Check specifically for empty list or numpy array BEFORE pd.isna
                is_list_or_array = isinstance(genre_data, (list, np.ndarray))
                is_empty = False
                if is_list_or_array:
                    try:
                        is_empty = (len(genre_data) == 0)
                    except TypeError:
                        is_empty = True

                if is_empty: # Skip if it's an empty list/array
                    continue
                # --- End REFINED Checks ---

                # 3. Now attempt to iterate
                try:
                    if is_list_or_array:
                        for g in genre_data:
                            if isinstance(g, str) and g: genres.add(g)
                except TypeError:
                    logging.warning(f"Genre data for artist {aid} was not None, empty, or list/array: type={type(genre_data)}, value={genre_data}")

        # Return statement is part of _get_genres
        return ', '.join(sorted(list(genres))) if genres else None
    # --- Nested Helper Function _get_genres Definition ENDS HERE ---


    # --- Nested Helper Function _get_popularity Definition STARTS HERE ---
    def _get_popularity(ids_str):
        # (Implementation from previous version)
        if pd.isna(ids_str) or not isinstance(ids_str, str): return None
        # Create the 'ids' list INSIDE this function
        ids = [aid.strip() for aid in ids_str.split(',') if aid.strip()]
        pops = []
        for aid in ids:
             data = artist_map.get(aid)
             pop_val = data.get('artist_fetched_popularity') if data else None
             if pd.notna(pop_val):
                  try: pops.append(float(pop_val))
                  except (ValueError, TypeError): pass
        return sum(pops) / len(pops) if pops else None
    # --- Nested Helper Function _get_popularity Definition ENDS HERE ---


    # --- Apply the helper functions ---
    track_df_out = track_df.copy()
    # These apply calls happen AFTER the functions are defined
    track_df_out['artist_genres'] = track_df_out['artist_ids'].apply(_get_genres)
    track_df_out['artist_popularity'] = track_df_out['artist_ids'].apply(_get_popularity)

    # Apply final types from ENRICHMENT_SCHEMA
    track_df_out = apply_data_types(track_df_out, {'artist_genres': ENRICHMENT_SCHEMA['artist_genres'],
                                                  'artist_popularity': ENRICHMENT_SCHEMA['artist_popularity']})

    # Logging and return
    null_genre_count = track_df_out['artist_genres'].isnull().sum()
    if null_genre_count == len(track_df_out):
         logging.warning("Applied artist data, but 'artist_genres' column is STILL all null. Check cache content/API.")
    else:
         logging.info(f"Finished applying artist data. Null artist_genres: {null_genre_count}/{len(track_df_out)}")
    return track_df_out
# --- Album Data Fetching & Caching ---
def fetch_album_details(spotify_client: spotipy.Spotify, album_ids: List[str]) -> Optional[pd.DataFrame]:
    """Fetches album details, returns DataFrame with schema types applied."""
    # (Implementation from previous version - unchanged)
    unique_album_ids = list(set(aid for aid in album_ids if aid))
    album_data_list = []; fetched_count = 0; total_ids = len(unique_album_ids)
    if not total_ids: return None
    logging.info(f"Fetching details for {total_ids} unique album IDs.")
    fetch_timestamp = pd.Timestamp.now(tz='UTC')
    for i in range(0, total_ids, ALBUM_BATCH_SIZE):
        batch_ids = unique_album_ids[i:min(i + ALBUM_BATCH_SIZE, total_ids)]
        try:
            results = call_spotify_api(spotify_client.albums, albums=batch_ids)
            if results and results.get('albums'):
                for album in results['albums']:
                    if album and album.get('id'):
                        album_data_list.append({ 'album_id': album['id'], 'album_fetched_genres_list': album.get('genres', []), 'album_fetched_popularity': album.get('popularity'), 'album_fetched_label': album.get('label'), 'album_last_fetched': fetch_timestamp }); fetched_count += 1
        except Exception as e: logging.error(f"Failed album batch fetch index {i}: {e}", exc_info=True)
    if not album_data_list: logging.warning("No album data fetched."); return None
    logging.info(f"Fetched data for {fetched_count}/{total_ids} albums.")
    album_df = pd.DataFrame(album_data_list)
    album_df = apply_data_types(album_df, ALBUM_CACHE_SCHEMA); return album_df

def load_album_cache(cache_path: str) -> pd.DataFrame:
    """Loads the album cache DataFrame, applies schema types."""
    # (Implementation from previous version - unchanged)
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path); logging.info(f"Loaded {len(df)} albums from cache: {cache_path}")
            if 'album_id' not in df.columns: logging.warning("Album cache missing 'album_id'. Ignoring."); return pd.DataFrame()
            if 'album_last_fetched' in df.columns: df['album_last_fetched'] = pd.to_datetime(df['album_last_fetched'], errors='coerce', utc=True)
            df = apply_data_types(df, ALBUM_CACHE_SCHEMA); return df
        except Exception as e: logging.error(f"Error loading album cache: {e}. Ignoring.", exc_info=True)
    else: logging.info(f"Album cache not found: {cache_path}.")
    return pd.DataFrame()

def update_album_cache(spotify_client: Optional[spotipy.Spotify], required_album_ids: Set[str],
                       cache_path: str, force_refresh: bool = False) -> pd.DataFrame:
    """Manages the album cache and returns the final complete album DataFrame."""
    # (Implementation from previous version - unchanged)
    existing_album_df = load_album_cache(cache_path)
    cached_ids = set(existing_album_df['album_id']) if 'album_id' in existing_album_df else set()
    ids_to_fetch = list(required_album_ids - cached_ids) if not force_refresh else list(required_album_ids)
    if force_refresh and not existing_album_df.empty: logging.info("Forcing refresh of album cache."); existing_album_df = pd.DataFrame(); cached_ids = set()
    logging.info(f"Album Cache: Required={len(required_album_ids)}, Cached={len(cached_ids)}, To Fetch={len(ids_to_fetch)}")
    newly_fetched_df = None
    if ids_to_fetch and spotify_client: newly_fetched_df = fetch_album_details(spotify_client, ids_to_fetch)
    elif not spotify_client and ids_to_fetch: logging.warning("Need new albums but no Spotify client.")
    final_df = existing_album_df
    if newly_fetched_df is not None and not newly_fetched_df.empty:
        final_df = pd.concat([existing_album_df, newly_fetched_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['album_id'], keep='last').reset_index(drop=True)
        logging.info(f"Saving updated album cache ({len(final_df)} total) to {cache_path}")
        export_data(final_df, cache_path)
    elif not existing_album_df.empty: logging.info("Using only cached album data.")
    else: logging.warning("No album data available (cached or fetched)."); return pd.DataFrame()
    return final_df

def apply_album_data_to_tracks(track_df: pd.DataFrame, album_data_df: pd.DataFrame) -> pd.DataFrame:
    """Applies album cache data to track DataFrame."""
    # (Implementation from previous version - unchanged, ensures ENRICHMENT_SCHEMA types applied at end)
    if album_data_df is None or album_data_df.empty:
        logging.warning("No album data provided to apply.")
        track_df_out = track_df.copy()
        if 'album_genres' not in track_df_out.columns: track_df_out['album_genres'] = pd.Series(dtype='string')
        if 'album_label' not in track_df_out.columns: track_df_out['album_label'] = pd.Series(dtype='string')
        if 'album_popularity' not in track_df_out.columns: track_df_out['album_popularity'] = pd.Series(dtype='Int32')
        return track_df_out
    logging.info(f"Applying album data ({len(album_data_df)} albums) to {len(track_df)} tracks...")
    lookup_cols = ['album_id', 'album_fetched_genres_list', 'album_fetched_popularity', 'album_fetched_label']
    if not all(col in album_data_df.columns for col in lookup_cols):
        missing = [c for c in lookup_cols if c not in album_data_df.columns]; logging.error(f"Album data missing required columns: {missing}."); return track_df
    album_data_to_merge = album_data_df[lookup_cols].copy()
    album_data_to_merge['album_genres'] = album_data_to_merge['album_fetched_genres_list'].apply(lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else None)
    album_data_to_merge.rename(columns={'album_fetched_popularity': 'album_popularity','album_fetched_label': 'album_label'}, inplace=True)
    merge_cols = ['album_id', 'album_genres', 'album_popularity', 'album_label']
    album_data_to_merge = album_data_to_merge[[col for col in merge_cols if col in album_data_to_merge.columns]]
    if 'album_id' not in album_data_to_merge.columns: logging.error("Prepared album data missing 'album_id'."); return track_df
    track_df_out = pd.merge(track_df, album_data_to_merge, on='album_id', how='left', suffixes=('', '_album_detail'))
    # Apply final types for added columns using enrichment schema
    track_df_out = apply_data_types(track_df_out, {k: v for k, v in ENRICHMENT_SCHEMA.items() if k.startswith('album_')})
    logging.info("Finished applying album data.")
    return track_df_out

# --- Data Analysis ---
# (Keep analyze_genres function as it was)
def analyze_genres(track_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Analyzes genres based on track popularity and track artist data."""
    # (Implementation remains the same)
    logging.info("Starting genre analysis...")
    required_cols = ['artist_genres', 'artist_popularity', 'popularity']
    missing_cols = [col for col in required_cols if col not in track_df.columns]
    if missing_cols: logging.warning(f"Missing cols for genre analysis: {missing_cols}. Skipping."); return None
    if track_df['artist_genres'].isnull().all(): logging.warning("Column 'artist_genres' is all null. Skipping."); return None
    temp_df = track_df[required_cols + ['track_id']].dropna(subset=['artist_genres']).copy()
    if temp_df.empty: logging.warning("No rows with artist genres after dropna."); return None
    temp_df['genre_list'] = temp_df['artist_genres'].str.split(',')
    genre_exploded_df = temp_df.explode('genre_list')
    genre_exploded_df['Genre'] = genre_exploded_df['genre_list'].str.strip()
    genre_exploded_df = genre_exploded_df[genre_exploded_df['Genre'].str.len() > 0].drop(columns=['genre_list'])
    if genre_exploded_df.empty: logging.warning("No valid genres after explode/clean."); return None
    genre_exploded_df['artist_popularity'] = pd.to_numeric(genre_exploded_df['artist_popularity'], errors='coerce')
    genre_exploded_df['popularity'] = pd.to_numeric(genre_exploded_df['popularity'], errors='coerce')
    analysis_results = genre_exploded_df.groupby('Genre').agg(
        track_count=('track_id', 'size'),
        avg_artist_popularity=('artist_popularity', 'mean'),
        avg_track_popularity=('popularity', 'mean') ).reset_index()
    analysis_results.rename(columns={'track_count': 'Track Count', 'avg_artist_popularity': 'Average Artist Popularity', 'avg_track_popularity': 'Average Song Popularity'}, inplace=True)
    analysis_df = analysis_results.sort_values(by='Track Count', ascending=False).reset_index(drop=True)
    logging.info(f"Genre analysis complete. Found {len(analysis_df)} unique genres."); return analysis_df

# --- Data Export ---
# (Keep export_data function as it was)
def export_data(df: Optional[pd.DataFrame], file_path: str, export_format: str = "parquet") -> None:
    """Exports the DataFrame to a file (Parquet or CSV)."""
    # (Implementation remains the same)
    if df is None or df.empty: logging.warning(f"Empty DataFrame. Skipping export: {file_path}"); return
    export_dir = os.path.dirname(file_path)
    try:
        # Attempt to create directory(ies) if they don't exist
        os.makedirs(export_dir, exist_ok=True)
    except OSError as e:
        # Log error and exit the function if directory creation fails
        logging.error(f"Could not create directory {export_dir}: {e}")
        return # Stop if directory can't be created
    fmt = export_format.lower(); allowed = ["parquet", "csv"];
    if fmt not in allowed: fmt = "parquet"; logging.error(f"Invalid format: {export_format}. Defaulting to {fmt}.")
    base, _ = os.path.splitext(file_path); full_path = f"{base}.{fmt}"
    try:
        logging.info(f"Exporting {len(df)} rows to {full_path}...")
        if fmt == "csv": df.to_csv(full_path, index=False, encoding='utf-8-sig')
        elif fmt == "parquet": df.to_parquet(full_path, index=False, engine='pyarrow', compression='snappy')
        logging.info(f"Export successful: {full_path}")
    except ImportError as e:
         if 'pyarrow' in str(e) and fmt == "parquet": logging.error("PyArrow missing. `pip install pyarrow`")
         elif 'fastparquet' in str(e) and fmt == "parquet": logging.error("FastParquet missing. `pip install fastparquet`")
         else: logging.error(f"ImportError: {e}")
    except Exception as e: logging.error(f"Error exporting data: {e}", exc_info=True)

# --- Main Execution Pipeline (Refactored) ---
def main(
    spotify_client: Optional[spotipy.Spotify] = None,
    max_tracks: Optional[int] = None,
    skip_base_fetch: bool = False,
    fetch_artist_details: bool = True,
    force_artist_refresh: bool = False,
    fetch_albums: bool = True, # Keep album fetching enabled
    # Removed fetch_audio flag
    ) -> None:
    """
    Main function: data retrieval, enrichment (artist cache, album cache), analysis, export.
    """
    start_time = time.time()
    run_config = f"skip_base={skip_base_fetch}, fetch_artist={fetch_artist_details}, force_artist={force_artist_refresh}, fetch_album={fetch_albums}"
    logging.info(f"=== Starting Data Pipeline ({run_config}) ===")

    # Ensure Spotify Client
    if not spotify_client:
        logging.info("Spotify client not provided. Attempting creation...")
        spotify_client = create_spotify_client()
        if not spotify_client: logging.error("Failed to create Spotify client. Aborting."); return

    # === Step 1: Get Base Track Data ===
    initial_df = None
    if not skip_base_fetch:
        logging.info("Pipeline Step 1: Fetching base tracks from API...")
        initial_df = get_and_process_saved_tracks(spotify_client, max_tracks)
        if initial_df is None or initial_df.empty: logging.error("Failed to retrieve base tracks. Aborting."); return
        export_data(initial_df, PARQUET_BASE_TRACKS)
    else:
        logging.info(f"Pipeline Step 1: Loading base tracks from {PARQUET_BASE_TRACKS}...")
        try: initial_df = pd.read_parquet(PARQUET_BASE_TRACKS)
        except FileNotFoundError: logging.error(f"Base file not found: {PARQUET_BASE_TRACKS}. Run with skip_base_fetch=False."); return
        except Exception as e: logging.error(f"Error loading base data: {e}", exc_info=True); return
        logging.info(f"Loaded {len(initial_df)} tracks.")

    if initial_df is None or initial_df.empty: logging.error("Base DataFrame empty. Aborting."); return
    current_df = initial_df.copy() # Start enrichment pipeline with base data

    # === Step 2: Artist Enrichment (Using Cache) ===
    if fetch_artist_details:
        logging.info("Pipeline Step 2: Managing Artist Cache & Enrichment...")
        try:
             required_artist_ids = set()
             id_col = 'artist_ids' # Track artist IDs
             if id_col in current_df.columns:
                   # Basic validation: non-null, 22 chars, alphanumeric
                   id_pattern = re.compile(r'^[a-zA-Z0-9]{22}$')
                   for ids_str in current_df[id_col].dropna():
                       if isinstance(ids_str, str):
                           required_artist_ids.update(aid.strip() for aid in ids_str.split(',') if aid.strip() and id_pattern.match(aid.strip()))
             if not required_artist_ids: logging.warning("No valid required artist IDs found.")
             else:
                 artist_data_df = update_artist_cache(spotify_client, required_artist_ids, PARQUET_ARTIST_CACHE, force_artist_refresh)
                 if artist_data_df is not None and not artist_data_df.empty:
                     current_df = apply_artist_data_to_tracks(current_df, artist_data_df) # Applies fix for ValueError
                     export_data(current_df, PARQUET_ARTIST_ENRICHED) # Save intermediate
                 else: logging.warning("Artist data empty after cache update. Enrichment failed/skipped.")
        except Exception as e: logging.error(f"Error during artist enrichment: {e}", exc_info=True)
    else: logging.info("Pipeline Step 2: Skipping Artist Detail enrichment.")

    # === Step 3: Audio Feature Enrichment (Removed) ===
    logging.info("Pipeline Step 3: Audio Feature Enrichment - SKIPPED (API Restricted).")

    # === Step 4: Album Enrichment (Using Cache) ===
    if fetch_albums:
        logging.info("Pipeline Step 4: Managing Album Cache & Enrichment...")
        try:
            required_album_ids = set()
            id_col = 'album_id'
            if id_col in current_df.columns:
                # Basic validation for album IDs
                id_pattern = re.compile(r'^[a-zA-Z0-9]{22}$')
                required_album_ids.update(aid for aid in current_df[id_col].dropna().unique() if isinstance(aid, str) and id_pattern.match(aid))
            if not required_album_ids: logging.warning("No valid required album IDs found.")
            else:
                # Note: force_refresh for albums might be less necessary, defaulting to False
                album_data_df = update_album_cache(spotify_client, required_album_ids, PARQUET_ALBUM_CACHE, force_refresh=False)
                if album_data_df is not None and not album_data_df.empty:
                    current_df = apply_album_data_to_tracks(current_df, album_data_df)
                    export_data(current_df, PARQUET_ALBUM_ENRICHED) # Save intermediate
                else: logging.warning("Album data empty after cache update. Enrichment failed/skipped.")
        except Exception as e: logging.error(f"Error during album enrichment: {e}", exc_info=True)
    else: logging.info("Pipeline Step 4: Skipping Album Detail enrichment.")

    # === Step 5: Save Final Enriched Data ===
    logging.info("Pipeline Step 5: Saving final combined dataset...")
    export_data(current_df, FINAL_DATA_PATH)

    # === Step 6: Analyze Genres ===
    required_analysis_cols = ['artist_genres', 'artist_popularity', 'popularity']
    # Check if artist details were fetched *AND* if the columns actually exist in the final df
    run_analysis = fetch_artist_details and all(col in current_df.columns for col in required_analysis_cols)
    genre_analysis_df = None
    if run_analysis:
        logging.info("Pipeline Step 6: Performing Genre Analysis...")
        genre_analysis_df = analyze_genres(current_df)
        if genre_analysis_df is not None and not genre_analysis_df.empty:
            export_data(genre_analysis_df, PARQUET_GENRE_ANALYSIS)
        else: logging.warning("Genre analysis returned empty or None.")
    else:
        missing_cols_for_analysis = [col for col in required_analysis_cols if col not in current_df.columns]
        logging.info(f"Skipping Genre Analysis (fetch_artist_details={fetch_artist_details}, missing_cols={missing_cols_for_analysis}).")

    # === Processing Complete ===
    end_time = time.time()
    logging.info(f"--- Processing Complete (Total time: {end_time - start_time:.2f} seconds) ---")
    logging.info("File Generation Summary:")
    # Provide summary based on expected paths
    for name, path in [('Base Tracks', PARQUET_BASE_TRACKS), ('Artist Cache', PARQUET_ARTIST_CACHE),
                       ('Artist Enriched', PARQUET_ARTIST_ENRICHED), ('Album Cache', PARQUET_ALBUM_CACHE),
                       ('Album Enriched', PARQUET_ALBUM_ENRICHED), ('Final Data File', FINAL_DATA_PATH),
                       ('Genre Analysis', PARQUET_GENRE_ANALYSIS)]:
        if os.path.exists(path): logging.info(f"  - {name:<18}: {path} (Exists)")
        # Optional: Add more logic here to check flags vs existence for warnings
    logging.info("--- End of Summary ---")

# --- Main execution block ---
if __name__ == "__main__":
    # Example: Use existing base file, fetch artist/album details using cache
    print("\nRunning pipeline: Using existing Base + Efficient Artist & Album Enrichment...\n")
    client = create_spotify_client()
    if client:
        main(
            client,
            skip_base_fetch=True,         # Set to False for a full refresh of base tracks
            fetch_artist_details=True,   # Enable/disable artist enrichment
            force_artist_refresh=False,  # Set to True to rebuild artist cache
            fetch_albums=True            # Enable/disable album enrichment
        )
    else:
        print("\nCould not create Spotify client. Exiting.\n")