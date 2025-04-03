# %%capture
# # Use this in Jupyter to suppress pip install output if needed
# # %pip install spotipy pandas python-dotenv pyarrow fastparquet requests python-dateutil

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials # Added ClientCredentials for testing later if needed
from dotenv import load_dotenv
import pandas as pd
import logging
from typing import Optional, List, Dict, Any, Set
import time
import re # For ID validation
import requests # For ConnectionError handling
from dateutil.parser import isoparse # For robust date parsing if needed

# --- Configuration ---
# TODO: Update paths if needed
DEFAULT_EXPORT_PATH = '/Users/dougstrouth/Library/Mobile Documents/com~apple~CloudDocs/Documents/Code/datasets/spotify'
ENV_FILE_PATH = "/Users/dougstrouth/Documents/.env"

# --- Constants ---
ARTIST_BATCH_SIZE = 50
TRACK_FETCH_LIMIT = 50
ALBUM_BATCH_SIZE = 20 # Max for albums endpoint
# AUDIO_FEATURES_BATCH_SIZE = 100 # Not needed anymore

INITIAL_DELAY = 1
MAX_RETRIES = 5

# File Paths
PARQUET_BASE_TRACKS = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_base.parquet')
PARQUET_ARTIST_CACHE = os.path.join(DEFAULT_EXPORT_PATH, 'artist_details.parquet')
PARQUET_ALBUM_CACHE = os.path.join(DEFAULT_EXPORT_PATH, 'album_details.parquet') # New Cache File
PARQUET_ARTIST_ENRICHED = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_artist_enriched.parquet') # Intermediate
PARQUET_ALBUM_ENRICHED = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_album_enriched.parquet') # Intermediate (After Album)
FINAL_DATA_PATH = os.path.join(DEFAULT_EXPORT_PATH, 'saved_tracks_enriched_final.parquet') # Final usable data
PARQUET_GENRE_ANALYSIS = os.path.join(DEFAULT_EXPORT_PATH, 'genre_analysis.parquet')

# --- Central Schema Definitions ---
# Base track schema + added columns
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
    'release_datetime': 'datetime64[ns, UTC]', 'release_year': 'Int64'
}

# Schema for the artist cache file
ARTIST_CACHE_SCHEMA = {
    'artist_id': 'string',
    'artist_fetched_genres_list': 'object', # Store list; handle carefully on load
    'artist_fetched_popularity': 'Int32',
    'artist_fetched_name': 'string',
    'artist_last_fetched': 'datetime64[ns, UTC]'
}

# Schema for the new album cache file
ALBUM_CACHE_SCHEMA = {
    'album_id': 'string',
    'album_fetched_genres_list': 'object', # Store list
    'album_fetched_popularity': 'Int32',
    'album_fetched_label': 'string',
    'album_last_fetched': 'datetime64[ns, UTC]',
    # Add other fetched album fields like copyrights, external_ids if desired
}

# Schema for columns added during enrichment
ENRICHMENT_SCHEMA = {
    'artist_genres': 'string',      # From track artists
    'artist_popularity': 'Float64', # From track artists
    'album_genres': 'string',       # From album details
    'album_label': 'string',        # From album details
    'album_popularity': 'Int32'     # From album details
}

# Combine schemas for the final DataFrame structure check (optional)
# FINAL_SCHEMA = {**BASE_SCHEMA, **ENRICHMENT_SCHEMA}


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING) # Quieter logs from requests


# --- Spotify Client & API Wrapper ---
# (Keep create_spotify_client and call_spotify_api as in the previous full script)
def create_spotify_client() -> Optional[spotipy.Spotify]:
    try:
        if not os.path.exists(ENV_FILE_PATH):
            logging.warning(f".env file not found at {ENV_FILE_PATH}. Trying current dir.")
            env_path = ".env"
        else: env_path = ENV_FILE_PATH
        load_dotenv(dotenv_path=env_path)
        client_id = os.getenv("spotify_client_id")
        client_secret = os.getenv("spotify_client_secret")
        redirect_uri = os.getenv("spotify_redirect_uri")
        if not all([client_id, client_secret, redirect_uri]):
            logging.error("Missing Spotify credentials."); return None
        if 'localhost' in redirect_uri: logging.warning("Redirect URI uses 'localhost'. Use 'http://127.0.0.1:PORT/'.")
        scopes = "user-library-read"
        # Explicit cache path can help avoid issues
        cache_path = os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '.', ".spotify_cache")
        sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri,
                scope=scopes, open_browser=False, cache_handler=spotipy.CacheFileHandler(cache_path=cache_path)
            )
        )
        user_info = sp.current_user()
        logging.info(f"Spotify client created and authenticated for user: {user_info.get('display_name','N/A')}")
        return sp
    except Exception as e: logging.error(f"Error creating Spotify client: {e}", exc_info=True); return None

def call_spotify_api(api_function: callable, *args, **kwargs) -> Any:
    delay = INITIAL_DELAY
    func_name = getattr(api_function, '__name__', 'unknown_api_function')
    logging.debug(f"Calling API: {func_name} Args: {args} Kwargs: {kwargs}")
    for attempt in range(MAX_RETRIES):
        try:
            results = api_function(*args, **kwargs); return results
        except spotipy.exceptions.SpotifyException as e:
            log_prefix = f"Spotify API error calling {func_name} (Attempt {attempt + 1}/{MAX_RETRIES})"
            log_details = f"Status={e.http_status}, Code={e.code}, URL={e.url}, Msg='{e.msg}' | Kwargs={kwargs}" # Avoid logging args potentially
            if e.http_status == 429:
                retry_after = int(e.headers.get('Retry-After', delay))
                logging.warning(f"{log_prefix}. Rate limit exceeded. Retrying in {retry_after}s...")
                time.sleep(retry_after); delay = min(delay * 2, 60)
            elif e.http_status in [401, 403]:
                 logging.error(f"{log_prefix}. Auth/Permission Error: {log_details}")
                 if attempt >= 1 or e.http_status == 403: raise # Don't retry 403 usually
                 time.sleep(delay*2); delay = min(delay * 2, 60)
            else:
                logging.error(f"{log_prefix}: {log_details}", exc_info=True)
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
# (Keep _extract helpers and process_track_item from previous full script)
def _extract_track_level_data(track: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'track_id': track.get('id'), 'track_name': track.get('name'), 'popularity': track.get('popularity'),
        'duration_ms': track.get('duration_ms'), 'explicit': track.get('explicit'), 'isrc': track.get('external_ids', {}).get('isrc'),
        'track_uri': track.get('uri'), 'track_href': track.get('href'), 'track_external_url_spotify': track.get('external_urls', {}).get('spotify'),
        'track_preview_url': track.get('preview_url'), 'track_is_local': track.get('is_local', False), 'track_is_playable': track.get('is_playable'),
        'track_disc_number': track.get('disc_number'), 'track_number': track.get('track_number'), 'track_type': track.get('type'),
        'track_available_markets': ','.join(track.get('available_markets', [])) or None,
    }
def _extract_track_artist_data(track_artists: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_ids, valid_names, valid_uris = [], [], []
    for artist in track_artists:
        if artist and artist.get('id'):
            valid_ids.append(artist['id']); valid_names.append(artist.get('name')); valid_uris.append(artist.get('uri'))
    return { 'artist_ids': ', '.join(filter(None, valid_ids)) or None, 'artist_names': ', '.join(filter(None, valid_names)) or None, 'track_artist_uris': ', '.join(filter(None, valid_uris)) or None }
def _extract_album_level_data(album: Dict[str, Any]) -> Dict[str, Any]:
    images = album.get('images', [])
    return {
        'album_id': album.get('id'), 'album_name': album.get('name'), 'album_release_date': album.get('release_date'),
        'album_uri': album.get('uri'), 'album_href': album.get('href'), 'album_external_url_spotify': album.get('external_urls', {}).get('spotify'),
        'album_type': album.get('album_type'), 'album_release_date_precision': album.get('release_date_precision'),
        'album_total_tracks': album.get('total_tracks'), 'album_available_markets': ','.join(album.get('available_markets', [])) or None,
        'album_image_url_64': next((img['url'] for img in images if img and img.get('height') == 64), None),
        'album_image_url_300': next((img['url'] for img in images if img and img.get('height') == 300), None),
        'album_image_url_640': next((img['url'] for img in images if img and img.get('height') == 640), None),
    }
def _extract_album_artist_data(album_artists: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_ids, valid_names = [], []
    for artist in album_artists:
        if artist and artist.get('id'): valid_ids.append(artist['id']); valid_names.append(artist.get('name'))
    return { 'album_artist_ids': ', '.join(filter(None, valid_ids)) or None, 'album_artist_names': ', '.join(filter(None, valid_names)) or None }

def process_track_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        added_at = item.get('added_at'); track = item.get('track')
        if not track or not track.get('id'): return None
        album = track.get('album', {}); track_artists = track.get('artists', []); album_artists = album.get('artists', [])
        processed_data = {'added_at': added_at}
        processed_data.update(_extract_track_level_data(track)); processed_data.update(_extract_track_artist_data(track_artists))
        processed_data.update(_extract_album_level_data(album)); processed_data.update(_extract_album_artist_data(album_artists))
        if not processed_data.get('track_id'): return None
        # Optional: Strict schema filtering
        # return {k: v for k, v in processed_data.items() if k in BASE_SCHEMA}
        return processed_data
    except Exception as e: logging.error(f"Error processing track item: {e}", exc_info=True); return None

def apply_data_types(df: pd.DataFrame, schema_map: Dict[str, str]) -> pd.DataFrame:
    # (Keep implementation from previous full script - it was already refactored)
    if df is None or df.empty: return df
    logging.info(f"Applying data types based on schema ({len(schema_map)} columns)...")
    df_out = df.copy(); converted_cols, skipped_cols, error_cols = 0, 0, 0
    for col, dtype_str in schema_map.items():
        if col not in df_out.columns: skipped_cols += 1; continue
        current_dtype = df_out[col].dtype
        try:
            target_dtype = pd.core.dtypes.common.pandas_dtype(dtype_str)
            # Basic check first - more complex below
            if current_dtype == target_dtype and not pd.api.types.is_object_dtype(current_dtype): # Re-apply for object sometimes needed
                 converted_cols +=1; continue
            # Handle conversions
            if dtype_str == 'datetime64[ns, UTC]': df_out[col] = pd.to_datetime(df_out[col], errors='coerce', utc=True)
            elif dtype_str.startswith('Int'): df_out[col] = pd.to_numeric(df_out[col], errors='coerce').astype(target_dtype)
            elif dtype_str.startswith('float'): df_out[col] = pd.to_numeric(df_out[col], errors='coerce').astype(target_dtype)
            elif dtype_str == 'boolean':
                map_dict = {'true': True, 'false': False, '1': True, '0': False, 1: True, 0: False, 1.0: True, 0.0: False}
                if pd.api.types.is_string_dtype(df_out[col]) or pd.api.types.is_object_dtype(df_out[col]):
                    df_out[col] = df_out[col].astype(str).str.lower().map(map_dict) # Map first
                df_out[col] = df_out[col].astype(target_dtype) # Then cast to nullable boolean
            else: df_out[col] = df_out[col].astype(target_dtype) # Includes 'string' type
            converted_cols += 1
        except Exception as e:
            logging.warning(f"Could not convert '{col}' to {dtype_str}. Current: {current_dtype}. Error: {e}")
            error_cols += 1
    logging.info(f"Type conversion: Applied={converted_cols}, Skipped={skipped_cols}, Errors={error_cols}")
    return df_out

def _parse_release_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parses album_release_date into datetime and year columns."""
    if 'album_release_date' in df.columns:
         logging.info("Parsing 'album_release_date' -> 'release_datetime', 'release_year'.")
         df_out = df.copy()
         df_out['release_datetime'] = pd.to_datetime(df_out['album_release_date'], errors='coerce')
         df_out['release_year'] = df_out['release_datetime'].dt.year.astype('Int64') # Use nullable Int64
         return df_out
    return df

# --- Data Retrieval ---
def get_and_process_saved_tracks(spotify_client: spotipy.Spotify, max_tracks: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Retrieves, processes, and type-converts saved tracks."""
    # (Keep pagination loop from previous script, calling the new process_track_item)
    # ... (Loop logic as before) ...
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
    logging.info(f"Created DataFrame shape: {df.shape}. Initial columns: {df.columns.tolist()}")
    # Apply types using central schema
    df = apply_data_types(df, BASE_SCHEMA)
    # Parse dates
    df = _parse_release_date_columns(df)
    logging.info("Base track DataFrame prepared.")
    return df


# --- Artist Data Fetching & Caching (Refactored) ---
# (Keep load_artist_cache, fetch_artist_data, update_artist_cache, apply_artist_data_to_tracks)
# Make sure fetch_artist_data uses ARTIST_CACHE_SCHEMA when applying types
def load_artist_cache(cache_path: str) -> pd.DataFrame:
    # (Implementation from previous response)
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logging.info(f"Loaded {len(df)} artists from cache: {cache_path}")
            if 'artist_id' not in df.columns:
                 logging.warning("Artist cache missing 'artist_id'. Ignoring."); return pd.DataFrame()
            # Ensure list type for genres after loading from parquet if needed - depends on save format
            # Let's trust pyarrow handles object type containing lists correctly for now.
            if 'artist_last_fetched' in df.columns:
                 df['artist_last_fetched'] = pd.to_datetime(df['artist_last_fetched'], errors='coerce', utc=True)
            # Apply known schema types
            df = apply_data_types(df, ARTIST_CACHE_SCHEMA)
            return df
        except Exception as e: logging.error(f"Error loading artist cache: {e}. Ignoring.", exc_info=True)
    else: logging.info(f"Artist cache not found: {cache_path}.")
    return pd.DataFrame()

def fetch_artist_data(spotify_client: spotipy.Spotify, artist_ids: List[str]) -> Optional[pd.DataFrame]:
    # (Implementation from previous response, returns DataFrame)
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
                        artist_data_list.append({
                            'artist_id': artist['id'],
                            'artist_fetched_genres_list': artist.get('genres', []), # Keep as list
                            'artist_fetched_popularity': artist.get('popularity'),
                            'artist_fetched_name': artist.get('name'),
                            'artist_last_fetched': fetch_timestamp
                        }); fetched_count += 1
            # logging.info(f"Fetched artist batch {i//ARTIST_BATCH_SIZE + 1}") # Less verbose
        except Exception as e: logging.error(f"Failed artist batch fetch index {i}: {e}", exc_info=True)
    if not artist_data_list: logging.warning("No artist data fetched."); return None
    logging.info(f"Fetched data for {fetched_count}/{total_ids} artists.")
    artist_df = pd.DataFrame(artist_data_list)
    # Apply types based on cache schema
    artist_df = apply_data_types(artist_df, ARTIST_CACHE_SCHEMA)
    return artist_df


def update_artist_cache(spotify_client: Optional[spotipy.Spotify], required_artist_ids: Set[str],
                       cache_path: str, force_refresh: bool = False) -> pd.DataFrame:
    # (Implementation from previous response)
    existing_artist_df = load_artist_cache(cache_path)
    cached_ids = set(existing_artist_df['artist_id']) if 'artist_id' in existing_artist_df else set()
    ids_to_fetch = list(required_artist_ids - cached_ids) if not force_refresh else list(required_artist_ids)
    if force_refresh and not existing_artist_df.empty:
        logging.info("Force refresh enabled, will overwrite existing cache entries.")
        existing_artist_df = pd.DataFrame() # Clear cache if forcing full refresh
        cached_ids = set()

    logging.info(f"Artist Cache: Required={len(required_artist_ids)}, Cached={len(cached_ids)}, To Fetch={len(ids_to_fetch)}")
    newly_fetched_df = None
    if ids_to_fetch and spotify_client:
        newly_fetched_df = fetch_artist_data(spotify_client, ids_to_fetch)
    elif not spotify_client and ids_to_fetch: logging.warning("Need new artists but no Spotify client.")

    final_df = existing_artist_df
    if newly_fetched_df is not None and not newly_fetched_df.empty:
        final_df = pd.concat([existing_artist_df, newly_fetched_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['artist_id'], keep='last').reset_index(drop=True)
        logging.info(f"Saving updated artist cache ({len(final_df)} total) to {cache_path}")
        export_data(final_df, cache_path)
    elif not existing_artist_df.empty : logging.info("Using only cached artist data.")
    else: logging.warning("No artist data available (cached or fetched).")

    return final_df

def apply_artist_data_to_tracks(track_df: pd.DataFrame, artist_data_df: pd.DataFrame) -> pd.DataFrame:
    """Applies artist cache data to track DataFrame. FIX for genre list handling."""
    if artist_data_df is None or artist_data_df.empty:
        logging.warning("No artist data to apply.")
        # Ensure columns exist
        track_df_out = track_df.copy()
        if 'artist_genres' not in track_df_out.columns: track_df_out['artist_genres'] = pd.Series(dtype='string')
        if 'artist_popularity' not in track_df_out.columns: track_df_out['artist_popularity'] = pd.Series(dtype='Float64')
        return track_df_out

    logging.info(f"Applying artist data ({len(artist_data_df)} artists) to {len(track_df)} tracks...")
    lookup_cols = ['artist_id', 'artist_fetched_genres_list', 'artist_fetched_popularity']
    if not all(col in artist_data_df.columns for col in lookup_cols):
        missing = [c for c in lookup_cols if c not in artist_data_df.columns]
        logging.error(f"Artist data missing required columns: {missing}. Cannot apply."); return track_df

    # Create map - ensure index is string, data types are correct
    artist_map = artist_data_df.set_index('artist_id')[lookup_cols[1:]].to_dict('index')

    def _get_genres(ids_str):
        if pd.isna(ids_str) or not isinstance(ids_str, str): return None
        ids = [aid.strip() for aid in ids_str.split(',') if aid.strip()]
        genres = set()
        for aid in ids:
            data = artist_map.get(aid)
            # *** FIX: Check if data exists AND genre list is valid ***
            if data and isinstance(data.get('artist_fetched_genres_list'), list):
                 genres.update(g for g in data['artist_fetched_genres_list'] if isinstance(g, str)) # Ensure genres are strings
        return ', '.join(sorted(list(genres))) if genres else None

    def _get_popularity(ids_str):
        if pd.isna(ids_str) or not isinstance(ids_str, str): return None
        ids = [aid.strip() for aid in ids_str.split(',') if aid.strip()]
        pops = []
        for aid in ids:
            data = artist_map.get(aid)
            # *** FIX: Check if data exists AND popularity is valid number ***
            pop_val = data.get('artist_fetched_popularity') if data else None
            if pd.notna(pop_val): # Check for NaN/None robustly
                 try: pops.append(float(pop_val)) # Convert to float JIC
                 except (ValueError, TypeError): pass # Ignore if not convertible
        return sum(pops) / len(pops) if pops else None

    track_df_out = track_df.copy()
    track_df_out['artist_genres'] = track_df_out['artist_ids'].apply(_get_genres)
    track_df_out['artist_popularity'] = track_df_out['artist_ids'].apply(_get_popularity)

    # Ensure final types match ENRICHMENT_SCHEMA
    track_df_out['artist_genres'] = track_df_out['artist_genres'].astype(EXPECTED_SCHEMA.get('artist_genres', 'string'))
    track_df_out['artist_popularity'] = track_df_out['artist_popularity'].astype(EXPECTED_SCHEMA.get('artist_popularity', 'Float64'))

    null_genre_count = track_df_out['artist_genres'].isnull().sum()
    if null_genre_count == len(track_df_out):
         logging.warning("Applied artist data, but 'artist_genres' column is STILL all null. Check cache content and lookup logic.")
    else:
         logging.info(f"Finished applying artist data. Null genres: {null_genre_count}/{len(track_df_out)}")

    return track_df_out

# --- Album Data Fetching & Caching (New) ---

def fetch_album_details(spotify_client: spotipy.Spotify, album_ids: List[str]) -> Optional[pd.DataFrame]:
    """Fetches album details (genres, label, popularity) from Spotify API."""
    unique_album_ids = list(set(aid for aid in album_ids if aid))
    album_data_list = []
    fetched_count = 0
    total_ids = len(unique_album_ids)
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
                        album_data_list.append({
                            'album_id': album['id'],
                            'album_fetched_genres_list': album.get('genres', []), # Album specific genres
                            'album_fetched_popularity': album.get('popularity'),
                            'album_fetched_label': album.get('label'),
                            'album_last_fetched': fetch_timestamp
                        })
                        fetched_count += 1
            logging.info(f"Fetched album batch {i//ALBUM_BATCH_SIZE + 1}/{ (total_ids + ALBUM_BATCH_SIZE - 1)//ALBUM_BATCH_SIZE }")
        except Exception as e:
            logging.error(f"Failed album batch fetch index {i}: {e}", exc_info=True)

    if not album_data_list: logging.warning("No album data fetched."); return None
    logging.info(f"Fetched data for {fetched_count}/{total_ids} albums.")
    album_df = pd.DataFrame(album_data_list)
    # Apply types based on cache schema
    album_df = apply_data_types(album_df, ALBUM_CACHE_SCHEMA)
    return album_df

def load_album_cache(cache_path: str) -> pd.DataFrame:
    """Loads the album cache DataFrame."""
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            logging.info(f"Loaded {len(df)} albums from cache: {cache_path}")
            if 'album_id' not in df.columns:
                 logging.warning("Album cache missing 'album_id'. Ignoring."); return pd.DataFrame()
            if 'album_last_fetched' in df.columns:
                 df['album_last_fetched'] = pd.to_datetime(df['album_last_fetched'], errors='coerce', utc=True)
            # Apply known schema types
            df = apply_data_types(df, ALBUM_CACHE_SCHEMA)
            return df
        except Exception as e: logging.error(f"Error loading album cache: {e}. Ignoring.", exc_info=True)
    else: logging.info(f"Album cache not found: {cache_path}.")
    return pd.DataFrame()

def update_album_cache(
    spotify_client: Optional[spotipy.Spotify],
    required_album_ids: Set[str],
    cache_path: str,
    force_refresh: bool = False
    ) -> pd.DataFrame:
    """Manages the album cache: loads, fetches missing, combines, saves."""
    existing_album_df = load_album_cache(cache_path)
    cached_ids = set(existing_album_df['album_id']) if 'album_id' in existing_album_df else set()
    ids_to_fetch = list(required_album_ids - cached_ids) if not force_refresh else list(required_album_ids)
    if force_refresh and not existing_album_df.empty:
        logging.info("Forcing refresh of all required albums."); existing_album_df = pd.DataFrame(); cached_ids = set()

    logging.info(f"Album Cache: Required={len(required_album_ids)}, Cached={len(cached_ids)}, To Fetch={len(ids_to_fetch)}")
    newly_fetched_df = None
    if ids_to_fetch and spotify_client:
        newly_fetched_df = fetch_album_details(spotify_client, ids_to_fetch)
    elif not spotify_client and ids_to_fetch: logging.warning("Need new albums but no Spotify client.")

    final_df = existing_album_df
    if newly_fetched_df is not None and not newly_fetched_df.empty:
        final_df = pd.concat([existing_album_df, newly_fetched_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['album_id'], keep='last').reset_index(drop=True)
        logging.info(f"Saving updated album cache ({len(final_df)} total) to {cache_path}")
        export_data(final_df, cache_path)
    elif not existing_album_df.empty: logging.info("Using only cached album data.")
    else: logging.warning("No album data available (cached or fetched).")

    return final_df

def apply_album_data_to_tracks(track_df: pd.DataFrame, album_data_df: pd.DataFrame) -> pd.DataFrame:
    """Applies cached/fetched album details (genres, label, pop) to the track DataFrame."""
    if album_data_df is None or album_data_df.empty:
        logging.warning("No album data provided to apply.")
        # Ensure columns exist
        track_df_out = track_df.copy()
        if 'album_genres' not in track_df_out.columns: track_df_out['album_genres'] = pd.Series(dtype='string')
        if 'album_label' not in track_df_out.columns: track_df_out['album_label'] = pd.Series(dtype='string')
        if 'album_popularity' not in track_df_out.columns: track_df_out['album_popularity'] = pd.Series(dtype='Int32')
        return track_df_out

    logging.info(f"Applying album data ({len(album_data_df)} albums) to {len(track_df)} tracks...")
    lookup_cols = ['album_id', 'album_fetched_genres_list', 'album_fetched_popularity', 'album_fetched_label']
    if not all(col in album_data_df.columns for col in lookup_cols):
        missing = [c for c in lookup_cols if c not in album_data_df.columns]
        logging.error(f"Album data missing required columns: {missing}. Cannot apply."); return track_df

    # Prepare columns to merge (select and rename)
    album_data_to_merge = album_data_df[lookup_cols].copy()
    # Convert genre list to string
    album_data_to_merge['album_genres'] = album_data_to_merge['album_fetched_genres_list'].apply(
        lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else None
    )
    album_data_to_merge.rename(columns={
        'album_fetched_popularity': 'album_popularity',
        'album_fetched_label': 'album_label'
    }, inplace=True)

    # Select final columns for merge, ensuring album_id is present
    merge_cols = ['album_id', 'album_genres', 'album_popularity', 'album_label']
    album_data_to_merge = album_data_to_merge[[col for col in merge_cols if col in album_data_to_merge.columns]]

    if 'album_id' not in album_data_to_merge.columns:
         logging.error("Prepared album data unexpectedly missing 'album_id'. Cannot merge.")
         return track_df

    # Merge into track DataFrame
    track_df_out = pd.merge(track_df, album_data_to_merge, on='album_id', how='left', suffixes=('', '_album_detail'))

    # Optional: Handle potential column name conflicts if base data had similar names (e.g., popularity)
    # Example: if 'popularity_album_detail' in track_df_out.columns: ...

    # Apply final types based on ENRICHMENT_SCHEMA
    track_df_out = apply_data_types(track_df_out, ENRICHMENT_SCHEMA)

    logging.info("Finished applying album data.")
    return track_df_out


# --- Data Analysis ---
# (Keep analyze_genres function as it was)
def analyze_genres(track_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Analyzes genres based on track popularity and artist data."""
    logging.info("Starting genre analysis...")
    required_cols = ['artist_genres', 'artist_popularity', 'popularity'] # Using track artist genres/pop
    # ... (rest of implementation is the same) ...
    missing_cols = [col for col in required_cols if col not in track_df.columns]
    if missing_cols: logging.warning(f"Missing cols for genre analysis: {missing_cols}. Skipping."); return None
    if track_df['artist_genres'].isnull().all(): logging.warning("Column 'artist_genres' is all null. Skipping."); return None
    temp_df = track_df[required_cols + ['track_id']].dropna(subset=['artist_genres']).copy()
    if temp_df.empty: return None
    temp_df['genre_list'] = temp_df['artist_genres'].str.split(',')
    genre_exploded_df = temp_df.explode('genre_list')
    genre_exploded_df['Genre'] = genre_exploded_df['genre_list'].str.strip()
    genre_exploded_df = genre_exploded_df[genre_exploded_df['Genre'].str.len() > 0].drop(columns=['genre_list'])
    if genre_exploded_df.empty: return None
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
    # ... (Implementation remains the same) ...
    if df is None or df.empty: logging.warning(f"Empty DataFrame. Skipping export: {file_path}"); return
    export_dir = os.path.dirname(file_path)
    try: os.makedirs(export_dir, exist_ok=True)
    except OSError as e: logging.error(f"Could not create dir {export_dir}: {e}"); return
    allowed_formats = ["parquet", "csv"]; fmt = export_format.lower()
    if fmt not in allowed_formats: fmt = "parquet"; logging.error(f"Invalid format: {export_format}. Defaulting to {fmt}.")
    base, _ = os.path.splitext(file_path); full_path = f"{base}.{fmt}"
    try:
        logging.info(f"Exporting {len(df)} rows to {full_path}...")
        if fmt == "csv": df.to_csv(full_path, index=False, encoding='utf-8-sig')
        elif fmt == "parquet": df.to_parquet(full_path, index=False, engine='pyarrow', compression='snappy')
        logging.info(f"Export successful: {full_path}")
    except ImportError as e:
         if 'pyarrow' in str(e) and fmt == "parquet": logging.error("PyArrow not found. `pip install pyarrow`")
         elif 'fastparquet' in str(e) and fmt == "parquet": logging.error("FastParquet not found. `pip install fastparquet`")
         else: logging.error(f"ImportError during export: {e}")
    except Exception as e: logging.error(f"Error exporting data: {e}", exc_info=True)


# --- Main Execution Pipeline (Refactored) ---
def main(
    spotify_client: Optional[spotipy.Spotify] = None,
    max_tracks: Optional[int] = None,
    skip_base_fetch: bool = False,
    fetch_artist_details: bool = True,
    force_artist_refresh: bool = False,
    # fetch_audio: bool = False, # Removed
    fetch_albums: bool = True # <<< Enable album fetching by default now
    ) -> None:
    """
    Main function: data retrieval, enrichment (artist cache, album cache), analysis, export.
    """
    start_time = time.time()
    logging.info("=== Starting Data Pipeline ===")

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
        except FileNotFoundError: logging.error(f"Base file not found: {PARQUET_BASE_TRACKS}."); return
        except Exception as e: logging.error(f"Error loading base data: {e}", exc_info=True); return
        logging.info(f"Loaded {len(initial_df)} tracks.")

    if initial_df is None or initial_df.empty: logging.error("Base DataFrame empty. Aborting."); return
    current_df = initial_df.copy()

    # === Step 2: Artist Enrichment (Using Cache) ===
    artist_data_df = None # Define scope outside if
    if fetch_artist_details:
        logging.info("Pipeline Step 2: Managing Artist Cache & Enrichment...")
        try:
             required_artist_ids = set()
             id_col = 'artist_ids'
             if id_col in current_df.columns:
                   for ids_str in current_df[id_col].dropna():
                       if isinstance(ids_str, str):
                           # Basic validation included here
                           required_artist_ids.update(aid.strip() for aid in ids_str.split(',') if aid.strip() and len(aid.strip()) == 22 and aid.strip().isalnum())
             if not required_artist_ids: logging.warning("No valid required artist IDs found.")
             else:
                 artist_data_df = update_artist_cache(spotify_client, required_artist_ids, PARQUET_ARTIST_CACHE, force_artist_refresh)
                 if artist_data_df is not None and not artist_data_df.empty:
                     current_df = apply_artist_data_to_tracks(current_df, artist_data_df)
                     export_data(current_df, PARQUET_ARTIST_ENRICHED) # Save intermediate
                 else: logging.warning("Artist data empty after cache update. Enrichment skipped.")
        except Exception as e: logging.error(f"Error during artist enrichment: {e}", exc_info=True)
    else: logging.info("Pipeline Step 2: Skipping Artist Detail enrichment.")


    # === Step 3: Audio Feature Enrichment (Removed) ===
    logging.info("Pipeline Step 3: Audio Feature Enrichment - SKIPPED (API Restricted).")


    # === Step 4: Album Enrichment (Using Cache) ===
    album_data_df = None # Define scope
    if fetch_albums:
        logging.info("Pipeline Step 4: Managing Album Cache & Enrichment...")
        try:
            required_album_ids = set()
            id_col = 'album_id'
            if id_col in current_df.columns:
                # Basic validation for album IDs might differ - adapt if needed
                required_album_ids.update(aid for aid in current_df[id_col].dropna().unique() if isinstance(aid, str) and len(aid) == 22 and aid.isalnum())
            if not required_album_ids: logging.warning("No valid required album IDs found.")
            else:
                album_data_df = update_album_cache(spotify_client, required_album_ids, PARQUET_ALBUM_CACHE, force_refresh=False) # Typically don't force album refresh unless needed
                if album_data_df is not None and not album_data_df.empty:
                    current_df = apply_album_data_to_tracks(current_df, album_data_df)
                    export_data(current_df, PARQUET_ALBUM_ENRICHED) # Save intermediate
                else: logging.warning("Album data empty after cache update. Enrichment skipped.")
        except Exception as e: logging.error(f"Error during album enrichment: {e}", exc_info=True)
    else:
        logging.info("Pipeline Step 4: Skipping Album Detail enrichment.")


    # === Step 5: Save Final Enriched Data ===
    logging.info("Pipeline Step 5: Saving final combined dataset...")
    export_data(current_df, FINAL_DATA_PATH)


    # === Step 6: Analyze Genres ===
    required_analysis_cols = ['artist_genres', 'artist_popularity', 'popularity']
    run_analysis = fetch_artist_details and all(col in current_df.columns for col in required_analysis_cols)
    genre_analysis_df = None
    if run_analysis:
        logging.info("Pipeline Step 6: Performing Genre Analysis...")
        genre_analysis_df = analyze_genres(current_df)
        if genre_analysis_df is not None and not genre_analysis_df.empty:
            export_data(genre_analysis_df, PARQUET_GENRE_ANALYSIS)
        else: logging.warning("Genre analysis returned empty or None.")
    else: logging.info("Skipping Genre Analysis (Artist details not fetched or cols missing).")


    # === Processing Complete ===
    end_time = time.time()
    logging.info(f"--- Processing Complete (Total time: {end_time - start_time:.2f} seconds) ---")
    logging.info("File Generation Summary:")
    # Simplified checks focusing on final outputs and caches
    if os.path.exists(PARQUET_BASE_TRACKS): logging.info(f"  - Base Tracks:       {PARQUET_BASE_TRACKS} (Exists)")
    if os.path.exists(PARQUET_ARTIST_CACHE): logging.info(f"  - Artist Cache:      {PARQUET_ARTIST_CACHE} (Exists)")
    if fetch_artist_details and os.path.exists(PARQUET_ARTIST_ENRICHED): logging.info(f"  - Artist Enriched:   {PARQUET_ARTIST_ENRICHED} (Exists - Intermediate)")
    if fetch_albums:
         if os.path.exists(PARQUET_ALBUM_CACHE): logging.info(f"  - Album Cache:       {PARQUET_ALBUM_CACHE} (Exists)")
         if os.path.exists(PARQUET_ALBUM_ENRICHED): logging.info(f"  - Album Enriched:    {PARQUET_ALBUM_ENRICHED} (Exists - Intermediate)")
    else: logging.info("  - Album Enriched:    Step skipped.")
    if os.path.exists(FINAL_DATA_PATH): logging.info(f"  - Final Data File:   {FINAL_DATA_PATH} (Exists)")
    if run_analysis and genre_analysis_df is not None and not genre_analysis_df.empty and os.path.exists(PARQUET_GENRE_ANALYSIS):
         logging.info(f"  - Genre Analysis:    {PARQUET_GENRE_ANALYSIS} (Exists)")
    else: logging.info("  - Genre Analysis:    Not generated or skipped.")


if __name__ == "__main__":
    print("Running pipeline: Using existing Base + Efficient Artist & Album Enrichment...")
    client = create_spotify_client()
    if client:
        main(
            client,
            skip_base_fetch=False,
            fetch_artist_details=True,
            force_artist_refresh=True,
            fetch_albums=True # Enable album fetching
        )
    else:
        print("Could not create Spotify client. Exiting.")