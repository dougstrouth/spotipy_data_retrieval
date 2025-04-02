import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import pandas as pd


# Load environment variables
load_dotenv("/Users/dougstrouth/Documents/.env")

# Spotify credentials from the .env file
SPOTIFY_CLIENT_ID = os.getenv("spotify_client_id")
SPOTIFY_CLIENT_SECRET = os.getenv("spotify_client_secret")
SPOTIFY_REDIRECT_URI = os.getenv("spotify_redirect_uri")


# Function to authenticate and create Spotify client
def create_spotify_client():
    try:
        sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope="playlist-read-private,user-library-read",
            )
        )
        return sp
    except Exception as e:
        print(f"Error creating Spotify client: {e}")
        return None


# Function to extract playlist details
def get_playlist_data(sp: spotipy.Spotify, playlist_url):
    # Extract playlist ID from the URL
    playlist_id = playlist_url.split("/")[-1].split("?")[0]

    # Fetch playlist information
    playlist_data = sp.playlist(playlist_id)
    playlist_name = playlist_data["name"]
    df = extract_playlist_data(playlist_data)
    return df


import pandas as pd


def extract_playlist_data(playlist_response):
    """
    Extracts playlist data from a Spotify API response and returns a DataFrame.

    Args:
      playlist_response: The JSON response from the Spotify API for a playlist.

    Returns:
      A pandas DataFrame containing the extracted data.
    """

    playlist_name = playlist_response["name"]
    tracks = playlist_response["tracks"]["items"]

    data = []
    for i, item in enumerate(tracks):
        track = item["track"]
        album = track["album"]
        data.append(
            {
                "playlist_name": playlist_name,
                "order_number": i + 1,
                "song_url": track["external_urls"]["spotify"],
                "title": track["name"],
                "artist": track["artists"][0]["name"],
                "playlist_url": playlist_response["external_urls"]["spotify"],
                "song_added_date": item["added_at"],
                "song_length_ms": track["duration_ms"],
                "album_type": album["album_type"],
                "album_total_tracks": album["total_tracks"],
                "explicit": track["explicit"],
                "popularity": track["popularity"],  # Added popularity
            }
        )

    df = pd.DataFrame(data)
    return df


def export_playlist_to_csv(df, data_loc):
    """
    Exports a playlist DataFrame to a CSV file with a filename based on the playlist title and creation date.

    Args:
      df: The DataFrame containing the playlist data.
      data_loc: The directory where the CSV file should be saved.
    """

    # Extract the playlist title and creation date
    title = df["title"].iloc[0]

    # Construct the filename
    filename = f"{data_loc}/{title}_playlist.csv"

    # Export the DataFrame to CSV
    df.to_csv(filename, index=False)


# Function to summarize the playlist data
def summarize_playlist(playlist_df: pd.DataFrame) -> str:
    # Calculate total number of songs in the playlist
    total_songs = len(playlist_df)
    output_str = f"Total Songs: {total_songs}\n"

    # Calculate average song added date (assuming it's a datetime column)
    if not playlist_df.empty:
        avg_added_date = (
            pd.to_datetime(playlist_df["song_added_date"]).mean().strftime("%Y-%m-%d")
        )
        output_str += f"Average Song Added Date: {avg_added_date}\n"
    else:
        output_str += "Average Song Added Date: Not available\n"

    # Get the first and last song in the playlist
    if not playlist_df.empty:
        first_song = playlist_df.iloc[0]
        last_song = playlist_df.iloc[-1]
        output_str += f"First Song: {first_song['title']} by {first_song['artist']}\n"
        output_str += f"Last Song: {last_song['title']} by {last_song['artist']}\n"

    # Calculate most popular artist (by counting occurrences)
    if not playlist_df.empty:
        most_popular_artist = playlist_df["artist"].mode()[0]
        output_str += f"Most Popular Artist: {most_popular_artist}\n"

    return output_str



def create_dataframe(savedTracks):
    """Creates a DataFrame from a list of saved tracks with additional fields."""
    columns = [
        'added_at', 'track_name', 'album_name', 'artist_names', 'album_release_date',
        'duration_ms', 'explicit', 'popularity', 'track_id', 'album_id', 'artist_ids', 'isrc'
    ]
    df = pd.DataFrame(columns=columns)

    for x in range(len(savedTracks['items'])):
        row = savedTracks['items'][x]
        track = row['track']
        album = track['album']
        #print("album",album)
        artists = track['artists']

        added_at = row['added_at']
        track_name = track['name']
        album_name = album['name']
        artist_names = ', '.join([artist['name'] for artist in artists]) # Join artist names
        album_release_date = album['release_date']
        duration_ms = track['duration_ms']
        explicit = track['explicit']
        popularity = track['popularity']
        track_id = track['id']
        album_id = album['id']
        artist_ids = ', '.join([artist['id'] for artist in artists])
        isrc = track['external_ids'].get('isrc') # Handle missing isrc

        df.loc[len(df)] = [
            added_at, track_name, album_name, artist_names, album_release_date,
            duration_ms, explicit, popularity, track_id, album_id, artist_ids, isrc
        ]

    return df


def get_and_process_saved_tracks(spotify_client, max_tracks=None):
    """Retrieves and processes saved tracks from the Spotify API using offset."""

    limit = 20  # Number of tracks to retrieve per request
    offset = 0  # Initial offset
    all_tracks_df = None  # Initialize DataFrame
    total_tracks_retrieved = 0 #track total tracks retrieved

    while True:
        results = spotify_client.current_user_saved_tracks(limit=limit, offset=offset)
        all_tracks_df = process_saved_tracks(results, all_tracks_df)  # Process and append

        total_tracks_retrieved += len(results['items'])

        if max_tracks and total_tracks_retrieved >= max_tracks:
            break

        if len(results['items']) < limit:  # Check if there are more tracks
            break  # No more tracks

        offset += limit  # Increment offset

    return all_tracks_df

import pandas as pd

def process_saved_tracks(saved_tracks_data, existing_data=None):
    """Processes saved track data from Spotify API calls and appends it to an existing DataFrame."""

    data = []
    for item in saved_tracks_data['items']:
        track_data = {}
        for key, value in item.items():
            if key == 'track':
                for track_key, track_value in value.items():
                    if track_key == 'album':
                        for album_key, album_value in track_value.items():
                            if album_key == 'artists':
                                track_data['album_artists'] = ', '.join([artist['name'] for artist in album_value])
                                track_data['album_artist_ids'] = ', '.join([artist['id'] for artist in album_value])
                                for artist_index, artist in enumerate(album_value):
                                    for artist_key, artist_value in artist.items():
                                        track_data[f"album_artist_{artist_index}_{artist_key}"] = artist_value
                            elif album_key == 'images':
                                for image_index, image in enumerate(album_value):
                                    for image_key, image_value in image.items():
                                        track_data[f"album_image_{image_index}_{image_key}"] = image_value
                            elif album_key == 'available_markets':
                                track_data['album_available_markets'] = ', '.join(album_value)
                            elif album_key == 'external_urls':
                                for url_key, url_value in album_value.items():
                                    track_data[f"album_external_urls_{url_key}"] = url_value
                            else:
                                track_data[f"album_{album_key}"] = album_value
                    elif track_key == 'artists':
                        track_data['track_artists'] = ', '.join([artist['name'] for artist in track_value])
                        track_data['track_artist_ids'] = ', '.join([artist['id'] for artist in track_value])
                        for artist_index, artist in enumerate(track_value):
                            for artist_key, artist_value in artist.items():
                                track_data[f"track_artist_{artist_index}_{artist_key}"] = artist_value
                    elif track_key == 'external_ids':
                        for external_id_key, external_id_value in track_value.items():
                            track_data[f"track_external_ids_{external_id_key}"] = external_id_value
                    elif track_key == 'external_urls':
                        for url_key, url_value in track_value.items():
                            track_data[f"track_external_urls_{url_key}"] = url_value
                    elif track_key == 'available_markets':
                        track_data['track_available_markets'] = ', '.join(track_value)
                    else:
                        track_data[f"track_{track_key}"] = track_value
            else:
                track_data[key] = value

        data.append(track_data)

    new_df = pd.DataFrame(data)

    if existing_data is None:
        return new_df
    else:
        return pd.concat([existing_data, new_df], ignore_index=True)