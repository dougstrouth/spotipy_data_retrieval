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
                scope="playlist-read-private",
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

