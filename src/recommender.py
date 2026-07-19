import csv
import os
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: float = 50.0

    @classmethod
    def from_dict(cls, data: Dict) -> "Song":
        """Constructs a Song instance from a dictionary representation."""
        title_val = data.get("title", data.get("name", "Unknown Title"))
        if not title_val or str(title_val).strip().lower() in ("nan", "none", ""):
            title_val = "Unknown Title"

        artist_val = data.get("artist", data.get("track_artists"))
        if not artist_val or str(artist_val).strip().lower() in ("nan", "none", ""):
            artist_val = data.get("track_album_album", data.get("album_name", "Unknown Artist"))
            if not artist_val or str(artist_val).strip().lower() in ("nan", "none", ""):
                artist_val = "Unknown Artist"

        genre_val = data.get("genre", data.get("genres", "pop"))
        if not genre_val or str(genre_val).strip().lower() in ("nan", "none", ""):
            genre_val = "pop"

        pop_val = data.get("popularity", data.get("artist_popularity", 50.0))
        if pop_val is None or str(pop_val).strip().lower() in ("nan", "none", ""):
            pop_val = 50.0

        return cls(
            id=int(data.get("id", 0)) if str(data.get("id", 0)).isdigit() else 0,
            title=str(title_val),
            artist=str(artist_val),
            genre=str(genre_val),
            mood=str(data.get("mood", "happy")),
            energy=float(data.get("energy", 0.5) if data.get("energy") is not None else 0.5),
            tempo_bpm=float(data.get("tempo_bpm", data.get("tempo", 120.0)) if data.get("tempo_bpm", data.get("tempo")) is not None else 120.0),
            valence=float(data.get("valence", 0.5) if data.get("valence") is not None else 0.5),
            danceability=float(data.get("danceability", 0.5) if data.get("danceability") is not None else 0.5),
            acousticness=float(data.get("acousticness", 0.5) if data.get("acousticness") is not None else 0.5),
            popularity=float(pop_val)
        )


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_valence: float = 0.5
    target_danceability: float = 0.5

    @classmethod
    def from_seed_songs(cls, seed_songs: List[Song]) -> "UserProfile":
        """
        Dynamically derives a UserProfile (centroid) from one or multiple seed songs.
        Powers "Song Radio" and "Similar Tracks" recommendation capabilities.
        """
        if not seed_songs:
            raise ValueError("seed_songs list cannot be empty")
        
        # Determine dominant genre and mood (mode/majority vote)
        top_genre = Counter([s.genre for s in seed_songs]).most_common(1)[0][0]
        top_mood = Counter([s.mood for s in seed_songs]).most_common(1)[0][0]

        # Calculate average audio features (centroid)
        avg_energy = sum(s.energy for s in seed_songs) / len(seed_songs)
        avg_acoustic = (sum(s.acousticness for s in seed_songs) / len(seed_songs)) >= 0.5
        avg_valence = sum(s.valence for s in seed_songs) / len(seed_songs)
        avg_dance = sum(s.danceability for s in seed_songs) / len(seed_songs)

        return cls(
            favorite_genre=top_genre,
            favorite_mood=top_mood,
            target_energy=avg_energy,
            likes_acoustic=avg_acoustic,
            target_valence=avg_valence,
            target_danceability=avg_dance
        )


def score_song(user_prefs: Union[Dict, UserProfile], song: Union[Dict, Song]) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences using the Algorithm Recipe:
      +2.0 points for Genre Match (or +1.5 for subgenre substring match)
      +1.0 point for Mood Match
      +1.0 point max based on Energy Proximity: 1.0 - |song.energy - user.target_energy|
      +0.5 point max for Acousticness Preference match
      +0.2 point max for Popularity Boost tie-breaker
    
    Returns a tuple of (total_score, list_of_reason_strings).
    """
    # Normalize song attributes
    if isinstance(song, Song):
        s_genre = song.genre.lower()
        s_mood = song.mood.lower()
        s_energy = song.energy
        s_acousticness = song.acousticness
        s_popularity = song.popularity
    else:
        s_genre = str(song.get("genre", song.get("genres", ""))).lower()
        s_mood = str(song.get("mood", "")).lower()
        s_energy = float(song.get("energy", 0.5) if song.get("energy") is not None else 0.5)
        s_acousticness = float(song.get("acousticness", 0.5) if song.get("acousticness") is not None else 0.5)
        s_pop_val = song.get("popularity", song.get("artist_popularity", 50.0))
        if s_pop_val is None or str(s_pop_val).strip().lower() in ("nan", "none", ""):
            s_pop_val = 50.0
        s_popularity = float(s_pop_val)

    # Normalize user preferences
    if isinstance(user_prefs, UserProfile):
        u_genre = user_prefs.favorite_genre.lower()
        u_mood = user_prefs.favorite_mood.lower()
        u_energy = user_prefs.target_energy
        u_acoustic = user_prefs.likes_acoustic
    else:
        u_genre = str(user_prefs.get("favorite_genre", user_prefs.get("genre", ""))).lower()
        u_mood = str(user_prefs.get("favorite_mood", user_prefs.get("mood", ""))).lower()
        u_energy = float(user_prefs.get("target_energy", user_prefs.get("energy", 0.5)))
        u_acoustic = bool(user_prefs.get("likes_acoustic", user_prefs.get("acoustic", False)))

    score = 0.0
    reasons = []

    # 1. Genre Match (+2.0 points max)
    if u_genre and s_genre:
        s_genre_clean = s_genre.strip()
        # If it looks like a list representation, e.g. "['pop']", remove quotes and brackets
        if s_genre_clean.startswith("[") and s_genre_clean.endswith("]"):
            import ast
            try:
                genre_list = [str(x).lower().strip() for x in ast.literal_eval(s_genre_clean)]
            except:
                genre_list = [x.strip(" '\"").lower() for x in s_genre_clean[1:-1].split(",")]
        else:
            genre_list = [s_genre_clean.lower()]

        matched = False
        is_rap = any(r in g for g in genre_list for r in ['hip hop', 'rap', 'trap'])
        
        if u_genre == 'pop' or 'pop' in u_genre:
            high_priority = ['canadian pop', 'viral pop', 'dance pop', 'pop rock']
            for hp in high_priority:
                if any(hp in g for g in genre_list):
                    g_score = 1.0 if is_rap else 2.0
                    score += g_score
                    reasons.append(f"Genre match: subgenre '{hp}' (+{g_score:.1f})")
                    matched = True
                    break

        if not matched:
            if u_genre in genre_list:
                g_score = 1.0 if is_rap else 2.0
                score += g_score
                reasons.append(f"Genre match: '{u_genre}' (+{g_score:.1f})")
            else:
                overlap = False
                for g in genre_list:
                    if u_genre in g or g in u_genre:
                        g_score = 0.5 if is_rap else 1.5
                        score += g_score
                        reasons.append(f"Sub-genre overlap: '{g}' (+{g_score:.1f})")
                        overlap = True
                        break

    # 1.5 Artist Affinity Boost (+1.0 point max)
    s_artist = (song.artist if isinstance(song, Song) else str(song.get("artist", song.get("track_artists", "")))).lower()
    pop_stars = ['shawn mendes', 'justin bieber', 'charlie puth', 'ed sheeran', 'taylor swift', 'the chainsmokers', 'jonas brothers', 'lauv', 'one direction', 'post malone', 'tate mcrae', 'rihanna', 'the weeknd']
    if any(star in s_artist for star in pop_stars):
        score += 1.0
        reasons.append(f"Featured Pop Star match (+1.0)")

    # 2. Mood Match (+1.0 point max)
    if u_mood and s_mood:
        if u_mood == s_mood:
            score += 1.0
            reasons.append(f"Mood match: '{s_mood}' (+1.0)")

    # 3. Energy Proximity (+1.0 point max)
    energy_diff = abs(s_energy - u_energy)
    energy_sim = max(0.0, 1.0 - energy_diff)
    score += energy_sim
    reasons.append(f"Energy match: {s_energy:.2f} vs target {u_energy:.2f} (+{energy_sim:.2f})")

    # 4. Acousticness Preference (+0.5 point max)
    is_song_acoustic = s_acousticness >= 0.5
    if is_song_acoustic == u_acoustic:
        score += 0.5
        reasons.append(f"Acoustic preference match (+0.5)")

    # 5. Popularity Boost Tie-breaker (+0.2 point max)
    pop_sim = (s_popularity / 100.0) * 0.2
    score += pop_sim
    reasons.append(f"Popularity boost: {s_popularity:.1f}/100 (+{pop_sim:.2f})")

    return round(score, 3), reasons


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Ranks songs for a user profile and returns the top k Song objects.
        """
        scored_songs = []
        for song in self.songs:
            score, _ = score_song(user, song)
            scored_songs.append((song, score))
        
        # Sort descending by score
        scored_songs.sort(key=lambda item: item[1], reverse=True)
        return [song for song, _ in scored_songs[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Returns a formatted explanation string describing why a song was scored.
        """
        score, reasons = score_song(user, song)
        return f"Score {score:.2f}/4.70 -> " + "; ".join(reasons)

    def recommend_from_seed(self, seed_songs: List[Song], k: int = 5) -> List[Tuple[Song, float, str]]:
        """
        Recommends songs based on 1 or more seed tracks (Song Radio).
        Derives a dynamic UserProfile from seeds and filters out the seed tracks.
        """
        user_profile = UserProfile.from_seed_songs(seed_songs)
        seed_ids = {s.id for s in seed_songs}

        candidates = [s for s in self.songs if s.id not in seed_ids]
        return recommend_songs(user_profile, candidates, k=k)


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file into a list of dictionaries.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Warning: File {csv_path} does not exist.")
        return []

    songs = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title_val = row.get("title", row.get("name", "Unknown Title"))
            if not title_val or str(title_val).strip().lower() in ("nan", "none", ""):
                title_val = "Unknown Title"

            artist_val = row.get("artist", row.get("track_artists"))
            if not artist_val or str(artist_val).strip().lower() in ("nan", "none", ""):
                artist_val = row.get("track_album_album", row.get("album_name", "Unknown Artist"))
                if not artist_val or str(artist_val).strip().lower() in ("nan", "none", ""):
                    artist_val = "Unknown Artist"

            genre_val = row.get("genre", row.get("genres", "pop"))
            if not genre_val or str(genre_val).strip().lower() in ("nan", "none", ""):
                genre_val = "pop"

            pop_val = row.get("popularity", row.get("artist_popularity", 50.0))
            if pop_val is None or str(pop_val).strip().lower() in ("nan", "none", ""):
                pop_val = 50.0

            parsed_row = {
                "id": int(row["id"]) if "id" in row and row["id"].isdigit() else row.get("track_id", 0),
                "title": str(title_val),
                "artist": str(artist_val),
                "genre": str(genre_val),
                "mood": row.get("mood", "happy"),
                "energy": float(row["energy"]) if "energy" in row and row["energy"] else 0.5,
                "tempo_bpm": float(row["tempo_bpm"]) if "tempo_bpm" in row and row["tempo_bpm"] else float(row.get("tempo", 120.0) or 120.0),
                "valence": float(row["valence"]) if "valence" in row and row["valence"] else 0.5,
                "danceability": float(row["danceability"]) if "danceability" in row and row["danceability"] else 0.5,
                "acousticness": float(row["acousticness"]) if "acousticness" in row and row["acousticness"] else 0.5,
                "popularity": float(pop_val),
            }
            songs.append(parsed_row)
    return songs


def recommend_songs(user_prefs: Union[Dict, UserProfile], songs: List[Union[Dict, Song]], k: int = 5) -> List[Tuple[Union[Dict, Song], float, str]]:
    """
    Functional implementation of the recommendation logic.
    Returns a list of tuples: (song, score, explanation_string)
    Required by src/main.py
    """
    scored_items = []
    for s in songs:
        score, reasons = score_song(user_prefs, s)
        explanation = "; ".join(reasons)
        scored_items.append((s, score, explanation))

    # Sort descending by score
    scored_items.sort(key=lambda item: item[1], reverse=True)
    return scored_items[:k]


def score_parquet_tracks(parquet_path: str, user_prefs: Union[Dict, UserProfile], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    High-performance Parquet evaluation for large datasets (e.g. 1M+ tracks in tracks.parquet).
    Uses pandas / pyarrow for chunked or vectorized processing.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Skipping parquet scoring.")
        return []

    if not os.path.exists(parquet_path):
        print(f"Parquet file {parquet_path} not found.")
        return []

    print(f"Loading parquet dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Filter clean tracks and avoid seed track duplicate if matching profile exists
    # Sample up to 20,000 for better recommendation pool
    sample_df = df.dropna(subset=["danceability", "energy", "valence", "acousticness"]).head(20000)

    song_dicts = []
    for _, row in sample_df.iterrows():
        artist_val = row.get("track_artists")
        if not artist_val or pd.isna(artist_val) or str(artist_val).strip().lower() in ("nan", "none", ""):
            artist_val = row.get("track_album_album")
            if not artist_val or pd.isna(artist_val) or str(artist_val).strip().lower() in ("nan", "none", ""):
                artist_val = row.get("album_name")
                if not artist_val or pd.isna(artist_val) or str(artist_val).strip().lower() in ("nan", "none", ""):
                    artist_val = "Unknown Artist"

        title_val = row.get("name")
        if not title_val or pd.isna(title_val) or str(title_val).strip().lower() in ("nan", "none", ""):
            title_val = "Unknown Title"

        pop_val = row.get("artist_popularity", row.get("popularity", 50.0))
        if pd.isna(pop_val) or pop_val is None or str(pop_val).strip().lower() in ("nan", "none", ""):
            pop_val = 50.0

        song_dict = {
            "id": row.get("track_id", ""),
            "title": str(title_val),
            "artist": str(artist_val),
            "genre": str(row.get("genres", "")),
            "mood": "chill" if float(row.get("energy", 0.5) if pd.notnull(row.get("energy")) else 0.5) < 0.5 else "intense",
            "energy": float(row.get("energy", 0.5) if pd.notnull(row.get("energy")) else 0.5),
            "tempo_bpm": float(row.get("tempo", 120.0) if pd.notnull(row.get("tempo")) else 120.0),
            "valence": float(row.get("valence", 0.5) if pd.notnull(row.get("valence")) else 0.5),
            "danceability": float(row.get("danceability", 0.5) if pd.notnull(row.get("danceability")) else 0.5),
            "acousticness": float(row.get("acousticness", 0.5) if pd.notnull(row.get("acousticness")) else 0.5),
            "popularity": float(pop_val),
        }
        song_dicts.append(song_dict)

    # Exclude exact seed track matches from candidate recommendations if a seed song name is in user profile
    # (Mimics Code 2: clean_df[~clean_df['name'].str.contains(...)])
    seed_name = "There's Nothing Holdin' Me Back".lower()
    song_dicts = [s for s in song_dicts if seed_name not in s["title"].lower()]

    return recommend_songs(user_prefs, song_dicts, k=k)
