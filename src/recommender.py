import csv
import os
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py aswell as interactive_recommender.py
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
    favorite_artist: str = ""

    @classmethod
    def from_seed_songs(cls, seed_songs: List[Song]) -> "UserProfile":
        """
        Dynamically derives a UserProfile (centroid) from one or multiple seed songs.
        Powers "Song Radio" and "Similar Tracks" recommendation capabilities.
        """
        if not seed_songs:
            raise ValueError("seed_songs list cannot be empty")
        
        # Determine dominant genre, mood, and artist (mode/majority vote)
        top_genre = Counter([s.genre for s in seed_songs]).most_common(1)[0][0]
        top_mood = Counter([s.mood for s in seed_songs]).most_common(1)[0][0]
        artists = [s.artist for s in seed_songs if s.artist and s.artist.lower() != "unknown artist"]
        top_artist = Counter(artists).most_common(1)[0][0] if artists else ""

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
            target_danceability=avg_dance,
            favorite_artist=top_artist
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
    u_artist = (user_prefs.favorite_artist if isinstance(user_prefs, UserProfile) and hasattr(user_prefs, "favorite_artist") else str(user_prefs.get("favorite_artist", user_prefs.get("artist", "")))).lower()
    if u_artist and u_artist != "unknown artist" and (u_artist in s_artist or s_artist in u_artist):
        score += 1.0
        reasons.append(f"Artist match: '{s_artist}' (+1.0)")

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

    # 5. Valence, Danceability & Tempo Proximity (+0.5 point max)
    if isinstance(song, Song):
        s_valence = song.valence
        s_dance = song.danceability
        s_tempo = song.tempo_bpm
    else:
        s_valence = float(song.get("valence", 0.5) if song.get("valence") is not None else 0.5)
        s_dance = float(song.get("danceability", 0.5) if song.get("danceability") is not None else 0.5)
        s_tempo = float(song.get("tempo_bpm", song.get("tempo", 120.0)) if song.get("tempo_bpm", song.get("tempo")) is not None else 120.0)

    if isinstance(user_prefs, UserProfile):
        u_valence = user_prefs.target_valence
        u_dance = user_prefs.target_danceability
    else:
        u_valence = float(user_prefs.get("target_valence", user_prefs.get("valence", 0.5)))
        u_dance = float(user_prefs.get("target_danceability", user_prefs.get("danceability", 0.5)))

    val_sim = max(0.0, 1.0 - abs(s_valence - u_valence))
    dance_sim = max(0.0, 1.0 - abs(s_dance - u_dance))
    score += (val_sim * 0.25) + (dance_sim * 0.25)
    reasons.append(f"Valence ({s_valence:.2f}) & Danceability ({s_dance:.2f}) proximity (+{(val_sim * 0.25 + dance_sim * 0.25):.2f})")

    # 6. Popularity Boost Tie-breaker (+0.2 point max)
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
        scored_songs = recommend_songs(user, self.songs, k=k)
        return [song for song, _, _ in scored_songs]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Returns a formatted explanation string describing why a song was scored.
        """
        score, reasons = score_song(user, song)
        return f"Score {score:.2f}/5.20 -> " + "; ".join(reasons)

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


def recommend_songs(user_prefs: Union[Dict, UserProfile], songs: List[Union[Dict, Song]], k: int = 5, max_per_artist: int = 2) -> List[Tuple[Union[Dict, Song], float, str]]:
    """
    Functional implementation of the recommendation logic.
    Returns a list of tuples: (song, score, explanation_string)
    Required by src/main.py
    """
    import re
    def norm_t(title_str):
        t = str(title_str).lower()
        t = re.sub(r'\s*\(.*?\)', '', t)
        t = re.sub(r'\s*\[.*?\]', '', t)
        t = re.sub(r'\s*-\s*.*', '', t)
        return t.strip()

    scored_items = []
    seen_titles = set()
    artist_counts = {}

    all_scored = []
    for s in songs:
        score, reasons = score_song(user_prefs, s)
        explanation = "; ".join(reasons)
        all_scored.append((s, score, explanation))

    all_scored.sort(key=lambda item: item[1], reverse=True)

    for s, score, explanation in all_scored:
        s_title = s.title if isinstance(s, Song) else s.get('title', s.get('name', ''))
        s_artist = (s.artist if isinstance(s, Song) else str(s.get('artist', s.get('track_artists', '')))).strip().lower()

        # Enforce max tracks per artist limit if max_per_artist > 0
        if max_per_artist > 0 and s_artist and artist_counts.get(s_artist, 0) >= max_per_artist:
            continue

        nt = norm_t(s_title)
        if nt and nt in seen_titles:
            continue
        if nt:
            seen_titles.add(nt)

        if s_artist:
            artist_counts[s_artist] = artist_counts.get(s_artist, 0) + 1

        scored_items.append((s, score, explanation))
        if len(scored_items) == k:
            break

    return scored_items


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
        part1 = os.path.join(os.path.dirname(parquet_path), "tracks_part1.parquet")
        part2 = os.path.join(os.path.dirname(parquet_path), "tracks_part2.parquet")
        if os.path.exists(part1) and os.path.exists(part2):
            print("Combining split parquet files (part1 & part2)...")
            df1 = pd.read_parquet(part1)
            df2 = pd.read_parquet(part2)
            df = pd.concat([df1, df2])
        else:
            print(f"Parquet file {parquet_path} not found.")
            return []
    else:
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


# =============================================================================
# Recommender ( Cosine Similarity Engine)
# =============================================================================

class ProductionRecommender:
    """
 music recommender using multi-signal weighted scoring:
      - 35% Audio Feature Cosine Similarity (7D vector)
      - 25% Genre Overlap (Jaccard similarity on parsed subgenre sets)
      - 20% Artist Affinity (exact match + same-cluster boost)
      - 10% Popularity (normalized artist_popularity)
      - 10% Release Year Proximity (exponential decay over 15-year window)

    Operates on the full parquet dataset (~900K tracks) with smart indexing.
    """

    # Weights for composite scoring — tuned for musical relevance
    W_AUDIO = 0.35       # Audio profile is the strongest signal
    W_GENRE = 0.15       # Base-genre match (broad, not subgenre-specific)
    W_ARTIST = 0.15      # Same artist boost
    W_POPULARITY = 0.20  # Mainstream hit priority (uses track-level popularity)
    W_YEAR = 0.15        # Era proximity — favor contemporary hits

    AUDIO_FEATURES = [
        "energy", "valence", "danceability", "acousticness",
        "speechiness", "liveness", "instrumentalness", "tempo_norm"
    ]

    # Base genre mapping: normalizes subgenres to parent categories
    # so "canadian pop" matches "dance pop" (both → "pop")
    BASE_GENRE_MAP = {
        'pop': 'pop', 'dance pop': 'pop', 'canadian pop': 'pop',
        'viral pop': 'pop', 'electropop': 'pop', 'art pop': 'pop',
        'indie pop': 'pop', 'synth-pop': 'pop', 'k-pop': 'pop',
        'europop': 'pop', 'pop rock': 'pop', 'pop rap': 'pop',
        'chamber pop': 'pop', 'dream pop': 'pop', 'power pop': 'pop',
        'neo mellow': 'pop', 'acoustic cover': 'pop',
        'rock': 'rock', 'alternative rock': 'rock', 'indie rock': 'rock',
        'classic rock': 'rock', 'garage rock': 'rock', 'punk rock': 'rock',
        'modern rock': 'rock', 'irish rock': 'rock',
        'permanent wave': 'rock', 'new wave': 'rock',
        'hip hop': 'hip hop', 'rap': 'hip hop', 'trap': 'hip hop',
        'conscious hip hop': 'hip hop', 'north carolina hip hop': 'hip hop',
        'r&b': 'r&b', 'contemporary r&b': 'r&b',
        'canadian contemporary r&b': 'r&b', 'urban contemporary': 'r&b',
        'edm': 'edm', 'house': 'edm', 'tech house': 'edm',
        'deep house': 'edm', 'tropical house': 'edm',
        'electronic': 'edm', 'electro house': 'edm',
        'country': 'country', 'classic country': 'country',
        'country rock': 'country',
        'jazz': 'jazz', 'soul': 'soul', 'funk': 'soul',
        'metal': 'metal', 'heavy metal': 'metal',
        'latin': 'latin', 'reggaeton': 'latin',
        'classical': 'classical', 'lo-fi': 'lo-fi',
        'lofi': 'lo-fi', 'lo-fi cover': 'lo-fi',
    }

    def __init__(self, parquet_path: str = "docs/tracks.parquet"):
        import pandas as pd
        import numpy as np
        import ast
        import re

        self._pd = pd
        self._np = np

        # Load dataset
        if not os.path.exists(parquet_path):
            part1 = os.path.join(os.path.dirname(parquet_path), "tracks_part1.parquet")
            part2 = os.path.join(os.path.dirname(parquet_path), "tracks_part2.parquet")
            if os.path.exists(part1) and os.path.exists(part2):
                df1 = pd.read_parquet(part1)
                df2 = pd.read_parquet(part2)
                df = pd.concat([df1, df2], ignore_index=True)
            else:
                raise FileNotFoundError(f"Cannot find {parquet_path} or split parts")
        else:
            df = pd.read_parquet(parquet_path)

        # Drop rows missing audio features (only ~478 out of 900K)
        df = df.dropna(subset=["energy", "valence", "danceability", "acousticness"])

        # --- Resolve artist names ---
        # Step 1: Build a lookup table from tracks that DO have track_artists.
        # Key = (artist_popularity, genres, artist_followers) → artist name.
        # This resolves ~71% of the 800K tracks missing track_artists.
        has_artist_mask = df["track_artists"].notna()
        if has_artist_mask.any():
            has_artist_df = df[has_artist_mask].copy()
            has_artist_df["_artist_key"] = (
                has_artist_df["artist_popularity"].astype(str) + "|" +
                has_artist_df["genres"].astype(str) + "|" +
                has_artist_df["artist_followers"].astype(str)
            )
            # Build lookup: for each key, take the most common artist name
            artist_lookup = has_artist_df.groupby("_artist_key")["track_artists"].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            )

            # Step 2: Apply lookup to tracks missing track_artists
            no_artist_mask = df["track_artists"].isna()
            df.loc[no_artist_mask, "_artist_key"] = (
                df.loc[no_artist_mask, "artist_popularity"].astype(str) + "|" +
                df.loc[no_artist_mask, "genres"].astype(str) + "|" +
                df.loc[no_artist_mask, "artist_followers"].astype(str)
            )
            df.loc[no_artist_mask, "resolved_artist"] = (
                df.loc[no_artist_mask, "_artist_key"].map(artist_lookup)
            )
            # Fill any remaining nulls with album_name, then "Unknown Artist"
            df["resolved_artist"] = df["resolved_artist"].fillna(df["track_artists"])
            df["resolved_artist"] = df["resolved_artist"].fillna(df["album_name"])
            df["resolved_artist"] = df["resolved_artist"].fillna("Unknown Artist")
        else:
            df["resolved_artist"] = df["track_artists"].fillna(df["album_name"]).fillna("Unknown Artist")

        df["resolved_artist"] = df["resolved_artist"].astype(str).str.strip()
        df.loc[df["resolved_artist"].str.lower().isin(["nan", "none", ""]), "resolved_artist"] = "Unknown Artist"
        df["artist_lower"] = df["resolved_artist"].str.lower()

        # --- Resolve track name ---
        df["resolved_name"] = df["name"].fillna("Unknown Title").astype(str).str.strip()
        df.loc[df["resolved_name"].str.lower().isin(["nan", "none", ""]), "resolved_name"] = "Unknown Title"
        df["name_lower"] = df["resolved_name"].str.lower()

        # --- Parse genres from stringified lists ---
        def parse_genres(g):
            if pd.isna(g) or not g or str(g).strip() in ("", "[]", "nan", "None"):
                return set()
            g = str(g).strip()
            if g.startswith("["):
                try:
                    return set(str(x).lower().strip() for x in ast.literal_eval(g))
                except:
                    return set(x.strip(" '\"").lower() for x in g[1:-1].split(",") if x.strip())
            return {g.lower()}

        df["genre_set"] = df["genres"].apply(parse_genres)

        # --- Parse release year ---
        def extract_year(d):
            if pd.isna(d) or not d:
                return 2020  # default
            d = str(d).strip()
            if len(d) >= 4 and d[:4].isdigit():
                return int(d[:4])
            return 2020

        df["release_year"] = df["album_release_date"].apply(extract_year)

        # --- Fill NaN audio features with 0.0 ---
        for feat in self.AUDIO_FEATURES:
            if feat == "tempo_norm":
                # Normalize tempo to 0-1 range (50-200 BPM → 0-1)
                raw_tempo = pd.to_numeric(df.get("tempo", 120.0), errors="coerce").fillna(120.0)
                df["tempo_norm"] = ((raw_tempo - 50.0) / 150.0).clip(0, 1)
            elif feat in df.columns:
                df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
            else:
                df[feat] = 0.0

        # --- Normalize popularity (track-level, with artist fallback) ---
        # Track popularity is more indicative of actual hits than artist-level
        track_pop = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0.0)
        artist_pop = pd.to_numeric(df.get("artist_popularity", 50.0), errors="coerce").fillna(50.0)
        # Use the higher of track or artist popularity
        df["norm_popularity"] = np.maximum(track_pop, artist_pop) / 100.0

        # --- Build audio feature matrix (numpy) ---
        self.df = df.reset_index(drop=True)
        self.audio_matrix = self.df[self.AUDIO_FEATURES].values.astype(np.float64)

        # Pre-compute norms for cosine similarity
        self._norms = np.linalg.norm(self.audio_matrix, axis=1, keepdims=True)
        self._norms[self._norms == 0] = 1e-10  # avoid division by zero

        # --- Build search indices ---
        self._name_index = list(self.df["resolved_name"])
        self._artist_index = list(self.df["resolved_artist"])
        self._name_lower_list = list(self.df["name_lower"])
        self._artist_lower_list = list(self.df["artist_lower"])

        # Build deduplicated lookup for fuzzy search (unique song+artist combos)
        self._build_search_catalog()

    def _build_search_catalog(self):
        """
        Build deduplicated catalogs for fast fuzzy search.
        Keeps only the highest-popularity version of each song+artist pair.
        """
        import re

        df = self.df
        np = self._np

        # Normalize title for dedup: strip parentheticals, remix tags, etc.
        def norm_title(t):
            t = str(t).lower().strip()
            t = re.sub(r'\s*[\(\[].*?[\)\]]', '', t)  # remove (feat. X), [Deluxe], etc.
            t = re.sub(r'\s*-\s*(remix|remaster|radio edit|live|deluxe|bonus).*', '', t, flags=re.IGNORECASE)
            return t.strip()

        df["_norm_title"] = df["resolved_name"].apply(norm_title)
        df["_dedup_key"] = df["_norm_title"] + " || " + df["artist_lower"]

        # Keep highest popularity version of each song
        idx = df.groupby("_dedup_key")["norm_popularity"].idxmax()
        self.catalog = df.loc[idx].reset_index(drop=True)

        # Rebuild name/artist search lists from catalog
        self._catalog_names = list(self.catalog["resolved_name"])
        self._catalog_artists = list(self.catalog["resolved_artist"])
        self._catalog_names_lower = list(self.catalog["name_lower"])
        self._catalog_artists_lower = list(self.catalog["artist_lower"])

        # Unique artist list for artist search
        unique_artists = self.catalog.drop_duplicates(subset=["artist_lower"])
        self._unique_artists = list(unique_artists["resolved_artist"])
        self._unique_artists_lower = list(unique_artists["artist_lower"])

    def fuzzy_search_songs(self, query: str, limit: int = 15, score_cutoff: int = 55) -> List[Dict]:
        """
        Two-pass song search:
          Pass 1: Fast exact/substring match against title OR title+artist combo
          Pass 2: Fuzzy fallback for misspellings via rapidfuzz

        Handles single titles ("Starboy") and title+artist queries ("blinding lights the weeknd").
        """
        from rapidfuzz import fuzz, process

        query_lower = query.strip().lower()
        query_len = len(query_lower)

        # ── Pass 1: Exact / Substring match ──────────────────────────────
        # 1a. Match title exact or substring
        exact_mask = (self.catalog["name_lower"] == query_lower) | \
                     self.catalog["name_lower"].str.contains(query_lower, na=False, regex=False)

        # 1b. If no direct title match, try title + artist combo match (for queries like "blinding lights the weeknd")
        if not exact_mask.any():
            combo = self.catalog["name_lower"] + " " + self.catalog["artist_lower"]
            exact_mask = combo.str.contains(query_lower, na=False, regex=False)

        if exact_mask.any():
            exact_df = self.catalog[exact_mask].copy()
            # Sort by highest norm_popularity, then track popularity
            pop_col = "popularity" if "popularity" in exact_df.columns else "norm_popularity"
            exact_df.sort_values(by=["norm_popularity", pop_col], ascending=[False, False], inplace=True)
            exact_df = exact_df.head(limit)

            matches = []
            seen = set()
            for _, row in exact_df.iterrows():
                dedup_key = row["_dedup_key"]
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                matches.append({
                    "catalog_idx": row.name,  # pandas index = catalog position
                    "name": row["resolved_name"],
                    "artist": row["resolved_artist"],
                    "album": str(row.get("album_name", "")),
                    "genres": row["genre_set"],
                    "popularity": row["norm_popularity"],
                    "year": row["release_year"],
                    "match_score": 100.0,
                })
            if matches:
                return matches[:limit]

        # ── Pass 2: Fuzzy fallback for misspellings ──────────────────────
        scorer = fuzz.token_sort_ratio if len(query_lower.split()) > 1 else fuzz.WRatio

        results = process.extract(
            query_lower,
            self._catalog_names_lower,
            scorer=scorer,
            limit=limit * 10,
            score_cutoff=max(score_cutoff - 10, 40)
        )

        seen = set()
        matches = []
        for match_str, score, idx in results:
            match_len = len(match_str)
            if query_len > 4 and match_len < query_len * 0.4:
                continue

            secondary_score = fuzz.token_set_ratio(query_lower, match_str)
            combined_score = (score * 0.6) + (secondary_score * 0.4)

            if combined_score < score_cutoff:
                continue

            row = self.catalog.iloc[idx]
            dedup_key = row["_dedup_key"]
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            matches.append({
                "catalog_idx": idx,
                "name": row["resolved_name"],
                "artist": row["resolved_artist"],
                "album": str(row.get("album_name", "")),
                "genres": row["genre_set"],
                "popularity": row["norm_popularity"],
                "year": row["release_year"],
                "match_score": combined_score,
            })
            if len(matches) >= limit * 3:
                break

        # Sort by fuzzy score desc, then popularity desc, then trim
        matches.sort(key=lambda x: (-x["match_score"], -x["popularity"]))

        # Post-pass: ensure highest popularity version is included
        if matches:
            top_names = set(m["name"].lower() for m in matches[:5])

            for name_lower in top_names:
                exact_mask = self.catalog["name_lower"] == name_lower
                if exact_mask.any():
                    best_row = self.catalog[exact_mask].sort_values(
                        "norm_popularity", ascending=False
                    ).iloc[0]
                    dedup_key = best_row["_dedup_key"]
                    if not any(m.get("_dk") == dedup_key or
                              (m["name"] == best_row["resolved_name"] and
                               m["artist"] == best_row["resolved_artist"])
                              for m in matches):
                        matches.append({
                            "catalog_idx": best_row.name,
                            "name": best_row["resolved_name"],
                            "artist": best_row["resolved_artist"],
                            "album": str(best_row.get("album_name", "")),
                            "genres": best_row["genre_set"],
                            "popularity": best_row["norm_popularity"],
                            "year": best_row["release_year"],
                            "match_score": matches[0]["match_score"],
                        })

            matches.sort(key=lambda x: (-x["match_score"], -x["popularity"]))

        return matches[:limit]

    def fuzzy_search_artists(self, query: str, limit: int = 10, score_cutoff: int = 55) -> List[str]:
        """
        Fuzzy search for artist names. Returns list of matched artist names.
        Uses partial_ratio to handle substring matching.
        """
        from rapidfuzz import process, fuzz

        query_lower = query.strip().lower()

        results = process.extract(
            query_lower,
            self._unique_artists_lower,
            scorer=fuzz.partial_ratio,
            limit=limit * 3,
            score_cutoff=score_cutoff
        )

        scored = []
        for match_str, partial_score, idx in results:
            wratio = fuzz.WRatio(query_lower, match_str)
            combined = (wratio * 0.4) + (partial_score * 0.6)
            len_ratio = min(len(query_lower), len(match_str)) / max(len(query_lower), len(match_str), 1)
            combined += len_ratio * 5
            scored.append((self._unique_artists[idx], combined))

        scored.sort(key=lambda x: -x[1])
        return [name for name, _ in scored[:limit]]

    def get_artist_tracks(self, artist_name: str, limit: int = 15) -> List[Dict]:
        """
        Get top tracks by a specific artist, sorted by track popularity.
        """
        artist_lower = artist_name.strip().lower()
        exact_mask = self.catalog["artist_lower"] == artist_lower
        contains_mask = self.catalog["artist_lower"].str.contains(artist_lower, regex=False, na=False)

        artist_df = self.catalog[exact_mask | contains_mask].copy()
        if artist_df.empty:
            return []

        # Sort by norm_popularity desc, then popularity desc
        pop_col = "popularity" if "popularity" in artist_df.columns else "norm_popularity"
        artist_df.sort_values(by=["norm_popularity", pop_col], ascending=[False, False], inplace=True)
        artist_df = artist_df.head(limit)

        tracks = []
        for _, row in artist_df.iterrows():
            tracks.append({
                "catalog_idx": row.name if hasattr(row, 'name') else 0,
                "name": row["resolved_name"],
                "artist": row["resolved_artist"],
                "album": str(row.get("album_name", "")),
                "genres": row["genre_set"],
                "popularity": row["norm_popularity"],
                "year": row["release_year"],
            })
        return tracks

    def recommend(self, seed_catalog_idx: int, k: int = 10, max_per_artist: int = 3) -> List[Dict]:
        """
        Generate top-k recommendations from a seed track using 5-signal scoring.

        Uses the FULL dataset (not just catalog) so we score every track,
        then deduplicates and limits per-artist in the final results.
        """
        import re
        np = self._np

        seed_row = self.catalog.iloc[seed_catalog_idx]
        seed_name = str(seed_row["resolved_name"])
        seed_artist = str(seed_row["resolved_artist"]).lower()
        seed_genres = seed_row["genre_set"]
        seed_year = seed_row["release_year"]

        # Get the original df index for this catalog entry to pull the audio vector
        # We need to find this track in self.df
        seed_track_id = seed_row.get("track_id", None)
        seed_name_lower = seed_row["name_lower"]

        # Find seed in main df by matching track_id or name+artist
        if seed_track_id and str(seed_track_id) != "nan":
            seed_mask = self.df["track_id"] == seed_track_id
            if seed_mask.any():
                seed_df_idx = self.df[seed_mask].index[0]
            else:
                seed_df_idx = 0
        else:
            seed_mask = (self.df["name_lower"] == seed_name_lower) & (self.df["artist_lower"] == seed_artist)
            if seed_mask.any():
                seed_df_idx = self.df[seed_mask].index[0]
            else:
                seed_df_idx = 0

        seed_vector = self.audio_matrix[seed_df_idx].reshape(1, -1)
        seed_norm = np.linalg.norm(seed_vector)
        if seed_norm == 0:
            seed_norm = 1e-10

        # --- Signal 1: Audio Similarity (Euclidean distance-based) ---
        # Euclidean distance is more sensitive to actual feature differences
        # than cosine similarity, giving better differentiation for pop songs
        diffs = self.audio_matrix - seed_vector
        euclidean_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        # Normalize: max possible distance in 7D unit cube = sqrt(7) ≈ 2.65
        max_dist = np.sqrt(len(self.AUDIO_FEATURES))
        audio_sims = np.clip(1.0 - (euclidean_dists / max_dist), 0, 1)

        # --- Signal 2: Base-Genre Similarity ---
        # Extract base genres from subgenre labels for broader matching:
        # "canadian pop" + "canadian contemporary r&b" → {"pop", "r&b"}
        def extract_base_genres(genre_set):
            base = set()
            for g in genre_set:
                g_lower = g.lower().strip()
                # Check direct mapping
                if g_lower in self.BASE_GENRE_MAP:
                    base.add(self.BASE_GENRE_MAP[g_lower])
                else:
                    # Substring match: "swedish tropical house" → "edm" (via "house")
                    for subgenre, parent in self.BASE_GENRE_MAP.items():
                        if subgenre in g_lower or g_lower in subgenre:
                            base.add(parent)
                            break
            return base

        seed_base_genres = extract_base_genres(seed_genres)

        genre_sims = np.zeros(len(self.df), dtype=np.float64)
        if seed_base_genres:
            for i, gs in enumerate(self.df["genre_set"]):
                if gs:
                    cand_base = extract_base_genres(gs)
                    if cand_base:
                        # Simple overlap: if they share ANY base genre, it's a match
                        if seed_base_genres & cand_base:
                            base_sim = 1.0
                        else:
                            base_sim = 0.0

                        # Bonus for exact subgenre overlap (0.2 max)
                        exact_overlap = len(seed_genres & gs)
                        exact_bonus = min(exact_overlap * 0.1, 0.2)

                        genre_sims[i] = min(base_sim + exact_bonus, 1.0)

        # --- Signal 3: Artist Affinity ---
        artist_sims = np.zeros(len(self.df), dtype=np.float64)
        artist_lower_arr = self.df["artist_lower"].values
        for i, a in enumerate(artist_lower_arr):
            if a == seed_artist:
                artist_sims[i] = 1.0
            elif seed_artist in str(a) or str(a) in seed_artist:
                artist_sims[i] = 0.5

        # --- Signal 4: Hybrid Popularity (Track Hits Priority) ---
        # 75% pure track popularity + 25% artist popularity
        # This ensures hit singles (Cruel Summer, Levitating, Shape of You, Sunflower)
        # beat non-singles and obscure background tracks.
        track_pops = self._pd.to_numeric(self.df.get("popularity", 0), errors="coerce").fillna(0.0).values / 100.0
        artist_pops = self.df["norm_popularity"].values.astype(np.float64)
        pop_scores = (track_pops * 0.75) + (artist_pops * 0.25)

        # --- Signal 5: Year Proximity (15-year decay) ---
        years = self.df["release_year"].values.astype(np.float64)
        year_diffs = np.abs(years - seed_year)
        year_sims = np.clip(1.0 - (year_diffs / 15.0), 0, 1)

        # --- Composite Score ---
        # Weights tuned for hit recommendation quality:
        # Audio 35%, Popularity 35%, Genre 15%, Artist 15%
        final_scores = (
            0.35 * audio_sims +
            0.35 * pop_scores +
            0.15 * genre_sims +
            0.15 * artist_sims
        )

        # --- Rank, deduplicate, and filter ---
        ranked_indices = np.argsort(-final_scores)

        def norm_title(t):
            t = str(t).lower().strip()
            t = re.sub(r'\s*[\(\[].*?[\)\]]', '', t)
            t = re.sub(r'\s*-\s*(remix|remaster|radio edit|live|deluxe|bonus).*', '', t, flags=re.IGNORECASE)
            return t.strip()

        seed_norm_title = norm_title(seed_name)
        seen_titles = {seed_norm_title}
        artist_counts = {}
        results = []

        for idx in ranked_indices:
            idx = int(idx)
            row = self.df.iloc[idx]
            track_name = str(row["resolved_name"])
            track_artist = str(row["resolved_artist"])
            track_artist_lower = str(row["artist_lower"])

            nt = norm_title(track_name)

            # Skip seed track
            if nt == seed_norm_title and track_artist_lower == seed_artist:
                continue

            # Deduplicate by normalized title
            if nt in seen_titles:
                continue
            seen_titles.add(nt)

            # Limit per artist
            if max_per_artist > 0:
                count = artist_counts.get(track_artist_lower, 0)
                if count >= max_per_artist:
                    continue
                artist_counts[track_artist_lower] = count + 1

            score = float(final_scores[idx])
            album = str(row.get("album_name", ""))

            # Build explanation
            explanations = []
            if audio_sims[idx] > 0.01:
                explanations.append(f"Audio similarity: {audio_sims[idx]:.2f}")
            if genre_sims[idx] > 0.01:
                explanations.append(f"Genre overlap: {genre_sims[idx]:.2f}")
            if artist_sims[idx] > 0.01:
                explanations.append(f"Artist match: {artist_sims[idx]:.1f}")
            explanations.append(f"Popularity: {pop_scores[idx]:.2f}")
            if year_sims[idx] > 0.01:
                explanations.append(f"Era match: {year_sims[idx]:.2f}")

            results.append({
                "name": track_name,
                "artist": track_artist,
                "album": album,
                "score": score,
                "explanation": "; ".join(explanations),
                "audio_sim": float(audio_sims[idx]),
                "genre_sim": float(genre_sims[idx]),
                "artist_sim": float(artist_sims[idx]),
                "pop_score": float(pop_scores[idx]),
                "year_sim": float(year_sims[idx]),
            })

            if len(results) >= k:
                break

        return results

