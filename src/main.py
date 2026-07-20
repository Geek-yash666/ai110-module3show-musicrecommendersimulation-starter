"""
Command line runner for the Music Recommender Simulation.
Demonstrates:
  1. Small dataset loading (data/songs.csv)
  2. Profile Differentiation: "Intense Rock" vs. "Chill Lofi"
  3. Single-song and Multi-song Seed Recommendations (Track Radio)
"""

import os
import sys

# Support running directly as a script (python3 src/main.py) or as a module (python3 -m src.main)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    load_songs,
    recommend_songs,
)


def print_recommendations(title: str, recommendations: list) -> None:
    print(f"\n==========================================")
    print(f"🎵 {title}")
    print(f"==========================================")
    for i, rec in enumerate(recommendations, 1):
        if isinstance(rec, dict):
            # Dict output from ProductionRecommender
            name = rec.get("name", "Unknown")
            artist = rec.get("artist", "Unknown")
            score = rec.get("score", 0.0)
            explanation = rec.get("explanation", "")
            print(f"{i}. {name} — {artist} (Score: {score:.2f})")
            print(f"   Reason: {explanation}")
        elif len(rec) == 3:
            # Tuple output from recommend_songs
            song, score, explanation = rec
            s_title = song.title if isinstance(song, Song) else song.get('title', song.get('name', ''))
            s_artist = song.artist if isinstance(song, Song) else song.get('artist', song.get('track_artists', ''))
            print(f"{i}. {s_title} — {s_artist} (Score: {score:.2f})")
            print(f"   Reason: {explanation}")
        else:
            print(f"{i}. {rec}")
    print()


def main() -> None:
    csv_path = "data/songs.csv"
    songs_data = load_songs(csv_path)
    song_objects = [Song.from_dict(s) for s in songs_data]
    recommender = Recommender(song_objects)

    print(f"Loaded {len(song_objects)} songs from {csv_path}.\n")

    # -------------------------------------------------------------
    # 1. Profile Differentiation: "Intense Rock" vs "Chill Lofi"
    # -------------------------------------------------------------
    intense_rock_profile = UserProfile(
        favorite_genre="rock",
        favorite_mood="intense",
        target_energy=0.85,
        likes_acoustic=False
    )

    chill_lofi_profile = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.35,
        likes_acoustic=True
    )

    # Simple recommendation output via functional interface
    rock_recs_func = recommend_songs(intense_rock_profile, song_objects, k=3)
    print_recommendations("User Taste Profile 1: Intense Rock (Small Dataset)", rock_recs_func)

    lofi_recs_func = recommend_songs(chill_lofi_profile, song_objects, k=3)
    print_recommendations("User Taste Profile 2: Chill Lofi (Small Dataset)", lofi_recs_func)

    # -------------------------------------------------------------
    # 2. Seed-Song / Track Radio Mode (Single and Multi-Song Seed)
    # -------------------------------------------------------------
    if len(song_objects) >= 4:
        seed_track_1 = song_objects[1]  # Midnight Coding (lofi)
        seed_track_2 = song_objects[3]  # Library Rain (lofi)

        print(f"🎧 Generating Track Radio from seed track: '{seed_track_1.title}' ({seed_track_1.genre}, {seed_track_1.mood})")
        single_seed_recs = recommender.recommend_from_seed([seed_track_1], k=3)
        print_recommendations("Single-Song Seed Recommendations (Track Radio - Small Dataset)", single_seed_recs)

        print(f"🎧 Generating Multi-Song Radio from seeds: '{seed_track_1.title}' + '{seed_track_2.title}'")
        multi_seed_recs = recommender.recommend_from_seed([seed_track_1, seed_track_2], k=3)
        print_recommendations("Multi-Song Seed Recommendations (Playlist Radio - Small Dataset)", multi_seed_recs)

if __name__ == "__main__":
    main()
