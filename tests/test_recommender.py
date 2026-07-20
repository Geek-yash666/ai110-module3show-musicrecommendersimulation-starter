import sys
import os
import numpy as np
import pandas as pd
import pytest
# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import Song, UserProfile, Recommender, ProductionRecommender, score_song, recommend_songs

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
        Song(
            id=3,
            title="Storm Runner",
            artist="Voltline",
            genre="rock",
            mood="intense",
            energy=0.91,
            tempo_bpm=152,
            valence=0.48,
            danceability=0.66,
            acousticness=0.10,
        )
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_differentiation_intense_rock_vs_chill_lofi():
    rock_user = UserProfile(
        favorite_genre="rock",
        favorite_mood="intense",
        target_energy=0.85,
        likes_acoustic=False,
    )
    lofi_user = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.35,
        likes_acoustic=True,
    )
    rec = make_small_recommender()

    rock_recs = rec.recommend(rock_user, k=3)
    lofi_recs = rec.recommend(lofi_user, k=3)

    # Rock profile should top-rank "Storm Runner" (rock/intense)
    assert rock_recs[0].genre == "rock"
    assert rock_recs[0].title == "Storm Runner"

    # Lofi profile should top-rank "Chill Lofi Loop" (lofi/chill)
    assert lofi_recs[0].genre == "lofi"
    assert lofi_recs[0].title == "Chill Lofi Loop"


def test_from_seed_songs_multi_track():
    seed1 = Song(
        id=10, title="Lofi 1", artist="Artist A", genre="lofi", mood="chill",
        energy=0.3, tempo_bpm=75, valence=0.5, danceability=0.4, acousticness=0.8
    )
    seed2 = Song(
        id=11, title="Lofi 2", artist="Artist B", genre="lofi", mood="chill",
        energy=0.4, tempo_bpm=85, valence=0.6, danceability=0.6, acousticness=0.9
    )

    derived_profile = UserProfile.from_seed_songs([seed1, seed2])
    assert derived_profile.favorite_genre == "lofi"
    assert derived_profile.favorite_mood == "chill"
    assert abs(derived_profile.target_energy - 0.35) < 0.001
    assert derived_profile.likes_acoustic is True


def test_recommend_from_seed():
    rec = make_small_recommender()
    seed_song = rec.songs[1]  # Chill Lofi Loop

    recs = rec.recommend_from_seed([seed_song], k=1)
    assert len(recs) == 1
    # Seed song itself should be excluded from recommendations
    recommended_song = recs[0][0]
    assert recommended_song.id != seed_song.id


def make_production_recommender() -> ProductionRecommender:
    """Small in-memory production catalog for ranking-policy tests."""
    rows = [
        ("seed-a", "Seed A", "Known Artist", {"pop"}, 0.70, 80, 80),
        ("seed-b", "Seed B", "Second Artist", {"pop"}, 0.80, 70, 70),
        ("unknown-1", "Unknown One", "Unknown Artist", {"pop"}, 0.72, 85, 85),
        ("unknown-1-remaster", "Unknown One (Remaster)", "Unknown Artist", {"pop"}, 0.72, 84, 84),
        ("unknown-2", "Unknown Two", "Unknown Artist", {"pop"}, 0.71, 84, 84),
        ("discovery", "Discovery Track", "Explorer", {"rock"}, 0.73, 75, 75),
        ("cover-a", "Shared Song", "Cover Artist A", {"pop"}, 0.73, 70, 70),
        ("cover-b", "Shared Song", "Cover Artist B", {"pop"}, 0.73, 69, 69),
        ("audio-match", "Audio Match", "Indie Artist", {"pop"}, 0.70, 0, 0),
        ("pop-hit", "Pop Hit", "Hit Artist", {"pop"}, 0.00, 100, 100),
    ]
    records = []
    for track_id, name, artist, genres, energy, track_pop, artist_pop in rows:
        records.append({
            "track_id": track_id, "resolved_name": name, "name_lower": name.lower(),
            "resolved_artist": artist, "artist_lower": artist.lower(), "genre_set": genres,
            "album_name": "Test Album", "energy": energy, "valence": 0.60,
            "danceability": 0.70, "acousticness": 0.20, "speechiness": 0.10,
            "liveness": 0.10, "instrumentalness": 0.00, "tempo_norm": 0.50,
            "norm_track_popularity": track_pop / 100, "norm_artist_popularity": artist_pop / 100,
            "norm_popularity": max(track_pop, artist_pop) / 100,
            "popularity": track_pop, "release_year": 2020,
        })
    engine = ProductionRecommender.__new__(ProductionRecommender)
    engine._np = np
    engine._pd = pd
    engine.df = pd.DataFrame(records)
    engine.catalog = engine.df.copy().reset_index(drop=True)
    engine.audio_matrix = engine.df[engine.AUDIO_FEATURES].values.astype(float)
    engine._build_search_catalog()
    return engine


def catalog_index(engine: ProductionRecommender, name: str) -> int:
    return int(engine.catalog.index[engine.catalog["resolved_name"] == name][0])


def test_production_playlist_radio_and_unknown_artist_cap():
    engine = make_production_recommender()
    results = engine.recommend_from_seeds(
        [catalog_index(engine, "Seed A"), catalog_index(engine, "Seed B")], k=3, max_per_artist=1
    )

    assert all(rec["name"] not in {"Seed A", "Seed B"} for rec in results)
    assert sum(rec["artist"] == "Unknown Artist" for rec in results) == 2
    assert set(results[0]["contributions"]) == {"audio", "popularity", "genre", "artist"}


def test_production_diverse_mode_includes_discovery_pick():
    engine = make_production_recommender()
    results = engine.recommend_from_seeds([catalog_index(engine, "Seed A")], k=3, discovery_ratio=0.34)

    assert any(rec["discovery"] for rec in results)


def test_production_search_handles_exact_and_fuzzy_queries():
    engine = make_production_recommender()

    exact = engine.fuzzy_search_songs("seed a", limit=1)
    fuzzy = engine.fuzzy_search_songs("unknwon two", limit=1)

    assert exact[0]["name"] == "Seed A"
    assert fuzzy[0]["name"] == "Unknown Two"


def test_production_presets_change_ranking_priority():
    engine = make_production_recommender()
    pop_hit_idx = int(engine.df.index[engine.df["resolved_name"] == "Pop Hit"][0])
    engine.audio_matrix[pop_hit_idx] = 0.0

    seed_index = catalog_index(engine, "Seed A")
    similar = engine.recommend(seed_index, k=9, weights=engine.SCORING_PRESETS["similar"])
    popular = engine.recommend(seed_index, k=9, weights=engine.SCORING_PRESETS["popular"])
    similar_names = [rec["name"] for rec in similar]
    popular_names = [rec["name"] for rec in popular]

    assert similar_names.index("Audio Match") < similar_names.index("Pop Hit")
    assert popular_names.index("Pop Hit") < popular_names.index("Audio Match")


def test_production_deduplicates_editions_but_keeps_covers():
    engine = make_production_recommender()
    results = engine.recommend(catalog_index(engine, "Seed A"), k=10, max_per_artist=0)
    names_and_artists = {(rec["name"], rec["artist"]) for rec in results}

    assert sum(rec["artist"] == "Unknown Artist" and rec["name"].startswith("Unknown One") for rec in results) == 1
    assert ("Shared Song", "Cover Artist A") in names_and_artists
    assert ("Shared Song", "Cover Artist B") in names_and_artists


def test_production_rejects_missing_or_invalid_seed_configuration():
    engine = make_production_recommender()

    with pytest.raises(ValueError, match="cannot be empty"):
        engine.recommend_from_seeds([])
    with pytest.raises(ValueError, match="Unknown scoring weights"):
        engine.recommend(catalog_index(engine, "Seed A"), weights={"not_a_signal": 1.0})
