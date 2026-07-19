import sys
import os
# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import Song, UserProfile, Recommender, score_song, recommend_songs

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
