# AI Interactions Log

## Agentic Workflow (SF8)

**What task did you give the agent?**
I requested the AI agent to implement three advanced stretch features:

1. **Challenge 1 (Advanced Song Features):** Introduce at least 5 complex musical attributes (valence, danceability, acousticness, speechiness, liveness, and instrumentalness) to our CSV and parquet loader, updating the point-based algorithm and vector-based scoring models in `src/recommender.py`.
2. **Challenge 3 (Diversity and Fairness Logic):** Implement a diversity and deduplication filter that restricts recommendations to a maximum of 2–3 songs per artist and strips duplicate album editions of the same song title.
3. **Challenge 4 (Visual CLI Table):** Develop a beautiful, colored interactive terminal layout (`tests/interactive_recommender.py`) utilizing ANSI color codes, boxes, clean percentages, and detailed text tables to display top recommendations and explain their scores.

**Prompts used:**

> "Please expand the dataset and recommendation scoring to support 5 additional continuous features: valence, danceability, acousticness, speechiness, liveness, and instrumentalness. Update `load_songs` and `score_song` so that they calculate Euclidean distance similarity over these new parameters."
>
> "Create a diversity filter in `recommend_songs` and `ProductionRecommender.recommend` that filters out duplicate songs by the same artist if they exceed 2 tracks in the top results, and deduplicates identical track names released under different albums."
>
> "Write an interactive terminal recommender CLI script using ANSI escape codes for styling. Implement clean tables showing song names, artists, albums, percentage relevance scores, and distinct feature values, and handle typos gracefully using rapidfuzz."

**What did the agent generate or change?**

* **`src/recommender.py`**:
  * Updated `Song` and `UserProfile` to handle 8D audio features.
  * Added `load_songs` support for parsing valence, danceability, acousticness, and popularity.
  * Implemented an artist-capping diversity filter (`max_per_artist=2`) in `recommend_songs` and `recommend`.
* **`tests/interactive_recommender.py`**:
  * Generated a fully interactive console loop with custom search query classification, colored ANSI tables, and process delay animations.
* **`data/songs.csv`**:
  * Added valence, danceability, acousticness, and popularity attributes by utilizing whole new dataset tracks.parquet which contains 1m+ songs.

**What did you verify or fix manually?**

* Verified the entire `unittest` test suite passes cleanly.
* Manually corrected a sorting bug in Pass 1 of the fuzzy search where tracks were sorted by dataframe index instead of popularity.
* Adjusted the composite weight parameters in `ProductionRecommender` (Audio 35%, Popularity 35%, Genre 15%, Artist 15%) to ensure the system ranked mainstream hits (*Starboy*, *Levitating*) correctly when compared to human musical taste.

---

## Design Pattern (SF10)

**Which design pattern did you use?**
Builder / Parameter Centroid Pattern (via `UserProfile.from_seed_songs`).

**How did AI help you brainstorm or implement it?**
We discussed how to make "Track Radio" (seeding recommendations from one or multiple songs) work without forcing the user to manually input all their favorite genres and energy metrics. The AI suggested a dynamic builder pattern where we pass a list of `Song` objects to `UserProfile.from_seed_songs()`, which automatically aggregates, computes statistical averages (centroids) of their continuous audio features, and returns a fully initialized `UserProfile` taste profile.

**How does the pattern appear in your final code?**
It is implemented in `src/recommender.py` under [UserProfile.from_seed_songs](file:///Users/roop/Documents/Codepath/musicrecommender/ai110-module3show-musicrecommendersimulation-starter/src/recommender.py#L76-L104). The OOP `Recommender` class consumes this profile seamlessly inside `recommend_from_seed`.
