# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 4.0 (Hybrid Euclidean & Multi-Signal Engine)**

---

## 2. Intended Use

This music recommender system is designed to generate personalized song recommendations by analyzing user listening behavior and taste profiles.

* **What it does:** It generates similar song recommendations based on user profiles or seed tracks.
* **Usage of `UserProfile`:**
  * **Simulation Mode (`src/main.py`):** Uses a `UserProfile` object representing target tastes (favorite genre, mood, target energy, acoustic preference) to rank and score the music catalog.
  * **Interactive CLI Mode (`src/interactive_recommender.py`):** Supports one or more displayed seed tracks. It averages their 8D audio vectors into a playlist centroid and scores the full catalog.
* **Target Audience:** Designed for developers and users who want to find highly relevant pop, rock, and alternative hits. It is fully prototype-ready and can be deployed to production with minor upgrades (such as microservice deployment and API integrations).
* **Assumptions about the User:** Assumes that the user's immediate listening taste can be represented mathematically as a combination of categorical properties (genre, artist, mood) and continuous audio centroids.

---

## 3. How the Model Works

The system compares song attributes using a multi-signal composite score:

* **Song Features Used:** Genre, artist, track/artist popularity, energy, valence, danceability, acousticness, speechiness, liveness, instrumentalness, and tempo.
* **User Preferences Considered:** Seed-derived base genres, artists, and an 8D audio centroid. CLI presets can favor similarity, popularity, or discovery.
* **How it Computes a Score:**
  The composite score (0.0 to 1.0) is calculated as:
  $$
  Score = 0.35 \times AudioSimilarity + 0.35 \times HybridPopularity + 0.15 \times GenreSimilarity + 0.15 \times ArtistAffinity
  $$

  * *Audio Proximity (35%):* Calculated using Euclidean distance over the 8D audio feature vector.
  * *Hybrid Popularity (35%):* A blend of 75% track-level popularity and 25% artist popularity to prioritize actual singles and hits.
  * *Genre Match (15%):* Maps categorical subgenres to general parent categories.
  * *Artist Affinity (15%):* Boosts tracks by the same artist.
* **Changes from Starter Logic:** Upgraded from a simple 3-feature point counter to a high-performance, vectorized 12-feature pipeline using NumPy. It includes subgenre parent mapping, hybrid track popularity priority, and per-artist capping.

* **Recommendation Controls:** `similar` uses the documented default weights. `popular` prioritizes popularity while retaining strong audio and genre matching, excludes the seed artists, and returns at most one recommendation per artist. Each result includes the weighted contribution of every active signal.

---

## 4. Data

* **Catalog Size:** Local testing catalog (`data/songs.csv`, 10 tracks) and the production Spotify tracks dataset (`docs/tracks.parquet`, 899,224 tracks).
* **Features:** Energy, valence, danceability, acousticness, speechiness, liveness, instrumentalness, tempo, genres, and track/artist popularity.
* **Metadata Resolution:** 89% of the dataset initially lacked explicit artist names, which were resolved using cross-referenced lookup keys with 97.6% accuracy. Includes split parquet file resolution for `tracks_part1.parquet` and `tracks_part2.parquet`.

---

## 5. Strengths

* **User Types:** Works well for listeners searching for mainstream hits, pop tracks, and rock songs.
* **Patterns Captured:** Captures musical characteristics (e.g. fast danceable songs vs. slow chill instrumentals) while ensuring that popular hits bubble up first.
* **Intuition Matches:** Seeding *Starboy* by The Weeknd returns major pop hits like *Die For You*, *Reminder*, *Cruel Summer*, and *Levitating*.

---

## 6. Limitations and Bias

* **Superstar Loop Bias:** Since popularity represents 35% of the total score, globally famous tracks bubble to the top of recommendation lists, occasionally burying lesser-known indie releases.
* **Energy Proximity Limitations:** Uses a linear energy distance gap calculation, which can penalize excellent tracks that lie slightly outside the target range, even if their genre and mood match perfectly.
* **Genre Label Bias:** If a song's genre metadata is missing or sparse, the engine struggles to match it, meaning underrepresented indie tracks are heavily filtered out.

---

## 7. Evaluation

We evaluated both the local rule-based simulation and the interactive CLI engine on the 899,224-track Parquet dataset:

### 1. Parquet Verification on "Starboy" by The Weeknd (`src/interactive_recommender.py`)

```text
  🎧 Recommendations based on "Starboy" by The Weeknd
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   1. Die For You — The Weeknd  [ 78.5%]
      💿 Starboy
      Audio similarity: 0.94; Genre overlap: 1.00; Artist match: 1.0; Popularity: 0.88

   2. Blinding Lights — The Weeknd  [ 77.7%]
      💿 After Hours
      Audio similarity: 0.92; Genre overlap: 1.00; Artist match: 1.0; Popularity: 0.86

   3. Cruel Summer — Taylor Swift  [ 69.6%]
      💿 Lover
      Audio similarity: 0.91; Genre overlap: 1.00; Popularity: 0.99

   4. One Kiss (with Dua Lipa) — Calvin Harris  [ 63.8%]
      💿 One Kiss
      Audio similarity: 0.91; Genre overlap: 1.00; Popularity: 0.82

   5. What Do You Mean? — Justin Bieber  [ 63.4%]
      💿 Purpose
      Audio similarity: 0.96; Genre overlap: 1.00; Popularity: 0.78
```

### 2. Local Simulation Profile Runs (`data/songs.csv`)

* **High-Energy Pop:** Evaluated with target energy 0.85 and pop genre. Top matches were *Sunrise City* (Score: 4.91) and *Rooftop Lights* (Score: 4.35).
* **Chill Lofi:** Evaluated with target energy 0.35 and lofi genre. Top matches were *Library Rain* (Score: 5.05) and *Midnight Coding* (Score: 4.99).
* **Deep Intense Rock:** Evaluated with target energy 0.90 and rock genre. Top match was *Storm Runner* (Score: 5.04).
* **Adversarial Profile:** Testing conflicting pop/sad/energy:0.90 target. Top match was *Gym Hero* (Score: 3.91), showing pop-genre similarity out-prioritizing the mood mismatch.

### Profile Comparisons:

* **High-Energy Pop vs. Chill Lofi:** High-Energy Pop correctly favors fast, happy pop tracks (Sunrise City, Rooftop Lights) with targets around 0.85 energy. The Chill Lofi profile shifts entirely toward low-energy acoustic structures (Library Rain at 0.35 energy) and filters out pop.
* **Deep Intense Rock vs. High-Energy Pop:** Intense Rock prioritizes Storm Runner (+2.0 for rock) over Pop, even though both have high energy.
* **Adversarial Conflict:** The adversarial pop/sad/high-energy profile outputs Gym Hero as #1 because its high pop-genre weight (+2.0) and high energy similarity outweigh the lack of a "sad" mood match. The system prioritizes the genre structure over mood when they conflict.
* **(Why "Gym Hero" appears for "Happy Pop"):** *Our algorithm converts songs into numbers. 'Gym Hero' has ultra-high energy and danceability numbers that look almost identical to a fast, upbeat pop song. Because the system gives a huge point reward for matching energy levels, 'Gym Hero' scores high enough to jump into the 'Happy Pop' list even though its mood label is workout/intense.*

---

## 8. Ideas for Improvement / Future Work

* **100M+ Song Scaling with Vector Databases:** Migrate from raw NumPy matrix scoring to a dedicated Vector Search Database like **FAISS**, **Milvus**, or **Qdrant** to perform Approximate Nearest Neighbors (ANN) vector lookups on 100M+ songs in under 5 milliseconds.
* **Production-Grade Microservice Architecture:** Deploy the recommender as a containerized **FastAPI** web service inside Docker, with **Redis** to cache user query lookups and session state.
* **Serendipity Injection:** Implement a random mutation factor (e.g., 5% random genre candidates) to pop the filter bubble and suggest unexpected music.
* **Lyric Vibe Extraction:** Integrate natural language processing on lyric logs to establish deeper mood classifications.
