# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

This music recommender system provides personalized, high-performance song recommendations using a hybrid multi-signal scoring model. It implements multidimensional cosine similarity over audio features, subgenre matching, artist affinity, popularity scaling, and release year proximity, fully scaling to evaluate datasets with over 899,000+ tracks.

---

## How The System Works

### **Song Features & User Profile**

This system models songs using **multi-dimensional audio features**:

**Song Attributes:**

- **Genre** (categorical): Primary music classification and subgenre lists (e.g. `['canadian pop', 'pop', 'viral pop']`)
- **Mood** (categorical): Emotional context — chill, intense, happy, focused, relaxed, moody
- **Audio Features** (numerical, 0–1 scale):
  - **Energy**: Intensity/activity level
  - **Valence**: Musical positivity
  - **Danceability**: Rhythmic suitability
  - **Acousticness**: Real instruments vs. production
  - **Speechiness**: Presence of spoken words
  - **Liveness**: Presence of an audience/live performance
- **Popularity**: Popularity metric (0–100) representing artist or track global popularity
- **Release Date / Year**: Track release year, used to prioritize recency proximity

**UserProfile:**

- **Favorite genre** & **favorite mood**: Primary user intent (derived from seeds or entered directly)
- **Target energy** & **audio centroids**: Average audio profile computed from seed songs
- **Likes Acoustic**: Boolean user preference

### **Finalized "Algorithm Recipe" (Composite Weighted Scoring)**

For each song candidate, the recommender calculates a **composite score** (0.0 to 1.0) using five distinct signals:

$$Score = 0.35 \times AudioSimilarity + 0.25 \times GenreSimilarity + 0.20 \times ArtistSimilarity + 0.10 \times Popularity + 0.10 \times YearProximity$$

1. **Audio Feature Cosine Similarity (35%)**: Calculates the multidimensional cosine similarity between the candidate's audio features (`danceability`, `energy`, `valence`, `acousticness`, `speechiness`, `liveness`) and the user's profile centroid.
2. **Genre Similarity (25%)**: Parses subgenres and computes matching overlap. If the seed is Pop, high-priority pop subgenres (`canadian pop`, `dance pop`, etc.) receive full points. To prevent hip-hop/rap tracks from crowding out pop, a penalty is applied if `hip hop` or `rap` overlap with pure pop requests.
3. **Artist Similarity (20%)**: Grants a high similarity boost if the artist matches the seed artist directly, or matches a predefined related pop star cluster.
4. **Popularity (10%)**: Integrates normalized global popularity (0.0 to 1.0) to prioritize high-engagement tracks as a tie-breaker.
5. **Release Year Proximity (10%)**: Calculates year distance decay over a 15-year window to favor songs released closer to the seed track's era.

### **Potential Biases & Risks**

* **Genre Over-Prioritization Bias**: Because genre similarity and artist affinity represent a combined **45%** of the total score, the system might over-prioritize genre matching, occasionally ignoring great songs in different genres that match the user's mood or energy profile.
* **Superstar / Popularity Loop Bias**: The **10%** popularity signal ensures that among otherwise similar matches, globally famous tracks (like Taylor Swift or Justin Bieber hits) bubble to the top. This can cause a popularity loop, reinforcing popular songs while burying obscure indie tracks.
* **Release Year Decay**: The **10%** release year proximity can filter out older classics that match the mood perfectly, confining users to a narrow release era.

### **Ranking Rule: Choosing What to Show**

Once scored, songs are ranked and filtered:
1. Sort by final score (descending)
2. Filter out exact seed track duplicates
3. Deduplicate by song name to prevent multiple album versions of the same track from flooding the results.
4. Return the top $k$ recommendations.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

```
Loaded 10 songs from data/songs.csv.

==========================================
🎵 User Taste Profile 1: Intense Rock
==========================================
1. Storm Runner — Voltline (Score: 4.54)
   Reason: Genre match: 'rock' (+2.0); Mood match: 'intense' (+1.0); Energy match: 0.91 vs target 0.85 (+0.94); Acoustic preference match (+0.5); Popularity boost: 50.0/100 (+0.10)
2. Gym Hero — Max Pulse (Score: 2.52)
   Reason: Mood match: 'intense' (+1.0); Energy match: 0.93 vs target 0.85 (+0.92); Acoustic preference match (+0.5); Popularity boost: 50.0/100 (+0.10)
3. Sunrise City — Neon Echo (Score: 1.57)
   Reason: Energy match: 0.82 vs target 0.85 (+0.97); Acoustic preference match (+0.5); Popularity boost: 50.0/100 (+0.10)


==========================================
🎵 User Taste Profile 2: Chill Lofi
==========================================
1. Library Rain — Paper Lanterns (Score: 4.60)
   Reason: Genre match: 'lofi' (+2.0); Mood match: 'chill' (+1.0); Energy match: 0.35 vs target 0.35 (+1.00); Acoustic preference match (+0.5); Popularity boost: 50.0/100 (+0.10)
2. Midnight Coding — LoRoom (Score: 4.53)
   Reason: Genre match: 'lofi' (+2.0); Mood match: 'chill' (+1.0); Energy match: 0.42 vs target 0.35 (+0.93); Acoustic preference match (+0.5); Popularity boost: 50.0/100 (+0.10)
3. Focus Flow — LoRoom (Score: 3.55)
   Reason: Genre match: 'lofi' (+2.0); Energy match: 0.40 vs target 0.35 (+0.95); Acoustic preference match (+0.5); Popularity boost: 50.0/100 (+0.10)
```

---

## Experiments You Tried

- **Genre Weighting (+2.0 vs +0.5)**: When genre weight was reduced from +2.0 to +0.5, cross-genre high-energy pop tracks ("Gym Hero") ranked higher than rock tracks for an intense rock profile. Keeping genre at +2.0 ensured categorical intent remained primary.
- **Single vs Multi-Song Seed Profiles**: We tested deriving dynamic user profiles from multiple seed tracks (`UserProfile.from_seed_songs([track1, track2])`). This computed a centroid of audio features (energy, acousticness, valence) which accurately generated playlist radio recommendations without manual user parameter entry.
- **Large Dataset Parquet Evaluation (`docs/tracks.parquet`)**: Evaluated 1M+ tracks dataset using PyArrow/Pandas sub-genre substring matching (`modern alternative rock`, `garage rock`), proving sub-genre matching correctly retrieves relevant tracks at scale.

---

## Limitations and Risks

- **Static Audio Features**: Does not analyze lyrics, timbre, or temporal music structure.
- **Filter Bubbles & Popularity Bias**: Pure content similarity can trap users in narrow genre loops without active novelty injection or popularity re-ranking.
- **Cold-Start Tracks**: Requires pre-calculated audio features (energy, acousticness) to score new releases.

---

## Reflection

Recommendation engines turn unstructured user listening behavior into numerical target profiles and score candidate tracks through multi-attribute proximity formulas. Weighting categorical constraints (genre/mood) above continuous signals (energy/acousticness) creates strong personalization while maintaining serendipity.

Bias in recommendation systems manifests through genre over-representation and popularity feedback loops. Mitigating these risks requires dynamic feature weighting, diversity re-ranking, and cold-start candidate retrieval strategies.

