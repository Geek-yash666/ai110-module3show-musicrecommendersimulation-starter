# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  
**VibeFinder 1.0 (Hybrid Cosine & Multi-Signal Engine)**

---

## 2. Intended Use  
This model is designed to generate personalized song recommendations by analyzing user listening behavior and taste profiles. 

* **Type of Recommendations:** Generates top-k similar tracks based on user preference profiles or seed song attributes (generating similar playlists/track radios).
* **Assumptions:** Assumes the user has consistent acoustic and energetic preferences and that their taste aligns with traditional structural genre metadata.
* **Context:** Built for classroom exploration, simulated user taste testing, and small-to-production scale catalog ranking (tested on up to 899,000+ tracks).

---

## 3. How the Model Works  
The model calculates a match score for each track candidate using a five-signal weighted scoring system:

1. **Audio Similarity (35%):** Measures physical music traits (energy, valence, danceability, acousticness, speechiness, liveness, and normalized tempo) using Euclidean proximity over an 8D vector space.
2. **Genre Similarity (15%):** Applies matching rules to base parent genres. Sharing any parent category (like `pop` or `rock`) awards full points. Exact subgenre overlap earns a small bonus.
3. **Artist Affinity (15%):** Grants points if the track is by the user's favorite artist.
4. **Popularity Signal (20%):** Weights 75% track popularity + 25% artist popularity to bubble hit tracks up for major pop queries.
5. **Year Proximity (15%):** Prioritizes contemporary releases matching the user's era target using a decay curve.

---

## 4. Data  
* **Catalog Size:** Tested on a localized catalog (`data/songs.csv`, 10 tracks) and the production-scale catalog (`docs/tracks.parquet`, 899,224 tracks).
* **Representation:** Tracks are annotated with energy, valence, acousticness, tempo, genres, and release year.
* **Limitations of the Data:** Lacks lyric sentiment analysis, vocal timbre properties, and cultural context. 89% of the dataset initially lacked explicit `track_artists` names, which were resolved using cross-referenced lookup keys with 97.6% accuracy.

---

## 5. Strengths  
* **Strong Contextual Grouping:** Consistently groups mainstream pop hits together when queried with major hit seeds like *Starboy* or *Levitating*.
* **Typo Handling:** Uses token-based C++ fuzzy matching via `rapidfuzz` to handle user input errors smoothly.
* **Highly Responsive Centroids:** Calculates accurate feature averages (energy, danceability) from seed playlists to generate relevant playlist radios.

---

## 6. Limitations and Bias  
* **Popularity Loop Bias:** Because popularity represents 20% of the scoring formula, mainstream hits (like Taylor Swift or Justin Bieber tracks) bubble up faster than lesser-known indie releases. This can bury obscure artists in a positive feedback loop.
* **Energy Proximity Limitations:** Uses a linear energy distance gap calculation, which can penalize excellent tracks that lie slightly outside the target range, even if their genre and mood match perfectly.
* **Genre Label Bias:** If a song's genre metadata is missing or sparse, the engine struggles to match it, meaning underrepresented indie tracks are heavily filtered out.

---

## 7. Evaluation  

We ran a simulation test on 4 diverse profiles using the local `songs.csv` catalog:

### 1. High-Energy Pop
```
1. Sunrise City — Neon Echo (Score: 4.91)
   Reason: Genre match: 'pop' (+2.0); Mood match: 'happy' (+1.0); Energy match: 0.82 vs target 0.85 (+0.97); Acoustic preference match (+0.5); Valence (0.84) & Danceability (0.79) proximity (+0.34); Popularity boost: 50.0/100 (+0.10)
2. Rooftop Lights — Indigo Parade (Score: 4.35)
   Reason: Sub-genre overlap: 'indie pop' (+1.5); Mood match: 'happy' (+1.0); Energy match: 0.76 vs target 0.85 (+0.91); Acoustic preference match (+0.5); Valence (0.81) & Danceability (0.82) proximity (+0.34); Popularity boost: 50.0/100 (+0.10)
3. Gym Hero — Max Pulse (Score: 3.86)
   Reason: Genre match: 'pop' (+2.0); Energy match: 0.93 vs target 0.85 (+0.92); Acoustic preference match (+0.5); Valence (0.77) & Danceability (0.88) proximity (+0.34); Popularity boost: 50.0/100 (+0.10)
```

### 2. Chill Lofi
```
1. Library Rain — Paper Lanterns (Score: 5.05)
   Reason: Genre match: 'lofi' (+2.0); Mood match: 'chill' (+1.0); Energy match: 0.35 vs target 0.35 (+1.00); Acoustic preference match (+0.5); Valence (0.60) & Danceability (0.58) proximity (+0.46); Popularity boost: 50.0/100 (+0.10)
2. Midnight Coding — LoRoom (Score: 4.99)
   Reason: Genre match: 'lofi' (+2.0); Mood match: 'chill' (+1.0); Energy match: 0.42 vs target 0.35 (+0.93); Acoustic preference match (+0.5); Valence (0.56) & Danceability (0.62) proximity (+0.45); Popularity boost: 50.0/100 (+0.10)
3. Focus Flow — LoRoom (Score: 4.00)
   Reason: Genre match: 'lofi' (+2.0); Energy match: 0.40 vs target 0.35 (+0.95); Acoustic preference match (+0.5); Valence (0.59) & Danceability (0.60) proximity (+0.45); Popularity boost: 50.0/100 (+0.10)
```

### 3. Deep Intense Rock
```
1. Storm Runner — Voltline (Score: 5.04)
   Reason: Genre match: 'rock' (+2.0); Mood match: 'intense' (+1.0); Energy match: 0.91 vs target 0.90 (+0.99); Acoustic preference match (+0.5); Valence (0.48) & Danceability (0.66) proximity (+0.45); Popularity boost: 50.0/100 (+0.10)
2. Gym Hero — Max Pulse (Score: 2.91)
   Reason: Mood match: 'intense' (+1.0); Energy match: 0.93 vs target 0.90 (+0.97); Acoustic preference match (+0.5); Valence (0.77) & Danceability (0.88) proximity (+0.34); Popularity boost: 50.0/100 (+0.10)
```

### 4. Adversarial Profile (Conflicting: pop/sad/energy: 0.90)
```
1. Gym Hero — Max Pulse (Score: 3.91)
   Reason: Genre match: 'pop' (+2.0); Energy match: 0.93 vs target 0.90 (+0.97); Acoustic preference match (+0.5); Valence (0.77) & Danceability (0.88) proximity (+0.34); Popularity boost: 50.0/100 (+0.10)
2. Sunrise City — Neon Echo (Score: 3.86)
   Reason: Genre match: 'pop' (+2.0); Energy match: 0.82 vs target 0.90 (+0.92); Acoustic preference match (+0.5); Valence (0.84) & Danceability (0.79) proximity (+0.34); Popularity boost: 50.0/100 (+0.10)
```

### Comparison & Musical Analysis:
* **High-Energy Pop vs. Chill Lofi:** High-Energy Pop correctly favors fast, happy pop tracks (Sunrise City, Rooftop Lights) with targets around 0.85 energy. The Chill Lofi profile shifts entirely toward low-energy acoustic structures (Library Rain at 0.35 energy) and filters out pop.
* **Deep Intense Rock vs. High-Energy Pop:** Intense Rock prioritizes Storm Runner (+2.0 for rock) over Pop, even though both have high energy.
* **Adversarial Conflict:** The adversarial pop/sad/high-energy profile outputs Gym Hero as #1 because its high pop-genre weight (+2.0) and high energy similarity outweigh the lack of a "sad" mood match. The system prioritizes the genre structure over mood when they conflict.

---

## 8. Future Work  
* **Serendipity Injection:** Implement a random mutation factor (e.g., 5% random genre candidates) to pop the filter bubble.
* **Lyric Vibe Extraction:** Integrate natural language processing on lyric logs to establish deeper mood classifications.
* **Collaborative Filtering:** Transition to a hybrid model that incorporates co-listening matrix data.

---

## 9. Personal Reflection  
Developing and evaluating recommender systems highlighted that a mathematical centroid can translate complex human music tastes into simple distance checks. I realized that popularity signals must be carefully tuned to prevent superstars from completely dominating recommendation feeds, illustrating the balance between familiarity and discovery.
