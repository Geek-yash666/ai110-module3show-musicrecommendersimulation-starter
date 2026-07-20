# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  
**VibeFinder 1.0 (Hybrid Cosine & Multi-Signal Engine)**

---

## 2. Goal / Task  
This recommender system suggests songs to users based on their listening taste profiles. It analyzes musical attributes (like tempo, energy, and acoustic properties) alongside metadata (genre, artist, popularity) to rank and surface tracks that align with user preferences.

---

## 3. Intended Use and Non-Intended Use  
* **Intended Use:** Designed for classroom exploration, simulated user taste testing, and content-based recommendation modeling. It is well-suited for creating track radios or recommending similar songs to seed playlists.
* **Non-Intended Use:** Not intended for high-security environments, financial forecasting, or professional playlist curation where real-time streaming constraints, commercial licensing, or social graphs are required.

---

## 4. Data Used  
* **Catalog Size:** Tested on a localized catalog (`data/songs.csv`, 10 tracks) and a production-scale Spotify tracks catalog (`docs/tracks.parquet`, 899,224 tracks).
* **Features:** Energy, valence, acousticness, tempo, genres, and release year.
* **Limitations of the Data:** The dataset lacks vocal timbre profiles, instrumental complexity, lyrics, and cultural/societal context. 89% of the dataset initially lacked explicit `track_artists` names, which were resolved using cross-referenced lookup keys with 97.6% accuracy.

---

## 5. Algorithm Summary  
The model calculates a composite score for each candidate song using five weighted factors:
* **Audio Proximity (35%):** We calculate the Euclidean distance over an 8D vector of musical features (including energy, valence, and tempo) to measure musical alignment.
* **Popularity (35%):** We blend 75% track popularity and 25% artist popularity to ensure hit tracks bubble up for popular seeds.
* **Genre Match (15%):** We check if the track shares any base parent genre (e.g. pop, rock, rap) with the user profile or seed track.
* **Artist Match (15%):** We grant points if the track is by the user's favorite artist.
* **Year Proximity (15%):** We prioritize songs released in the same decade or era.

---

## 6. Observed Behavior / Biases  
* **Superstar Loop Bias:** Since popularity represents 35% of the total score, globally famous tracks (like Taylor Swift or Justin Bieber hits) bubble to the top of recommendation lists, occasionally burying lesser-known indie releases.
* **Energy Proximity Limitations:** Uses a linear energy distance gap calculation, which can penalize excellent tracks that lie slightly outside the target range, even if their genre and mood match perfectly.
* **Genre Label Bias:** If a song's genre metadata is missing or sparse, the engine struggles to match it, meaning underrepresented indie tracks are heavily filtered out.

---

## 7. Evaluation Process  

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

### Profile Comparisons:
* **High-Energy Pop vs. Chill Lofi:** High-Energy Pop correctly favors fast, happy pop tracks (Sunrise City, Rooftop Lights) with targets around 0.85 energy. The Chill Lofi profile shifts entirely toward low-energy acoustic structures (Library Rain at 0.35 energy) and filters out pop.
* **Deep Intense Rock vs. High-Energy Pop:** Intense Rock prioritizes Storm Runner (+2.0 for rock) over Pop, even though both have high energy.
* **Adversarial Conflict:** The adversarial pop/sad/high-energy profile outputs Gym Hero as #1 because its high pop-genre weight (+2.0) and high energy similarity outweigh the lack of a "sad" mood match. The system prioritizes the genre structure over mood when they conflict.

---

## 8. Ideas for Improvement  
* **Serendipity Injection:** Implement a random mutation factor (e.g., 5% random genre candidates) to pop the filter bubble and suggest unexpected music.
* **Lyric Vibe Extraction:** Integrate natural language processing on lyric logs to establish deeper mood classifications.
* **Collaborative Filtering:** Transition to a hybrid model that incorporates co-listening matrix data.

---

## 9. Personal Reflection  

* **Biggest Learning Moment:** Implementing the two-pass search logic. Discovering that C++ rapidfuzz truncated results before sorting by popularity taught me that database-level ordering parameters must be established *before* applying fuzzy thresholds.
* **AI Tool Integration:** AI was highly effective in generating vectorized matrix operations and parsing nested genre structures. However, I had to double-check search result deduplication logic as early iterations incorrectly matched tracks by index rather than by unique track identifiers.
* **Surprising Algorithms:** I was surprised by how natural a basic Euclidean distance check over 8 audio parameters feels. Simply checking distance across values like valence and energy produces recommendations that intuitively align with human taste.
* **Next Steps:** If I extended this project, I would implement collaborative filtering based on user playlist overlap and add real-time feedback loops (e.g., "skip" count penalties) to dynamically alter the user's taste centroid.
