# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

### **Song Features & User Profile**

This system models songs using **multi-dimensional audio features**:

**Song Attributes:**

- **Genre** (categorical): Primary music classification — hard constraint for filtering
- **Mood** (categorical): Emotional context — chill, intense, happy, focused, relaxed, moody
- **Audio Features** (numerical, 0–1 scale):
  - **Energy**: Intensity/activity level → captures workout vs. focus vibes
  - **Valence**: Musical positivity → separates sad from happy songs within a mood
  - **Danceability**: Rhythmic suitability → distinguishes party from contemplative tracks
  - **Acousticness**: Real instruments vs. production → user preference signal
  - **Tempo (BPM)**: Song speed → fine-tunes within mood clusters

*At scale:* This expands to **30+ features** including loudness, speechiness, instrumentalness, liveness, key, mode, release date, popularity, explicit content — plus collaborative signals (skips, saves, playlist co-occurrence) and contextual data (time of day, device, weather).

**UserProfile:**

- **Favorite genre** & **favorite mood**: Primary user intent (40% + 35% weight)
- **Target energy**: How aroused/intense the user wants to feel (15% weight)
- **Listening history** (at scale): Implicit preferences from plays, skips, saves, session length
- **Listening context** (at scale): Time of day, device, location, session type (workout, focus, party, discovery)

### **Scoring Rule: Turning Preferences into Scores**

For each song, we calculate a **composite score** (0–1) that rewards closeness to user preferences:

```
Score = 0.40 × Genre Match 
       + 0.35 × Mood Match 
       + 0.15 × Energy Proximity 
       + 0.10 × Acousticness Preference

Genre Match:  1.0 if song.genre == user.favorite_genre, else 0.5
Mood Match:   1.0 - |song.mood_energy - user.target_energy| (Gaussian or linear distance)
Energy:       Gaussian curve centered on user.target_energy (rewards closeness, not extremes)
Acousticness: User preference boolean weighted lightly
```

**Why this formula?**

- **Genre is weighted highest (40%)**: Categorical mismatch = instant skip; collaborative filtering shows genre is the #1 reason users engage
- **Mood is second (35%)**: Direct signal of user intent; captures emotional context better than valence alone
- **Energy is third (15%)**: Fine-tunes within mood; small changes matter (0.7 energy is very different from 0.3)
- **Acousticness is lightest (10%)**: Preference signal, but lower priority than core mood/energy match

**At scale:** Real systems also incorporate:

- **Novelty penalty**: Down-weight songs the user has heard before
- **Popularity boost**: New releases + trending songs get slight boost (discovery vs. exploitation trade-off)
- **Diversity penalty**: If last 3 recommendations were all indie, decrease indie weight in next batch
- **Cold-start handling**: For new songs, use audio features + collaborative similarity; for new users, use demographic + explicit preference data

### **Ranking Rule: Choosing What to Show**

Once scored, songs are **ranked and personalized**:

```
1. Sort by score (descending)
2. Apply context filters: time of day, device, session type
3. Inject diversity: avoid showing 5 similar songs in a row
4. Add novelty: shuffle within score bands to surface discoveries
5. Personalize order: recent liked songs bubble up slightly
```

**Why separate scoring from ranking?**

- **Scoring** answers: "Is this song good for this user?" (local evaluation)
- **Ranking** answers: "In what order should I show good songs?" (global optimization for engagement)

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

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*:

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this
