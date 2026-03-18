Problem Statement:
Build an AI system that analyzes a user’s browsing history for a selectable time window (last 3, 4, or 5 days) and identifies:
Browsing patterns based on time (hour/day/session behavior)
Dominant website categories (social/media/learning/shopping/etc.)
Behavior clusters (types of sessions)
Deep-learning-based predictions/anomalies (next category prediction or unusual sessions)
Correlation between browsing behavior and RAM usage (system + browser RAM), highlighting memory-heavy patterns and giving actionable recommendations.

Business Use Cases:
Employee productivity analytics (privacy-safe / category-level only): Identify distraction-heavy hours and recommend focus blocks.
Digital wellbeing tools: Detect late-night social media loops and give healthier usage suggestions.
IT performance optimization: Find websites/categories that cause high browser RAM usage and recommend tab/extension management.
Cybersecurity behavior baselining (high-level): Detect unusual browsing sessions that deviate from typical patterns (anomaly detection).
EdTech learner behavior insights: Understand study vs distraction sessions and recommend optimal learning schedules.
Device performance support: Predict RAM spikes and suggest actions before slowdown occurs.
Approach:
Learners should implement the project in 7 modules:
1) Data Collection
Extract browsing history events from browser SQLite DB (Chrome/Edge history).
Log RAM metrics periodically (every 5–10 seconds):
system RAM used/available
browser process RAM (Chrome/Edge total)
Optional (bonus): active tab tracking to estimate true “time spent”.
2) Data Preprocessing
Clean URL → extract domain
Remove/obfuscate sensitive URL parts (query strings)
Map domain → category using a curated mapping table
Convert timestamps to local time; create hour, date, day_name
3) Sessionization (Core Step)
Build sessions using inactivity threshold (e.g., 15 minutes gap = new session)
For each session compute summary stats (counts, ratios, switching behavior)


4) RAM Correlation (Time Alignment)
Join browsing events with RAM logs using nearest timestamp merge
Produce RAM stats per session/category (mean, peak)


5) Pattern Discovery (Unsupervised)
Cluster sessions or days using engineered features
Label clusters with interpretation rules (top categories + time + RAM)


6) Deep Learning (Pick at least one)
Option A: LSTM/GRU Next-Category Prediction
Input: category sequence per session
Output: next category / probability of social-media next
Option B: Autoencoder Anomaly Detection
Input: session feature vector
Output: anomaly score for unusual sessions (time + switching + RAM spikes)


7) Recommendation Engine + Reporting
Generate recommendations from rules + model signals
Output report + optional dashboard
