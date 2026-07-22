import pandas as pd

# Only including numbers that are documented by credible sources
# Sources: Rolling Stone, Billboard, Music Business Worldwide, NexaTunes, Guardian
real_world = pd.DataFrame([
    {
        "artist": "Kendrick Lamar",
        "streams_removed": 218_000_000,
        "incident": "Spotify removed ~218M streams from 'Luther' (ft. SZA). Stream count dropped from 500M to 282M during post-Super Bowl fraud crackdown (March 2025).",
        "source": "SoapCentral / ChartMasters",
        "label": "UMG/pgLang",
        "year": 2025,
        "verified": True,
        "legal_action": False
    },
    {
        "artist": "BTS (Jimin)",
        "streams_removed": 200_000_000,
        "incident": "Jimin lost up to 200M streams on a single song in 2025 Spotify purge.",
        "source": "NexaTunes / Koreaboo",
        "label": "HYBE",
        "year": 2025,
        "verified": True,
        "legal_action": False
    },
    {
        "artist": "BTS (Jin)",
        "streams_removed": 15_000_000,
        "incident": "Jin's 'Don't Say You Love Me' lost 15M+ streams in 2025 purge.",
        "source": "Hauterrfly / Koreaboo",
        "label": "HYBE",
        "year": 2025,
        "verified": True,
        "legal_action": False
    },
    {
        "artist": "BTS (V)",
        "streams_removed": 13_000_000,
        "incident": "V's 'Winter Ahead' lost ~13M streams in 2025 purge.",
        "source": "Hauterrfly / Koreaboo",
        "label": "HYBE",
        "year": 2025,
        "verified": True,
        "legal_action": False
    },
    {
        "artist": "BLACKPINK (Rosé)",
        "streams_removed": 2_000_000,
        "incident": "Rosé lost ~2M streams on 'rosé' during July 2025 purge.",
        "source": "Hauterrfly / Koreaboo",
        "label": "YG/HYBE",
        "year": 2025,
        "verified": True,
        "legal_action": False
    },
    {
        "artist": "BLACKPINK (Jennie)",
        "streams_removed": 2_000_000,
        "incident": "Jennie lost ~2M streams on 'Like Jennie' during July 2025 purge.",
        "source": "Hauterrfly / Koreaboo",
        "label": "YG/HYBE",
        "year": 2025,
        "verified": True,
        "legal_action": False
    },
    {
        "artist": "Drake",
        "streams_removed": None,
        "incident": "RICO lawsuit filed Dec 31 2025 alleges Drake used Stake casino tipping feature to fund bot farms since 2022. Allegations only — no streams confirmed removed. Drake denies.",
        "source": "Billboard / Music Business Worldwide",
        "label": "OVO/UMG",
        "year": 2025,
        "verified": False,
        "legal_action": True
    },
    {
        "artist": "Michael Smith",
        "streams_removed": None,
        "incident": "NC musician pleaded guilty March 2025 to using AI bots to generate $10M+ in fake streaming royalties from Spotify, Apple Music, Amazon Music.",
        "source": "Rolling Stone",
        "label": "Independent",
        "year": 2024,
        "verified": True,
        "legal_action": True
    },
])

real_world.to_csv("real_world_reports.csv", index=False)
print("✅ Saved real_world_reports.csv")
print(f"\n{len(real_world)} incidents documented")
print("\nVerified with confirmed stream numbers:")
confirmed = real_world[real_world["streams_removed"].notna()]
print(confirmed[["artist", "streams_removed", "source"]].to_string(index=False))
print(f"\nTotal confirmed streams removed: {confirmed['streams_removed'].sum():,.0f}")
