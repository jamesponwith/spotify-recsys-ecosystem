# Data card

## Sources

| Dataset | Role | Size used | Licence / access |
|---|---|---|---|
| [Spotify Million Playlist Dataset (MPD)](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) | Playlists, titles, track metadata, co-occurrence | 100 slices = 100 000 playlists | Research use; obtained via a public HuggingFace mirror (`jaxliu/Spotify_Million_Playlist_Dataset_Challenge`) |
| [Spotify audio features dump](https://huggingface.co/datasets/ozefe/spotify_audio_features) | Numeric audio features per track id | 255 M rows scanned → 159 308 matched | Public HuggingFace dataset |
| [Spotify tracks dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) | `explicit` flag + genre labels | 114 000 rows | Public HuggingFace dataset |

MPD was collected by Spotify from public US playlists created between January
2010 and October 2017. It is **not** a random sample of listening: it
over-represents US English-language mainstream music of that period, and every
finding here inherits that skew.

## Built catalog

| Quantity | Value |
|---|---|
| Playlists retained | 98 334 |
| Distinct tracks seen | 679 889 |
| Catalog tracks (≥ 4 playlists) | 159 338 |
| Playlist–track interactions | 5 881 943 |
| Median playlist length | 44 tracks |
| Folksonomy tags (≥ 5 playlists) | 4 569 |
| Track–tag entries | 4 369 399 |
| Audio-feature coverage | **99.98 %** of catalog tracks |
| `explicit` flag coverage | **2.76 %** of catalog tracks |

## Processing

1. **Playlist filter.** Keep playlists with 5–250 tracks. Degenerate and
   pathological lengths distort co-occurrence statistics.
2. **Track filter.** Keep tracks appearing on ≥ 4 playlists. Below that,
   co-occurrence is indistinguishable from noise. This drops 76 % of distinct
   tracks but only a small share of *impressions* — it is the long tail of
   one-off additions.
3. **Deduplication.** A track counts once per playlist even if repeated.
4. **Title tokenisation.** Lowercase, strip accents and emoji, drop a
   playlist-specific stoplist (`playlist`, `mix`, `songs`, `fav`, …), emit
   unigrams + bigrams, and canonicalise decade forms (`90s`, `'90s`, `1990s` →
   `1990s`; a bare `00s`/`10s` resolves to the 2000s/2010s, correct for a
   2010–2017 corpus).
5. **Feature join.** Audio features join on the Spotify track id parsed from
   the MPD `track_uri`. Duplicates resolve to the highest-popularity row.

## Known limitations

**The explicit flag is only 2.76 % observed.** This is the sharpest limitation
in the project. `exclude_explicit` filters tracks *known* to be explicit; it
cannot vouch for unlabelled ones. The system reports this on every affected
response rather than implying a guarantee, and the constraint is reported as
`no_known_explicit`. Making this a real guarantee needs a licensed metadata
feed — it is not solvable from these sources.

**No release dates.** Neither source carries a release year, so era targeting
is *inferred from playlist titles*: tracks on playlists named "90s throwbacks"
accumulate a `1990s` tag. This is a genuine behavioural signal but it is noisy
and reflects when listeners *file* music, not when it was released. Treat era
results as "music people associate with the 90s", not "music released in the
1990s".

**Temporal skew.** MPD ends in 2017. There is no music after it, and recency
effects in the data are frozen at that date.

**Popularity skew.** Playlists are heavily popularity-biased. Every metric in
`docs/EVALUATION.md` is reported alongside catalog coverage and Gini precisely
because accuracy alone rewards recommending the head of the distribution.

**Geographic and language skew.** Predominantly US, predominantly English. The
folksonomy vocabulary is English; a non-English query degrades to the lexical
and collaborative channels.

**Cold-start items.** A track on few playlists has a thin tag profile and a
noisy embedding. The `min_track_playlists` filter hides this rather than
solving it; a production system would need content-based audio embeddings to
place genuinely new tracks.

## Privacy

MPD playlists are public and were released by Spotify with user identifiers
removed; only a numeric `pid` remains. Nothing in this pipeline attempts to
re-identify creators, and no per-user modelling is performed — all
personalisation signals here are item-level co-occurrence.

## Reproducing

```bash
make data      # download MPD slices + audio features
make build     # catalog, interactions, folksonomy
```

The build is deterministic given the same inputs and `BuildConfig`; slice order
is sorted numerically rather than by filesystem order so runs are comparable.
