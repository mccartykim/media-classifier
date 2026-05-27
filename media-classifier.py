"""Media classifier for Jellyfin library organization.

Scans source directories and creates categorized symlinks in /srv/media/.
Multi-signal pipeline: anitopy + regex fast-path, AniList/Wikipedia/ffprobe
evidence gathering, scoring, and local LLM arbiter for ambiguous cases.
"""

import argparse
import errno
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from rapidfuzz import fuzz as rf_fuzz
except ImportError:
    rf_fuzz = None

try:
    import anitopy
except ImportError:
    anitopy = None

# Configuration (defaults — overridden by --config or env vars)
SOURCE_DIRS = [
    "/mnt/media-drive/putio/chill.institute",
    "/mnt/media-drive/putio/Items shared with you/Parsimony",
]

MEDIA_BASE = Path("/srv/media")
STATE_DIR = Path(os.environ.get("STATE_DIRECTORY", "/var/lib/media-classifier"))
STATE_FILE = STATE_DIR / "state.json"

FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")

CLASSIFIER_VERSION = 2
CACHE_TTL_DAYS = 30

LLM_SHOW_CACHE_FILE = STATE_DIR / "llm_show_cache.json"
NEEDS_CLASSIFY_REVIEW_LOG = STATE_DIR / "needs-classify-review.log"
_llm_show_lock = threading.Lock()

# Manual aliases for shows whose folder names vary too widely for normalization
# alone (abbreviations, alternate titles). Map any raw or normalized form to a
# canonical folder name. Keys are matched case-insensitively after normalization.
SHOW_ALIASES = {}

TYPE_DIRS = {
    "movie": MEDIA_BASE / "Movies",
    "tv": MEDIA_BASE / "TV Shows",
    "anime": MEDIA_BASE / "Anime",
}

MEDIA_EXTENSIONS = {
    ".mkv", ".mp4", ".avi", ".wmv", ".flv", ".mov", ".m4v",
    ".webm", ".ts", ".m2ts", ".mpg", ".mpeg",
}

SUBTITLE_EXTENSIONS = {
    ".srt", ".ass", ".ssa", ".sub", ".idx", ".sup", ".vtt",
}

# --- Known fansub/encode groups (fast-path → anime) ---
KNOWN_FANSUB_GROUPS = {
    "subsplease", "erai-raws", "horriblesubs", "commie", "gjm",
    "judas", "ssa", "asw", "anime time", "anime-time", "ember",
    "mtbb", "scy", "tsundere", "kametsu", "cbm", "db", "ctr",
    "sallysubs", "lostyears", "bludragon", "reaktor", "cleo",
    "hakata ramen", "yameii", "golumpa", "smugcat", "toonsHub",
    "tenrai-sensei", "breeze", "dhd", "hr", "sam", "pixel",
    "cerberus", "kawaiika-raws", "damedesuyo", "asenshi",
    "vivid", "coalgirls", "thora", "chotab", "beatrice-raws",
    "moozzi2", "yousei-raws", "uccuss",
}

# --- Regex patterns ---
TV_EPISODE_RE = re.compile(r"(?:^|[.\s/])S\d{1,2}E\d{1,3}", re.IGNORECASE)
TV_SEASON_RE = re.compile(r"(?:^|[.\s/])S\d{1,2}(?:[.\s/]|$)", re.IGNORECASE)
TV_SEASON_WORD_RE = re.compile(r"(?:^|[.\s/])Season[.\s]\d{1,2}", re.IGNORECASE)
TV_XN_RE = re.compile(r"(?:^|[.\s/])\d{1,2}x\d{2}", re.IGNORECASE)
MOVIE_YEAR_RE = re.compile(r"[.\s(](?:19|20)\d{2}[.\s)]")
CRC_RE = re.compile(r"\[[\dA-Fa-f]{8}\]")
BRACKET_PREFIX_RE = re.compile(r"^\[(.+?)\]\s")


def has_tv_pattern(name):
    return any(p.search(name) for p in [TV_EPISODE_RE, TV_SEASON_RE, TV_SEASON_WORD_RE, TV_XN_RE])


def load_config(config_path):
    """Load JSON config file and apply settings to globals."""
    global SOURCE_DIRS, MEDIA_BASE, TYPE_DIRS, OLLAMA_HOST, OLLAMA_MODEL, FFPROBE_PATH, SHOW_ALIASES

    with open(config_path) as f:
        cfg = json.load(f)

    if "sourceDirs" in cfg:
        SOURCE_DIRS = cfg["sourceDirs"]
    if "mediaBase" in cfg:
        MEDIA_BASE = Path(cfg["mediaBase"])
    if "categories" in cfg:
        TYPE_DIRS = {k: MEDIA_BASE / v for k, v in cfg["categories"].items()}
    else:
        TYPE_DIRS = {
            "movie": MEDIA_BASE / "Movies",
            "tv": MEDIA_BASE / "TV Shows",
            "anime": MEDIA_BASE / "Anime",
        }
    if "ollamaHost" in cfg:
        OLLAMA_HOST = cfg["ollamaHost"]
    if "ollamaModel" in cfg:
        OLLAMA_MODEL = cfg["ollamaModel"]
    if "ffprobePath" in cfg:
        FFPROBE_PATH = cfg["ffprobePath"]
    if "showAliases" in cfg:
        SHOW_ALIASES = dict(cfg["showAliases"])


# =============================================================================
# State management
# =============================================================================

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            data = json.load(f)
        # Migrate v1 state
        if "version" not in data:
            data = {
                "version": CLASSIFIER_VERSION,
                "anilist_cache": {},
                "wikipedia_cache": {},
                "processed": data.get("processed", {}),
            }
        return data
    return {
        "version": CLASSIFIER_VERSION,
        "anilist_cache": {},
        "wikipedia_cache": {},
        "processed": {},
    }


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_FILE)


def should_reprocess(entry):
    """Re-process if version bumped and confidence was low or unknown."""
    v = entry.get("classifier_version", 1)
    if v >= CLASSIFIER_VERSION:
        return False
    return entry.get("confidence") in ("low", None) or entry.get("type") == "unknown"


def cache_expired(cache_entry):
    fetched = cache_entry.get("fetched_at", "")
    if not fetched:
        return True
    try:
        ts = datetime.fromisoformat(fetched)
        age = datetime.now(timezone.utc) - ts
        return age.days > CACHE_TTL_DAYS
    except (ValueError, TypeError):
        return True


# =============================================================================
# Stage 1: Filename parsing
# =============================================================================

def parse_filename(name):
    """Parse a media filename (or relative path) into structured signals.

    When name contains '/', path components are checked for signals like
    fansub groups and CRCs that often appear in parent directory names.
    """
    # Use the full path for signal detection, filename for primary parsing
    filename = Path(name).name
    full_path = name

    info = {
        "original": name,
        "cleaned_title": None,
        "episode": None,
        "season": None,
        "release_group": None,
        "has_crc": bool(CRC_RE.search(full_path)),
        "has_bracket_prefix": bool(BRACKET_PREFIX_RE.match(filename)) or any(
            BRACKET_PREFIX_RE.match(part) for part in Path(name).parts[:-1]
        ),
        "bracket_group": None,
        "year": None,
        "has_tv_pattern": has_tv_pattern(full_path),
    }

    # Extract bracket group from filename or parent dirs
    bm = BRACKET_PREFIX_RE.match(filename)
    if bm:
        info["bracket_group"] = bm.group(1).strip()
    else:
        for part in Path(name).parts[:-1]:
            bm = BRACKET_PREFIX_RE.match(part)
            if bm:
                info["bracket_group"] = bm.group(1).strip()
                break

    # Try anitopy — feed it the most informative path component
    if anitopy:
        try:
            parsed = anitopy.parse(filename)
            title = parsed.get("anime_title")
            episode = parsed.get("episode_number")

            # For episodic files, prefer parent directory for show title.
            # Episode filenames often contain episode titles ("Crimson Bullet")
            # rather than show titles ("Undead Unluck").
            if len(Path(name).parts) > 1:
                parent_parsed = anitopy.parse(Path(name).parts[-2])
                parent_title = parent_parsed.get("anime_title") or parent_parsed.get("file_name")
                if not info["release_group"]:
                    info["release_group"] = parent_parsed.get("release_group")
                # Use parent title when: filename has no title, title is bare number,
                # or filename is episodic and parent has a real show name
                if parent_title and (
                    not title
                    or re.match(r"^\d+$", title)
                    or (episode and parent_title and len(parent_title) > len(title or ""))
                ):
                    title = parent_title

            info["cleaned_title"] = title or parsed.get("file_name", filename)
            info["episode"] = episode
            info["season"] = parsed.get("anime_season")
            info["release_group"] = info["release_group"] or parsed.get("release_group")
        except Exception:
            pass

    # Regex fallback for title extraction
    if not info["cleaned_title"]:
        # Use filename for title extraction, fall back to parent dir if bare
        title_source = filename
        if len(Path(name).parts) > 1 and re.match(r"^\d+\.\w+$", filename):
            title_source = Path(name).parts[-2]
        # Strip bracket prefix
        clean = BRACKET_PREFIX_RE.sub("", title_source)
        # Strip extension
        clean = re.sub(r"\.\w{2,4}$", "", clean)
        # Strip resolution, codec, source tags
        clean = re.sub(
            r"[.\s](?:\d{3,4}p|(?:x|h)\.?26[45]|HEVC|AVC|AAC|FLAC|BluRay|"
            r"BDRip|WEB-?(?:DL|Rip)|HDRip|DVDRip|AMZN|NF|HULU|CR|10bit|"
            r"HDR|DTS|Atmos|PROPER|REPACK|REMUX).*$",
            "", clean, flags=re.IGNORECASE,
        )
        # Strip season/episode markers
        clean = re.sub(r"[.\s]S\d{1,2}(?:E\d{1,3})?.*$", "", clean, flags=re.IGNORECASE)
        # Strip year
        clean = re.sub(r"[.\s(](?:19|20)\d{2}[.\s)]?.*$", "", clean)
        # Dots to spaces
        clean = clean.replace(".", " ").strip()
        info["cleaned_title"] = clean

    # Extract year from full path (find all candidates, pick the valid one)
    for ym in re.finditer(r"(?:19|20)\d{2}", full_path):
        year = int(ym.group(0))
        if 1920 <= year <= 2030:
            info["year"] = year
            break

    return info


# =============================================================================
# Stage 2: Fast-path classification
# =============================================================================

def classify_fast_path(info):
    """High-confidence classification without API calls.
    Returns (type, confidence) or (None, None).
    """
    group = (info.get("release_group") or info.get("bracket_group") or "").lower()

    # Known fansub group → anime
    if group and group in KNOWN_FANSUB_GROUPS:
        return "anime", "high"

    # anitopy parsed + release group + CRC → anime
    if info.get("release_group") and info.get("has_crc") and info.get("episode"):
        return "anime", "high"

    # Bracket prefix + CRC → anime
    if info.get("has_bracket_prefix") and info.get("has_crc"):
        return "anime", "high"

    # CRC hash + episode = fansub release → anime
    if info.get("has_crc") and info.get("episode"):
        return "anime", "high"

    # S01E01 pattern, no bracket prefix, no anime signals → tv
    if info.get("has_tv_pattern") and not info.get("has_bracket_prefix") and not info.get("has_crc"):
        if group not in KNOWN_FANSUB_GROUPS:
            return "tv", "high"

    # Year in name, no episode markers, no bracket prefix → movie
    if info.get("year") and not info.get("has_tv_pattern") and not info.get("episode"):
        if not info.get("has_bracket_prefix"):
            return "movie", "high"

    return None, None


# =============================================================================
# Stage 3: Evidence gathering
# =============================================================================

def normalize_title(title):
    """Normalize title for cache keys and API queries."""
    if not title:
        return ""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


_last_anilist_call = 0.0
_last_wikipedia_call = 0.0

def query_anilist(title, state):
    """Query AniList GraphQL API for anime matches."""
    global _last_anilist_call
    key = normalize_title(title)
    if not key:
        return None

    # Check cache
    cached = state.get("anilist_cache", {}).get(key)
    if cached and not cache_expired(cached):
        return cached.get("results")

    # Rate limit: 1 second between non-cached calls
    elapsed = time.monotonic() - _last_anilist_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _last_anilist_call = time.monotonic()

    query = """
    query ($search: String) {
      Page(page: 1, perPage: 5) {
        media(search: $search, type: ANIME, sort: POPULARITY_DESC) {
          title { romaji english native }
          format
          episodes
          popularity
          countryOfOrigin
          genres
          averageScore
        }
      }
    }
    """
    payload = json.dumps({"query": query, "variables": {"search": title}}).encode()

    try:
        req = urllib.request.Request(
            "https://graphql.anilist.co",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "MediaClassifier/2.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            results = data.get("data", {}).get("Page", {}).get("media", [])

        # Cache
        state.setdefault("anilist_cache", {})[key] = {
            "results": results,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        return results
    except Exception as e:
        print(f"  [WARN] AniList query failed for '{title}': {e}", file=sys.stderr)
        return None


def query_wikipedia(title, state):
    """Query Wikipedia MediaWiki API for context."""
    global _last_wikipedia_call
    key = normalize_title(title)
    if not key:
        return None

    cached = state.get("wikipedia_cache", {}).get(key)
    if cached and not cache_expired(cached):
        return cached.get("summary")

    # Rate limit: 1 second between non-cached calls
    elapsed = time.monotonic() - _last_wikipedia_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _last_wikipedia_call = time.monotonic()

    try:
        # Search for the title
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            f"action=query&list=search&srsearch={urllib.request.quote(title)}"
            "&srnamespace=0&srlimit=3&format=json"
        )
        req = urllib.request.Request(search_url, headers={"User-Agent": "MediaClassifier/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        results = data.get("query", {}).get("search", [])
        if not results:
            state.setdefault("wikipedia_cache", {})[key] = {
                "summary": None,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            return None

        # Get extract of first result
        page_id = results[0]["pageid"]
        extract_url = (
            "https://en.wikipedia.org/w/api.php?"
            f"action=query&pageids={page_id}&prop=extracts"
            "&exintro=true&explaintext=true&exsentences=3&format=json"
        )
        req = urllib.request.Request(extract_url, headers={"User-Agent": "MediaClassifier/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        pages = data.get("query", {}).get("pages", {})
        extract = pages.get(str(page_id), {}).get("extract", "")

        summary = extract[:500] if extract else None
        state.setdefault("wikipedia_cache", {})[key] = {
            "summary": summary,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        return summary
    except Exception as e:
        print(f"  [WARN] Wikipedia query failed for '{title}': {e}", file=sys.stderr)
        return None


def probe_file(filepath):
    """Run ffprobe on a media file and extract classification signals."""
    if not Path(filepath).exists():
        return None

    try:
        result = subprocess.run(
            [
                FFPROBE_PATH, "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(filepath),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})

        audio_langs = []
        sub_formats = []
        sub_langs = []

        for s in streams:
            codec_type = s.get("codec_type")
            lang = s.get("tags", {}).get("language", "und")

            if codec_type == "audio":
                audio_langs.append(lang)
            elif codec_type == "subtitle":
                sub_formats.append(s.get("codec_name", "unknown"))
                sub_langs.append(lang)

        duration = None
        dur_str = fmt.get("duration")
        if dur_str:
            try:
                duration = float(dur_str)
            except (ValueError, TypeError):
                pass

        return {
            "audio_langs": audio_langs,
            "sub_formats": sub_formats,
            "sub_langs": sub_langs,
            "duration": duration,
            "num_audio": len(audio_langs),
            "num_subs": len(sub_formats),
        }
    except Exception as e:
        print(f"  [WARN] ffprobe failed for '{filepath}': {e}", file=sys.stderr)
        return None


def gather_evidence(info, filepath, state):
    """Gather all evidence signals for ambiguous titles."""
    title = info.get("cleaned_title", "")
    evidence = {"anilist": None, "ffprobe": None, "wikipedia": None}

    evidence["anilist"] = query_anilist(title, state)
    evidence["ffprobe"] = probe_file(filepath)
    evidence["wikipedia"] = query_wikipedia(title, state)

    return evidence


# =============================================================================
# Stage 4: Score & decide
# =============================================================================

def score_evidence(info, evidence):
    """Score evidence to classify media. Returns (type, confidence) or (None, None)."""
    scores = {"anime": 0.0, "tv": 0.0, "movie": 0.0}
    reasons = []

    ffprobe = evidence.get("ffprobe")
    anilist = evidence.get("anilist")
    wikipedia = evidence.get("wikipedia")

    # --- AniList signals ---
    if anilist:
        top = anilist[0] if anilist else None
        if top:
            pop = top.get("popularity", 0)
            origin = top.get("countryOfOrigin", "")
            fmt = top.get("format", "")

            # Strong match: Japanese origin + high popularity
            if origin == "JP" and pop > 10000:
                scores["anime"] += 3.0
                reasons.append(f"anilist: JP anime, pop={pop}")
            elif origin == "JP":
                scores["anime"] += 2.0
                reasons.append(f"anilist: JP anime, pop={pop}")
            elif pop > 5000:
                scores["anime"] += 1.0
                reasons.append(f"anilist: non-JP, pop={pop}")

            # Movie format on AniList (anime movie)
            if fmt == "MOVIE":
                scores["anime"] += 1.0
                reasons.append("anilist: movie format")
    else:
        # No AniList result = negative signal for anime
        scores["tv"] += 1.0
        scores["movie"] += 0.5
        reasons.append("anilist: no match")

    # --- ffprobe signals ---
    if ffprobe:
        audio = ffprobe.get("audio_langs", [])
        subs = ffprobe.get("sub_formats", [])
        duration = ffprobe.get("duration")

        # Japanese audio
        if "jpn" in audio or "ja" in audio or "jap" in audio:
            scores["anime"] += 3.0
            reasons.append("ffprobe: JP audio")

        # Non-Japanese, non-English audio → likely not anime
        non_jp_en = [l for l in audio if l not in ("jpn", "ja", "jap", "eng", "en", "und")]
        if non_jp_en:
            scores["tv"] += 1.5
            scores["anime"] -= 1.0
            reasons.append(f"ffprobe: non-JP/EN audio ({', '.join(non_jp_en)})")

        # ASS/SSA subtitles = fansub
        ass_subs = [s for s in subs if s in ("ass", "ssa")]
        if ass_subs:
            scores["anime"] += 1.5
            reasons.append("ffprobe: ASS/SSA subs (fansub)")

        # Duration
        if duration:
            if duration < 1800:  # < 30 min = episode
                scores["anime"] += 0.5
                scores["tv"] += 0.5
                reasons.append(f"ffprobe: short ({duration / 60:.0f}min)")
            elif duration > 5400:  # > 90 min = movie
                scores["movie"] += 2.0
                reasons.append(f"ffprobe: long ({duration / 60:.0f}min)")

    # --- Wikipedia signals ---
    if wikipedia:
        wiki_lower = wikipedia.lower()
        jp_indicators = ["japanese", "manga", "anime", "light novel", "visual novel"]
        western_indicators = ["american", "british", "danish", "french", "german",
                              "canadian", "australian", "netflix", "hbo"]

        if any(ind in wiki_lower for ind in jp_indicators):
            scores["anime"] += 2.0
            reasons.append("wikipedia: JP media context")
        if any(ind in wiki_lower for ind in western_indicators):
            scores["tv"] += 1.5
            scores["movie"] += 0.5
            reasons.append("wikipedia: western media context")

    # --- Structural signals ---
    if info.get("has_tv_pattern"):
        scores["tv"] += 1.0
        scores["anime"] += 0.5  # Anime can have S01E01 too

    if info.get("year") and not info.get("has_tv_pattern") and not info.get("episode"):
        scores["movie"] += 1.5

    # --- Decision ---
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    second_best = sorted(scores.values(), reverse=True)[1]
    margin = best_score - second_best

    if best_score >= 3.0 and margin >= 2.0:
        return best_type, "high", scores, reasons
    if best_score >= 2.0 and margin >= 1.0:
        return best_type, "medium", scores, reasons

    return None, None, scores, reasons


# =============================================================================
# Stage 5: LLM arbiter
# =============================================================================

def classify_llm(info, evidence, scores, reasons):
    """Use local LLM via Ollama as final arbiter for ambiguous cases."""
    # Build context prompt with all evidence
    parts = [f"Classify this media file for a Jellyfin library."]
    parts.append(f"Filename: {info['original']}")

    if info.get("cleaned_title"):
        parts.append(f"Parsed title: {info['cleaned_title']}")

    # AniList evidence
    anilist = evidence.get("anilist")
    if anilist and len(anilist) > 0:
        top = anilist[0]
        titles = top.get("title", {})
        title_str = titles.get("english") or titles.get("romaji") or "unknown"
        parts.append(
            f"AniList: '{title_str}' ({top.get('countryOfOrigin', '?')}, "
            f"format={top.get('format', '?')}, eps={top.get('episodes', '?')}, "
            f"popularity={top.get('popularity', '?')})"
        )
    else:
        parts.append("AniList: No anime match found")

    # ffprobe evidence
    ffprobe = evidence.get("ffprobe")
    if ffprobe:
        dur = ffprobe.get("duration")
        dur_str = f"{dur / 60:.0f}min" if dur else "unknown"
        audio_str = ", ".join(ffprobe.get("audio_langs", [])) or "none"
        sub_fmts = ", ".join(ffprobe.get("sub_formats", [])) or "none"
        parts.append(f"ffprobe: duration={dur_str}, audio=[{audio_str}], sub_formats=[{sub_fmts}]")
    else:
        parts.append("ffprobe: no data")

    # Wikipedia evidence
    wikipedia = evidence.get("wikipedia")
    if wikipedia:
        parts.append(f"Wikipedia: {wikipedia[:300]}")
    else:
        parts.append("Wikipedia: no relevant result")

    # Current scores
    parts.append(f"Scoring signals: {', '.join(reasons) if reasons else 'none'}")
    if scores:
        parts.append(f"Current scores: anime={scores.get('anime', 0):.1f}, "
                      f"tv={scores.get('tv', 0):.1f}, movie={scores.get('movie', 0):.1f}")

    parts.append(
        '\nBased on ALL evidence above, classify as exactly one of: anime, tv, movie.\n'
        'Respond with ONLY a JSON object: {"category": "anime"|"tv"|"movie"} /no_think'
    )

    prompt = "\n".join(parts)

    try:
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": {"type": "object", "properties": {"category": {"type": "string", "enum": ["anime", "tv", "movie"]}}, "required": ["category"]},
            "options": {"temperature": 0.1, "num_predict": 50},
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            text = data.get("response", "").strip()

        # Parse JSON response
        result = json.loads(text)
        category = result.get("category", "").lower()
        if category in TYPE_DIRS:
            return category, "medium"
    except Exception as e:
        print(f"  [WARN] LLM arbiter failed: {e}", file=sys.stderr)

    return None, None


# =============================================================================
# Pipeline orchestration
# =============================================================================

def classify(name, filepath, state):
    """Run the full classification pipeline.
    Returns (type, confidence, signals_dict).
    """
    # Stage 1: Parse filename
    info = parse_filename(name)

    # Stage 2: Fast-path
    media_type, confidence = classify_fast_path(info)
    if media_type:
        return media_type, confidence, {"method": "fast_path"}

    # Stage 3: Gather evidence
    evidence = gather_evidence(info, filepath, state)

    # Stage 4: Score
    media_type, confidence, scores, reasons = score_evidence(info, evidence)
    if media_type:
        signals = {
            "method": "scored",
            "scores": scores,
            "reasons": reasons,
        }
        ffprobe = evidence.get("ffprobe")
        if ffprobe:
            signals["ffprobe_lang"] = ffprobe.get("audio_langs", [])
            signals["ffprobe_subs"] = ffprobe.get("sub_formats", [])
        signals["anilist"] = bool(evidence.get("anilist"))
        return media_type, confidence, signals

    # Stage 5: LLM arbiter
    media_type, confidence = classify_llm(info, evidence, scores, reasons)
    if media_type:
        signals = {
            "method": "llm_arbiter",
            "model": OLLAMA_MODEL,
            "scores": scores,
            "reasons": reasons,
        }
        return media_type, confidence, signals

    # Stage 6: Unclassified
    return "unknown", "none", {"method": "unclassified", "scores": scores, "reasons": reasons}


# =============================================================================
# Symlink management
# =============================================================================

_BONUS_DIRS = {
    "extras", "featurettes", "deleted scenes", "behind the scenes",
    "specials", "bonus", "bonus features", "interviews", "trailers",
    "sample", "subs", "subtitles", "webisodes", "podcasts encoded",
    "mini video podcasts", "movie featurettes", "other",
    # Specific extras that appear as directory names
    "3d model commentary", "3d models with commentary", "animatics",
    "alternate ending", "intro commercials", "international clips",
}

# Pattern matching "S08 1080p Bluray", "S01 480p DVD", etc.
_SEASON_PREFIX_RE = re.compile(r"^S(\d{1,2})\b", re.IGNORECASE)


def _is_bonus_dir(name):
    """Check if a directory name looks like bonus/extras content."""
    lower = name.lower()
    if lower in _BONUS_DIRS:
        return True
    # Match patterns like "Featurettes 5.2", "Deleted Scenes 1"
    base = re.sub(r"\s+[\d.]+$", "", lower).strip()
    if base in _BONUS_DIRS:
        return True
    return False


def _is_season_dir(name):
    """Check if a directory name is a season directory.

    Matches: "Season 1", "S08 1080p Bluray", "S01 480p DVD", etc.
    """
    if re.match(r"^Season\s+(\d+)$", name, re.IGNORECASE):
        return True
    if _SEASON_PREFIX_RE.match(name):
        return True
    return False


def _season_from_dir(name):
    """Extract season number from a season directory name."""
    m = re.match(r"^Season\s+(\d+)$", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = _SEASON_PREFIX_RE.match(name)
    if m:
        return int(m.group(1))
    return None


def _infer_show_folder(source):
    """Infer show name and season from the source file's directory structure.

    Returns (show_name, season_num_or_None).
    Uses the parent directory name as the show folder name when it looks like
    a show/season directory. Falls back to anitopy parsing.
    """
    source_root_names = {Path(d).name for d in SOURCE_DIRS}

    # Get season/episode from filename
    season_match = re.search(r"S(\d{1,2})", source.name, re.IGNORECASE)
    file_season = int(season_match.group(1)) if season_match else None

    parent = source.parent
    parent_name = parent.name

    # Check if parent is a season directory ("Season 1", "S08 1080p Bluray", etc.)
    if _is_season_dir(parent_name):
        season_num = _season_from_dir(parent_name)
        grandparent = parent.parent
        # If the season dir lives inside a bonus dir (e.g. Extras/Season 3/),
        # this is bonus content — skip it entirely
        if _is_bonus_dir(grandparent.name):
            return None, None
        if grandparent.name and grandparent.name not in source_root_names:
            return _clean_show_name(grandparent.name), season_num
        return None, None

    # Walk up past bonus/extras directories to find the real show
    if _is_bonus_dir(parent_name):
        cur = parent.parent
        while cur.name and (_is_bonus_dir(cur.name) or _is_season_dir(cur.name)) and cur.name not in source_root_names:
            cur = cur.parent
        if cur.name and cur.name not in source_root_names:
            # Return None for season to skip these files (they're extras)
            return None, None
        return None, None

    # Skip if parent is a source root directory (e.g., "chill.institute", "Parsimony")
    if parent_name in source_root_names:
        # File is loose in a source root — extract show name from filename
        show_name = _show_name_from_filename(source.name)
        if show_name:
            return show_name, file_season
        return None, None

    # Parent is not a season dir — it's likely the show dir itself
    show_name = _clean_show_name(parent_name)
    if show_name:
        return show_name, file_season

    return None, None


def _show_name_from_filename(filename):
    """Extract show name from a media filename for folder naming.

    Used when a file is loose in a source root directory with no parent
    folder to derive the show name from.
    """
    stem = Path(filename).stem
    if anitopy:
        try:
            parsed = anitopy.parse(filename)
            title = parsed.get("anime_title")
            if title:
                return title
        except Exception:
            pass
    # Fallback: strip S01E01 and everything after
    cleaned = re.sub(r"[.\s]S\d{2}E\d{2}.*", "", stem, flags=re.IGNORECASE)
    # Replace dots with spaces
    cleaned = cleaned.replace(".", " ").strip()
    return cleaned if cleaned else None


def _clean_show_name(name):
    """Clean up a directory name to use as a Jellyfin show folder name.

    Strips release info like resolution, codec, group tags from folder names
    while preserving the show title and year.
    """
    if not name:
        return None

    # Try anitopy for structured parsing
    if anitopy:
        try:
            parsed = anitopy.parse(name)
            title = parsed.get("anime_title")
            if title:
                # Preserve year if present
                year_match = re.search(r"\(?((?:19|20)\d{2})\)?", name)
                if year_match and year_match.group(1) not in title:
                    title = f"{title} ({year_match.group(1)})"
                return title
        except Exception:
            pass

    # Fallback: strip common release info patterns
    cleaned = name
    # Remove "Season N-M" or "S01-S05" suffixes
    cleaned = re.sub(r"\s*Season\s+\d[\d-]*(?:\s+S\d{2}(?:-S\d{2})?)?(?:\s*\(.*\))*\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*S\d{2}(?:-S\d{2})?\s*(?:\(.*\))*\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    # Remove resolution/codec suffix in parens
    cleaned = re.sub(r"\s*\((?:\d{3,4}p|BluRay|WEB|HEVC|x26[45]).*\)$", "", cleaned, flags=re.IGNORECASE).strip()
    # Remove bracket groups like [eztv.re]
    cleaned = re.sub(r"\s*\[.*?\]\s*$", "", cleaned).strip()
    # Remove trailing dots/dashes
    cleaned = cleaned.rstrip(".-_ ")

    return cleaned if cleaned else name


# Release tokens that mark the boundary between show title and release info.
# When found in a normalized key, everything from that token onward is dropped.
_RELEASE_BOUNDARY_TOKENS = (
    "season", "complete", "bluray", "blu ray", "web dl", "webdl", "webrip",
    "hdtv", "dvdrip", "hdrip", "brrip", "amzn", "nf", "hmax", "atvp",
    "x264", "x265", "h264", "h265", "hevc", "xvid", "aac", "ac3",
    "1080p", "720p", "480p", "2160p", "4k",
)


def _normalize_show_key(name):
    """Produce a comparison key for show-name dedup.

    Collapses case, '&'/'and', year suffixes, punctuation, and trailing
    release-tag noise so that 'Law & Order Special Victims Unit (1999)' and
    'Law and Order SVU Season 13 Complete WEB x264' collapse to comparable
    forms (modulo abbreviations, which SHOW_ALIASES handles).
    """
    if not name:
        return ""
    s = name.lower()
    s = s.replace("&", " and ")
    # Strip parenthesized or bare 4-digit years
    s = re.sub(r"\((?:19|20)\d{2}\)", " ", s)
    s = re.sub(r"\b(?:19|20)\d{2}\b", " ", s)
    # Strip S01/S01-S05/SxxExx tokens
    s = re.sub(r"\bs\d{1,2}(?:[-e]s?\d{1,3})?\b", " ", s)
    # Replace any non-alphanumeric with space
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())
    # Cut at first release-boundary token so trailing release info doesn't
    # poison the key. Keep everything before it as the title.
    for tok in _RELEASE_BOUNDARY_TOKENS:
        idx = s.find(f" {tok} ")
        if s.startswith(f"{tok} "):
            idx = 0
        if s.endswith(f" {tok}"):
            end_idx = len(s) - len(tok) - 1
            if idx == -1 or end_idx < idx:
                idx = end_idx
        if idx == 0:
            s = ""
            break
        if idx > 0:
            s = s[:idx].strip()
    return " ".join(s.split())


def _load_llm_show_cache():
    """Load the LLM show-verification cache from disk."""
    try:
        if LLM_SHOW_CACHE_FILE.exists():
            with open(LLM_SHOW_CACHE_FILE) as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_llm_show_cache(cache):
    """Persist the LLM show-verification cache to disk."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(LLM_SHOW_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass


def _log_needs_review(candidate, siblings, reason):
    """Append a line to needs-classify-review.log when LLM verification fails."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        line = f"[{ts}] candidate={candidate!r} siblings={siblings!r} reason={reason}\n"
        with open(NEEDS_CLASSIFY_REVIEW_LOG, "a") as f:
            f.write(line)
    except OSError:
        pass


def _llm_verify_new_show(candidate, siblings):
    """Ask the LLM whether candidate is actually one of the existing sibling shows.

    Returns the sibling name to reuse (MATCH N) or the original candidate (NEW).
    On LLM failure, logs to needs-classify-review.log and returns candidate unchanged.
    """
    # Build cache key: (normalized-candidate, sorted-tuple-of-sibling-normalized-keys)
    cand_key = _normalize_show_key(candidate)
    sib_keys = tuple(sorted(_normalize_show_key(s) for s in siblings))
    cache_key = f"{cand_key}|{'|'.join(sib_keys)}"

    cache = _load_llm_show_cache()
    if cache_key in cache:
        decision = cache[cache_key]
        if decision.get("verdict") == "MATCH" and decision.get("match_name") in siblings:
            return decision["match_name"]
        # NEW or stale cache (sibling list changed) → return candidate
        return candidate

    # Build prompt
    sibling_lines = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(siblings))
    prompt = (
        "You are a media library deduplicator. A new file is about to be classified "
        f'as a NEW show called "{candidate}". The following existing shows in the '
        "same library type have similar-looking names:\n\n"
        f"{sibling_lines}\n\n"
        "Is the new show actually one of the existing entries? Reply with EXACTLY "
        "one line:\n"
        '  MATCH <number>    if it is the same show as that entry (e.g., "MATCH 2")\n'
        "  NEW              if it is a different show\n\n"
        "If unsure, reply NEW. Do not invent shows that are not in the list."
    )

    with _llm_show_lock:
        try:
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 30},
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            text = data.get("response", "").strip()
        except Exception as e:
            _log_needs_review(candidate, siblings, f"llm_error: {e}")
            return candidate

    # Parse response
    match = re.match(r"^MATCH\s+(\d+)\s*$", text, re.IGNORECASE)
    if match:
        idx = int(match.group(1)) - 1  # 1-indexed in prompt
        if 0 <= idx < len(siblings):
            cache[cache_key] = {"verdict": "MATCH", "match_name": siblings[idx]}
            _save_llm_show_cache(cache)
            return siblings[idx]
        # Out-of-range index → log and fall through
        _log_needs_review(candidate, siblings, f"llm_match_index_out_of_range: {text!r}")
        cache[cache_key] = {"verdict": "NEW"}
        _save_llm_show_cache(cache)
        return candidate

    if text.upper().startswith("NEW"):
        cache[cache_key] = {"verdict": "NEW"}
        _save_llm_show_cache(cache)
        return candidate

    # Unparseable response
    _log_needs_review(candidate, siblings, f"llm_unparseable: {text!r}")
    return candidate


def _canonical_show_dir(target_dir, candidate):
    """Map a candidate show folder name to a canonical name.

    Applies SHOW_ALIASES first (so abbreviations like 'SVU' resolve), then
    checks for an existing directory in target_dir whose normalized key
    matches the candidate's — if so, reuses that name. Otherwise returns
    the candidate unchanged.
    """
    if not candidate:
        return candidate

    key = _normalize_show_key(candidate)

    # Alias lookup: keys may be raw strings or normalized keys
    if SHOW_ALIASES:
        if candidate in SHOW_ALIASES:
            return SHOW_ALIASES[candidate]
        for alias_src, alias_dst in SHOW_ALIASES.items():
            if _normalize_show_key(alias_src) == key:
                return alias_dst

    # Existing-dir lookup
    try:
        existing = [p.name for p in target_dir.iterdir() if p.is_dir()]
    except (FileNotFoundError, NotADirectoryError):
        existing = []

    for name in existing:
        if name == candidate:
            return candidate
        if _normalize_show_key(name) == key:
            return name

    return candidate


def _count_media_files(d):
    """Count media files anywhere under d."""
    count = 0
    try:
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS:
                count += 1
    except OSError:
        pass
    return count


def _season_dirs(show_dir):
    """List 'Season N' / 'S0N ...' subdirs of show_dir as {season_num: path}.

    Subdirs that don't parse as seasons are ignored — only collisions on the
    same season number are interesting for the merge tool.
    """
    out = {}
    try:
        for child in show_dir.iterdir():
            if not child.is_dir():
                continue
            if _is_season_dir(child.name):
                n = _season_from_dir(child.name)
                if n is not None:
                    out.setdefault(n, []).append(child)
    except OSError:
        pass
    return out


def _season_collisions(dirs):
    """Given a group's [(path, count), ...], return {season_num: [paths]} for
    seasons that appear in >1 of the group's show dirs.
    """
    by_season = {}
    for path, _ in dirs:
        for n, paths in _season_dirs(path).items():
            by_season.setdefault(n, []).extend(paths)
    return {n: ps for n, ps in by_season.items() if len(ps) > 1}


def _scan_dupe_groups(aliases=None):
    """Group sibling directories under each TYPE_DIR by normalized show key.

    If `aliases` is given, directories matching an alias are re-keyed under the
    alias canonical's normalized key. This collapses cross-key collisions
    (e.g. "Law and Order SVU" and "Law & Order Special Victims Unit") into a
    single group when an alias bridges them.

    Returns (within_type, cross_type, alias_canonical_by_key):
      within_type: {type_name: [(key, [(path, file_count), ...]), ...]}  groups with >1 dir
      cross_type:  [(key, [(type_name, path, file_count), ...])]         keys spanning >1 type
      alias_canonical_by_key: {effective_key: alias_canonical_name}
    """
    within_type = {}
    by_key_global = {}  # key -> [(type_name, path, count)]
    alias_canonical_by_key = {}

    for type_name, type_dir in TYPE_DIRS.items():
        if not type_dir.is_dir():
            continue
        by_key = {}
        try:
            children = sorted(type_dir.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            spec = _alias_lookup(aliases, child.name) if aliases else None
            if spec and spec.get("canonical"):
                key = _normalize_show_key(spec["canonical"])
                alias_canonical_by_key[key] = spec["canonical"]
            else:
                key = _normalize_show_key(child.name)
            if not key:
                continue
            count = _count_media_files(child)
            by_key.setdefault(key, []).append((child, count))
            by_key_global.setdefault(key, []).append((type_name, child, count))
        groups = sorted(
            ((k, v) for k, v in by_key.items() if len(v) > 1),
            key=lambda kv: kv[0],
        )
        if groups:
            within_type[type_name] = groups

    cross_type = sorted(
        (
            (k, v) for k, v in by_key_global.items()
            if len({t for t, _, _ in v}) > 1
        ),
        key=lambda kv: kv[0],
    )
    return within_type, cross_type, alias_canonical_by_key


def _suggest_canonical(entries):
    """Pick the (path, file_count) entry with the most files; tiebreak alphabetically."""
    return sorted(entries, key=lambda e: (-e[1], e[0].name))[0]


def report_duplicates(json_output=False):
    """Print a report of duplicate show directories across TYPE_DIRs.

    Within-type groups: sibling dirs that collapse to the same _normalize_show_key.
    Cross-type groups: same normalized key appears under more than one TYPE_DIR
    (e.g. a show split between TV Shows/ and Anime/).
    """
    within_type, cross_type, _ = _scan_dupe_groups()

    if json_output:
        out = {
            "within_type": {
                tn: [
                    {
                        "key": key,
                        "canonical_suggested": _suggest_canonical(dirs)[0].name,
                        "directories": [
                            {"path": str(p), "files": c} for p, c in dirs
                        ],
                        "season_collisions": {
                            str(n): [str(p) for p in ps]
                            for n, ps in sorted(_season_collisions(dirs).items())
                        },
                    }
                    for key, dirs in groups
                ]
                for tn, groups in within_type.items()
            },
            "cross_type": [
                {
                    "key": key,
                    "directories": [
                        {"type": t, "path": str(p), "files": c}
                        for t, p, c in entries
                    ],
                }
                for key, entries in cross_type
            ],
        }
        print(json.dumps(out, indent=2))
        return

    total_within = 0
    total_season_collisions = 0
    for type_name, groups in within_type.items():
        print(f"\n=== {type_name.upper()} ({TYPE_DIRS[type_name]}) ===")
        for key, dirs in groups:
            total_within += 1
            canonical_path = _suggest_canonical(dirs)[0]
            print(f"\nSHOW: {key}")
            for path, count in dirs:
                mark = "  ← suggested canonical" if path == canonical_path else ""
                print(f"  {path.name}/  ({count} files){mark}")
            collisions = _season_collisions(dirs)
            for n, paths in sorted(collisions.items()):
                total_season_collisions += 1
                print(f"  ! Season {n} present in {len(paths)} dirs:")
                for p in paths:
                    print(f"      {p}")

    if cross_type:
        print("\n=== CROSS-TYPE DUPLICATES ===")
        for key, entries in cross_type:
            print(f"\nSHOW: {key}")
            for t, p, c in entries:
                print(f"  [{t}] {p.name}/  ({c} files)")

    print(
        f"\nTotal within-type duplicate groups: {total_within}"
        f"\nTotal cross-type duplicate groups:  {len(cross_type)}"
        f"\nTotal season-level collisions:      {total_season_collisions}"
    )


# =============================================================================
# Phase 2: merge duplicate show dirs (--merge-dupes)
# =============================================================================

def _load_aliases(path):
    """Load alias spec from JSON or YAML. Returns {variant: {canonical, type}}.

    File format::

        {
          "aliases": {
            "Law and Order SVU 1999": {
              "canonical": "Law & Order Special Victims Unit (1999)",
              "type": "tv"
            },
            "Undead Unluck": {"type": "anime"}
          }
        }

    YAML accepted if PyYAML is installed. A bare string value is treated as
    ``{"canonical": <string>}``.
    """
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise SystemExit(
                f"PyYAML required to load {path!r}. Install pyyaml or convert to JSON."
            )
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)

    raw = data.get("aliases", {}) if isinstance(data, dict) else {}
    out = {}
    for variant, spec in raw.items():
        if isinstance(spec, str):
            out[variant] = {"canonical": spec, "type": None}
        elif isinstance(spec, dict):
            out[variant] = {
                "canonical": spec.get("canonical") or variant,
                "type": spec.get("type"),
            }
    return out


def _alias_lookup(aliases, name):
    """Find alias spec for a directory name. Matches raw name or normalized key."""
    if not aliases or not name:
        return None
    if name in aliases:
        return aliases[name]
    key = _normalize_show_key(name)
    if not key:
        return None
    for variant, spec in aliases.items():
        if _normalize_show_key(variant) == key:
            return spec
    return None


_YEAR_PAREN_RE = re.compile(r"\((?:19|20)\d{2}\)")


def _pick_canonical(entries, prefer_year_form=False, override_name=None,
                    alias_preferred_name=None):
    """Choose canonical (path, count) from [(path, count), ...].

    Priority:
      1. ``override_name`` matches a path's basename (user --canonical flag)
      2. ``alias_preferred_name`` matches a path's basename (from aliases file)
      3. highest file count
      4. ``prefer_year_form`` puts ``Name (YYYY)`` ahead of bare ``Name``
      5. alphabetical
    """
    if override_name:
        for e in entries:
            if e[0].name == override_name:
                return e
    if alias_preferred_name:
        for e in entries:
            if e[0].name == alias_preferred_name:
                return e

    def sort_key(e):
        path, count = e
        has_year = bool(_YEAR_PAREN_RE.search(path.name))
        year_rank = 0 if (prefer_year_form and has_year) else 1
        return (-count, year_rank, path.name)

    return sorted(entries, key=sort_key)[0]


def _parse_canonical_override(spec):
    """Parse 'KEY=NAME' into (normalized_key, name). Raises SystemExit on bad input."""
    if "=" not in spec:
        raise SystemExit(f"--canonical expects 'KEY=DIR_NAME', got: {spec!r}")
    key_part, name = spec.split("=", 1)
    return _normalize_show_key(key_part.strip()), name.strip()


def _move_path(src, dst):
    """Move src -> dst, recreating symlinks across filesystem boundaries.

    rename(2) fails with EXDEV on cross-fs moves; for symlinks we can recreate
    the link target on the other side and unlink the source.
    """
    try:
        src.rename(dst)
    except OSError as e:
        if e.errno == errno.EXDEV and src.is_symlink():
            target = os.readlink(src)
            os.symlink(target, dst)
            src.unlink()
        else:
            raise


def _merge_show_dir(src, dst, *, dry_run=True):
    """Move all entries from src into dst preserving relative layout.

    Existing entries at the destination: if both source and target are symlinks
    pointing to the same path, the source is dropped (safe duplicate). Otherwise
    the move is skipped and recorded as a conflict.

    Returns a stats dict::

        {
            "moves":       [(src_path, dst_path), ...],
            "same_target": [src_path, ...],          # dropped, identical symlink
            "conflicts":   [(src_path, dst_path, reason), ...],
            "src_removed": bool,                     # src dir emptied and rmdir'd
        }
    """
    src = Path(src)
    dst = Path(dst)
    moves, same_target, conflicts = [], [], []

    if not src.exists():
        return {"moves": moves, "same_target": same_target,
                "conflicts": conflicts, "src_removed": True}

    # Snapshot entries first — rglob is lazy and we mutate the tree as we go.
    entries = sorted(src.rglob("*"), key=lambda p: (len(p.parts), str(p)))

    for sp in entries:
        # Skip plain directories — we only move leaves; empty parents are
        # cleaned up at the end. A symlink to a directory is treated as a leaf.
        if sp.is_dir() and not sp.is_symlink():
            continue
        if not sp.exists() and not sp.is_symlink():
            continue  # may have been moved out from under us
        rel = sp.relative_to(src)
        target = dst / rel

        if target.exists() or target.is_symlink():
            if sp.is_symlink() and target.is_symlink():
                try:
                    if os.readlink(sp) == os.readlink(target):
                        same_target.append(sp)
                        if not dry_run:
                            sp.unlink()
                        continue
                except OSError:
                    pass
            conflicts.append((sp, target, "target exists"))
            continue

        moves.append((sp, target))
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            _move_path(sp, target)

    src_removed = False
    if not dry_run:
        # Bottom-up rmdir of newly-empty subdirs and src itself.
        subdirs = [p for p in src.rglob("*") if p.is_dir() and not p.is_symlink()]
        for d in sorted(subdirs, key=lambda p: -len(p.parts)):
            try:
                d.rmdir()
            except OSError:
                pass
        try:
            src.rmdir()
            src_removed = True
        except OSError:
            pass

    return {"moves": moves, "same_target": same_target,
            "conflicts": conflicts, "src_removed": src_removed}


def _resolve_cross_type_target(aliases, entries):
    """Pick (target_type, canonical_path, source_entries) for a cross-type group.

    Cross-type groups are only auto-mergeable when an alias declares the target
    type. Without that, the classifier can't tell which type is "right" and the
    group is returned to the caller for needs-review handling.

    Returns ``(target_type, canonical_path, [other_entries])`` or ``None``.
    """
    target_type = None
    for _t, path, _c in entries:
        spec = _alias_lookup(aliases, path.name)
        if spec and spec.get("type"):
            target_type = spec["type"]
            break
    if target_type is None:
        return None
    target_entries = [e for e in entries if e[0] == target_type]
    other_entries = [e for e in entries if e[0] != target_type]
    if not target_entries:
        return None
    # Pick canonical among the target-type entries
    canon_path, _ = _pick_canonical([(p, c) for _, p, c in target_entries])
    return target_type, canon_path, other_entries + [
        e for e in target_entries if e[1] != canon_path
    ]


def merge_duplicates(*, dry_run=True, aliases_path=None, prefer_year_form=False,
                     canonical_overrides=None, json_output=False):
    """Merge duplicate show directories in TYPE_DIRs.

    Default is dry-run; pass ``dry_run=False`` to actually move symlinks.

    Cross-type groups (same show under multiple TYPE_DIRs) require an aliases
    file declaring the target ``type``; otherwise they're listed under
    ``needs_review`` and skipped.
    """
    aliases = _load_aliases(aliases_path) if aliases_path else {}
    overrides = dict(_parse_canonical_override(s) for s in (canonical_overrides or []))

    within_type, cross_type, alias_canonical_by_key = _scan_dupe_groups(aliases=aliases)
    cross_keys = {k for k, _ in cross_type}

    report = {
        "dry_run": dry_run,
        "within_type": [],
        "cross_type": [],
        "needs_review": [],
    }

    def _do_merges(canonical_path, src_entries, *, type_label=None):
        merges = []
        for src_path, _ in src_entries:
            if src_path == canonical_path:
                continue
            stats = _merge_show_dir(src_path, canonical_path, dry_run=dry_run)
            merges.append({
                "from": str(src_path),
                "moved": len(stats["moves"]),
                "skipped_same_target": len(stats["same_target"]),
                "conflicts": [
                    {"src": str(s), "dst": str(d), "reason": r}
                    for s, d, r in stats["conflicts"]
                ],
                "src_removed": stats["src_removed"],
            })
        return merges

    # Within-type groups (skip keys we'll process under cross-type)
    for type_name, groups in within_type.items():
        for key, entries in groups:
            if key in cross_keys:
                continue
            canonical_path, _ = _pick_canonical(
                entries,
                prefer_year_form=prefer_year_form,
                override_name=overrides.get(key),
                alias_preferred_name=alias_canonical_by_key.get(key),
            )
            report["within_type"].append({
                "type": type_name,
                "key": key,
                "canonical": str(canonical_path),
                "merges": _do_merges(canonical_path, entries),
            })

    # Cross-type groups — need an alias to know which type is canonical
    for key, entries in cross_type:
        resolution = _resolve_cross_type_target(aliases, entries)
        if resolution is None:
            report["needs_review"].append({
                "key": key,
                "reason": "cross-type group requires aliases file entry with 'type'",
                "directories": [
                    {"type": t, "path": str(p), "files": c} for t, p, c in entries
                ],
            })
            continue
        target_type, canonical_path, src_entries = resolution
        # Apply --canonical override or alias preferred name if available
        target_type_entries = [(p, c) for t, p, c in entries if t == target_type]
        canonical_path, _ = _pick_canonical(
            target_type_entries,
            prefer_year_form=prefer_year_form,
            override_name=overrides.get(key),
            alias_preferred_name=alias_canonical_by_key.get(key),
        )
        # src_entries from resolver includes paths; rebuild merge list (path, count)
        merge_srcs = [(p, c) for _, p, c in entries if p != canonical_path]
        report["cross_type"].append({
            "key": key,
            "target_type": target_type,
            "canonical": str(canonical_path),
            "merges": _do_merges(canonical_path, merge_srcs),
        })

    if json_output:
        print(json.dumps(report, indent=2))
    else:
        _print_merge_report(report)

    return report


def _print_merge_report(report):
    """Human-readable rendering of merge_duplicates() output."""
    mode = "DRY-RUN" if report["dry_run"] else "APPLY"
    print(f"=== Merge duplicates ({mode}) ===")

    total_moves = total_same = total_conflicts = 0

    for grp in report["within_type"]:
        print(f"\n[{grp['type']}] {grp['key']}")
        print(f"  canonical: {grp['canonical']}")
        for m in grp["merges"]:
            total_moves += m["moved"]
            total_same += m["skipped_same_target"]
            total_conflicts += len(m["conflicts"])
            tag = " (removed)" if m["src_removed"] else ""
            print(
                f"    {m['from']}  →  moved={m['moved']} "
                f"same-target={m['skipped_same_target']} "
                f"conflicts={len(m['conflicts'])}{tag}"
            )
            for c in m["conflicts"]:
                print(f"      ! conflict: {c['src']}  vs  {c['dst']}  ({c['reason']})")

    for grp in report["cross_type"]:
        print(f"\n[CROSS → {grp['target_type']}] {grp['key']}")
        print(f"  canonical: {grp['canonical']}")
        for m in grp["merges"]:
            total_moves += m["moved"]
            total_same += m["skipped_same_target"]
            total_conflicts += len(m["conflicts"])
            tag = " (removed)" if m["src_removed"] else ""
            print(
                f"    {m['from']}  →  moved={m['moved']} "
                f"same-target={m['skipped_same_target']} "
                f"conflicts={len(m['conflicts'])}{tag}"
            )
            for c in m["conflicts"]:
                print(f"      ! conflict: {c['src']}  vs  {c['dst']}  ({c['reason']})")

    if report["needs_review"]:
        print("\n=== NEEDS REVIEW (no alias entry) ===")
        for grp in report["needs_review"]:
            print(f"\n  {grp['key']}  ({grp['reason']})")
            for d in grp["directories"]:
                print(f"    [{d['type']}] {d['path']}  ({d['files']} files)")

    print(
        f"\nTotal moves: {total_moves}"
        f"\nSame-target skips: {total_same}"
        f"\nConflicts: {total_conflicts}"
        f"\nNeeds-review groups: {len(report['needs_review'])}"
    )
    if report["dry_run"]:
        print("\n(dry-run — pass --apply to execute)")


# =============================================================================
# Phase 3: episode-level dedup (--report-episode-dupes / --merge-episode-dupes)
# =============================================================================

# Captures (season, episode) from an SxxEyy-style token at a delimiter boundary.
_EPISODE_TOKEN_RE = re.compile(r"(?:^|[.\s/_-])S(\d{1,2})E(\d{1,3})", re.IGNORECASE)

# In-memory ffprobe cache, keyed by (resolved_path, mtime_ns).
_EPISODE_PROBE_CACHE = {}


def _parse_episode_key(filename, season_hint=None):
    """Extract (season, episode) ints from a filename, or None if unrecognised.

    SxxEyy match wins. Falls back to anitopy when present (e.g. anime files that
    carry only an episode number); ``season_hint`` (from the enclosing Season N
    directory) fills in the season when anitopy can't.
    """
    m = _EPISODE_TOKEN_RE.search(filename)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    if anitopy:
        try:
            parsed = anitopy.parse(filename)
        except Exception:
            return None
        ep_raw = parsed.get("episode_number")
        sn_raw = parsed.get("anime_season")
        try:
            ep_n = int(ep_raw) if ep_raw is not None else None
        except (TypeError, ValueError):
            ep_n = None
        try:
            sn_n = int(sn_raw) if sn_raw is not None else None
        except (TypeError, ValueError):
            sn_n = None
        if ep_n is not None:
            if sn_n is None:
                sn_n = season_hint
            if sn_n is not None:
                return (sn_n, ep_n)
    return None


def _episode_probe(target_path):
    """Probe a media file for (height, has_subs, size, mtime). Cached.

    Cache key is (path, mtime_ns) so re-encodes invalidate. Returns None when
    the target is missing or ffprobe fails — callers must tolerate this and
    treat unknown signals as zero.
    """
    target_path = Path(target_path)
    try:
        st = target_path.stat()
    except OSError:
        return None
    key = (str(target_path), st.st_mtime_ns)
    if key in _EPISODE_PROBE_CACHE:
        return _EPISODE_PROBE_CACHE[key]

    info = {
        "height": 0,
        "has_subs": False,
        "sub_count": 0,
        "size": st.st_size,
        "mtime": st.st_mtime,
    }
    try:
        result = subprocess.run(
            [
                FFPROBE_PATH, "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(target_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout or "{}")
            for s in data.get("streams", []):
                t = s.get("codec_type")
                if t == "video":
                    try:
                        h = int(s.get("height") or 0)
                    except (TypeError, ValueError):
                        h = 0
                    if h > info["height"]:
                        info["height"] = h
                elif t == "subtitle":
                    info["sub_count"] += 1
            info["has_subs"] = info["sub_count"] > 0
    except Exception:
        pass

    _EPISODE_PROBE_CACHE[key] = info
    return info


def _external_subs_present(link_path):
    """True iff a subtitle file sits beside the symlink or beside its target.

    Matches the stem-prefix rule used by ``_link_companion_subs``.
    """
    link_path = Path(link_path)
    stem = link_path.stem
    try:
        for p in link_path.parent.iterdir():
            if p == link_path:
                continue
            if p.suffix.lower() in SUBTITLE_EXTENSIONS and p.stem.startswith(stem):
                return True
    except OSError:
        pass
    try:
        target = link_path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    try:
        for p in target.parent.iterdir():
            if p.suffix.lower() in SUBTITLE_EXTENSIONS and p.stem.startswith(target.stem):
                return True
    except OSError:
        pass
    return False


def _pick_episode_winner(links, prefer_release_group=None):
    """Pick which symlink to keep for one (season, episode) group.

    Priority (each beats everything below it): subtitle presence (internal +
    external) > video height > prefer-release-group regex hit > target file
    size > target mtime > filename (lexical ascending).

    Returns ``(winner, [losers])``.
    """
    prefer_re = (
        re.compile(prefer_release_group, re.IGNORECASE)
        if prefer_release_group else None
    )

    def score(link):
        probe = _episode_probe(link) or {}
        subs = (1 if probe.get("has_subs") else 0) + (
            1 if _external_subs_present(link) else 0
        )
        height = int(probe.get("height") or 0)
        size = int(probe.get("size") or 0)
        mtime = float(probe.get("mtime") or 0.0)
        group_match = 1 if (prefer_re and prefer_re.search(link.name)) else 0
        return (subs, height, group_match, size, mtime)

    # Stable two-pass sort: name ascending first, then score descending —
    # ties on score fall through to name in ascending order.
    by_name = sorted(links, key=lambda l: l.name)
    ranked = sorted(by_name, key=score, reverse=True)
    return ranked[0], ranked[1:]


def _scan_episode_dupes():
    """Walk TYPE_DIRS/<show>/<season>/ and return groups with >1 symlink per
    (season, episode). Each entry is a dict with type/show/season/episode/links.
    """
    out = []
    for type_name, type_dir in TYPE_DIRS.items():
        if not type_dir.is_dir():
            continue
        try:
            shows = sorted(type_dir.iterdir())
        except OSError:
            continue
        for show_dir in shows:
            if not show_dir.is_dir():
                continue
            try:
                season_children = sorted(show_dir.iterdir())
            except OSError:
                continue
            for season_dir in season_children:
                if not season_dir.is_dir() or season_dir.is_symlink():
                    continue
                if not _is_season_dir(season_dir.name):
                    continue
                season_num = _season_from_dir(season_dir.name)
                by_ep = {}
                try:
                    entries = sorted(season_dir.iterdir())
                except OSError:
                    continue
                for link in entries:
                    if not link.is_symlink():
                        continue
                    if link.suffix.lower() not in MEDIA_EXTENSIONS:
                        continue
                    key = _parse_episode_key(link.name, season_hint=season_num)
                    if key is None:
                        continue
                    by_ep.setdefault(key, []).append(link)
                for (sn, ep), links in sorted(by_ep.items()):
                    if len(links) > 1:
                        out.append({
                            "type": type_name,
                            "show": show_dir.name,
                            "season": sn,
                            "episode": ep,
                            "links": links,
                        })
    return out


def report_episode_duplicates(json_output=False, prefer_release_group=None):
    """Print or emit JSON of episode-level dedup candidates across TYPE_DIRs."""
    dupes = _scan_episode_dupes()
    enriched = []
    for d in dupes:
        winner, losers = _pick_episode_winner(
            d["links"], prefer_release_group=prefer_release_group,
        )
        enriched.append({**d, "winner": winner, "losers": losers})

    if json_output:
        payload = [
            {
                "type": d["type"],
                "show": d["show"],
                "season": d["season"],
                "episode": d["episode"],
                "winner": str(d["winner"]),
                "candidates": [
                    {
                        "path": str(l),
                        "target": (
                            os.readlink(l) if l.is_symlink() else None
                        ),
                        "is_winner": (l == d["winner"]),
                    }
                    for l in d["links"]
                ],
            }
            for d in enriched
        ]
        print(json.dumps(payload, indent=2))
        return

    if not enriched:
        print("No episode-level duplicates found.")
        return

    by_show = {}
    for d in enriched:
        by_show.setdefault((d["type"], d["show"]), []).append(d)

    total_losers = 0
    for (tname, show), entries in sorted(by_show.items()):
        print(f"\n=== {tname.upper()}: {show} ===")
        for d in sorted(entries, key=lambda e: (e["season"], e["episode"])):
            total_losers += len(d["losers"])
            print(
                f"S{d['season']:02d}E{d['episode']:02d}: "
                f"{len(d['links'])} encodings"
            )
            for link in d["links"]:
                mark = "  ← keep" if link == d["winner"] else ""
                print(f"  {link.name}{mark}")
    print(
        f"\nTotal episode-dupe groups: {len(enriched)}"
        f"\nSymlinks that would be unlinked: {total_losers}"
    )


def merge_episode_duplicates(*, dry_run=True, prefer_release_group=None,
                             json_output=False):
    """Unlink loser symlinks for each (season, episode) dupe group.

    Only the symlinks under TYPE_DIRs are touched — the file each symlink
    points at is never modified.
    """
    dupes = _scan_episode_dupes()
    report = {"dry_run": dry_run, "groups": [], "unlinked": 0}

    for d in dupes:
        winner, losers = _pick_episode_winner(
            d["links"], prefer_release_group=prefer_release_group,
        )
        unlinked = []
        for loser in losers:
            if not dry_run:
                try:
                    loser.unlink()
                except OSError:
                    continue
            unlinked.append(str(loser))
            report["unlinked"] += 1
        report["groups"].append({
            "type": d["type"],
            "show": d["show"],
            "season": d["season"],
            "episode": d["episode"],
            "winner": str(winner),
            "unlinked": unlinked,
        })

    if json_output:
        print(json.dumps(report, indent=2))
    else:
        mode = "DRY-RUN" if dry_run else "APPLY"
        print(f"=== Merge episode dupes ({mode}) ===")
        for g in report["groups"]:
            print(
                f"\n[{g['type']}] {g['show']} "
                f"S{g['season']:02d}E{g['episode']:02d}"
            )
            print(f"  keep:   {Path(g['winner']).name}")
            for u in g["unlinked"]:
                print(f"  unlink: {Path(u).name}")
        print(
            f"\nTotal groups: {len(report['groups'])}"
            f"\nTotal symlinks unlinked: {report['unlinked']}"
        )
        if dry_run:
            print("\n(dry-run — pass --apply to execute)")

    return report


def create_symlink(source, media_type):
    """Create a symlink in the appropriate Jellyfin media directory.

    For TV/anime: creates Show Name/Season N/ folder structure.
    For movies: places directly in Movies/.
    Also symlinks companion subtitle files.
    """
    target_dir = TYPE_DIRS.get(media_type)
    if not target_dir:
        return False

    source = Path(source)

    # Skip files inside bonus/extras directories (applies to all media types)
    source_root_names = {Path(d).name for d in SOURCE_DIRS}
    for ancestor in source.parents:
        if ancestor.name in source_root_names or not ancestor.name:
            break
        if _is_bonus_dir(ancestor.name):
            return False

    # For TV and anime, create show/season folder structure
    if media_type in ("tv", "anime"):
        show_name, season_num = _infer_show_folder(source)
        if show_name:
            show_name = _canonical_show_dir(target_dir, show_name)
            show_dir = target_dir / show_name
            if season_num is not None:
                dest_dir = show_dir / f"Season {season_num}"
            else:
                dest_dir = show_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Can't determine show folder — skip rather than dump flat
            return False
    else:
        dest_dir = target_dir

    link_path = dest_dir / source.name
    created = False

    if not link_path.exists() and not link_path.is_symlink():
        try:
            link_path.symlink_to(source)
            rel = link_path.relative_to(target_dir)
            print(f"  [LINK] {media_type}: {rel}")
            created = True
        except OSError as e:
            print(f"  [ERROR] Failed to create symlink for {source.name}: {e}", file=sys.stderr)
            return False

    # Symlink companion subtitle files
    # Always check even if video link already existed — subtitles may have arrived later
    _link_companion_subs(source, dest_dir, media_type, created)

    return created


def _link_companion_subs(source, target_dir, media_type, created_ref):
    """Find and symlink subtitle files for a video.

    Searches two patterns:
    1. Same directory, matching stem (e.g., Movie.Name.srt alongside Movie.Name.mkv)
    2. Subs/ subdirectory with episode-matching folder (e.g., Subs/Show.S01E01/2_English.srt)
    """
    # Pattern 1: Same directory, matching stem
    for sub_file in source.parent.iterdir():
        if sub_file.suffix.lower() in SUBTITLE_EXTENSIONS and sub_file.stem.startswith(source.stem):
            sub_link = target_dir / sub_file.name
            if not sub_link.exists() and not sub_link.is_symlink():
                try:
                    sub_link.symlink_to(sub_file)
                    print(f"  [LINK] {media_type}: {sub_file.name} (subtitle)")
                except OSError as e:
                    print(f"  [ERROR] Failed to symlink subtitle {sub_file.name}: {e}", file=sys.stderr)

    # Pattern 2: Subs/ subdirectory with episode-matching folder name
    subs_dir = source.parent / "Subs"
    if not subs_dir.is_dir():
        return

    for sub_folder in subs_dir.iterdir():
        if not sub_folder.is_dir():
            continue
        # Match folder name to video stem (folder often contains the video name)
        if not sub_folder.name.startswith(source.stem):
            continue
        for sub_file in sub_folder.iterdir():
            if sub_file.suffix.lower() in SUBTITLE_EXTENSIONS:
                # Rename to match video: VideoName.lang.srt
                lang = sub_file.stem.split("_", 1)[-1] if "_" in sub_file.stem else "en"
                sub_name = f"{source.stem}.{lang}{sub_file.suffix}"
                sub_link = target_dir / sub_name
                if not sub_link.exists() and not sub_link.is_symlink():
                    try:
                        sub_link.symlink_to(sub_file)
                        print(f"  [LINK] {media_type}: {sub_name} (subtitle from Subs/)")
                    except OSError as e:
                        print(f"  [ERROR] Failed to symlink subtitle {sub_name}: {e}", file=sys.stderr)


def find_media_files(source_dir):
    """Recursively find all media files in source directory.

    Returns list of (file_path, classification_name) tuples.
    classification_name is the relative path from source_dir, which
    includes parent directory names as additional classification signals.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        return []

    items = []
    for f in sorted(source_path.rglob("*")):
        if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS:
            if any(p.name.startswith(".") for p in f.relative_to(source_path).parents):
                continue
            if f.name.startswith("."):
                continue
            items.append((f, str(f.relative_to(source_path))))
    return items


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Classify media files for Jellyfin")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument(
        "--report-dupes",
        action="store_true",
        help="Scan TYPE_DIRs for duplicate show dirs (by _normalize_show_key) and exit",
    )
    parser.add_argument(
        "--report-dupes-json",
        action="store_true",
        help="Like --report-dupes but emit JSON for downstream tooling",
    )
    parser.add_argument(
        "--merge-dupes",
        action="store_true",
        help="Merge duplicate show dirs. Dry-run by default; pass --apply to execute.",
    )
    parser.add_argument(
        "--merge-dupes-json",
        action="store_true",
        help="Like --merge-dupes but emit JSON instead of a human-readable report.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --merge-dupes: actually move symlinks (omit for dry-run).",
    )
    parser.add_argument(
        "--aliases",
        help="Path to aliases JSON/YAML file (see _load_aliases docstring for format).",
    )
    parser.add_argument(
        "--prefer-year-form",
        action="store_true",
        help="When picking canonical, prefer dirs whose name contains '(YYYY)'.",
    )
    parser.add_argument(
        "--canonical",
        action="append",
        default=[],
        metavar="KEY=DIR_NAME",
        help="Override canonical pick for a group (repeatable). KEY is matched "
             "against the group's normalized show key.",
    )
    parser.add_argument(
        "--report-episode-dupes",
        action="store_true",
        help="Scan each Season N dir for multiple symlinks of the same SxxEyy and exit.",
    )
    parser.add_argument(
        "--report-episode-dupes-json",
        action="store_true",
        help="Like --report-episode-dupes but emit JSON for downstream tooling.",
    )
    parser.add_argument(
        "--merge-episode-dupes",
        action="store_true",
        help="Unlink loser symlinks for each (season, episode) dupe group. "
             "Dry-run by default; pass --apply to execute. Source files untouched.",
    )
    parser.add_argument(
        "--merge-episode-dupes-json",
        action="store_true",
        help="Like --merge-episode-dupes but emit JSON instead of a text report.",
    )
    parser.add_argument(
        "--prefer-release-group",
        metavar="REGEX",
        help="When picking the episode winner, prefer symlinks whose filename "
             "matches this regex (e.g. 'AMZN.*WEBRip|x265|ImE'). Beats size/mtime "
             "but loses to subs presence and video height.",
    )
    args = parser.parse_args()

    if args.config:
        load_config(args.config)

    if args.report_dupes or args.report_dupes_json:
        report_duplicates(json_output=args.report_dupes_json)
        return

    if args.merge_dupes or args.merge_dupes_json:
        merge_duplicates(
            dry_run=not args.apply,
            aliases_path=args.aliases,
            prefer_year_form=args.prefer_year_form,
            canonical_overrides=args.canonical,
            json_output=args.merge_dupes_json,
        )
        return

    if args.report_episode_dupes or args.report_episode_dupes_json:
        report_episode_duplicates(
            json_output=args.report_episode_dupes_json,
            prefer_release_group=args.prefer_release_group,
        )
        return

    if args.merge_episode_dupes or args.merge_episode_dupes_json:
        merge_episode_duplicates(
            dry_run=not args.apply,
            prefer_release_group=args.prefer_release_group,
            json_output=args.merge_episode_dupes_json,
        )
        return

    state = load_state()
    processed = state["processed"]
    stats = {"new": 0, "skipped": 0, "failed": 0, "reprocessed": 0}

    for source_dir in SOURCE_DIRS:
        items = find_media_files(source_dir)
        if not items:
            print(f"[WARN] No media found in: {source_dir}")
            continue

        print(f"[SCAN] {source_dir} ({len(items)} items)")

        for filepath, rel_path in items:
            key = str(filepath)
            # Use relative path (includes parent dirs) for richer classification signals
            name = rel_path

            # Check if already processed
            existing = processed.get(key)
            if existing and not should_reprocess(existing):
                # Still check for new companion subtitle files
                existing_type = existing.get("type")
                if existing_type and existing_type in TYPE_DIRS:
                    create_symlink(filepath, existing_type)
                stats["skipped"] += 1
                continue

            if existing:
                stats["reprocessed"] += 1
                print(f"  [REPROCESS] {name} (was {existing.get('type')})")

            # Classify using relative path for name, actual file for ffprobe
            media_type, confidence, signals = classify(name, str(filepath), state)

            if media_type == "unknown":
                print(f"  [REVIEW] Unclassified: {name}")
                processed[key] = {
                    "type": "unknown",
                    "confidence": "none",
                    "signals": signals,
                    "classifier_version": CLASSIFIER_VERSION,
                }
                stats["failed"] += 1
                continue

            print(f"  [{media_type.upper()}] {name} (confidence={confidence}, method={signals.get('method', '?')})")

            if create_symlink(filepath, media_type):
                processed[key] = {
                    "type": media_type,
                    "confidence": confidence,
                    "signals": signals,
                    "classifier_version": CLASSIFIER_VERSION,
                }
                stats["new"] += 1
            else:
                # Symlink already exists or failed
                processed[key] = {
                    "type": media_type,
                    "confidence": confidence,
                    "signals": signals,
                    "classifier_version": CLASSIFIER_VERSION,
                }
                stats["skipped"] += 1

    save_state(state)
    print(
        f"\n[DONE] New: {stats['new']}, Skipped: {stats['skipped']}, "
        f"Reprocessed: {stats['reprocessed']}, Failed: {stats['failed']}"
    )


if __name__ == "__main__":
    main()
