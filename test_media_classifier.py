"""Tests for media classifier pipeline.

Run unit tests:    python3 -m pytest test_media_classifier.py -v
Run live tests:    python3 -m pytest test_media_classifier.py -v --live
"""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

# Import from hyphenated filename
import importlib
sys.path.insert(0, str(Path(__file__).parent))
mc = importlib.import_module("media-classifier")


# =============================================================================
# Fixtures & helpers
# =============================================================================


@pytest.fixture
def empty_state():
    return {
        "version": mc.CLASSIFIER_VERSION,
        "anilist_cache": {},
        "wikipedia_cache": {},
        "processed": {},
    }


def make_ffprobe_result(audio_langs=None, sub_formats=None, duration=None):
    return {
        "audio_langs": audio_langs or [],
        "sub_formats": sub_formats or [],
        "sub_langs": [],
        "duration": duration,
        "num_audio": len(audio_langs or []),
        "num_subs": len(sub_formats or []),
    }


def make_anilist_result(title="Test", country="JP", fmt="TV", episodes=12, popularity=50000):
    return [{
        "title": {"romaji": title, "english": title, "native": None},
        "format": fmt,
        "episodes": episodes,
        "popularity": popularity,
        "countryOfOrigin": country,
        "genres": [],
        "averageScore": 75,
    }]


# =============================================================================
# Stage 1: Filename parsing
# =============================================================================

class TestParseFilename:
    def test_fansub_filename(self):
        info = mc.parse_filename("[SubsPlease] Jujutsu Kaisen - 01 (1080p).mkv")
        assert info["has_bracket_prefix"]
        assert info["bracket_group"] == "SubsPlease"
        if mc.anitopy:
            assert info["release_group"] is not None

    def test_scene_filename(self):
        info = mc.parse_filename("Breaking.Bad.S01E05.720p.BluRay.mkv")
        assert info["has_tv_pattern"]
        assert not info["has_bracket_prefix"]
        assert not info["has_crc"]

    def test_movie_filename(self):
        info = mc.parse_filename("The.Matrix.1999.2160p.UHD.BluRay.mkv")
        assert info["year"] == 1999
        assert not info["has_tv_pattern"]

    def test_crc_detection(self):
        info = mc.parse_filename("[MTBB] Cowboy Bebop [BD 1080p] [A1B2C3D4].mkv")
        assert info["has_crc"]
        assert info["has_bracket_prefix"]

    def test_title_extraction_scene(self):
        info = mc.parse_filename("Breaking.Bad.S01E05.720p.BluRay.mkv")
        assert "breaking bad" in info["cleaned_title"].lower()

    def test_title_extraction_movie(self):
        info = mc.parse_filename("The.Matrix.1999.2160p.UHD.BluRay.mkv")
        assert "matrix" in info["cleaned_title"].lower()

    def test_ambiguous_title(self):
        info = mc.parse_filename("Kingdom.S01E01.1080p.mkv")
        assert info["has_tv_pattern"]
        assert "kingdom" in info["cleaned_title"].lower()


# =============================================================================
# Stage 2: Fast-path classification
# =============================================================================

class TestFastPath:
    def test_known_fansub_group(self):
        info = mc.parse_filename("[SubsPlease] Jujutsu Kaisen - 01 (1080p).mkv")
        media_type, conf = mc.classify_fast_path(info)
        assert media_type == "anime"
        assert conf == "high"

    def test_erai_raws(self):
        info = mc.parse_filename("[Erai-raws] Frieren - 18 [1080p].mkv")
        media_type, conf = mc.classify_fast_path(info)
        assert media_type == "anime"
        assert conf == "high"

    def test_bracket_crc(self):
        info = mc.parse_filename("[MTBB] Cowboy Bebop [BD 1080p] [A1B2C3D4].mkv")
        media_type, conf = mc.classify_fast_path(info)
        assert media_type == "anime"
        assert conf == "high"

    def test_scene_tv(self):
        info = mc.parse_filename("Breaking.Bad.S01E05.720p.BluRay.mkv")
        media_type, conf = mc.classify_fast_path(info)
        assert media_type == "tv"
        assert conf == "high"

    def test_movie_year(self):
        info = mc.parse_filename("The.Matrix.1999.2160p.UHD.BluRay.mkv")
        media_type, conf = mc.classify_fast_path(info)
        assert media_type == "movie"
        assert conf == "high"

    def test_ambiguous_returns_none(self):
        info = mc.parse_filename("Kingdom.S01E01.1080p.mkv")
        # S01E01 without bracket prefix → tv fast-path
        media_type, conf = mc.classify_fast_path(info)
        assert media_type == "tv"

    def test_bare_name_no_fast_path(self):
        info = mc.parse_filename("Kingdom")
        media_type, conf = mc.classify_fast_path(info)
        assert media_type is None


# =============================================================================
# Stage 3: Evidence gathering (mocked)
# =============================================================================

class TestAniListQuery:
    def test_cached_result(self, empty_state):
        empty_state["anilist_cache"]["kingdom"] = {
            "results": make_anilist_result("Kingdom"),
            "fetched_at": "2099-01-01T00:00:00+00:00",
        }
        result = mc.query_anilist("Kingdom", empty_state)
        assert result is not None
        assert result[0]["title"]["romaji"] == "Kingdom"

    def test_expired_cache(self, empty_state):
        empty_state["anilist_cache"]["kingdom"] = {
            "results": make_anilist_result("Kingdom"),
            "fetched_at": "2020-01-01T00:00:00+00:00",
        }
        # Would need network; mock the URL open
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "data": {"Page": {"media": make_anilist_result("Kingdom")}}
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = mc.query_anilist("Kingdom", empty_state)
            assert result is not None

    def test_empty_title(self, empty_state):
        result = mc.query_anilist("", empty_state)
        assert result is None


class TestFfprobe:
    def test_parse_japanese_audio(self):
        ffprobe_output = json.dumps({
            "streams": [
                {"codec_type": "video", "codec_name": "hevc"},
                {"codec_type": "audio", "codec_name": "aac", "tags": {"language": "jpn"}},
                {"codec_type": "audio", "codec_name": "aac", "tags": {"language": "eng"}},
                {"codec_type": "subtitle", "codec_name": "ass", "tags": {"language": "eng"}},
            ],
            "format": {"duration": "1452.5"},
        })

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0, stdout=ffprobe_output, stderr=""
            )
            with mock.patch("pathlib.Path.exists", return_value=True):
                result = mc.probe_file("/fake/file.mkv")

        assert result is not None
        assert "jpn" in result["audio_langs"]
        assert "ass" in result["sub_formats"]
        assert result["duration"] == pytest.approx(1452.5)

    def test_parse_danish_audio(self):
        ffprobe_output = json.dumps({
            "streams": [
                {"codec_type": "audio", "codec_name": "aac", "tags": {"language": "dan"}},
                {"codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng"}},
            ],
            "format": {"duration": "3420.0"},
        })

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                returncode=0, stdout=ffprobe_output, stderr=""
            )
            with mock.patch("pathlib.Path.exists", return_value=True):
                result = mc.probe_file("/fake/riget.mkv")

        assert "dan" in result["audio_langs"]
        assert result["duration"] == pytest.approx(3420.0)

    def test_missing_file(self):
        result = mc.probe_file("/nonexistent/file.mkv")
        assert result is None


# =============================================================================
# Stage 4: Scoring
# =============================================================================

class TestScoring:
    def test_jp_audio_plus_anilist(self):
        info = mc.parse_filename("Kingdom.S01E01.1080p.mkv")
        evidence = {
            "anilist": make_anilist_result("Kingdom", country="JP", popularity=52000),
            "ffprobe": make_ffprobe_result(audio_langs=["jpn", "eng"], sub_formats=["ass"], duration=1452),
            "wikipedia": "Kingdom is a Japanese manga series written and illustrated by Yasuhisa Hara.",
        }
        media_type, conf, scores, reasons = mc.score_evidence(info, evidence)
        assert media_type == "anime"
        assert conf in ("high", "medium")

    def test_no_anilist_english_audio(self):
        info = mc.parse_filename("The.Office.S03E12.720p.mkv")
        evidence = {
            "anilist": None,
            "ffprobe": make_ffprobe_result(audio_langs=["eng"], sub_formats=["subrip"], duration=1320),
            "wikipedia": "The Office is an American mockumentary sitcom television series.",
        }
        media_type, conf, scores, reasons = mc.score_evidence(info, evidence)
        assert media_type == "tv"

    def test_danish_audio_no_anilist(self):
        info = mc.parse_filename("Riget.S01E01.DVDRip.mkv")
        evidence = {
            "anilist": None,
            "ffprobe": make_ffprobe_result(audio_langs=["dan"], sub_formats=["subrip"], duration=3420),
            "wikipedia": "The Kingdom is a Danish television series created by Lars von Trier.",
        }
        media_type, conf, scores, reasons = mc.score_evidence(info, evidence)
        assert media_type == "tv"

    def test_long_duration_movie(self):
        info = mc.parse_filename("Blade.Runner.2049.2017.2160p.mkv")
        evidence = {
            "anilist": None,
            "ffprobe": make_ffprobe_result(audio_langs=["eng"], duration=9864),
            "wikipedia": "Blade Runner 2049 is a 2017 American science fiction film.",
        }
        media_type, conf, scores, reasons = mc.score_evidence(info, evidence)
        assert media_type == "movie"

    def test_anime_movie_anilist(self):
        info = mc.parse_filename("One.Piece.Film.Red.2022.1080p.mkv")
        evidence = {
            "anilist": make_anilist_result("One Piece Film: Red", country="JP", fmt="MOVIE", popularity=80000),
            "ffprobe": make_ffprobe_result(audio_langs=["jpn"], sub_formats=["ass"], duration=6720),
            "wikipedia": "One Piece Film: Red is a 2022 Japanese animated fantasy action adventure film.",
        }
        media_type, conf, scores, reasons = mc.score_evidence(info, evidence)
        assert media_type == "anime"

    def test_castlevania_anilist_match(self):
        info = mc.parse_filename("Castlevania.S04E08.1080p.NF.mkv")
        evidence = {
            "anilist": make_anilist_result("Castlevania", country="JP", popularity=30000),
            "ffprobe": make_ffprobe_result(audio_langs=["eng", "jpn"], sub_formats=["subrip"], duration=1440),
            "wikipedia": None,
        }
        media_type, conf, scores, reasons = mc.score_evidence(info, evidence)
        assert media_type == "anime"


# =============================================================================
# Stage 5: LLM arbiter (mocked)
# =============================================================================

class TestLLMArbiter:
    def test_successful_classification(self):
        info = mc.parse_filename("Ambiguous.Title.S01E01.mkv")
        evidence = {
            "anilist": None,
            "ffprobe": make_ffprobe_result(audio_langs=["eng"]),
            "wikipedia": None,
        }

        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "response": '{"category": "tv"}',
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            media_type, conf = mc.classify_llm(info, evidence, {"anime": 1, "tv": 1, "movie": 0}, [])
            assert media_type == "tv"
            assert conf == "medium"

    def test_llm_failure_returns_none(self):
        info = mc.parse_filename("Ambiguous.Title.S01E01.mkv")
        evidence = {"anilist": None, "ffprobe": None, "wikipedia": None}

        with mock.patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            media_type, conf = mc.classify_llm(info, evidence, {}, [])
            assert media_type is None


# =============================================================================
# Full pipeline (mocked)
# =============================================================================

class TestFullPipeline:
    def test_fansub_fast_path(self, empty_state):
        media_type, conf, signals = mc.classify(
            "[SubsPlease] Jujutsu Kaisen - 01 (1080p).mkv", "/fake/file.mkv", empty_state
        )
        assert media_type == "anime"
        assert signals["method"] == "fast_path"

    def test_scene_tv_fast_path(self, empty_state):
        media_type, conf, signals = mc.classify(
            "Breaking.Bad.S01E05.720p.BluRay.mkv", "/fake/file.mkv", empty_state
        )
        assert media_type == "tv"
        assert signals["method"] == "fast_path"

    def test_movie_fast_path(self, empty_state):
        media_type, conf, signals = mc.classify(
            "The.Matrix.1999.2160p.UHD.BluRay.mkv", "/fake/file.mkv", empty_state
        )
        assert media_type == "movie"
        assert signals["method"] == "fast_path"


# =============================================================================
# State management
# =============================================================================

class TestState:
    def test_migrate_v1_state(self):
        v1 = {"processed": {"file.mkv": {"type": "tv"}}}
        # Simulate load
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(v1))):
            with mock.patch.object(Path, "exists", return_value=True):
                state = mc.load_state()
        assert state["version"] == mc.CLASSIFIER_VERSION
        assert "anilist_cache" in state
        assert "wikipedia_cache" in state

    def test_should_reprocess_old_version(self):
        entry = {"type": "unknown", "confidence": None, "classifier_version": 1}
        assert mc.should_reprocess(entry)

    def test_should_not_reprocess_current_version(self):
        entry = {"type": "anime", "confidence": "high", "classifier_version": mc.CLASSIFIER_VERSION}
        assert not mc.should_reprocess(entry)

    def test_should_reprocess_low_confidence(self):
        entry = {"type": "tv", "confidence": "low", "classifier_version": 1}
        assert mc.should_reprocess(entry)

    def test_cache_expiry(self):
        assert mc.cache_expired({"fetched_at": "2020-01-01T00:00:00+00:00"})
        assert not mc.cache_expired({"fetched_at": "2099-01-01T00:00:00+00:00"})
        assert mc.cache_expired({})
        assert mc.cache_expired({"fetched_at": ""})


# =============================================================================
# End-to-end test cases (integration, no network by default)
# =============================================================================

E2E_CASES = [
    ("[SubsPlease] Jujutsu Kaisen - 01 (1080p).mkv", "anime", "known fansub group"),
    ("[Erai-raws] Frieren - 18 [1080p].mkv", "anime", "known fansub group"),
    ("Breaking.Bad.S01E05.720p.BluRay.mkv", "tv", "S01E01 + no anime signals"),
    ("The.Matrix.1999.2160p.UHD.BluRay.mkv", "movie", "year + no episodes"),
    ("[MTBB] Cowboy Bebop [BD 1080p] [A1B2C3D4].mkv", "anime", "known group + CRC"),
    ("The.Office.S03E12.720p.mkv", "tv", "S01E01 + no anime signals"),
    ("Game.of.Thrones.S08E06.1080p.mkv", "tv", "S01E01 + no anime signals"),
    ("Inception.2010.1080p.BluRay.mkv", "movie", "year + no episodes"),
]


@pytest.mark.parametrize("filename,expected,reason", E2E_CASES)
def test_e2e_fast_path(filename, expected, reason, empty_state):
    """These should all be resolved by fast-path (no API calls needed)."""
    media_type, conf, signals = mc.classify(filename, "/fake/file.mkv", empty_state)
    assert media_type == expected, f"Expected {expected} for '{filename}' ({reason}), got {media_type}"
    assert signals["method"] == "fast_path"


# =============================================================================
# Config loading
# =============================================================================

class TestPathParsing:
    """Test that parent directory names provide classification signals."""

    def test_fansub_in_parent_dir(self):
        info = mc.parse_filename("[SubsPlease] Frieren/01.mkv")
        assert info["bracket_group"] == "SubsPlease"
        assert info["has_bracket_prefix"]

    def test_crc_in_parent_dir(self):
        info = mc.parse_filename("[MTBB] Cowboy Bebop [A1B2C3D4]/01.mkv")
        assert info["has_crc"]
        assert info["has_bracket_prefix"]

    def test_bare_episode_uses_parent_title(self):
        info = mc.parse_filename("Breaking Bad/01.mkv")
        assert "breaking bad" in info["cleaned_title"].lower()

    def test_tv_pattern_in_filename(self):
        info = mc.parse_filename("Some Show/S01E05.mkv")
        assert info["has_tv_pattern"]

    def test_year_in_parent_dir(self):
        info = mc.parse_filename("The Matrix (1999)/movie.mkv")
        assert info["year"] == 1999


class TestPathClassification:
    """Test full classification with path-based names."""

    def test_fansub_dir_fast_path(self, empty_state):
        media_type, conf, signals = mc.classify(
            "[SubsPlease] Frieren/01.mkv", "/fake/file.mkv", empty_state
        )
        assert media_type == "anime"
        assert signals["method"] == "fast_path"

    def test_scene_dir_fast_path(self, empty_state):
        media_type, conf, signals = mc.classify(
            "Breaking.Bad.S01E05.720p.BluRay/video.mkv", "/fake/file.mkv", empty_state
        )
        assert media_type == "tv"
        assert signals["method"] == "fast_path"


class TestConfigLoading:
    def test_load_config_overrides(self, tmp_path):
        config = {
            "sourceDirs": ["/tmp/test-source"],
            "mediaBase": "/tmp/test-media",
            "categories": {"movie": "Films", "tv": "Series", "anime": "Anime"},
            "ollamaHost": "http://test:11434",
            "ollamaModel": "test-model",
            "ffprobePath": "/usr/bin/ffprobe",
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        mc.load_config(str(config_file))

        assert mc.SOURCE_DIRS == ["/tmp/test-source"]
        assert mc.MEDIA_BASE == Path("/tmp/test-media")
        assert mc.OLLAMA_HOST == "http://test:11434"
        assert mc.OLLAMA_MODEL == "test-model"
        assert mc.FFPROBE_PATH == "/usr/bin/ffprobe"
        assert mc.TYPE_DIRS["movie"] == Path("/tmp/test-media/Films")


# =============================================================================
# Live API tests (require --live flag)
# =============================================================================

@pytest.mark.live
class TestLiveAniList:
    def test_kingdom_is_anime(self, empty_state):
        results = mc.query_anilist("Kingdom", empty_state)
        assert results is not None
        assert len(results) > 0
        assert results[0]["countryOfOrigin"] == "JP"

    def test_the_office_no_anime(self, empty_state):
        results = mc.query_anilist("The Office", empty_state)
        # The Office might return results but they shouldn't be highly popular JP anime
        if results:
            top = results[0]
            assert top.get("popularity", 0) < 10000 or top.get("countryOfOrigin") != "JP"

    def test_castlevania(self, empty_state):
        results = mc.query_anilist("Castlevania", empty_state)
        assert results is not None
        assert len(results) > 0

    def test_caching(self, empty_state):
        mc.query_anilist("Kingdom", empty_state)
        assert "kingdom" in empty_state["anilist_cache"]
        # Second call should use cache (no network)
        results = mc.query_anilist("Kingdom", empty_state)
        assert results is not None


@pytest.mark.live
class TestLiveWikipedia:
    def test_kingdom_anime(self, empty_state):
        result = mc.query_wikipedia("Kingdom anime", empty_state)
        assert result is not None
        assert "japanese" in result.lower() or "manga" in result.lower()

    def test_kingdom_lars_von_trier(self, empty_state):
        result = mc.query_wikipedia("The Kingdom Lars von Trier", empty_state)
        assert result is not None
        assert "danish" in result.lower() or "lars" in result.lower() or "trier" in result.lower()


# =============================================================================
# Symlink tests
# =============================================================================


class TestCreateSymlink:
    def test_movie_flat_with_subtitles(self, tmp_path):
        """Movies should be symlinked flat (no folder structure) with subtitles."""
        src_dir = tmp_path / "Movie (2023)"
        src_dir.mkdir()
        video = src_dir / "Movie.2023.1080p.mkv"
        video.write_text("video")
        srt = src_dir / "Movie.2023.1080p.srt"
        srt.write_text("subtitles")

        target = tmp_path / "Movies"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["movie"] = target
        try:
            result = mc.create_symlink(str(video), "movie")
            assert result is True
            assert (target / "Movie.2023.1080p.mkv").is_symlink()
            assert (target / "Movie.2023.1080p.srt").is_symlink()
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_tv_show_season_dir_structure(self, tmp_path):
        """TV with Season dir should create Show/Season N/ structure."""
        show_dir = tmp_path / "Breaking Bad (2008) Season 1-5"
        season_dir = show_dir / "Season 1"
        season_dir.mkdir(parents=True)
        video = season_dir / "Breaking Bad (2008) - S01E01 - Pilot.mkv"
        video.write_text("video")

        target = tmp_path / "TV Shows"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            result = mc.create_symlink(str(video), "tv")
            assert result is True
            assert (target / "Breaking Bad (2008)" / "Season 1" / video.name).is_symlink()
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_tv_show_infer_season_from_filename(self, tmp_path):
        """TV without Season dir should infer season from filename."""
        show_dir = tmp_path / "Creature Commandos"
        show_dir.mkdir()
        video = show_dir / "Creature.Commandos.S01E01.1080p.mkv"
        video.write_text("video")

        target = tmp_path / "TV Shows"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            result = mc.create_symlink(str(video), "tv")
            assert result is True
            assert (target / "Creature Commandos" / "Season 1" / video.name).is_symlink()
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_subs_subdirectory_with_folder_structure(self, tmp_path):
        """Subs/ subdirectory subtitles should land in the show/season folder."""
        show_dir = tmp_path / "My Show"
        show_dir.mkdir()
        video = show_dir / "My.Show.S01E01.1080p.mkv"
        video.write_text("video")

        subs_dir = show_dir / "Subs" / "My.Show.S01E01.1080p"
        subs_dir.mkdir(parents=True)
        (subs_dir / "2_English.srt").write_text("english subs")

        target = tmp_path / "TV Shows"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            result = mc.create_symlink(str(video), "tv")
            assert result is True
            dest = target / "My Show" / "Season 1"
            assert (dest / video.name).is_symlink()
            assert (dest / "My.Show.S01E01.1080p.English.srt").is_symlink()
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_s_prefix_season_dir(self, tmp_path):
        """S08 1080p Bluray should be treated as Season 8, not a show name."""
        show_dir = tmp_path / "It's Always Sunny in Philadelphia (2005) Season 1-13 S01-S13 (Mixed x265)"
        season_dir = show_dir / "S08 1080p Bluray"
        season_dir.mkdir(parents=True)
        video = season_dir / "It's Always Sunny in Philadelphia S08E01 Pop-Pop.mkv"
        video.write_text("video")

        target = tmp_path / "TV Shows"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            result = mc.create_symlink(str(video), "tv")
            assert result is True
            expected = target / "It's Always Sunny in Philadelphia (2005)" / "Season 8" / video.name
            assert expected.is_symlink()
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_bonus_dir_skipped(self, tmp_path):
        """Files in Featurettes/Extras dirs should not get symlinks."""
        show_dir = tmp_path / "Breaking Bad (2008) Season 1-5"
        feat_dir = show_dir / "Extras"
        feat_dir.mkdir(parents=True)
        video = feat_dir / "Inside Breaking Bad.mkv"
        video.write_text("video")

        target = tmp_path / "TV Shows"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            result = mc.create_symlink(str(video), "tv")
            assert result is False
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_nested_bonus_dir_skipped(self, tmp_path):
        """Deeply nested bonus content should not get symlinks."""
        show_dir = tmp_path / "Futurama (1999) Season 1-7"
        nested = show_dir / "Other" / "Futurama Movie (2008)" / "Featurettes"
        nested.mkdir(parents=True)
        video = nested / "Deleted Scenes.mkv"
        video.write_text("video")

        target = tmp_path / "TV Shows"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            result = mc.create_symlink(str(video), "tv")
            assert result is False
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_extras_season_subdir_skipped(self, tmp_path):
        """Extras/Season 3/file.mkv should be skipped (bonus content inside season subdirs)."""
        show_dir = tmp_path / "The Venture Bros. (Seasons 1-6)"
        extras_season = show_dir / "Extras" / "Season 3"
        extras_season.mkdir(parents=True)
        video = extras_season / "Extras - Deleted Scenes - ORB.mkv"
        video.write_text("video")

        target = tmp_path / "TV Shows"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        orig_source_dirs = mc.SOURCE_DIRS[:]
        mc.TYPE_DIRS["tv"] = target
        mc.SOURCE_DIRS = [str(tmp_path)]
        try:
            result = mc.create_symlink(str(video), "tv")
            assert result is False
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)
            mc.SOURCE_DIRS = orig_source_dirs

    def test_no_subtitle_no_crash(self, tmp_path):
        """Video without subtitles should still work."""
        src_dir = tmp_path / "Solo Show"
        src_dir.mkdir()
        video = src_dir / "Solo.Video.mkv"
        video.write_text("video")

        target = tmp_path / "TV Shows"
        target.mkdir()
        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            result = mc.create_symlink(str(video), "tv")
            assert result is True
            assert (target / "Solo Show" / "Solo.Video.mkv").is_symlink()
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)


class TestShowNameNormalization:
    """Regression: mc-pw3 — duplicate show dirs from inconsistent normalization."""

    def test_normalize_key_collapses_case_punct_year_amp(self):
        keys = {
            mc._normalize_show_key("Law and Order SVU"),
            mc._normalize_show_key("Law And Order SVU"),
            mc._normalize_show_key("Law and Order SVU 1999"),
            mc._normalize_show_key("Law & Order SVU (1999)"),
        }
        assert len(keys) == 1, f"expected all keys to match, got {keys}"

    def test_normalize_key_expands_abbreviations_to_spelled_out(self):
        """Abbreviated and spelled-out titles must collapse to one key.

        Regression for the real prod dupe: 'Law and Order SVU' and
        'Law & Order Special Victims Unit (1999)' landed in separate dirs
        because the normalizer left 'svu' != 'special victims unit'.
        The abbreviation expander bridges them without a manual aliases file.
        """
        assert mc._normalize_show_key("Law and Order SVU") == \
            mc._normalize_show_key("Law & Order Special Victims Unit (1999)")
        assert mc._normalize_show_key("Law and Order SVU") == \
            mc._normalize_show_key("Law And Order Special Victims Unit")
        # And the canonical key is the spelled-out form, not the abbreviation
        assert mc._normalize_show_key("Law and Order SVU") == \
            "law and order special victims unit"

    def test_normalize_key_other_abbreviations_collapse(self):
        """A couple more abbreviation bridges the expander ships with."""
        assert mc._normalize_show_key("Star Trek TNG") == \
            mc._normalize_show_key("Star Trek The Next Generation")
        assert mc._normalize_show_key("Better Call Saul") == \
            mc._normalize_show_key("BCS")

    def test_canonical_show_dir_reuses_existing(self, tmp_path):
        target = tmp_path / "TV Shows"
        target.mkdir()
        (target / "Breaking Bad (2008)").mkdir()

        # Different release-style names should map to the existing dir
        assert mc._canonical_show_dir(target, "Breaking Bad") == "Breaking Bad (2008)"
        assert mc._canonical_show_dir(target, "breaking.bad.2008") == "Breaking Bad (2008)"

    def test_canonical_show_dir_keeps_new_when_no_match(self, tmp_path):
        target = tmp_path / "TV Shows"
        target.mkdir()
        (target / "The Wire").mkdir()
        assert mc._canonical_show_dir(target, "Better Call Saul") == "Better Call Saul"

    def test_canonical_show_dir_uses_aliases(self, tmp_path):
        target = tmp_path / "TV Shows"
        target.mkdir()
        orig = mc.SHOW_ALIASES.copy()
        mc.SHOW_ALIASES.clear()
        mc.SHOW_ALIASES["Law and Order SVU"] = "Law & Order Special Victims Unit (1999)"
        try:
            # Direct hit
            assert mc._canonical_show_dir(target, "Law and Order SVU") == "Law & Order Special Victims Unit (1999)"
            # Normalized variant of alias key
            assert mc._canonical_show_dir(target, "Law.And.Order.SVU") == "Law & Order Special Victims Unit (1999)"
        finally:
            mc.SHOW_ALIASES.clear()
            mc.SHOW_ALIASES.update(orig)

    def test_canonical_show_dir_classify_aliases_routes_to_existing(self, tmp_path):
        """--aliases file lets a fresh variant route into the existing canonical dir."""
        target = tmp_path / "TV Shows"
        target.mkdir()
        (target / "Law & Order Special Victims Unit (1999)").mkdir()

        orig = mc.CLASSIFY_ALIASES.copy()
        mc.CLASSIFY_ALIASES.clear()
        mc.CLASSIFY_ALIASES["Law and Order SVU"] = {
            "canonical": "Law & Order Special Victims Unit (1999)",
            "type": "tv",
        }
        try:
            assert mc._canonical_show_dir(target, "Law and Order SVU") == \
                "Law & Order Special Victims Unit (1999)"
            # Variant of the alias key also resolves
            assert mc._canonical_show_dir(target, "Law.And.Order.SVU") == \
                "Law & Order Special Victims Unit (1999)"
        finally:
            mc.CLASSIFY_ALIASES.clear()
            mc.CLASSIFY_ALIASES.update(orig)

    def test_canonical_show_dir_classify_aliases_resolves_sibling_variant(self, tmp_path):
        """When the existing sibling is the variant and the candidate is canonical,
        alias resolution still collapses them onto the existing dir.
        """
        target = tmp_path / "TV Shows"
        target.mkdir()
        # Sibling is the variant form (mc-imk Phase 2 hasn't renamed yet)
        (target / "Law and Order SVU").mkdir()

        orig = mc.CLASSIFY_ALIASES.copy()
        mc.CLASSIFY_ALIASES.clear()
        mc.CLASSIFY_ALIASES["Law and Order SVU"] = {
            "canonical": "Law & Order Special Victims Unit (1999)",
            "type": "tv",
        }
        try:
            assert mc._canonical_show_dir(
                target, "Law & Order Special Victims Unit (1999)"
            ) == "Law and Order SVU"
        finally:
            mc.CLASSIFY_ALIASES.clear()
            mc.CLASSIFY_ALIASES.update(orig)

    def test_canonical_show_dir_classify_aliases_no_sibling_uses_canonical(self, tmp_path):
        """Empty TYPE_DIR with an alias entry → new dir uses the canonical name."""
        target = tmp_path / "TV Shows"
        target.mkdir()

        orig = mc.CLASSIFY_ALIASES.copy()
        mc.CLASSIFY_ALIASES.clear()
        mc.CLASSIFY_ALIASES["Law and Order SVU"] = {
            "canonical": "Law & Order Special Victims Unit (1999)",
            "type": "tv",
        }
        try:
            assert mc._canonical_show_dir(target, "Law and Order SVU") == \
                "Law & Order Special Victims Unit (1999)"
        finally:
            mc.CLASSIFY_ALIASES.clear()
            mc.CLASSIFY_ALIASES.update(orig)

    def test_duplicate_dirs_collapse_into_one(self, tmp_path):
        """The actual SVU bug: three release-style variants must land in one dir."""
        # First episode arrives — creates the canonical dir
        first_show = tmp_path / "Law & Order Special Victims Unit (1999)"
        first_season = first_show / "Season 6"
        first_season.mkdir(parents=True)
        first_video = first_season / "Law & Order Special Victims Unit (1999) - S06E14 - Game.mkv"
        first_video.write_text("v")

        # Subsequent episodes arrive with different release-style names
        second_show = tmp_path / "Law and Order SVU Season 13 Complete  WEB x264 [i_c]"
        second_show.mkdir()
        second_video = second_show / "Law and Order SVU s13e18 - Valentine's Day.mkv"
        second_video.write_text("v")

        third_show = tmp_path / "Law.And.Order.SVU.S24.COMPLETE.720p.AMZN.WEBRip.x264-GalaxyTV[TGx]"
        third_show.mkdir()
        third_video = third_show / "Law.And.Order.SVU.S24E22.720p.AMZN.WEBRip.x264-GalaxyTV.mkv"
        third_video.write_text("v")

        target = tmp_path / "TV Shows"
        target.mkdir()
        # Pre-create the canonical dir so the normalizer has something to match against
        (target / "Law & Order Special Victims Unit (1999)" / "Season 6").mkdir(parents=True)

        orig_type_dirs = mc.TYPE_DIRS.copy()
        orig_aliases = mc.SHOW_ALIASES.copy()
        mc.TYPE_DIRS["tv"] = target
        # Alias handles the SVU↔Special Victims Unit abbreviation
        mc.SHOW_ALIASES.clear()
        mc.SHOW_ALIASES["Law and Order SVU"] = "Law & Order Special Victims Unit (1999)"
        try:
            assert mc.create_symlink(str(first_video), "tv") is True
            assert mc.create_symlink(str(second_video), "tv") is True
            assert mc.create_symlink(str(third_video), "tv") is True

            show_dirs = sorted(p.name for p in target.iterdir() if p.is_dir())
            assert show_dirs == ["Law & Order Special Victims Unit (1999)"], show_dirs
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)
            mc.SHOW_ALIASES.clear()
            mc.SHOW_ALIASES.update(orig_aliases)

    def test_create_symlink_classify_aliases_routes_into_existing(self, tmp_path):
        """SVU dedup via --aliases file: variant file lands in the canonical dir."""
        # Canonical dir already exists (e.g. created by an earlier classify run)
        target = tmp_path / "TV Shows"
        (target / "Law & Order Special Victims Unit (1999)" / "Season 13").mkdir(parents=True)

        # Incoming file uses the SVU variant
        src = tmp_path / "Law.And.Order.SVU.S13.COMPLETE"
        src.mkdir()
        video = src / "Law.And.Order.SVU.S13E18.mkv"
        video.write_text("v")

        orig_type_dirs = mc.TYPE_DIRS.copy()
        orig_classify_aliases = mc.CLASSIFY_ALIASES.copy()
        mc.TYPE_DIRS["tv"] = target
        mc.CLASSIFY_ALIASES.clear()
        mc.CLASSIFY_ALIASES["Law and Order SVU"] = {
            "canonical": "Law & Order Special Victims Unit (1999)",
            "type": "tv",
        }
        try:
            assert mc.create_symlink(str(video), "tv") is True
            show_dirs = sorted(p.name for p in target.iterdir() if p.is_dir())
            assert show_dirs == ["Law & Order Special Victims Unit (1999)"], show_dirs
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)
            mc.CLASSIFY_ALIASES.clear()
            mc.CLASSIFY_ALIASES.update(orig_classify_aliases)

    def test_create_symlink_no_aliases_fresh_dir(self, tmp_path):
        """Fresh TYPE_DIR with no siblings → new dir uses the cleaned candidate."""
        target = tmp_path / "TV Shows"
        target.mkdir()

        src = tmp_path / "Better.Call.Saul.S01"
        src.mkdir()
        video = src / "Better.Call.Saul.S01E01.mkv"
        video.write_text("v")

        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            assert mc.create_symlink(str(video), "tv") is True
            show_dirs = [p.name for p in target.iterdir() if p.is_dir()]
            assert len(show_dirs) == 1, show_dirs
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_punct_and_case_variants_collapse_without_alias(self, tmp_path):
        """No alias needed for pure case/punctuation/year differences."""
        target = tmp_path / "TV Shows"
        target.mkdir()
        (target / "Breaking Bad (2008)").mkdir()

        # Source 1: parent dir uses dots and no year
        src1 = tmp_path / "Breaking.Bad.S01"
        src1.mkdir()
        v1 = src1 / "Breaking.Bad.S01E01.mkv"
        v1.write_text("v")

        # Source 2: parent dir uses different casing and year
        src2 = tmp_path / "breaking bad 2008"
        src2.mkdir()
        v2 = src2 / "breaking.bad.S02E01.mkv"
        v2.write_text("v")

        orig_type_dirs = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS["tv"] = target
        try:
            assert mc.create_symlink(str(v1), "tv") is True
            assert mc.create_symlink(str(v2), "tv") is True
            show_dirs = sorted(p.name for p in target.iterdir() if p.is_dir())
            assert show_dirs == ["Breaking Bad (2008)"], show_dirs
        finally:
            mc.TYPE_DIRS.update(orig_type_dirs)

    def test_load_config_show_aliases(self, tmp_path):
        config = {"showAliases": {"Foo": "Foo Bar Baz"}}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))
        orig = mc.SHOW_ALIASES.copy()
        try:
            mc.load_config(str(config_file))
            assert mc.SHOW_ALIASES == {"Foo": "Foo Bar Baz"}
        finally:
            mc.SHOW_ALIASES.clear()
            mc.SHOW_ALIASES.update(orig)


class TestReportDupes:
    """mc-imk: --report-dupes scans TYPE_DIRs and groups same-key sibling dirs."""

    def _stage(self, tmp_path, layout):
        """Build TYPE_DIRS under tmp_path. layout: {type_name: {show_dir: [files]}}.

        Returns a saved-state restore callable so tests can scope mutations.
        """
        type_dirs = {}
        for type_name, shows in layout.items():
            base = tmp_path / type_name
            base.mkdir()
            type_dirs[type_name] = base
            for show, files in shows.items():
                d = base / show
                d.mkdir()
                for f in files:
                    fp = d / f
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text("v")
        orig = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS.clear()
        mc.TYPE_DIRS.update(type_dirs)

        def restore():
            mc.TYPE_DIRS.clear()
            mc.TYPE_DIRS.update(orig)

        return restore

    def test_no_dupes_emits_empty_groups(self, tmp_path, capsys):
        restore = self._stage(tmp_path, {
            "tv": {"Andor": ["a.mkv"], "For All Mankind": ["b.mkv"]},
        })
        try:
            mc.report_duplicates()
            out = capsys.readouterr().out
            assert "Total within-type duplicate groups: 0" in out
            assert "Total cross-type duplicate groups:  0" in out
        finally:
            restore()

    def test_within_type_groups_collapsed_and_canonical_picked_by_filecount(
        self, tmp_path, capsys,
    ):
        restore = self._stage(tmp_path, {
            "tv": {
                # Three SVU variants that normalize to the same key.
                # The middle one has the most files → should be suggested canonical.
                "Law And Order SVU":         ["a.mkv"],
                "Law and Order SVU 1999":    ["b.mkv", "c.mkv", "d.mkv"],
                "Law And Order Special Victims Unit": ["e.mkv"],
                # A distinct show should NOT be grouped.
                "Andor":                     ["x.mkv"],
            },
        })
        try:
            mc.report_duplicates()
            out = capsys.readouterr().out
            # SVU group present, Andor not in any group
            assert "Law and Order SVU 1999" in out
            assert "← suggested canonical" in out
            # Canonical line should be on the highest-file-count dir
            canon_line = next(
                line for line in out.splitlines()
                if "← suggested canonical" in line
            )
            assert "Law and Order SVU 1999" in canon_line
            # The Special-Victims-Unit spelling normalizes differently, so it
            # forms its own (1-element) bucket — no group printed for it.
            assert "Special Victims Unit" not in canon_line
            assert "Total within-type duplicate groups: 1" in out
        finally:
            restore()

    def test_cross_type_detected_when_same_key_under_two_types(
        self, tmp_path, capsys,
    ):
        restore = self._stage(tmp_path, {
            "tv":    {"Undead Unluck": ["a.mkv"]},
            "anime": {"Undead Unluck": ["b.mkv"]},
        })
        try:
            mc.report_duplicates()
            out = capsys.readouterr().out
            assert "CROSS-TYPE DUPLICATES" in out
            assert "[tv]" in out and "[anime]" in out
            assert "Total cross-type duplicate groups:  1" in out
        finally:
            restore()

    def test_json_output_structure(self, tmp_path, capsys):
        restore = self._stage(tmp_path, {
            "tv": {
                "Andor": ["a.mkv"],
                "Andor (2022)": ["b.mkv", "c.mkv"],
            },
        })
        try:
            mc.report_duplicates(json_output=True)
            out = capsys.readouterr().out
            data = json.loads(out)
            assert "within_type" in data and "cross_type" in data
            tv_groups = data["within_type"]["tv"]
            assert len(tv_groups) == 1
            grp = tv_groups[0]
            assert grp["canonical_suggested"] == "Andor (2022)"
            assert {d["path"].rsplit("/", 1)[-1] for d in grp["directories"]} == {
                "Andor", "Andor (2022)",
            }
        finally:
            restore()

    def test_season_collision_flagged(self, tmp_path, capsys):
        """Kim's clue 3: same season number under two dirs of one group must be flagged."""
        # Two SVU dirs sharing the same normalized key, with overlapping Season 6.
        restore = self._stage(tmp_path, {
            "tv": {
                "Law and Order SVU":      [],
                "Law and Order SVU 1999": [],
            },
        })
        try:
            base = mc.TYPE_DIRS["tv"]
            (base / "Law and Order SVU" / "Season 6").mkdir()
            (base / "Law and Order SVU" / "Season 6" / "ep1.mkv").write_text("v")
            (base / "Law and Order SVU 1999" / "Season 6").mkdir()
            (base / "Law and Order SVU 1999" / "Season 6" / "ep2.mkv").write_text("v")
            # Non-colliding season so the canonical-size tiebreak is unambiguous.
            (base / "Law and Order SVU 1999" / "Season 7").mkdir()
            (base / "Law and Order SVU 1999" / "Season 7" / "ep3.mkv").write_text("v")

            mc.report_duplicates()
            out = capsys.readouterr().out
            assert "! Season 6 present in 2 dirs:" in out
            assert "Total season-level collisions:      1" in out
        finally:
            restore()

    def test_missing_type_dir_is_skipped(self, tmp_path, capsys):
        # Stage only tv; reference a nonexistent anime/movie dir.
        restore = self._stage(tmp_path, {"tv": {"Andor": ["a.mkv"]}})
        try:
            mc.TYPE_DIRS["anime"] = tmp_path / "does-not-exist"
            mc.report_duplicates()  # must not raise
            out = capsys.readouterr().out
            assert "Total within-type duplicate groups: 0" in out
        finally:
            restore()


# =============================================================================
# Phase 2: --merge-dupes
# =============================================================================

class TestMergeDupes:
    """mc-imk Phase 2: merge sibling dirs that --report-dupes identified."""

    def _stage_with_symlinks(self, tmp_path, layout, source_root=None):
        """Build TYPE_DIRS under tmp_path with SYMLINKS pointing into a source dir.

        layout: {type_name: {show_dir: [{"rel": "Season 1/ep.mkv", "src": "v1.mkv"}, ...]}}

        Each entry creates a symlink at <type>/<show>/<rel> pointing at <source>/<src>.
        Source files are deduplicated by name (same src across multiple entries → one
        physical file, multiple symlinks to it).
        """
        source = source_root or (tmp_path / "_src")
        source.mkdir(exist_ok=True)

        type_dirs = {}
        for type_name, shows in layout.items():
            base = tmp_path / type_name
            base.mkdir()
            type_dirs[type_name] = base
            for show, entries in shows.items():
                for entry in entries:
                    src_file = source / entry["src"]
                    if not src_file.exists():
                        src_file.write_text(entry["src"])
                    link = base / show / entry["rel"]
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(src_file)

        orig = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS.clear()
        mc.TYPE_DIRS.update(type_dirs)

        def restore():
            mc.TYPE_DIRS.clear()
            mc.TYPE_DIRS.update(orig)

        return restore

    # --- _pick_canonical -----------------------------------------------------

    def test_pick_canonical_prefers_most_files(self, tmp_path):
        a = tmp_path / "A"
        b = tmp_path / "B"
        chosen, _ = mc._pick_canonical([(a, 3), (b, 10)])
        assert chosen == b

    def test_pick_canonical_alphabetical_tiebreak(self, tmp_path):
        a = tmp_path / "Andor"
        b = tmp_path / "Andor (2022)"
        chosen, _ = mc._pick_canonical([(a, 12), (b, 12)])
        assert chosen == a  # "Andor" < "Andor (2022)" alphabetically

    def test_pick_canonical_prefer_year_form(self, tmp_path):
        a = tmp_path / "Andor"
        b = tmp_path / "Andor (2022)"
        chosen, _ = mc._pick_canonical([(a, 12), (b, 12)], prefer_year_form=True)
        assert chosen == b

    def test_pick_canonical_override_wins(self, tmp_path):
        a = tmp_path / "BigFile"
        b = tmp_path / "Andor (2022)"
        chosen, _ = mc._pick_canonical(
            [(a, 100), (b, 1)], override_name="Andor (2022)",
        )
        assert chosen == b

    def test_pick_canonical_alias_preferred_wins_over_count(self, tmp_path):
        a = tmp_path / "BigFile"
        b = tmp_path / "Special Form"
        chosen, _ = mc._pick_canonical(
            [(a, 100), (b, 1)], alias_preferred_name="Special Form",
        )
        assert chosen == b

    # --- _parse_canonical_override -------------------------------------------

    def test_parse_canonical_override_round_trip(self):
        key, name = mc._parse_canonical_override("Andor=Andor (2022)")
        assert key == mc._normalize_show_key("Andor")
        assert name == "Andor (2022)"

    def test_parse_canonical_override_bad_input_exits(self):
        with pytest.raises(SystemExit):
            mc._parse_canonical_override("no-equals-sign")

    # --- _load_aliases / _alias_lookup ---------------------------------------

    def test_load_aliases_json(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text(json.dumps({
            "aliases": {
                "Law and Order SVU 1999": {
                    "canonical": "Law & Order Special Victims Unit (1999)",
                    "type": "tv",
                },
                "Bare String": "Bare Canonical",
            }
        }))
        aliases = mc._load_aliases(str(p))
        assert aliases["Law and Order SVU 1999"]["canonical"] == \
            "Law & Order Special Victims Unit (1999)"
        assert aliases["Law and Order SVU 1999"]["type"] == "tv"
        assert aliases["Bare String"] == {"canonical": "Bare Canonical", "type": None}

    def test_alias_lookup_normalized_match(self):
        aliases = {"Law and Order SVU 1999": {"canonical": "X", "type": "tv"}}
        # Different casing/punctuation should still match via normalized key
        spec = mc._alias_lookup(aliases, "law & order svu 1999")
        assert spec is not None
        assert spec["canonical"] == "X"

    def test_alias_lookup_no_match(self):
        aliases = {"Foo": {"canonical": "Foo Bar", "type": "tv"}}
        assert mc._alias_lookup(aliases, "Different Show") is None
        assert mc._alias_lookup({}, "anything") is None
        assert mc._alias_lookup(None, "anything") is None

    # --- _merge_show_dir -----------------------------------------------------

    def test_merge_show_dir_moves_symlinks_into_target(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        src_show = tmp_path / "A"
        dst_show = tmp_path / "B"
        (src_show / "Season 1").mkdir(parents=True)
        (dst_show / "Season 1").mkdir(parents=True)
        ep_target = source / "ep1.mkv"
        ep_target.write_text("v")
        (src_show / "Season 1" / "ep1.mkv").symlink_to(ep_target)

        stats = mc._merge_show_dir(src_show, dst_show, dry_run=False)
        assert len(stats["moves"]) == 1
        assert stats["src_removed"]
        assert not src_show.exists()
        moved = dst_show / "Season 1" / "ep1.mkv"
        assert moved.is_symlink()
        assert os.readlink(moved) == str(ep_target)

    def test_merge_show_dir_dry_run_makes_no_changes(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        src_show = tmp_path / "A"
        dst_show = tmp_path / "B"
        (src_show / "Season 1").mkdir(parents=True)
        dst_show.mkdir()
        ep_target = source / "ep1.mkv"
        ep_target.write_text("v")
        link = src_show / "Season 1" / "ep1.mkv"
        link.symlink_to(ep_target)

        stats = mc._merge_show_dir(src_show, dst_show, dry_run=True)
        assert len(stats["moves"]) == 1
        assert not stats["src_removed"]
        # Source still present, destination still empty
        assert link.is_symlink()
        assert not (dst_show / "Season 1" / "ep1.mkv").exists()

    def test_merge_show_dir_same_target_skipped(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        ep_target = source / "ep1.mkv"
        ep_target.write_text("v")
        src_show = tmp_path / "A"
        dst_show = tmp_path / "B"
        (src_show / "Season 1").mkdir(parents=True)
        (dst_show / "Season 1").mkdir(parents=True)
        (src_show / "Season 1" / "ep1.mkv").symlink_to(ep_target)
        (dst_show / "Season 1" / "ep1.mkv").symlink_to(ep_target)

        stats = mc._merge_show_dir(src_show, dst_show, dry_run=False)
        assert stats["moves"] == []
        assert len(stats["same_target"]) == 1
        assert stats["conflicts"] == []
        # Source's duplicate symlink should be gone
        assert not (src_show / "Season 1" / "ep1.mkv").exists()
        # Destination still in place
        assert (dst_show / "Season 1" / "ep1.mkv").is_symlink()

    def test_merge_show_dir_differing_target_is_conflict(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        v1 = source / "v1.mkv"; v1.write_text("v1")
        v2 = source / "v2.mkv"; v2.write_text("v2")
        src_show = tmp_path / "A"
        dst_show = tmp_path / "B"
        (src_show / "Season 1").mkdir(parents=True)
        (dst_show / "Season 1").mkdir(parents=True)
        (src_show / "Season 1" / "ep1.mkv").symlink_to(v1)
        (dst_show / "Season 1" / "ep1.mkv").symlink_to(v2)

        stats = mc._merge_show_dir(src_show, dst_show, dry_run=False)
        assert stats["moves"] == []
        assert stats["same_target"] == []
        assert len(stats["conflicts"]) == 1
        # Source dir not removed because conflict left a file behind
        assert (src_show / "Season 1" / "ep1.mkv").exists()
        assert not stats["src_removed"]

    def test_merge_show_dir_merges_into_existing_season(self, tmp_path):
        """Season N in src merges into Season N in dst — not nested."""
        source = tmp_path / "src"
        source.mkdir()
        e1 = source / "e1.mkv"; e1.write_text("e1")
        e2 = source / "e2.mkv"; e2.write_text("e2")
        src_show = tmp_path / "A"
        dst_show = tmp_path / "B"
        (src_show / "Season 1").mkdir(parents=True)
        (dst_show / "Season 1").mkdir(parents=True)
        (src_show / "Season 1" / "ep1.mkv").symlink_to(e1)
        (dst_show / "Season 1" / "ep2.mkv").symlink_to(e2)

        mc._merge_show_dir(src_show, dst_show, dry_run=False)

        season1 = dst_show / "Season 1"
        assert sorted(p.name for p in season1.iterdir()) == ["ep1.mkv", "ep2.mkv"]
        # No nested "Season 1/Season 1" produced
        assert not (season1 / "Season 1").exists()

    # --- merge_duplicates orchestration -------------------------------------

    def test_merge_duplicates_within_type_dry_run(self, tmp_path, capsys):
        restore = self._stage_with_symlinks(tmp_path, {
            "tv": {
                "Andor":        [{"rel": "Season 1/ep1.mkv", "src": "andor_s1e1.mkv"}],
                "Andor (2022)": [
                    {"rel": "Season 2/ep1.mkv", "src": "andor_s2e1.mkv"},
                    {"rel": "Season 2/ep2.mkv", "src": "andor_s2e2.mkv"},
                ],
            },
        })
        try:
            report = mc.merge_duplicates(dry_run=True)
            assert report["dry_run"]
            assert len(report["within_type"]) == 1
            grp = report["within_type"][0]
            # Andor (2022) has more files → canonical
            assert grp["canonical"].endswith("Andor (2022)")
            # Both source dirs still on disk (dry-run)
            assert (tmp_path / "tv" / "Andor" / "Season 1" / "ep1.mkv").is_symlink()
            assert (tmp_path / "tv" / "Andor (2022)" / "Season 2" / "ep1.mkv").is_symlink()
        finally:
            restore()

    def test_merge_duplicates_within_type_apply(self, tmp_path):
        restore = self._stage_with_symlinks(tmp_path, {
            "tv": {
                "Andor":        [{"rel": "Season 1/ep1.mkv", "src": "a_s1e1.mkv"}],
                "Andor (2022)": [
                    {"rel": "Season 2/ep1.mkv", "src": "a_s2e1.mkv"},
                    {"rel": "Season 2/ep2.mkv", "src": "a_s2e2.mkv"},
                ],
            },
        })
        try:
            mc.merge_duplicates(dry_run=False, prefer_year_form=True)
            base = tmp_path / "tv"
            shows = sorted(p.name for p in base.iterdir())
            assert shows == ["Andor (2022)"]
            files = sorted(
                str(p.relative_to(base / "Andor (2022)"))
                for p in (base / "Andor (2022)").rglob("*")
                if p.is_symlink()
            )
            assert files == ["Season 1/ep1.mkv", "Season 2/ep1.mkv", "Season 2/ep2.mkv"]
        finally:
            restore()

    def test_merge_duplicates_alias_bridges_cross_key_collision(self, tmp_path):
        """SVU case: aliases let two normalized-key groups collapse into one."""
        restore = self._stage_with_symlinks(tmp_path, {
            "tv": {
                "Law & Order Special Victims Unit (1999)": [
                    {"rel": "Season 1/ep1.mkv", "src": "svu_s1e1.mkv"},
                    {"rel": "Season 1/ep2.mkv", "src": "svu_s1e2.mkv"},
                ],
                "Law and Order SVU 1999": [
                    {"rel": "Season 2/ep1.mkv", "src": "svu_s2e1.mkv"},
                ],
            },
        })
        aliases_file = tmp_path / "aliases.json"
        aliases_file.write_text(json.dumps({
            "aliases": {
                "Law and Order SVU 1999": {
                    "canonical": "Law & Order Special Victims Unit (1999)",
                    "type": "tv",
                },
            },
        }))
        try:
            report = mc.merge_duplicates(
                dry_run=False, aliases_path=str(aliases_file),
            )
            # No conflicts, no needs-review
            assert report["needs_review"] == []
            assert len(report["within_type"]) == 1
            grp = report["within_type"][0]
            assert grp["canonical"].endswith(
                "Law & Order Special Victims Unit (1999)"
            )
            # All seasons under the canonical dir
            base = tmp_path / "tv" / "Law & Order Special Victims Unit (1999)"
            seasons = sorted(p.name for p in base.iterdir())
            assert seasons == ["Season 1", "Season 2"]
            # Source dir gone
            assert not (tmp_path / "tv" / "Law and Order SVU 1999").exists()
        finally:
            restore()

    def test_merge_duplicates_cross_type_needs_alias(self, tmp_path):
        restore = self._stage_with_symlinks(tmp_path, {
            "tv":    {"Undead Unluck": [{"rel": "Season 1/e1.mkv", "src": "uu1.mkv"}]},
            "anime": {"Undead Unluck": [{"rel": "Season 1/e2.mkv", "src": "uu2.mkv"}]},
        })
        try:
            # No aliases → goes to needs_review
            report = mc.merge_duplicates(dry_run=True)
            assert len(report["needs_review"]) == 1
            assert report["needs_review"][0]["key"] == "undead unluck"
            # Both source dirs untouched
            assert (tmp_path / "tv" / "Undead Unluck").exists()
            assert (tmp_path / "anime" / "Undead Unluck").exists()
        finally:
            restore()

    def test_merge_duplicates_cross_type_apply_with_alias(self, tmp_path):
        restore = self._stage_with_symlinks(tmp_path, {
            "tv":    {"Undead Unluck": [{"rel": "Season 1/e_tv.mkv",    "src": "uu_tv.mkv"}]},
            "anime": {"Undead Unluck": [{"rel": "Season 1/e_anime.mkv", "src": "uu_an.mkv"}]},
        })
        aliases_file = tmp_path / "aliases.json"
        aliases_file.write_text(json.dumps({
            "aliases": {"Undead Unluck": {"type": "anime"}},
        }))
        try:
            mc.merge_duplicates(dry_run=False, aliases_path=str(aliases_file))
            # tv copy should be gone, anime has both episodes
            assert not (tmp_path / "tv" / "Undead Unluck").exists()
            anime_files = sorted(
                p.name for p in (tmp_path / "anime" / "Undead Unluck" / "Season 1").iterdir()
            )
            assert anime_files == ["e_anime.mkv", "e_tv.mkv"]
        finally:
            restore()

    def test_merge_duplicates_canonical_override(self, tmp_path):
        """User can force a specific dir to be canonical regardless of file counts."""
        restore = self._stage_with_symlinks(tmp_path, {
            "tv": {
                "Andor":        [{"rel": "S1/e.mkv", "src": "a1.mkv"}],
                "Andor (2022)": [
                    {"rel": "S2/e1.mkv", "src": "a2.mkv"},
                    {"rel": "S2/e2.mkv", "src": "a3.mkv"},
                    {"rel": "S2/e3.mkv", "src": "a4.mkv"},
                ],
            },
        })
        try:
            mc.merge_duplicates(
                dry_run=False,
                canonical_overrides=["Andor=Andor"],
            )
            # Despite Andor (2022) having more files, override forces "Andor"
            base = tmp_path / "tv"
            shows = sorted(p.name for p in base.iterdir())
            assert shows == ["Andor"]
        finally:
            restore()

    def test_merge_duplicates_no_dupes_returns_empty(self, tmp_path):
        restore = self._stage_with_symlinks(tmp_path, {
            "tv": {"Andor": [{"rel": "S1/e.mkv", "src": "a.mkv"}]},
        })
        try:
            report = mc.merge_duplicates(dry_run=True)
            assert report["within_type"] == []
            assert report["cross_type"] == []
            assert report["needs_review"] == []
        finally:
            restore()

    def test_merge_duplicates_json_output(self, tmp_path, capsys):
        restore = self._stage_with_symlinks(tmp_path, {
            "tv": {
                "Andor":        [{"rel": "S1/e.mkv",  "src": "a.mkv"}],
                "Andor (2022)": [
                    {"rel": "S2/e1.mkv", "src": "b1.mkv"},
                    {"rel": "S2/e2.mkv", "src": "b2.mkv"},
                ],
            },
        })
        try:
            mc.merge_duplicates(dry_run=True, json_output=True)
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["dry_run"]
            assert len(data["within_type"]) == 1
            assert data["within_type"][0]["canonical"].endswith("Andor (2022)")
        finally:
            restore()

    def test_merge_duplicates_conflict_reported_and_src_retained(self, tmp_path):
        """When source and dest have differing symlinks at the same path,
        the merge must skip with a conflict and leave the source dir in place."""
        source = tmp_path / "_src"
        source.mkdir()
        v1 = source / "v1.mkv"; v1.write_text("v1")
        v2 = source / "v2.mkv"; v2.write_text("v2")
        tv = tmp_path / "tv"
        (tv / "Andor" / "Season 1").mkdir(parents=True)
        (tv / "Andor (2022)" / "Season 1").mkdir(parents=True)
        (tv / "Andor" / "Season 1" / "ep1.mkv").symlink_to(v1)
        (tv / "Andor (2022)" / "Season 1" / "ep1.mkv").symlink_to(v2)

        orig = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS.clear()
        mc.TYPE_DIRS["tv"] = tv
        try:
            report = mc.merge_duplicates(dry_run=False, prefer_year_form=True)
            grp = report["within_type"][0]
            # Andor (2022) is canonical; Andor's ep1.mkv conflicts
            merge = grp["merges"][0]
            assert merge["moved"] == 0
            assert len(merge["conflicts"]) == 1
            assert not merge["src_removed"]
            # The conflicting source symlink survives
            assert (tv / "Andor" / "Season 1" / "ep1.mkv").is_symlink()
        finally:
            mc.TYPE_DIRS.clear()
            mc.TYPE_DIRS.update(orig)

    def test_merge_duplicates_yaml_aliases_skipped_without_pyyaml(self, tmp_path):
        """YAML aliases path raises SystemExit if PyYAML isn't installed.

        Skipped when PyYAML *is* present — we just want to confirm the failure
        mode is a clean SystemExit rather than an ImportError surfacing.
        """
        try:
            import yaml  # noqa: F401
            pytest.skip("PyYAML installed — can't exercise the missing-module path")
        except ImportError:
            pass
        p = tmp_path / "aliases.yaml"
        p.write_text("aliases: {}\n")
        with pytest.raises(SystemExit):
            mc._load_aliases(str(p))


# =============================================================================
# Phase 3: --report-episode-dupes / --merge-episode-dupes (mc-x2b)
# =============================================================================


class TestParseEpisodeKey:
    """_parse_episode_key extracts (season, episode) from filenames."""

    def test_standard_sxxeyy(self):
        assert mc._parse_episode_key("Show.S06E01.AMZN.WEBRip.mkv") == (6, 1)

    def test_anchored_at_start(self):
        assert mc._parse_episode_key("S01E12.mkv") == (1, 12)

    def test_lowercase(self):
        assert mc._parse_episode_key("show.s02e05.mkv") == (2, 5)

    def test_three_digit_episode(self):
        assert mc._parse_episode_key("Show.S01E125.mkv") == (1, 125)

    def test_no_match_returns_none(self):
        assert mc._parse_episode_key("movie.2022.1080p.mkv") is None

    def test_sxx_only_does_not_match(self):
        # S01 alone (no E) is a season marker, not an episode key.
        assert mc._parse_episode_key("Show.S01.Complete.mkv") is None

    def test_anitopy_fallback_uses_season_hint(self):
        if mc.anitopy is None:
            pytest.skip("anitopy not installed")
        # Bare anime episode form — anitopy will extract episode=12,
        # season comes from the directory hint.
        result = mc._parse_episode_key(
            "[SubsPlease] Show - 12 [1080p].mkv", season_hint=3,
        )
        assert result == (3, 12)

    def test_anitopy_fallback_without_hint_returns_none(self):
        if mc.anitopy is None:
            pytest.skip("anitopy not installed")
        # No SxxEyy, no season hint, anitopy may extract season=None →
        # we cannot form a key.
        result = mc._parse_episode_key(
            "[SubsPlease] Show - 12 [1080p].mkv", season_hint=None,
        )
        assert result is None


class TestExternalSubsPresent:
    """_external_subs_present checks sibling subtitle files by stem-prefix."""

    def test_link_side_subtitle_detected(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        target = src / "ep1.mkv"
        target.write_text("v")
        d = tmp_path / "Season 1"
        d.mkdir()
        link = d / "Show.S01E01.mkv"
        link.symlink_to(target)
        sub = d / "Show.S01E01.en.srt"
        sub.write_text("subs")
        assert mc._external_subs_present(link) is True

    def test_target_side_subtitle_detected(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        target = src / "Show.S01E01.mkv"
        target.write_text("v")
        (src / "Show.S01E01.en.srt").write_text("subs")
        d = tmp_path / "Season 1"
        d.mkdir()
        link = d / "Show.S01E01.mkv"
        link.symlink_to(target)
        assert mc._external_subs_present(link) is True

    def test_no_subs_returns_false(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        target = src / "ep1.mkv"
        target.write_text("v")
        d = tmp_path / "Season 1"
        d.mkdir()
        link = d / "Show.S01E01.mkv"
        link.symlink_to(target)
        assert mc._external_subs_present(link) is False


class TestEpisodeProbeCache:
    """_episode_probe caches by (path, mtime_ns) and degrades cleanly."""

    def test_returns_none_for_missing_file(self, tmp_path):
        missing = tmp_path / "nope.mkv"
        assert mc._episode_probe(missing) is None

    def test_cache_hit_avoids_subprocess(self, tmp_path, monkeypatch):
        f = tmp_path / "v.mkv"
        f.write_text("v")
        calls = {"n": 0}
        real_run = mc.subprocess.run

        def fake_run(*a, **kw):
            calls["n"] += 1
            return real_run(["true"], capture_output=True, text=True)

        mc._EPISODE_PROBE_CACHE.clear()
        monkeypatch.setattr(mc.subprocess, "run", fake_run)
        mc._episode_probe(f)
        mc._episode_probe(f)
        # Second call should hit cache → run invoked exactly once
        assert calls["n"] == 1

    def test_probe_with_mocked_ffprobe_extracts_height_and_subs(
        self, tmp_path, monkeypatch,
    ):
        f = tmp_path / "v.mkv"
        f.write_text("v")
        mc._EPISODE_PROBE_CACHE.clear()
        payload = json.dumps({
            "streams": [
                {"codec_type": "video", "height": 1080},
                {"codec_type": "audio", "tags": {"language": "eng"}},
                {"codec_type": "subtitle", "codec_name": "ass"},
            ],
        })

        def fake_run(*a, **kw):
            return type("R", (), {"returncode": 0, "stdout": payload, "stderr": ""})()

        monkeypatch.setattr(mc.subprocess, "run", fake_run)
        info = mc._episode_probe(f)
        assert info["height"] == 1080
        assert info["has_subs"] is True
        assert info["sub_count"] == 1


class TestPickEpisodeWinner:
    """_pick_episode_winner applies the bead's priority list."""

    def _mk_link(self, tmp_path, name, content="v", season_dir="Season 1"):
        src = tmp_path / "_src"
        src.mkdir(exist_ok=True)
        target = src / f"{name}.target"
        target.write_text(content)
        d = tmp_path / season_dir
        d.mkdir(exist_ok=True)
        link = d / name
        link.symlink_to(target)
        return link

    def _stub_probe(self, monkeypatch, probes):
        """probes: {filename: dict-with-height/has_subs/size/mtime}."""
        def fake(link):
            return probes.get(Path(link).name, {})
        monkeypatch.setattr(mc, "_episode_probe", fake)
        monkeypatch.setattr(mc, "_external_subs_present", lambda l: False)

    def test_subs_presence_beats_height(self, tmp_path, monkeypatch):
        a = self._mk_link(tmp_path, "a.S01E01.mkv")
        b = self._mk_link(tmp_path, "b.S01E01.mkv")
        self._stub_probe(monkeypatch, {
            "a.S01E01.mkv": {"has_subs": True, "height": 720, "size": 1, "mtime": 1},
            "b.S01E01.mkv": {"has_subs": False, "height": 1080, "size": 100, "mtime": 2},
        })
        winner, losers = mc._pick_episode_winner([a, b])
        assert winner == a
        assert losers == [b]

    def test_height_beats_release_group_regex(self, tmp_path, monkeypatch):
        a = self._mk_link(tmp_path, "S01E01.AMZN.WEBRip.mkv")
        b = self._mk_link(tmp_path, "S01E01.GENERIC.mkv")
        self._stub_probe(monkeypatch, {
            "S01E01.AMZN.WEBRip.mkv": {"height": 720, "size": 1, "mtime": 1},
            "S01E01.GENERIC.mkv":     {"height": 1080, "size": 1, "mtime": 1},
        })
        winner, _ = mc._pick_episode_winner([a, b], prefer_release_group="AMZN")
        assert winner == b  # 1080p wins despite AMZN regex match on the other

    def test_release_group_regex_beats_size(self, tmp_path, monkeypatch):
        a = self._mk_link(tmp_path, "S06E01.GENERIC.mkv")
        b = self._mk_link(tmp_path, "Law.Order.SVU.S06E01.AMZN.WEBRip.x265.ImE.mkv")
        # Equal height. Regex hit on b wins despite a being larger.
        self._stub_probe(monkeypatch, {
            "S06E01.GENERIC.mkv":
                {"height": 1080, "size": 5_000_000_000, "mtime": 1},
            "Law.Order.SVU.S06E01.AMZN.WEBRip.x265.ImE.mkv":
                {"height": 1080, "size": 1_500_000_000, "mtime": 1},
        })
        winner, _ = mc._pick_episode_winner(
            [a, b], prefer_release_group="AMZN.*WEBRip|x265|ImE",
        )
        assert winner == b

    def test_size_beats_mtime(self, tmp_path, monkeypatch):
        a = self._mk_link(tmp_path, "a.S01E01.mkv")
        b = self._mk_link(tmp_path, "b.S01E01.mkv")
        self._stub_probe(monkeypatch, {
            "a.S01E01.mkv": {"height": 720, "size": 100, "mtime": 1.0},
            "b.S01E01.mkv": {"height": 720, "size": 50,  "mtime": 999.0},
        })
        winner, _ = mc._pick_episode_winner([a, b])
        assert winner == a

    def test_name_lexical_fallback(self, tmp_path, monkeypatch):
        a = self._mk_link(tmp_path, "a.S01E01.mkv")
        b = self._mk_link(tmp_path, "b.S01E01.mkv")
        # All probe signals equal → lex ascending wins.
        equal = {"height": 720, "size": 1, "mtime": 1.0}
        self._stub_probe(monkeypatch, {
            "a.S01E01.mkv": equal, "b.S01E01.mkv": equal,
        })
        winner, _ = mc._pick_episode_winner([a, b])
        assert winner == a


class TestScanEpisodeDupes:
    """_scan_episode_dupes walks TYPE_DIRs and groups by (season, episode)."""

    def _stage(self, tmp_path, layout):
        """layout: {type_name: {show: {season_dir: [(filename, target_name)]}}}"""
        src = tmp_path / "_src"
        src.mkdir(exist_ok=True)
        type_dirs = {}
        for type_name, shows in layout.items():
            base = tmp_path / type_name
            base.mkdir()
            type_dirs[type_name] = base
            for show, seasons in shows.items():
                for season_dir, entries in seasons.items():
                    d = base / show / season_dir
                    d.mkdir(parents=True)
                    for fname, tgt in entries:
                        tgt_path = src / tgt
                        if not tgt_path.exists():
                            tgt_path.write_text(tgt)
                        (d / fname).symlink_to(tgt_path)
        orig = mc.TYPE_DIRS.copy()
        mc.TYPE_DIRS.clear()
        mc.TYPE_DIRS.update(type_dirs)

        def restore():
            mc.TYPE_DIRS.clear()
            mc.TYPE_DIRS.update(orig)

        return restore

    def test_no_dupes_returns_empty(self, tmp_path):
        restore = self._stage(tmp_path, {
            "tv": {"Show": {"Season 1": [
                ("Show.S01E01.mkv", "s1e1.mkv"),
                ("Show.S01E02.mkv", "s1e2.mkv"),
            ]}},
        })
        try:
            assert mc._scan_episode_dupes() == []
        finally:
            restore()

    def test_two_encodings_grouped(self, tmp_path):
        restore = self._stage(tmp_path, {
            "tv": {"SVU": {"Season 6": [
                ("S06E01.GENERIC.mkv",                            "g1.mkv"),
                ("Law.Order.SVU.S06E01.AMZN.WEBRip.x265.ImE.mkv", "a1.mkv"),
                ("S06E02.GENERIC.mkv",                            "g2.mkv"),
            ]}},
        })
        try:
            dupes = mc._scan_episode_dupes()
            assert len(dupes) == 1
            d = dupes[0]
            assert d["season"] == 6 and d["episode"] == 1
            assert len(d["links"]) == 2
            assert d["type"] == "tv"
            assert d["show"] == "SVU"
        finally:
            restore()

    def test_skips_non_season_dirs(self, tmp_path):
        restore = self._stage(tmp_path, {
            "tv": {"Show": {"Specials": [  # not a season dir
                ("Show.S00E01.mkv", "x1.mkv"),
                ("Show.S00E01.alt.mkv", "x2.mkv"),
            ]}},
        })
        try:
            # "Specials" is not a Season N dir → no scan, no dupes reported.
            assert mc._scan_episode_dupes() == []
        finally:
            restore()

    def test_movies_dir_ignored(self, tmp_path):
        # Movies live flat; no Season N dirs → no dupes.
        restore = self._stage(tmp_path, {
            "movie": {"Andor (2022)": {"Season 1": [
                ("Andor.S01E01.mkv", "x.mkv"),
                ("Andor.S01E01.alt.mkv", "y.mkv"),
            ]}},
        })
        # Movies normally don't have Season N — but if they do, treat consistently.
        # The function only filters on _is_season_dir, so movies' "Season 1" *would*
        # be scanned. Keep this test as a sanity check on TYPE_DIRS traversal.
        try:
            dupes = mc._scan_episode_dupes()
            assert len(dupes) == 1
            assert dupes[0]["type"] == "movie"
        finally:
            restore()


class TestReportEpisodeDuplicates:
    """report_episode_duplicates text + JSON output shape."""

    def _stage(self, tmp_path, layout):
        return TestScanEpisodeDupes._stage(self, tmp_path, layout)

    def test_empty_prints_no_dupes(self, tmp_path, capsys):
        restore = self._stage(tmp_path, {
            "tv": {"Show": {"Season 1": [("Show.S01E01.mkv", "x.mkv")]}},
        })
        try:
            mc.report_episode_duplicates()
            out = capsys.readouterr().out
            assert "No episode-level duplicates found." in out
        finally:
            restore()

    def test_text_report_lists_keep_and_drops(self, tmp_path, capsys, monkeypatch):
        restore = self._stage(tmp_path, {
            "tv": {"SVU": {"Season 6": [
                ("S06E01.GENERIC.mkv",                            "g.mkv"),
                ("Law.Order.SVU.S06E01.AMZN.WEBRip.x265.ImE.mkv", "a.mkv"),
            ]}},
        })
        # Force the AMZN-tagged file to win via the regex.
        monkeypatch.setattr(mc, "_episode_probe", lambda l: {
            "height": 1080, "size": 1, "mtime": 1.0,
            "has_subs": False, "sub_count": 0,
        })
        monkeypatch.setattr(mc, "_external_subs_present", lambda l: False)
        try:
            mc.report_episode_duplicates(prefer_release_group="AMZN|x265|ImE")
            out = capsys.readouterr().out
            assert "SVU" in out
            assert "S06E01: 2 encodings" in out
            # Winner line gets the "← keep" mark; the GENERIC file does not.
            keep_line = next(
                l for l in out.splitlines() if "← keep" in l
            )
            assert "AMZN" in keep_line
            assert "Symlinks that would be unlinked: 1" in out
        finally:
            restore()

    def test_json_output_shape(self, tmp_path, capsys, monkeypatch):
        restore = self._stage(tmp_path, {
            "tv": {"SVU": {"Season 6": [
                ("S06E01.GENERIC.mkv",                            "g.mkv"),
                ("Law.Order.SVU.S06E01.AMZN.WEBRip.x265.ImE.mkv", "a.mkv"),
            ]}},
        })
        monkeypatch.setattr(mc, "_episode_probe", lambda l: {
            "height": 1080, "size": 1, "mtime": 1.0,
            "has_subs": False, "sub_count": 0,
        })
        monkeypatch.setattr(mc, "_external_subs_present", lambda l: False)
        try:
            mc.report_episode_duplicates(
                json_output=True, prefer_release_group="AMZN",
            )
            data = json.loads(capsys.readouterr().out)
            assert len(data) == 1
            grp = data[0]
            assert grp["season"] == 6 and grp["episode"] == 1
            assert len(grp["candidates"]) == 2
            winners = [c for c in grp["candidates"] if c["is_winner"]]
            assert len(winners) == 1
            assert "AMZN" in winners[0]["path"]
        finally:
            restore()


class TestMergeEpisodeDuplicates:
    """merge_episode_duplicates dry-run vs apply, targets stay intact."""

    def _stage(self, tmp_path, layout):
        return TestScanEpisodeDupes._stage(self, tmp_path, layout)

    def test_dry_run_makes_no_changes(self, tmp_path, monkeypatch):
        restore = self._stage(tmp_path, {
            "tv": {"SVU": {"Season 6": [
                ("S06E01.A.mkv", "ga.mkv"),
                ("S06E01.B.mkv", "gb.mkv"),
            ]}},
        })
        monkeypatch.setattr(mc, "_episode_probe", lambda l: {
            "height": 1080, "size": 1, "mtime": 1.0,
        })
        monkeypatch.setattr(mc, "_external_subs_present", lambda l: False)
        try:
            report = mc.merge_episode_duplicates(dry_run=True)
            assert report["dry_run"] is True
            assert report["unlinked"] == 1
            season = tmp_path / "tv" / "SVU" / "Season 6"
            files = sorted(p.name for p in season.iterdir())
            # Both symlinks still on disk in dry-run mode.
            assert files == ["S06E01.A.mkv", "S06E01.B.mkv"]
        finally:
            restore()

    def test_apply_unlinks_losers_only(self, tmp_path, monkeypatch):
        restore = self._stage(tmp_path, {
            "tv": {"SVU": {"Season 6": [
                ("S06E01.GENERIC.mkv",                            "g.mkv"),
                ("Law.Order.SVU.S06E01.AMZN.WEBRip.x265.ImE.mkv", "a.mkv"),
            ]}},
        })
        monkeypatch.setattr(mc, "_episode_probe", lambda l: {
            "height": 1080, "size": 1, "mtime": 1.0,
        })
        monkeypatch.setattr(mc, "_external_subs_present", lambda l: False)
        try:
            mc.merge_episode_duplicates(
                dry_run=False, prefer_release_group="AMZN|x265|ImE",
            )
            season = tmp_path / "tv" / "SVU" / "Season 6"
            remaining = sorted(p.name for p in season.iterdir())
            # AMZN file should be the sole survivor.
            assert remaining == [
                "Law.Order.SVU.S06E01.AMZN.WEBRip.x265.ImE.mkv",
            ]
            # Target files (in _src) must be intact.
            src = tmp_path / "_src"
            assert (src / "g.mkv").exists()
            assert (src / "a.mkv").exists()
        finally:
            restore()

    def test_apply_no_dupes_is_noop(self, tmp_path):
        restore = self._stage(tmp_path, {
            "tv": {"Show": {"Season 1": [
                ("Show.S01E01.mkv", "x1.mkv"),
                ("Show.S01E02.mkv", "x2.mkv"),
            ]}},
        })
        try:
            report = mc.merge_episode_duplicates(dry_run=False)
            assert report["unlinked"] == 0
            assert report["groups"] == []
        finally:
            restore()

    def test_json_output_shape(self, tmp_path, capsys, monkeypatch):
        restore = self._stage(tmp_path, {
            "tv": {"SVU": {"Season 6": [
                ("S06E01.A.mkv", "a.mkv"),
                ("S06E01.B.mkv", "b.mkv"),
            ]}},
        })
        monkeypatch.setattr(mc, "_episode_probe", lambda l: {
            "height": 1080, "size": 1, "mtime": 1.0,
        })
        monkeypatch.setattr(mc, "_external_subs_present", lambda l: False)
        try:
            mc.merge_episode_duplicates(dry_run=True, json_output=True)
            data = json.loads(capsys.readouterr().out)
            assert data["dry_run"] is True
            assert data["unlinked"] == 1
            assert len(data["groups"]) == 1
            assert data["groups"][0]["season"] == 6
            assert data["groups"][0]["episode"] == 1
            assert len(data["groups"][0]["unlinked"]) == 1
        finally:
            restore()


class TestLLMShowVerification:
    """mc-5mc: LLM verification when creating a new top-level show dir."""

    def test_fuzzy_similarity_triggers_llm(self, tmp_path):
        """When a sibling has token_set_ratio >= 60, LLM is consulted."""
        target = tmp_path / "TV Shows"
        target.mkdir()
        (target / "Sousou no Frieren").mkdir()

        with mock.patch.object(mc, "_fuzzy_similar_siblings", return_value=["Sousou no Frieren"]) as mock_fuzz, \
             mock.patch.object(mc, "_llm_verify_new_show", return_value="Sousou no Frieren") as mock_llm:
            result = mc._canonical_show_dir(target, "Frieren")
            mock_fuzz.assert_called_once()
            mock_llm.assert_called_once_with("Frieren", ["Sousou no Frieren"])
            assert result == "Sousou no Frieren"

    def test_no_similar_siblings_skips_llm(self, tmp_path):
        """When no sibling is similar enough, LLM is not called."""
        target = tmp_path / "TV Shows"
        target.mkdir()
        (target / "Breaking Bad (2008)").mkdir()

        with mock.patch.object(mc, "_llm_verify_new_show", return_value="Breaking Bad (2008)") as mock_llm:
            result = mc._canonical_show_dir(target, "The Wire")
            mock_llm.assert_not_called()
            assert result == "The Wire"

    def test_no_existing_dirs_skips_llm(self, tmp_path):
        """When TYPE_DIR has no sibling dirs, LLM is not called."""
        target = tmp_path / "TV Shows"
        target.mkdir()

        with mock.patch.object(mc, "_llm_verify_new_show", return_value="anything") as mock_llm:
            result = mc._canonical_show_dir(target, "Frieren")
            mock_llm.assert_not_called()
            assert result == "Frieren"

    def test_llm_match_reuses_sibling(self, tmp_path):
        """LLM MATCH result reuses the sibling's exact dir name."""
        target = tmp_path / "Anime"
        target.mkdir()
        (target / "Sousou no Frieren").mkdir()

        cache_file = tmp_path / "cache.json"
        review_log = tmp_path / "review.log"

        with mock.patch.object(mc, "LLM_SHOW_CACHE_FILE", cache_file), \
             mock.patch.object(mc, "NEEDS_CLASSIFY_REVIEW_LOG", review_log), \
             mock.patch.object(mc, "OLLAMA_HOST", "http://localhost:11434"), \
             mock.patch.object(mc, "OLLAMA_MODEL", "test-model"), \
             mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "response": "MATCH 1",
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = mc._llm_verify_new_show("Frieren", ["Sousou no Frieren"])
            assert result == "Sousou no Frieren"

    def test_llm_new_proceeds_with_candidate(self, tmp_path):
        """LLM NEW result returns the original candidate."""
        cache_file = tmp_path / "cache.json"
        review_log = tmp_path / "review.log"

        with mock.patch.object(mc, "LLM_SHOW_CACHE_FILE", cache_file), \
             mock.patch.object(mc, "NEEDS_CLASSIFY_REVIEW_LOG", review_log), \
             mock.patch.object(mc, "OLLAMA_HOST", "http://localhost:11434"), \
             mock.patch.object(mc, "OLLAMA_MODEL", "test-model"), \
             mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "response": "NEW",
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = mc._llm_verify_new_show("Frieren", ["Sousou no Frieren"])
            assert result == "Frieren"

    def test_llm_failure_falls_through_to_new(self, tmp_path):
        """When LLM is unreachable, falls through to NEW and logs to review log."""
        cache_file = tmp_path / "cache.json"
        review_log = tmp_path / "review.log"

        with mock.patch.object(mc, "LLM_SHOW_CACHE_FILE", cache_file), \
             mock.patch.object(mc, "NEEDS_CLASSIFY_REVIEW_LOG", review_log), \
             mock.patch.object(mc, "OLLAMA_HOST", "http://localhost:11434"), \
             mock.patch.object(mc, "OLLAMA_MODEL", "test-model"), \
             mock.patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            result = mc._llm_verify_new_show("Frieren", ["Sousou no Frieren"])
            assert result == "Frieren"
            assert review_log.exists()
            content = review_log.read_text()
            assert "llm_error" in content
            assert "Frieren" in content

    def test_llm_unparseable_response_falls_through(self, tmp_path):
        """Unparseable LLM response falls through to NEW and logs."""
        cache_file = tmp_path / "cache.json"
        review_log = tmp_path / "review.log"

        with mock.patch.object(mc, "LLM_SHOW_CACHE_FILE", cache_file), \
             mock.patch.object(mc, "NEEDS_CLASSIFY_REVIEW_LOG", review_log), \
             mock.patch.object(mc, "OLLAMA_HOST", "http://localhost:11434"), \
             mock.patch.object(mc, "OLLAMA_MODEL", "test-model"), \
             mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "response": "I think it's the same show",
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = mc._llm_verify_new_show("Frieren", ["Sousou no Frieren"])
            assert result == "Frieren"
            assert review_log.exists()
            content = review_log.read_text()
            assert "llm_unparseable" in content

    def test_cache_hit_avoids_llm_call(self, tmp_path):
        """Cache hit returns stored decision without calling LLM."""
        cache_file = tmp_path / "cache.json"
        review_log = tmp_path / "review.log"
        cache_file.write_text(json.dumps({
            "frieren|sousou no frieren": {"verdict": "MATCH", "match_name": "Sousou no Frieren"},
        }))

        with mock.patch.object(mc, "LLM_SHOW_CACHE_FILE", cache_file), \
             mock.patch.object(mc, "NEEDS_CLASSIFY_REVIEW_LOG", review_log), \
             mock.patch("urllib.request.urlopen") as mock_urlopen:
            result = mc._llm_verify_new_show("Frieren", ["Sousou no Frieren"])
            mock_urlopen.assert_not_called()
            assert result == "Sousou no Frieren"

    def test_cache_new_hit_returns_candidate(self, tmp_path):
        """Cache hit with NEW verdict returns candidate unchanged."""
        cache_file = tmp_path / "cache.json"
        review_log = tmp_path / "review.log"
        cache_file.write_text(json.dumps({
            "the wire|breaking bad 2008": {"verdict": "NEW"},
        }))

        with mock.patch.object(mc, "LLM_SHOW_CACHE_FILE", cache_file), \
             mock.patch.object(mc, "NEEDS_CLASSIFY_REVIEW_LOG", review_log):
            result = mc._llm_verify_new_show("The Wire", ["Breaking Bad (2008)"])
            assert result == "The Wire"

    def test_llm_match_out_of_range_falls_through(self, tmp_path):
        """MATCH with out-of-range index falls through to NEW."""
        cache_file = tmp_path / "cache.json"
        review_log = tmp_path / "review.log"

        with mock.patch.object(mc, "LLM_SHOW_CACHE_FILE", cache_file), \
             mock.patch.object(mc, "NEEDS_CLASSIFY_REVIEW_LOG", review_log), \
             mock.patch.object(mc, "OLLAMA_HOST", "http://localhost:11434"), \
             mock.patch.object(mc, "OLLAMA_MODEL", "test-model"), \
             mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = json.dumps({
                "response": "MATCH 5",
            }).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = mc._llm_verify_new_show("Frieren", ["Sousou no Frieren"])
            assert result == "Frieren"
            assert review_log.exists()

    def test_exact_normalized_match_skips_llm(self, tmp_path):
        """When an exact normalized-key match exists, LLM is not called."""
        target = tmp_path / "TV Shows"
        target.mkdir()
        (target / "Breaking Bad (2008)").mkdir()

        with mock.patch.object(mc, "_llm_verify_new_show", return_value="should not be called") as mock_llm:
            result = mc._canonical_show_dir(target, "breaking.bad.2008")
            mock_llm.assert_not_called()
            assert result == "Breaking Bad (2008)"

    def test_alias_match_skips_llm(self, tmp_path):
        """When an alias matches, LLM is not called."""
        target = tmp_path / "TV Shows"
        target.mkdir()
        orig = mc.SHOW_ALIASES.copy()
        mc.SHOW_ALIASES.clear()
        mc.SHOW_ALIASES["Law and Order SVU"] = "Law & Order Special Victims Unit (1999)"
        try:
            with mock.patch.object(mc, "_llm_verify_new_show", return_value="should not be called") as mock_llm:
                result = mc._canonical_show_dir(target, "Law and Order SVU")
                mock_llm.assert_not_called()
                assert result == "Law & Order Special Victims Unit (1999)"
        finally:
            mc.SHOW_ALIASES.clear()
            mc.SHOW_ALIASES.update(orig)

    def test_multiple_similar_siblings_passed_to_llm(self, tmp_path):
        """All siblings with token_set_ratio >= 60 are passed to LLM."""
        target = tmp_path / "Anime"
        target.mkdir()
        (target / "Sousou no Frieren").mkdir()
        (target / "Frieren Beyond Journey's End").mkdir()
        (target / "Attack on Titan").mkdir()

        similar = ["Sousou no Frieren", "Frieren Beyond Journey's End"]
        with mock.patch.object(mc, "_fuzzy_similar_siblings", return_value=similar) as mock_fuzz, \
             mock.patch.object(mc, "_llm_verify_new_show", return_value="Frieren") as mock_llm:
            mc._canonical_show_dir(target, "Frieren")
            called_siblings = mock_llm.call_args[0][1]
            assert "Sousou no Frieren" in called_siblings
            assert "Frieren Beyond Journey's End" in called_siblings
            assert "Attack on Titan" not in called_siblings
