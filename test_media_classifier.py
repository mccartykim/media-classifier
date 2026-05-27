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
