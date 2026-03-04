import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False, help="Run live API tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: marks tests that hit real APIs")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="need --live to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
