import json
import pytest
from griptape import utils
from griptape.loaders.web_loader_2 import WebLoader
from tests.mocks.mock_embedding_driver import MockEmbeddingDriver

MAX_TOKENS = 50


class TestWebLoader:
    @pytest.fixture(autouse=True)
    def mock_playwright(self, mocker):
        playwright = mocker.MagicMock()
        playwright.__enter__.return_value.chromium.launch.return_value.__enter__.return_value.new_page.return_value.content.return_value = "foobar"
        mocker.patch("playwright.sync_api.sync_playwright", return_value=playwright)

    @pytest.fixture
    def loader(self):
        return WebLoader(max_tokens=MAX_TOKENS, embedding_driver=MockEmbeddingDriver())

    def test_load(self, loader):
        artifacts = loader.load("https://github.com/griptape-ai/griptape")

        assert len(artifacts) >= 1
        assert "foobar" in artifacts[0].value.lower()

        assert artifacts[0].embedding == [0, 1]

    def test_load_collection(self, loader):
        artifacts = loader.load_collection(
            ["https://github.com/griptape-ai/griptape", "https://github.com/griptape-ai/griptape-docs"]
        )

        assert list(artifacts.keys()) == [
            utils.str_to_hash("https://github.com/griptape-ai/griptape"),
            utils.str_to_hash("https://github.com/griptape-ai/griptape-docs"),
        ]
        assert "foobar" in [a.value for artifact_list in artifacts.values() for a in artifact_list][0].lower()

        assert list(artifacts.values())[0][0].embedding == [0, 1]

    def test_empty_page_string_response(self, loader, mocker):
        playwright = mocker.MagicMock()
        playwright.__enter__.return_value.chromium.launch.return_value.__enter__.return_value.new_page.return_value.content.return_value = ""
        mocker.patch("playwright.sync_api.sync_playwright", return_value=playwright)

        with pytest.raises(Exception, match="can't access URL"):
            loader.load("https://example.com/")

    def test_empty_page_none_response(self, loader, mocker):
        playwright = mocker.MagicMock()
        playwright.__enter__.return_value.chromium.launch.return_value.__enter__.return_value.new_page.return_value.content.return_value = None
        mocker.patch("playwright.sync_api.sync_playwright", return_value=playwright)

        with pytest.raises(Exception, match="can't access URL"):
            loader.load("https://example.com/")
