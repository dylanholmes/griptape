import re
from typing import Any

from attr import define

from griptape.utils import str_to_hash, execute_futures_dict, import_optional_dependency
from griptape.artifacts import TextArtifact
from griptape.loaders import BaseTextLoader


@define
class WebLoader(BaseTextLoader):
    def load(self, source: str, include_links: bool = True, *args, **kwargs) -> list[TextArtifact]:
        return self._load_page_to_artifacts(source, include_links)

    def load_collection(
        self, sources: list[str], include_links: bool = True, *args, **kwargs
    ) -> dict[str, list[TextArtifact]]:
        return execute_futures_dict(
            {
                str_to_hash(source): self.futures_executor.submit(self._load_page_to_artifacts, source, include_links)
                for source in sources
            }
        )

    def _load_page_to_artifacts(self, url: str, include_links: bool = True) -> list[TextArtifact]:
        page_text = self.extract_page_text(url, include_links)
        if page_text is None:
            return []

        return self._text_to_artifacts(page_text)

    def extract_page_text(self, url: str, include_links: bool = True) -> str:
        playwright_sync_api = import_optional_dependency("playwright.sync_api")
        bs4 = import_optional_dependency("bs4")

        sync_playwright = playwright_sync_api.sync_playwright
        BeautifulSoup = bs4.BeautifulSoup

        def strip_excessive_newlines_and_spaces(document: str) -> str:
            # collapse repeated spaces into one
            document = re.sub(r" +", " ", document)
            # remove trailing spaces
            document = re.sub(r" +[\n\r]", "\n", document)
            # remove repeated newlines
            document = re.sub(r"[\n\r]+", "\n", document)
            return document.strip()

        def strip_newlines(document: str) -> str:
            # HTML might contain newlines which are just whitespaces to a browser
            return re.sub(r"[\n\r]+", " ", document)

        def format_document_soup(
            document: Any,
            table_cell_separator: str = "\t"
        ) -> str:
            """Format html to a flat text document.

            The following goals:
            - Newlines from within the HTML are removed (as browser would ignore them as well).
            - Repeated newlines/spaces are removed (as browsers would ignore them).
            - Newlines only before and after headlines and paragraphs or when explicit (br or pre tag)
            - Table columns/rows are separated by newline
            - List elements are separated by newline and start with a hyphen
            """
            text = ""
            list_element_start = False
            verbatim_output = 0
            in_table = False
            last_added_newline = False
            for e in document.descendants:
                verbatim_output -= 1
                if isinstance(e, bs4.element.NavigableString):
                    if isinstance(e, (bs4.element.Comment, bs4.element.Doctype)):
                        continue
                    element_text = e.text
                    if in_table:
                        # Tables are represented in natural language with rows separated by newlines
                        # Can't have newlines then in the table elements
                        element_text = element_text.replace("\n", " ").strip()

                    # Some tags are translated to spaces but in the logic underneath this section, we
                    # translate them to newlines as a browser should render them such as with br
                    # This logic here avoids a space after newline when it shouldn't be there.
                    if last_added_newline and element_text.startswith(" "):
                        element_text = element_text[1:]
                        last_added_newline = False

                    if element_text:
                        content_to_add = (
                            element_text
                            if verbatim_output > 0
                            else strip_newlines(element_text)
                        )

                        # Don't join separate elements without any spacing
                        if (text and not text[-1].isspace()) and (
                            content_to_add and not content_to_add[0].isspace()
                        ):
                            text += " "

                        text += content_to_add

                        list_element_start = False
                elif isinstance(e, bs4.element.Tag):
                    # table is standard HTML element
                    if e.name == "table":
                        in_table = True
                    # tr is for rows
                    elif e.name == "tr" and in_table:
                        text += "\n"
                    # td for data cell, th for header
                    elif e.name in ["td", "th"] and in_table:
                        text += table_cell_separator
                    elif e.name == "/table":
                        in_table = False
                    elif in_table:
                        # don't handle other cases while in table
                        pass

                    elif e.name in ["p", "div"]:
                        if not list_element_start:
                            text += "\n"
                    elif e.name in ["h1", "h2", "h3", "h4"]:
                        text += "\n"
                        list_element_start = False
                        last_added_newline = True
                    elif e.name == "br":
                        text += "\n"
                        list_element_start = False
                        last_added_newline = True
                    elif e.name == "li":
                        text += "\n- "
                        list_element_start = True
                    elif e.name == "pre":
                        if verbatim_output <= 0:
                            verbatim_output = len(list(e.childGenerator()))
            return strip_excessive_newlines_and_spaces(text)

        with sync_playwright() as p:
            with p.chromium.launch(headless=True) as browser:
                page = browser.new_page()
                page.goto(url)
                content = page.content()

                if not content:
                    raise Exception("can't access URL")

                soup = BeautifulSoup(content, "html.parser")

                cybotCookiebotDialog = soup.find("div", {"id": "CybotCookiebotDialog"})
                if cybotCookiebotDialog:
                    cybotCookiebotDialog.extract()

                for undesired_element in ["sidebar","footer"]:
                    [tag.extract() for tag in soup.find_all(class_=lambda x: x and undesired_element in x.split())]

                for undesired_tag in ["nav","footer","meta","script","style","symbol","aside"]:
                    [tag.extract() for tag in soup.find_all(undesired_tag)]

                text = format_document_soup(soup)

                return text
    