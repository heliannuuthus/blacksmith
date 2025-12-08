"""HowToCook scraper implementation."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from gh import GitHubClient

from .models import Category, Dish, ParsedDish, ParsedTip, Tip
from .parser import MarkdownParser

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """File metadata without content."""

    name: str
    path: str
    sha: str
    category: str | None
    html_url: str


class HowToCookScraper:
    """Scraper for HowToCook repository."""

    OWNER = "Anduin2017"
    REPO = "HowToCook"
    BRANCH = "master"

    def __init__(
        self,
        token: str | None = None,
        use_gh_cli: bool = False,
        max_retries: int = 3,
    ):
        """
        Initialize scraper.

        Args:
            token: GitHub personal access token (optional)
            use_gh_cli: Use gh cli credentials (requires user consent)
            max_retries: Maximum number of retry attempts for network errors
        """
        logger.info("Initializing HowToCook scraper")
        self.client = GitHubClient(
            token=token,
            use_gh_cli=use_gh_cli,
            max_retries=max_retries,
        )
        self.parser = MarkdownParser()
        logger.debug("Scraper initialized for %s/%s", self.OWNER, self.REPO)

    def get_dish_categories(self) -> list[Category]:
        """
        Get all dish categories.

        Returns:
            List of dish categories
        """
        logger.info("Fetching dish categories")
        contents = self.client.get_contents(
            self.OWNER, self.REPO, "dishes", self.BRANCH
        )
        categories = []
        for item in contents:
            if item.type == "dir":
                categories.append(Category(name=item.name, path=item.path))
        logger.debug("Found %d dish categories", len(categories))
        return categories

    def get_tip_categories(self) -> list[Category]:
        """
        Get all tip categories.

        Returns:
            List of tip categories
        """
        logger.info("Fetching tip categories")
        contents = self.client.get_contents(
            self.OWNER, self.REPO, "tips", self.BRANCH
        )
        categories = []
        for item in contents:
            if item.type == "dir":
                categories.append(Category(name=item.name, path=item.path))
        logger.debug("Found %d tip categories", len(categories))
        return categories

    def list_dish_files(self, category: str | None = None) -> Iterator[FileInfo]:
        """
        List dish files without fetching content (for incremental sync).

        Args:
            category: Category name (None for all categories)

        Yields:
            FileInfo objects with metadata only
        """
        if category:
            logger.info("Listing dish files in category: %s", category)
            category_path = f"dishes/{category}"
            contents = self.client.get_contents(
                self.OWNER, self.REPO, category_path, self.BRANCH
            )
            for item in contents:
                if item.type == "file" and item.name.endswith(".md"):
                    yield FileInfo(
                        name=Path(item.name).stem,
                        path=item.path,
                        sha=item.sha,
                        category=category,
                        html_url=item.html_url,
                    )
        else:
            logger.info("Listing all dish files")
            categories = self.get_dish_categories()
            for cat in categories:
                yield from self.list_dish_files(category=cat.name)

    def list_tip_files(self, category: str | None = None) -> Iterator[FileInfo]:
        """
        List tip files without fetching content (for incremental sync).

        Args:
            category: Category name (None for all categories)

        Yields:
            FileInfo objects with metadata only
        """
        if category:
            logger.info("Listing tip files in category: %s", category)
            category_path = f"tips/{category}"
            contents = self.client.get_contents(
                self.OWNER, self.REPO, category_path, self.BRANCH
            )
            for item in contents:
                if item.type == "file" and item.name.endswith(".md"):
                    yield FileInfo(
                        name=Path(item.name).stem,
                        path=item.path,
                        sha=item.sha,
                        category=category,
                        html_url=item.html_url,
                    )
        else:
            logger.info("Listing all tip files")
            categories = self.get_tip_categories()
            for cat in categories:
                yield from self.list_tip_files(category=cat.name)

            # Root-level tips
            logger.debug("Listing root-level tip files")
            contents = self.client.get_contents(
                self.OWNER, self.REPO, "tips", self.BRANCH
            )
            for item in contents:
                if item.type == "file" and item.name.endswith(".md"):
                    yield FileInfo(
                        name=Path(item.name).stem,
                        path=item.path,
                        sha=item.sha,
                        category=None,
                        html_url=item.html_url,
                    )

    def fetch_dish_by_path(self, path: str, category: str) -> Dish:
        """Fetch a single dish by path."""
        logger.debug("Fetching dish: %s", path)
        file = self.client.get_file_content(self.OWNER, self.REPO, path, self.BRANCH)
        return Dish(
            name=Path(file.name).stem,
            category=category,
            path=file.path,
            sha=file.sha,
            content=file.content,
            html_url=file.html_url,
        )

    def fetch_tip_by_path(self, path: str, category: str | None) -> Tip:
        """Fetch a single tip by path."""
        logger.debug("Fetching tip: %s", path)
        file = self.client.get_file_content(self.OWNER, self.REPO, path, self.BRANCH)
        return Tip(
            name=Path(file.name).stem,
            category=category,
            path=file.path,
            sha=file.sha,
            content=file.content,
            html_url=file.html_url,
        )

    def fetch_dishes(
        self, category: str | None = None
    ) -> Iterator[Dish]:
        """
        Fetch all dishes or dishes in a specific category.

        Args:
            category: Category name (None for all categories)

        Yields:
            Dish objects
        """
        if category:
            # Fetch dishes from specific category
            logger.info("Fetching dishes from category: %s", category)
            category_path = f"dishes/{category}"
            files = self.client.get_all_files(
                self.OWNER, self.REPO, category_path, self.BRANCH, extension=".md"
            )
            for file in files:
                logger.debug("Processing dish: %s", file.name)
                yield Dish(
                    name=Path(file.name).stem,
                    category=category,
                    path=file.path,
                    sha=file.sha,
                    content=file.content,
                    html_url=file.html_url,
                )
        else:
            # Fetch all dishes from all categories
            logger.info("Fetching dishes from all categories")
            categories = self.get_dish_categories()
            for cat in categories:
                yield from self.fetch_dishes(category=cat.name)

    def fetch_tips(
        self, category: str | None = None
    ) -> Iterator[Tip]:
        """
        Fetch all tips or tips in a specific category.

        Args:
            category: Category name (None for all categories and root files)

        Yields:
            Tip objects
        """
        if category:
            # Fetch tips from specific category
            logger.info("Fetching tips from category: %s", category)
            category_path = f"tips/{category}"
            files = self.client.get_all_files(
                self.OWNER, self.REPO, category_path, self.BRANCH, extension=".md"
            )
            for file in files:
                logger.debug("Processing tip: %s", file.name)
                yield Tip(
                    name=Path(file.name).stem,
                    category=category,
                    path=file.path,
                    sha=file.sha,
                    content=file.content,
                    html_url=file.html_url,
                )
        else:
            # Fetch tips from all categories
            logger.info("Fetching tips from all categories")
            categories = self.get_tip_categories()
            for cat in categories:
                yield from self.fetch_tips(category=cat.name)

            # Also fetch root-level tip files
            logger.debug("Fetching root-level tip files")
            root_files = self.client.get_all_files(
                self.OWNER, self.REPO, "tips", self.BRANCH, extension=".md"
            )
            for file in root_files:
                # Skip if file is in a subdirectory (already fetched)
                if "/" not in file.path.replace("tips/", ""):
                    logger.debug("Processing root tip: %s", file.name)
                    yield Tip(
                        name=Path(file.name).stem,
                        category=None,
                        path=file.path,
                        sha=file.sha,
                        content=file.content,
                        html_url=file.html_url,
                    )

    def fetch_all_dishes(self) -> list[Dish]:
        """
        Fetch all dishes into a list.

        Returns:
            List of all dishes
        """
        return list(self.fetch_dishes())

    def fetch_all_tips(self) -> list[Tip]:
        """
        Fetch all tips into a list.

        Returns:
            List of all tips
        """
        return list(self.fetch_tips())

    def fetch_parsed_dishes(
        self, category: str | None = None
    ) -> Iterator[ParsedDish]:
        """
        Fetch and parse all dishes or dishes in a specific category.

        Args:
            category: Category name (None for all categories)

        Yields:
            ParsedDish objects with structured content
        """
        for dish in self.fetch_dishes(category=category):
            logger.debug("Parsing dish: %s", dish.name)
            parsed_content = self.parser.parse_dish(dish)
            yield ParsedDish(
                name=dish.name,
                category=dish.category,
                path=dish.path,
                sha=dish.sha,
                html_url=dish.html_url,
                parsed=parsed_content,
            )

    def fetch_parsed_tips(
        self, category: str | None = None
    ) -> Iterator[ParsedTip]:
        """
        Fetch and parse all tips or tips in a specific category.

        Args:
            category: Category name (None for all categories and root files)

        Yields:
            ParsedTip objects with structured content
        """
        for tip in self.fetch_tips(category=category):
            logger.debug("Parsing tip: %s", tip.name)
            parsed_content = self.parser.parse_tip(tip)
            yield ParsedTip(
                name=tip.name,
                category=tip.category,
                path=tip.path,
                sha=tip.sha,
                html_url=tip.html_url,
                parsed=parsed_content,
            )

    def parse_dish(self, dish: Dish) -> ParsedDish:
        """Parse a single dish."""
        parsed_content = self.parser.parse_dish(dish)
        return ParsedDish(
            name=dish.name,
            category=dish.category,
            path=dish.path,
            sha=dish.sha,
            html_url=dish.html_url,
            parsed=parsed_content,
        )

    def parse_tip(self, tip: Tip) -> ParsedTip:
        """Parse a single tip."""
        parsed_content = self.parser.parse_tip(tip)
        return ParsedTip(
            name=tip.name,
            category=tip.category,
            path=tip.path,
            sha=tip.sha,
            html_url=tip.html_url,
            parsed=parsed_content,
        )

    def fetch_all_parsed_dishes(self) -> list[ParsedDish]:
        """
        Fetch and parse all dishes into a list.

        Returns:
            List of all parsed dishes
        """
        return list(self.fetch_parsed_dishes())

    def fetch_all_parsed_tips(self) -> list[ParsedTip]:
        """
        Fetch and parse all tips into a list.

        Returns:
            List of all parsed tips
        """
        return list(self.fetch_parsed_tips())

