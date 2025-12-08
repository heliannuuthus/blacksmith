"""GitHub API client."""

import base64
import logging
import os
import subprocess
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .models import GitHubContent, GitHubDirectory, GitHubFile

logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_MIN_WAIT = 1  # seconds
DEFAULT_MAX_WAIT = 10  # seconds

# Retryable exceptions
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.NetworkError,
)


def get_token_from_gh_cli() -> str | None:
    """
    Get GitHub token from gh cli.
    
    Returns:
        Token string or None if gh cli not available/authenticated
    """
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info("Using token from gh cli")
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug("gh cli not available: %s", e)
    return None


def get_token(token: str | None = None, use_gh_cli: bool = False) -> str | None:
    """
    Get GitHub token from various sources.
    
    Priority:
    1. Explicitly provided token
    2. Environment variable GH_TOKEN / GITHUB_TOKEN
    3. gh cli (`gh auth token`) - only if use_gh_cli=True
    
    Args:
        token: Explicitly provided token
        use_gh_cli: Whether to use gh cli credentials (requires user consent)
        
    Returns:
        GitHub token or None
    """
    if token:
        logger.debug("Using explicitly provided token")
        return token
    
    # Check environment variables
    env_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if env_token:
        logger.info("Using token from environment variable")
        return env_token
    
    # Try gh cli only if explicitly allowed
    if use_gh_cli:
        return get_token_from_gh_cli()
    
    return None


def create_retry_decorator(max_retries: int = DEFAULT_MAX_RETRIES):
    """Create a retry decorator with specified max retries."""
    return retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=DEFAULT_MIN_WAIT, max=DEFAULT_MAX_WAIT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class GitHubClient:
    """GitHub REST API client with retry support."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        use_gh_cli: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token (optional)
            base_url: Custom base URL (defaults to GitHub API)
            timeout: Request timeout in seconds
            use_gh_cli: Use gh cli credentials (requires user consent)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "blacksmith-github-client",
        }
        
        # Get token
        resolved_token = get_token(token, use_gh_cli=use_gh_cli)
        
        if resolved_token:
            self.headers["Authorization"] = f"token {resolved_token}"
            logger.debug("GitHub client initialized with token")
        else:
            logger.warning("GitHub client initialized without token (rate limited)")
        logger.info("GitHub client ready, base_url=%s, max_retries=%d", self.base_url, max_retries)

    def _request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> httpx.Response:
        """Make HTTP request to GitHub API with retry."""
        url = f"{self.base_url}{endpoint}"
        
        @create_retry_decorator(self.max_retries)
        def do_request() -> httpx.Response:
            logger.debug("Request: %s %s", method, url)
            with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                response = client.request(method, url, **kwargs)
                logger.debug(
                    "Response: %s %s (status=%d)",
                    method,
                    endpoint,
                    response.status_code,
                )
                # Retry on 5xx errors
                if response.status_code >= 500:
                    logger.warning("Server error %d, will retry", response.status_code)
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
        
        return do_request()

    def _download(self, url: str) -> str:
        """Download content from URL with retry."""
        @create_retry_decorator(self.max_retries)
        def do_download() -> str:
            logger.debug("Downloading: %s", url)
            with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                response = client.get(url)
                if response.status_code >= 500:
                    logger.warning("Server error %d, will retry", response.status_code)
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.text
        
        return do_download()

    def get_contents(
        self, owner: str, repo: str, path: str = "", ref: str = "master"
    ) -> list[GitHubContent]:
        """
        Get repository contents.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Path in repository (empty for root)
            ref: Branch/tag/commit (default: master)

        Returns:
            List of GitHubContent items
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref} if ref else {}
        logger.info("Fetching contents: %s/%s path=%s ref=%s", owner, repo, path, ref)
        response = self._request("GET", endpoint, params=params)
        data = response.json()

        # Handle single file response
        if isinstance(data, dict):
            logger.debug("Single file response: %s", data.get("name"))
            return [GitHubContent(**data)]

        # Handle directory listing
        logger.debug("Directory listing: %d items", len(data))
        return [GitHubContent(**item) for item in data]

    def get_file_content(
        self, owner: str, repo: str, path: str, ref: str = "master"
    ) -> GitHubFile:
        """
        Get file content with decoded text.

        Args:
            owner: Repository owner
            repo: Repository name
            path: File path in repository
            ref: Branch/tag/commit (default: master)

        Returns:
            GitHubFile with decoded content
        """
        logger.info("Fetching file content: %s/%s path=%s", owner, repo, path)
        contents = self.get_contents(owner, repo, path, ref)
        if not contents:
            logger.error("File not found: %s", path)
            raise ValueError(f"File not found: {path}")
        if contents[0].type != "file":
            logger.error("Path is not a file: %s", path)
            raise ValueError(f"Path is not a file: {path}")

        content_item = contents[0]
        # Prefer download_url for raw content (more reliable)
        if content_item.download_url:
            logger.debug("Downloading from URL: %s", content_item.download_url)
            decoded = self._download(content_item.download_url)
        elif content_item.content:
            logger.debug("Decoding base64 content for: %s", path)
            # Fallback to base64 decoded content
            decoded = base64.b64decode(content_item.content).decode("utf-8")
        else:
            logger.error("File has no content: %s", path)
            raise ValueError(f"File has no content: {path}")
        
        logger.debug("File content fetched: %s (%d bytes)", path, len(decoded))

        return GitHubFile(
            name=content_item.name,
            path=content_item.path,
            sha=content_item.sha,
            size=content_item.size,
            html_url=content_item.html_url,
            content=decoded,
        )

    def get_directory_tree(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: str = "master",
        recursive: bool = False,
    ) -> GitHubDirectory:
        """
        Get directory tree recursively.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Directory path (empty for root)
            ref: Branch/tag/commit (default: master)
            recursive: Whether to recursively fetch subdirectories

        Returns:
            GitHubDirectory with all items
        """
        logger.info(
            "Fetching directory tree: %s/%s path=%s recursive=%s",
            owner, repo, path, recursive
        )
        items: list[GitHubContent] = []
        contents = self.get_contents(owner, repo, path, ref)

        for item in contents:
            items.append(item)
            if recursive and item.type == "dir":
                logger.debug("Recursing into directory: %s", item.path)
                subdir = self.get_directory_tree(
                    owner, repo, item.path, ref, recursive=True
                )
                items.extend(subdir.items)

        logger.debug("Directory tree fetched: %s (%d items)", path, len(items))
        return GitHubDirectory(path=path, items=items)

    def get_all_files(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: str = "master",
        extension: str | None = None,
    ) -> list[GitHubFile]:
        """
        Get all files in a directory tree.

        Args:
            owner: Repository owner
            repo: Repository name
            path: Directory path (empty for root)
            ref: Branch/tag/commit (default: master)
            extension: Filter by file extension (e.g., '.md')

        Returns:
            List of GitHubFile objects
        """
        logger.info(
            "Fetching all files: %s/%s path=%s extension=%s",
            owner, repo, path, extension
        )
        directory = self.get_directory_tree(owner, repo, path, ref, recursive=True)
        files: list[GitHubFile] = []

        for item in directory.items:
            if item.type == "file":
                if extension and not item.name.endswith(extension):
                    continue
                try:
                    file_content = self.get_file_content(owner, repo, item.path, ref)
                    files.append(file_content)
                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", item.path, e)
                    continue

        logger.info("Fetched %d files from %s", len(files), path)
        return files
