"""GitHub API data models."""

from typing import Literal
from pydantic import BaseModel, Field


class GitHubContent(BaseModel):
    """GitHub content item (file or directory)."""

    name: str
    path: str
    sha: str
    size: int
    url: str
    html_url: str
    git_url: str
    download_url: str | None
    type: Literal["file", "dir"]
    content: str | None = None  # Base64 encoded content for files
    encoding: str | None = None  # Usually "base64" for files


class GitHubFile(BaseModel):
    """GitHub file with decoded content."""

    name: str
    path: str
    sha: str
    size: int
    html_url: str
    content: str
    encoding: str = "utf-8"


class GitHubDirectory(BaseModel):
    """GitHub directory listing."""

    path: str
    items: list[GitHubContent] = Field(default_factory=list)

