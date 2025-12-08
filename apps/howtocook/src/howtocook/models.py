"""HowToCook data models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Category(BaseModel):
    """Recipe category."""

    name: str
    path: str


class Dish(BaseModel):
    """Dish recipe."""

    name: str
    category: str
    path: str
    sha: str  # Git blob SHA for incremental sync
    content: str
    html_url: str


class Tip(BaseModel):
    """Cooking tip."""

    name: str
    category: str | None
    path: str
    sha: str  # Git blob SHA for incremental sync
    content: str
    html_url: str


class ParsedDish(BaseModel):
    """Parsed dish with structured content."""

    name: str
    category: str
    path: str
    sha: str
    html_url: str
    parsed: dict[str, Any]


class ParsedTip(BaseModel):
    """Parsed tip with structured content."""

    name: str
    category: str | None
    path: str
    sha: str
    html_url: str
    parsed: dict[str, Any]


class FileEntry(BaseModel):
    """File entry in manifest."""

    path: str
    sha: str
    category: str | None = None
    synced_at: datetime = Field(default_factory=datetime.now)


class SyncManifest(BaseModel):
    """Sync manifest for incremental updates."""

    version: int = 1
    last_sync: datetime = Field(default_factory=datetime.now)
    files: dict[str, FileEntry] = Field(default_factory=dict)  # path -> FileEntry

    def needs_update(self, path: str, sha: str) -> bool:
        """Check if file needs update."""
        if path not in self.files:
            return True
        return self.files[path].sha != sha

    def update_entry(self, path: str, sha: str, category: str | None = None) -> None:
        """Update or add file entry."""
        self.files[path] = FileEntry(
            path=path,
            sha=sha,
            category=category,
            synced_at=datetime.now(),
        )
        self.last_sync = datetime.now()

    def remove_stale(self, current_paths: set[str]) -> list[str]:
        """Remove entries not in current paths, return removed paths."""
        stale = [p for p in self.files if p not in current_paths]
        for p in stale:
            del self.files[p]
        return stale

