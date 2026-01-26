"""URL and frontmatter parsing utilities."""

from typing import NamedTuple
from urllib.parse import urlparse

import yaml


class RepoInfo(NamedTuple):
    """Repository information parsed from URL."""

    owner: str
    repo: str
    path: str
    ref: str  # branch/tag/commit


def parse_repo_url(url: str) -> RepoInfo:
    """
    Parse various GitHub URL formats into RepoInfo.
    
    Supports:
    - https://github.com/owner/repo/blob/branch/path/to/file.md
    - https://github.com/owner/repo.git
    - https://github.com/owner/repo
    - git@github.com:owner/repo.git
    - owner/repo
    - owner/repo/path/to/file.md
    - owner/repo@branch
    - owner/repo@branch/path/to/file.md
    
    Args:
        url: Repository URL in various formats
        
    Returns:
        RepoInfo with owner, repo, path, and ref
    """
    # Default values
    owner = ""
    repo = ""
    path = "SKILL.md"
    ref = "main"
    
    original_url = url
    url = url.strip()
    
    # Handle SSH git URLs (git@github.com:owner/repo.git)
    if url.startswith("git@"):
        # Format: git@github.com:owner/repo.git or git@github.com:owner/repo.git/path
        if ":" not in url:
            raise ValueError(f"Invalid git SSH URL format: {url}")
        
        # Split host and repo part
        _, repo_part = url.split(":", 1)
        # Remove .git suffix if present
        repo_part = repo_part.rstrip("/").rstrip(".git")
        
        parts = repo_part.split("/")
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1]
            if len(parts) > 2:
                path = "/".join(parts[2:])
        else:
            raise ValueError(f"Invalid git SSH URL format: {url}")
    
    # Handle full HTTPS URLs
    elif url.startswith("https://"):
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1].rstrip(".git")
            
            # Check if it's a blob URL (file path)
            if len(parts) >= 4 and parts[2] == "blob":
                ref = parts[3]
                path = "/".join(parts[4:]) if len(parts) > 4 else "SKILL.md"
            elif len(parts) >= 3:
                # Assume remaining parts are path (but not blob format)
                path = "/".join(parts[2:])
            else:
                # Root level, use default path
                path = "SKILL.md"
        else:
            raise ValueError(f"Invalid GitHub URL format: {url}")
    
    # Handle short format (owner/repo)
    else:
        # Remove .git suffix if present
        url = url.rstrip("/").rstrip(".git")
        
        # Check for @branch syntax
        if "@" in url:
            repo_part, ref_part = url.split("@", 1)
            # Check if ref_part contains path
            if "/" in ref_part:
                ref, path = ref_part.split("/", 1)
            else:
                ref = ref_part
            url = repo_part
        
        parts = url.split("/")
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1]
            if len(parts) > 2:
                path = "/".join(parts[2:])
        elif len(parts) == 1:
            # Just owner, need repo name
            raise ValueError(f"Repository name missing in: {original_url}")
        else:
            raise ValueError(f"Invalid format: {original_url}")
    
    # Normalize path - if it doesn't end with .md, try common names
    if not path.endswith((".md", ".markdown")):
        # If path is empty or just a directory, append SKILL.md
        if not path or path == "SKILL.md":
            path = "SKILL.md"
        elif "/" in path:
            # Path exists, append SKILL.md
            path = f"{path}/SKILL.md" if not path.endswith("/") else f"{path}SKILL.md"
        else:
            # Single directory name, append SKILL.md
            path = f"{path}/SKILL.md"
    
    # Ensure ref is set
    if not ref:
        ref = "main"
    
    return RepoInfo(owner=owner, repo=repo, path=path, ref=ref)


def parse_frontmatter(content: str) -> dict[str, str]:
    """
    Parse YAML frontmatter from markdown content.
    
    Args:
        content: Markdown content with optional YAML frontmatter
        
    Returns:
        Dictionary of frontmatter metadata, empty dict if no frontmatter
    """
    # Check for frontmatter delimiter
    if not content.startswith("---"):
        return {}
    
    # Find the end of frontmatter
    lines = content.split("\n")
    if len(lines) < 2:
        return {}
    
    # Find closing ---
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    
    if end_idx is None:
        return {}
    
    # Extract frontmatter content
    frontmatter_lines = lines[1:end_idx]
    frontmatter_text = "\n".join(frontmatter_lines)
    
    try:
        metadata = yaml.safe_load(frontmatter_text)
        return metadata if isinstance(metadata, dict) else {}
    except yaml.YAMLError as e:
        # If YAML parsing fails, return empty dict
        return {}
