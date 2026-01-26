"""Skill downloader implementation."""

import logging
from pathlib import Path
from typing import Optional

import httpx
from gh import GitHubClient

from .parser import RepoInfo, parse_repo_url, parse_frontmatter

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_DIR = Path.home() / ".cluade" / "skills"


class SkillDownloader:
    """Download skill.md files from GitHub repositories."""

    def __init__(
        self,
        token: Optional[str] = None,
        use_gh_cli: bool = True,
        skills_dir: Optional[Path] = None,
        max_retries: int = 3,
    ):
        """
        Initialize skill downloader.

        Args:
            token: GitHub personal access token (optional, overrides gh cli)
            use_gh_cli: Use gh cli credentials (default: True)
            skills_dir: Directory to save skills (default: ~/.cluade/skills)
            max_retries: Maximum number of retry attempts
        """
        self.client = GitHubClient(token=token, use_gh_cli=use_gh_cli, max_retries=max_retries)
        self.skills_dir = skills_dir or DEFAULT_SKILLS_DIR
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Skill downloader initialized, skills_dir=%s", self.skills_dir)

    def download(self, url: str, ref: Optional[str] = None) -> Path:
        """
        Download skill.md from a GitHub repository.

        Args:
            url: Repository URL in various formats
            ref: Optional branch/tag/commit (overrides URL ref)

        Returns:
            Path to the downloaded skill.md file

        Raises:
            ValueError: If URL is invalid or skill name cannot be determined
            FileNotFoundError: If skill.md file is not found
        """
        # Parse URL
        repo_info = parse_repo_url(url)
        
        # Override ref if provided
        if ref:
            repo_info = RepoInfo(
                owner=repo_info.owner,
                repo=repo_info.repo,
                path=repo_info.path,
                ref=ref,
            )
        
        logger.info(
            "Downloading skill from %s/%s path=%s ref=%s",
            repo_info.owner,
            repo_info.repo,
            repo_info.path,
            repo_info.ref,
        )
        
        # Try different common skill file names
        skill_files = [
            repo_info.path,
            "SKILL.md",
            "skill.md",
            "Skill.md",
        ]
        
        # If path doesn't end with .md, try appending SKILL.md
        if not repo_info.path.endswith((".md", ".markdown")):
            skill_files.insert(1, f"{repo_info.path}/SKILL.md")
        
        # Try different branch names if needed
        refs_to_try = [repo_info.ref]
        if repo_info.ref == "main":
            refs_to_try.append("master")
        elif repo_info.ref == "master":
            refs_to_try.append("main")
        
        # Try to download the file
        content = None
        actual_path = None
        actual_ref = None
        
        for ref_to_try in refs_to_try:
            for skill_path in skill_files:
                try:
                    logger.debug("Trying to fetch: %s (ref=%s)", skill_path, ref_to_try)
                    file_content = self.client.get_file_content(
                        repo_info.owner,
                        repo_info.repo,
                        skill_path,
                        ref=ref_to_try,
                    )
                    content = file_content.content
                    actual_path = skill_path
                    actual_ref = ref_to_try
                    logger.info("Found skill file: %s (ref=%s)", skill_path, ref_to_try)
                    break
                except Exception as e:
                    logger.debug("Failed to fetch %s (ref=%s): %s", skill_path, ref_to_try, e)
                    continue
            
            if content is not None:
                break
        
        if content is None:
            raise FileNotFoundError(
                f"Could not find skill.md file in {repo_info.owner}/{repo_info.repo} "
                f"(tried refs: {', '.join(refs_to_try)})"
            )
        
        # Parse frontmatter to get skill name
        metadata = parse_frontmatter(content)
        skill_name = metadata.get("name")
        
        if not skill_name:
            # Try to infer from repo name or path
            if actual_path and actual_path != "SKILL.md":
                # Use directory name or file stem
                path_parts = actual_path.split("/")
                skill_name = path_parts[-2] if len(path_parts) > 1 else repo_info.repo
            else:
                skill_name = repo_info.repo
            
            logger.warning(
                "No 'name' field in frontmatter, using inferred name: %s",
                skill_name,
            )
        
        # Create skill directory
        skill_dir = self.skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        # Save skill.md
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        
        logger.info("Skill downloaded to: %s", skill_file)
        
        return skill_file

    def download_from_raw_url(self, raw_url: str, skill_name: Optional[str] = None) -> Path:
        """
        Download skill.md from a raw GitHub URL.

        Args:
            raw_url: Raw GitHub URL (e.g., https://raw.githubusercontent.com/...)
            skill_name: Optional skill name (if not provided, will try to parse from content)

        Returns:
            Path to the downloaded skill.md file
        """
        logger.info("Downloading from raw URL: %s", raw_url)
        
        # Download content
        with httpx.Client(timeout=30.0) as client:
            response = client.get(raw_url)
            response.raise_for_status()
            content = response.text
        
        # Parse frontmatter to get skill name
        metadata = parse_frontmatter(content)
        if not skill_name:
            skill_name = metadata.get("name")
        
        if not skill_name:
            raise ValueError(
                "Cannot determine skill name. Provide skill_name or ensure frontmatter has 'name' field."
            )
        
        # Create skill directory
        skill_dir = self.skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        
        # Save skill.md
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(content, encoding="utf-8")
        
        logger.info("Skill downloaded to: %s", skill_file)
        
        return skill_file
