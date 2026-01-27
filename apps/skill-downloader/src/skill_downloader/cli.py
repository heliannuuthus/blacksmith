"""CLI for skill downloader."""

import logging
import subprocess
import sys
from pathlib import Path

import click

from .downloader import SkillDownloader, DEFAULT_SKILLS_DIR

logger = logging.getLogger(__name__)


def setup_logging(verbose: int) -> None:
    """Setup logging."""
    level = logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def check_gh_auth() -> bool:
    """
    Check if gh cli is authenticated.
    
    Returns:
        True if authenticated, False otherwise
    """
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def ensure_gh_auth() -> None:
    """
    Ensure gh cli is authenticated. If not, prompt user to authenticate.
    
    Raises:
        SystemExit: If gh cli is not available or not authenticated
    """
    # Check if gh cli is installed
    try:
        subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            timeout=5,
            check=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        click.echo("✗ Error: gh cli is not installed.", err=True)
        click.echo("Please install gh cli first:", err=True)
        click.echo("  https://cli.github.com/", err=True)
        raise SystemExit(1)
    
    # Check if authenticated
    if not check_gh_auth():
        click.echo("✗ Error: gh cli is not authenticated.", err=True)
        click.echo("Please authenticate first:", err=True)
        click.echo("  gh auth login", err=True)
        raise SystemExit(1)


@click.group()
@click.option("--token", envvar="GITHUB_TOKEN", help="GitHub token (overrides gh cli)")
@click.option("--skills-dir", type=click.Path(path_type=Path), default=DEFAULT_SKILLS_DIR, help="Skills directory")
@click.option("--retries", "-r", type=int, default=3, help="Retry attempts")
@click.option("-v", "--verbose", count=True, help="Verbosity (-v, -vv)")
@click.pass_context
def cli(ctx: click.Context, token: str | None, skills_dir: Path, retries: int, verbose: int) -> None:
    """Download skill.md files from GitHub repositories."""
    setup_logging(verbose)
    
    # If no token provided, ensure gh cli is authenticated
    if not token:
        ensure_gh_auth()
    
    ctx.ensure_object(dict)
    ctx.obj["downloader"] = SkillDownloader(
        token=token,
        use_gh_cli=True,  # Always use gh cli if no token provided
        skills_dir=skills_dir,
        max_retries=retries,
    )


@cli.command()
@click.argument("url")
@click.option("--ref", "-r", help="Branch/tag/commit (overrides URL ref)")
@click.pass_context
def download(ctx: click.Context, url: str, ref: str | None) -> None:
    """
    Download skill.md from a GitHub repository.
    
    URL formats supported:
    - https://github.com/owner/repo/blob/branch/path/to/skill.md
    - https://github.com/owner/repo.git
    - https://github.com/owner/repo
    - owner/repo
    - owner/repo/path/to/skill.md
    - owner/repo@branch
    - owner/repo@branch/path/to/skill.md
    """
    downloader = ctx.obj["downloader"]
    
    try:
        skill_file = downloader.download(url, ref=ref)
        click.echo(f"✓ Skill downloaded to: {skill_file}")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("raw_url")
@click.option("--name", "-n", help="Skill name (if not in frontmatter)")
@click.pass_context
def download_raw(ctx: click.Context, raw_url: str, name: str | None) -> None:
    """
    Download skill.md from a raw GitHub URL.
    
    Example: https://raw.githubusercontent.com/owner/repo/branch/path/to/SKILL.md
    """
    downloader = ctx.obj["downloader"]
    
    try:
        skill_file = downloader.download_from_raw_url(raw_url, skill_name=name)
        click.echo(f"✓ Skill downloaded to: {skill_file}")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option("--skills-dir", type=click.Path(path_type=Path), default=DEFAULT_SKILLS_DIR, help="Skills directory")
def list(skills_dir: Path) -> None:
    """List installed skills."""
    if not skills_dir.exists():
        click.echo(f"Skills directory does not exist: {skills_dir}")
        return
    
    skills = [d for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]
    
    if not skills:
        click.echo("No skills installed.")
        return
    
    click.echo(f"Installed skills ({len(skills)}):")
    for skill_dir in sorted(skills):
        skill_file = skill_dir / "SKILL.md"
        # Try to read frontmatter for description
        try:
            content = skill_file.read_text(encoding="utf-8")
            from .parser import parse_frontmatter
            metadata = parse_frontmatter(content)
            description = metadata.get("description", "")
            desc_text = f" - {description}" if description else ""
            click.echo(f"  {skill_dir.name}{desc_text}")
        except Exception:
            click.echo(f"  {skill_dir.name}")


if __name__ == "__main__":
    cli()
