"""CLI for HowToCook scraper."""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from .models import SyncManifest
from .scraper import FileInfo, HowToCookScraper

logger = logging.getLogger(__name__)

MANIFEST_FILE = ".sync-manifest.json"
DEFAULT_CONCURRENCY = 8


def setup_logging(verbose: int) -> None:
    """Setup logging."""
    level = logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def sanitize_filename(name: str) -> str:
    """Sanitize filename."""
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name.strip()


def load_manifest(output_dir: Path) -> SyncManifest:
    """Load sync manifest."""
    path = output_dir / MANIFEST_FILE
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return SyncManifest(**json.load(f))
    return SyncManifest()


def save_manifest(output_dir: Path, manifest: SyncManifest) -> None:
    """Save sync manifest."""
    path = output_dir / MANIFEST_FILE
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)


# ============ CLI Group ============

@click.group()
@click.option("--token", envvar="GITHUB_TOKEN", help="GitHub token")
@click.option("--use-gh-cli", is_flag=True, help="Use gh cli credentials")
@click.option("--retries", "-r", type=int, default=3, help="Retry attempts")
@click.option("-v", "--verbose", count=True, help="Verbosity (-v, -vv)")
@click.pass_context
def cli(ctx: click.Context, token: str | None, use_gh_cli: bool, retries: int, verbose: int) -> None:
    """HowToCook data scraper CLI."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["scraper"] = HowToCookScraper(token=token, use_gh_cli=use_gh_cli, max_retries=retries)


# ============ Sync Commands ============

def sync_files(
    scraper: HowToCookScraper,
    list_fn,
    fetch_fn,
    output_path: Path,
    category: str | None,
    parsed: bool,
    force: bool,
    concurrency: int,
) -> dict:
    """Generic sync logic for dishes/tips."""
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = SyncManifest() if force else load_manifest(output_path)
    
    # List files
    file_infos = list(list_fn(category=category))
    current_paths = {f.path for f in file_infos}
    
    # Diff
    to_update = [f for f in file_infos if manifest.needs_update(f.path, f.sha)]
    stale = manifest.remove_stale(current_paths)
    
    click.echo(f"Found {len(file_infos)}: {len(to_update)} new/updated, {len(stale)} deleted")
    
    if not to_update and not stale:
        click.echo("Everything up to date!")
        return {"saved": 0, "deleted": 0, "errors": 0}
    
    # Delete stale
    deleted = 0
    for path in stale:
        entry = manifest.files.get(path)
        if entry:
            file_path = output_path / sanitize_filename(entry.category or "root") / (sanitize_filename(Path(path).stem) + ".json")
            if file_path.exists():
                file_path.unlink()
                deleted += 1
    
    # Sync
    saved = 0
    errors = 0
    lock = threading.Lock()
    
    def sync_one(info: FileInfo):
        nonlocal saved, errors
        try:
            item = fetch_fn(info.path, info.category)
            if parsed:
                item = scraper.parse_dish(item) if hasattr(item, 'category') else scraper.parse_tip(item)
            
            cat = getattr(item, 'category', None) or "root"
            cat_dir = output_path / sanitize_filename(cat)
            cat_dir.mkdir(parents=True, exist_ok=True)
            
            name = item.parsed.get("title", item.name) if parsed else item.name
            file_path = cat_dir / (sanitize_filename(name) + ".json")
            
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(item.model_dump(), f, ensure_ascii=False, indent=2)
            
            with lock:
                manifest.update_entry(info.path, info.sha, info.category)
                saved += 1
        except Exception as e:
            logger.error("Failed %s: %s", info.path, e)
            with lock:
                errors += 1
    
    with click.progressbar(length=len(to_update), label="Syncing") as bar:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(sync_one, info) for info in to_update]
            for _ in as_completed(futures):
                bar.update(1)
    
    save_manifest(output_path, manifest)
    return {"saved": saved, "deleted": deleted, "errors": errors}


@cli.command()
@click.option("-o", "--output-dir", default="datasets/howtocook/dishes", show_default=True)
@click.option("-c", "--category", help="Category filter")
@click.option("-p", "--parsed", is_flag=True, help="Parse markdown")
@click.option("-f", "--force", is_flag=True, help="Force full sync")
@click.option("-j", "--concurrency", type=int, default=DEFAULT_CONCURRENCY, show_default=True)
@click.pass_context
def dishes(ctx, output_dir, category, parsed, force, concurrency):
    """Sync dishes with incremental updates."""
    scraper = ctx.obj["scraper"]
    result = sync_files(
        scraper,
        scraper.list_dish_files,
        scraper.fetch_dish_by_path,
        Path(output_dir),
        category, parsed, force, concurrency,
    )
    click.echo(f"\nSaved {result['saved']}, deleted {result['deleted']}, errors {result['errors']}")


@cli.command()
@click.option("-o", "--output-dir", default="datasets/howtocook/tips", show_default=True)
@click.option("-c", "--category", help="Category filter")
@click.option("-p", "--parsed", is_flag=True, help="Parse markdown")
@click.option("-f", "--force", is_flag=True, help="Force full sync")
@click.option("-j", "--concurrency", type=int, default=DEFAULT_CONCURRENCY, show_default=True)
@click.pass_context
def tips(ctx, output_dir, category, parsed, force, concurrency):
    """Sync tips with incremental updates."""
    scraper = ctx.obj["scraper"]
    result = sync_files(
        scraper,
        scraper.list_tip_files,
        scraper.fetch_tip_by_path,
        Path(output_dir),
        category, parsed, force, concurrency,
    )
    click.echo(f"\nSaved {result['saved']}, deleted {result['deleted']}, errors {result['errors']}")


@cli.command()
@click.pass_context
def categories(ctx):
    """List all categories."""
    scraper = ctx.obj["scraper"]
    
    click.echo("Dish categories:")
    for cat in scraper.get_dish_categories():
        click.echo(f"  - {cat.name}")
    
    click.echo("\nTip categories:")
    for cat in scraper.get_tip_categories():
        click.echo(f"  - {cat.name}")


@cli.command()
@click.option("-o", "--output-dir", required=True, help="Directory to check")
def status(output_dir):
    """Show sync status."""
    output_path = Path(output_dir)
    if not output_path.exists():
        click.echo(f"Not found: {output_path}")
        return
    
    manifest = load_manifest(output_path)
    if not manifest.files:
        click.echo("No sync history.")
        return
    
    click.echo(f"Last sync: {manifest.last_sync.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"Files: {len(manifest.files)}")
    
    by_cat = {}
    for e in manifest.files.values():
        cat = e.category or "root"
        by_cat[cat] = by_cat.get(cat, 0) + 1
    
    click.echo("\nBy category:")
    for cat, count in sorted(by_cat.items()):
        click.echo(f"  {cat}: {count}")


# ============ Detect Command ============

@cli.command()
@click.option("-o", "--output-dir", default="datasets/howtocook/dishes", show_default=True)
@click.option("-c", "--category", help="Category filter")
@click.option("-j", "--concurrency", type=int, default=DEFAULT_CONCURRENCY, show_default=True)
def detect(output_dir, category, concurrency):
    """
    Full workflow: fetch -> parse -> evaluate -> refine -> save.
    
    Uses LLM (Qwen) to evaluate and refine recipes.
    Requires DASHSCOPE_API_KEY environment variable.
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.environ.get("DASHSCOPE_API_KEY"):
        click.echo("Error: DASHSCOPE_API_KEY required", err=True)
        click.echo("Set in .env or: export DASHSCOPE_API_KEY=sk-xxx", err=True)
        raise SystemExit(1)
    
    from .workflow import RecipeDetector
    
    click.echo("=" * 50)
    click.echo("Recipe Detection Workflow")
    click.echo("=" * 50)
    click.echo(f"Output: {output_dir}")
    click.echo(f"Category: {category or 'all'}")
    click.echo(f"Concurrency: {concurrency}")
    click.echo()
    
    detector = RecipeDetector()
    result = detector.run(output_dir=output_dir, category=category, concurrency=concurrency)
    
    click.echo()
    click.echo("=" * 50)
    click.echo("Results")
    click.echo("=" * 50)
    click.echo(f"Total:   {result['total']}")
    click.echo(f"Fetched: {result['fetched']}")
    click.echo(f"Refined: {result['refined']}")
    click.echo(f"Saved:   {result['saved']}")
    click.echo(f"Errors:  {result['errors']}")


if __name__ == "__main__":
    cli()
