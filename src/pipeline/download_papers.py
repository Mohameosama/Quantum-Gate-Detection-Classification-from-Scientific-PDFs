"""
Module for downloading papers from arXiv.
Handles paper retrieval and storage.
"""
import arxiv
import os
import time
from typing import List, Optional, Set, Tuple
import logging
from pathlib import Path

#from config import PDF_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_paper_list(paper_list_file: str) -> List[str]:
    """
    Load the list of arXiv paper IDs from a file.
    
    Parameters
    ----------
    paper_list_file : str
        Path to the file containing arXiv paper IDs.
    
    Returns
    -------
    List[str]
        List of arXiv paper IDs in order.
    """
    with open(paper_list_file, 'r') as f:
        papers = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(papers)} papers from list")
    return papers


def extract_arxiv_id(arxiv_string: str) -> str:
    """
    Extract clean arXiv ID from string like 'arXiv:2408.02837'.
    
    Parameters
    ----------
    arxiv_string : str
        String containing arXiv ID (e.g., 'arXiv:2408.02837').
    
    Returns
    -------
    str
        Clean arXiv ID (e.g., '2408.02837').
    """
    if arxiv_string.startswith('arXiv:'):
        return arxiv_string[6:]  # Remove 'arXiv:' prefix
    return arxiv_string


def cached_arxiv_ids(pdf_dir: str) -> Set[str]:
    """
    Return a set of cached arXiv IDs already present in the pdf directory.

    Parameters
    ----------
    pdf_dir : str
        Directory containing cached PDFs named like '<arxiv_id>.pdf'.

    Returns
    -------
    Set[str]
        Set of arXiv IDs (filename stems) found in the directory.
    """
    p = Path(pdf_dir)
    if not p.exists():
        return set()
    out: Set[str] = set()
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() == ".pdf":
            out.add(f.stem)
    return out


def compute_resume_index(paper_ids: List[str], cached_ids: Set[str]) -> int:
    """
    Compute the 0-based index of the first paper that is NOT cached, assuming you want
    to resume from the longest cached prefix.

    If the cached set is not a perfect prefix (e.g., gaps), this returns the first gap.
    """
    for i, pid in enumerate(paper_ids):
        if extract_arxiv_id(pid) not in cached_ids:
            return i
    return len(paper_ids)


def download_paper(arxiv_id: str, output_dir: str, max_retries: int = 3) -> Optional[str]:
    """
    Download a single paper from arXiv.
    
    Parameters
    ----------
    arxiv_id : str
        The arXiv ID of the paper to download (e.g., '2408.02837').
    output_dir : str
        Directory where the PDF will be saved.
    max_retries : int
        Maximum number of retry attempts if download fails.
    
    Returns
    -------
    Optional[str]
        Path to the downloaded PDF file, or None if download failed.
    """
    clean_id = extract_arxiv_id(arxiv_id)
    output_path = os.path.join(output_dir, f"{clean_id}.pdf")
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        logger.info(f"Paper {clean_id} already downloaded, skipping.")
        return output_path
    
    # Try downloading with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading paper {clean_id} (attempt {attempt + 1}/{max_retries})...")
            
            # Search for the paper
            search = arxiv.Search(id_list=[clean_id])
            paper = next(search.results())
            
            # Download the PDF
            paper.download_pdf(dirpath=output_dir, filename=f"{clean_id}.pdf")
            
            logger.info(f"Successfully downloaded: {clean_id}")
            logger.info(f"  Title: {paper.title}")
            logger.info(f"  Published: {paper.published}")
            
            # Small delay to be respectful to arXiv servers
            time.sleep(1)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading {clean_id} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {clean_id} after {max_retries} attempts")
                return None


def download_papers_batch(paper_ids: List[str], output_dir: str, limit: Optional[int] = None) -> List[str]:
    """
    Download multiple papers from arXiv.
    
    Parameters
    ----------
    paper_ids : List[str]
        List of arXiv paper IDs to download.
    output_dir : str
        Directory where PDFs will be saved.
    limit : Optional[int]
        Maximum number of papers to download. If None, download all.
    
    Returns
    -------
    List[str]
        List of paths to successfully downloaded PDF files.
    """
    downloaded_paths = []
    papers_to_download = paper_ids[:limit] if limit else paper_ids
    
    logger.info(f"Starting batch download of {len(papers_to_download)} papers...")
    
    for idx, paper_id in enumerate(papers_to_download, 1):
        logger.info(f"\nProcessing paper {idx}/{len(papers_to_download)}: {paper_id}")
        
        pdf_path = download_paper(paper_id, output_dir)
        if pdf_path:
            downloaded_paths.append(pdf_path)
        
        # Add a small delay between downloads
        if idx < len(papers_to_download):
            time.sleep(3)
    
    logger.info(f"\nDownload complete. Successfully downloaded {len(downloaded_paths)}/{len(papers_to_download)} papers.")
    return downloaded_paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and cache arXiv PDFs from a paper list.")
    ap.add_argument("--paper-list", required=True, help="Path to paper_list_<exam_id>.txt")
    ap.add_argument("--pdf-dir", default="downloaded_pdfs", help="Output directory for cached PDFs")
    ap.add_argument("--start-line", type=int, default=1, help="1-based line to start from in the paper list")
    ap.add_argument("--resume-from-cache", action="store_true", help="Auto-resume from the first non-cached paper in the list")
    ap.add_argument("--only-missing", action="store_true", help="Download only papers missing from pdf-dir (preserves order)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit; otherwise download only first N papers")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be downloaded and exit (no downloads)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    papers = load_paper_list(args.paper_list)
    limit = args.limit if args.limit and args.limit > 0 else None

    # Determine starting index (0-based)
    start_idx = max(0, int(args.start_line) - 1)
    if args.resume_from_cache:
        cached = cached_arxiv_ids(str(pdf_dir))
        start_idx = compute_resume_index(papers, cached)
        logger.info(f"Resuming from list index {start_idx} (1-based line {start_idx + 1})")

    papers = papers[start_idx:]

    if args.only_missing:
        cached = cached_arxiv_ids(str(pdf_dir))
        before = len(papers)
        papers = [p for p in papers if extract_arxiv_id(p) not in cached]
        logger.info(f"Filtered only-missing: {before} -> {len(papers)} papers to download")

    if limit is not None:
        papers = papers[:limit]

    if args.dry_run:
        logger.info(f"[DRY RUN] Would download {len(papers)} PDFs into {pdf_dir.resolve()}")
        for i, pid in enumerate(papers[:20], 1):
            logger.info(f"[DRY RUN] {i:03d}: {extract_arxiv_id(pid)}")
        raise SystemExit(0)

    downloaded = download_papers_batch(papers, str(pdf_dir), limit=None)
    print(f"[DONE] downloaded/available PDFs: {len(downloaded)} in {pdf_dir.resolve()}")

