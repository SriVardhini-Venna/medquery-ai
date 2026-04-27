"""
pubmed_fetcher.py
Fetches paper abstracts from PubMed using NCBI E-utilities (free, no key required).
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import time

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, max_results: int = 25, api_key: str = None) -> List[str]:
    """Search PubMed and return a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids: List[str], api_key: str = None) -> List[Dict]:
    """Fetch full paper metadata + abstracts for a list of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=20)
    resp.raise_for_status()

    papers = []
    root = ET.fromstring(resp.content)

    for article in root.findall(".//PubmedArticle"):
        try:
            # Title
            title_el = article.find(".//ArticleTitle")
            title = (title_el.text or "No title").strip()

            # Abstract (handles structured abstracts with labels)
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                ((el.get("Label", "") + ": ") if el.get("Label") else "") + (el.text or "")
                for el in abstract_parts
            ).strip()

            if not abstract:
                continue

            # PMID
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            # Year
            year_el = article.find(".//PubDate/Year")
            if year_el is None:
                year_el = article.find(".//PubDate/MedlineDate")
            year = (year_el.text or "Unknown")[:4] if year_el is not None else "Unknown"

            # Authors (first 3 + et al.)
            all_authors = article.findall(".//Author")
            author_names = []
            for auth in all_authors[:3]:
                last = auth.find("LastName")
                fore = auth.find("ForeName")
                if last is not None:
                    name = last.text
                    if fore is not None and fore.text:
                        name += f" {fore.text[0]}."
                    author_names.append(name)
            author_str = ", ".join(author_names)
            if len(all_authors) > 3:
                author_str += " et al."

            # Journal
            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else ""

            # Keywords
            keywords = [
                kw.text for kw in article.findall(".//Keyword") if kw.text
            ]

            papers.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": author_str,
                "year": year,
                "journal": journal,
                "keywords": ", ".join(keywords[:8]),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        except Exception:
            continue

    return papers


def fetch_papers_for_topic(topic: str, max_results: int = 25) -> List[Dict]:
    """High-level helper: search + fetch in one call."""
    pmids = search_pubmed(topic, max_results=max_results)
    time.sleep(0.34)  # NCBI rate limit: 3 req/sec without key
    return fetch_abstracts(pmids)
