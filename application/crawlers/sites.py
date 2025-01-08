import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def normalize_url(url):
    """
    Normalize the URL by removing fragments and trailing slashes.
    """
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="").geturl()
    return normalized.rstrip("/")

def extract_links(url):
    """
    Extracts all valid 'https://' links from a webpage.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        links = set()

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(url, href)
            if full_url.startswith("https://"):
                links.add(normalize_url(full_url))

        return links

    except requests.exceptions.RequestException as e:
        print(f"[Sites] Error fetching {url}: {e}")
        return set()


def dfs_crawl(url, visited, domain, depth=0, max_depth=2):
    """
    Performs DFS traversal to crawl all links within the same domain.
    Stops if the maximum depth is reached.
    """
    if depth > max_depth:
        return

    url = normalize_url(url)
    if url in visited:
        return

    visited.add(url)
    print(f"[Sites] Crawling URL: {url}")

    links = extract_links(url)

    for link in links:
        if urlparse(link).netloc == domain and link not in visited and link is not url:
            dfs_crawl(link, visited, domain, depth + 1, max_depth)


# Main function
def crawl_website(start_url):
    """
    Crawls all pages within the same domain starting from start_url.
    """
    domain = urlparse(start_url).netloc
    visited = set()

    dfs_crawl(start_url, visited, domain)
    return visited


if __name__ == "__main__":
    url = "https://en.cppreference.com/"
    print(f"[Sites] Starting crawl at {url}...\n")
    visited_links = crawl_website(url)
    print(f"\n[Sites] Crawled {len(visited_links)} links within the domain.")
