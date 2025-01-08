import requests
from loguru import logger
from bs4 import BeautifulSoup

from domain.documents import SiteDocument
from base import BaseCrawler

class SiteCrawler(BaseCrawler):
    site_doc = SiteDocument

    def extract(self, link: str, **kwargs) -> None:
        old_doc = self.site_doc.find(link=link)
        if old_doc is not None:
            logger.info(f"[Site Crawler] Site already exists in the database: {link}")
            return
        
        logger.info(f"[Site Crawler] Starting scrapping URL: {link}")

        response = requests.get(link)

        if response.status_code == 200:
            html_content = response.content
        else:
            raise Exception

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for tag in soup(["script", "style", "code"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            
            instance = self.site_doc(
                content=text,
                link=link,
                platform='site'
            )

            instance.save()

        except Exception:
            raise

        logger.info(f"[Site Crawler] Finished scraping URL: {link}")
