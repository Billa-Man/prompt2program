import re
import requests
from loguru import logger
from bs4 import BeautifulSoup

from domain.documents import SiteDocument
from etl.base import BaseCrawler

class SiteCrawler(BaseCrawler):
    site_doc = SiteDocument

    def extract(self, link: str, **kwargs) -> None:
        old_doc = self.site_doc.find(link=link)
        if old_doc is not None:
            logger.info(f"Site already exists in the database: {link}")
            return
        
        logger.info(f"Starting scrapping URL: {link}")

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
            clean_text = re.sub(r'\s+', ' ', text)
            clean_text = re.sub(r'\S+@\S+', '[email]', clean_text)
            clean_text = re.sub(r'\[ edit \]', '', clean_text)
            clean_text = re.sub(r'\[edit\]', '', clean_text)
            clean_text = re.sub(r'Create account Log in Namespaces Page Discussion Variants Views View Edit History Actions', '', clean_text)
            clean_text = re.sub(r'Navigation Support us Recent changes FAQ Offline version Toolbox What links here Related changes Upload file Special pages Printable version Permanent link Page information In other languages Česky Deutsch Español Français Italiano 日本語 한국어 Polski Português Русский 中文 This page was last modified on 2 August 2024, at 21:20. Privacy policy About cppreference.com Disclaimers', '', clean_text)

            instance = self.site_doc(
                content=clean_text,
                link=link,
                platform='site'
            )

            instance.save()

        except Exception:
            raise

        logger.info(f"Finished scraping URL: {link}")
