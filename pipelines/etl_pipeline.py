import sys
import os

from etl.extract.sites import crawl_website
from etl.extract.site_crawler import SiteCrawler
from url_links import crawl_urls

from clearml import PipelineController

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

site_crawler = SiteCrawler()

pipe = PipelineController(
    name="ETL", project="prompt2program", version="1.0.0",
    repo="",
)


# Extract URLs
pipe.add_function_step(
    name="extract_urls",
    function=crawl_website,
    function_kwargs=dict(start_url=crawl_urls),
    function_return=['urls']
)

# Clean websites and store them
pipe.add_function_step(
    name="extract_and_clean_info",
    parents=['extract_urls'],
    function=site_crawler.extract,
    function_kwargs=dict(link='${extract_urls.urls}')
)

# pipe.add_function_step(
#     name="load",
#     function=,
# )

pipe.set_default_execution_queue("default")

if __name__ == "__main__":
    pipe.start_locally(run_pipeline_steps_locally=True)