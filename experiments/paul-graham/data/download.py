import time
from pathlib import Path
from urllib.parse import urljoin, urldefrag

import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm


def main():
    root_url = "https://www.paulgraham.com/articles.html"
    skip_urls = {
        root_url,
        "https://www.paulgraham.com/index.html",
    }

    root_contents = requests.get(root_url).text
    soup = BeautifulSoup(root_contents, "html.parser")

    urls = set()
    for a in soup.find_all("a", href=True):
        url = urljoin(root_url, a["href"])
        url, _ = urldefrag(url)
        if url not in skip_urls and url.startswith("https://www.paulgraham.com/"):
            urls.add(url)

    all_text = ""
    for url in tqdm(sorted(urls)):
        all_text += "<|article-start|>\n\n"
        downloaded = trafilatura.fetch_url(url)
        all_text += trafilatura.extract(downloaded)
        all_text += "\n\n<|article-end|>\n\n"
        time.sleep(2)  ## be polite and don't hammer the server


    (Path(__file__).parent / "input.txt").write_text(all_text)




if __name__ == "__main__":
    main()
