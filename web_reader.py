import requests
from bs4 import BeautifulSoup

def read_website(url):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    text = ""
    for p in soup.find_all("p"):
        text += p.get_text() + "\n"

    return text.strip()
