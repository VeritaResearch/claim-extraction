import requests
import tempfile
from urllib.parse import urlparse, urljoin
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
from atlas.document_processing import split_sentences, pdf_to_text

SOURCES = {"congressional record": "https://www.congress.gov/congressional-record"}


def get_congressional_statements():
    """
    Fetches the most recent congressional record from the official website.
    """
    url = SOURCES["congressional record"]

    # Set up Selenium WebDriver
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Firefox(options=options)

    # Load the page
    driver.get(url)

    # Wait for the page to load (if necessary)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    text = [line for line in soup.text.split("\n") if "Entire Issue" in line][0]
    pdf_url = soup.find("a", string=text)["href"]
    parsed_url = urlparse(SOURCES["congressional record"])
    pdf_url = urljoin(f"{parsed_url.scheme}://{parsed_url.netloc}", pdf_url)

    # Download the PDF to a temporary file
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(response.content)
        temp_pdf_path = temp_pdf.name

    # Process the PDF (if needed)
    text, _ = pdf_to_text(temp_pdf_path)
    sentences = split_sentences(text)
    return sentences
