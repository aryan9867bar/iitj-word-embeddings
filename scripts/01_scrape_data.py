# Collect textual data from IIT Jodhpur sources for Word2Vec training

import requests
from bs4 import BeautifulSoup
import os
import time
import re
import sys


# CONFIGURATION
# Base directory for raw data output
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# HTTP headers to mimic a standard browser request
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36'),
    'Accept-Language': 'en-US,en;q=0.9',
}

# Delay between requests in seconds (to be respectful to the server)
REQUEST_DELAY = 1.5

# URLs — organized by category (using the new www.iitj.ac.in URL structure)

# SOURCE 1: Academic Regulations (MANDATORY)
# The full regulations are available as a single long page
ACADEMIC_REGULATION_URLS = [
    ("https://www.iitj.ac.in/office-of-academics/en/academic-regulations",
     "Academic Regulations"),
]

# SOURCE 2: Department pages — about, vision, mission of each department
DEPARTMENT_URLS = [
    ("https://www.iitj.ac.in/computer-science-engineering/", "CSE Department"),
    ("https://www.iitj.ac.in/electrical-engineering/", "Electrical Engineering"),
    ("https://www.iitj.ac.in/mechanical-engineering/", "Mechanical Engineering"),
    ("https://www.iitj.ac.in/civil-and-infrastructure-engineering/", "Civil Engineering"),
    ("https://www.iitj.ac.in/chemistry/en/chemistry", "Chemistry"),
    ("https://www.iitj.ac.in/physics/", "Physics"),
    ("https://www.iitj.ac.in/mathematics/", "Mathematics"),
    ("https://www.iitj.ac.in/bioscience-bioengineering", "Bioscience & Bioengineering"),
    ("https://www.iitj.ac.in/materials-engineering/en/materials-engineering", "Materials Engineering"),
    ("https://www.iitj.ac.in/chemical-engineering/", "Chemical Engineering"),
]

# SOURCE 3: Faculty profiles — names, designations, research interests
FACULTY_URLS = [
    ("https://www.iitj.ac.in/People/List?dept=computer-science-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd&ln=en",
     "CSE Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=electrical-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
     "EE Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=mechanical-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
     "ME Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=physics&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
     "Physics Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=mathematics&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
     "Mathematics Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=chemistry&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
     "Chemistry Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=civil-and-infrastructure-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
     "Civil Engineering Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=bioscience-bioengineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
     "Bioscience Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=school-of-artificial-intelligence-data-science&c=ce26246f-00c9-4286-bb4c-7f023b4c5460",
     "AI & Data Science Faculty"),
    ("https://www.iitj.ac.in/People/List?dept=school-of-liberal-arts&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd",
     "Liberal Arts Faculty"),
]

# SOURCE 4: Academic program pages — BTech, MTech, PhD descriptions
PROGRAM_URLS = [
    ("https://www.iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
     "BTech Programs"),
    ("https://www.iitj.ac.in/Master-of-Technology/en/Master-of-Technology",
     "MTech Programs"),
    ("https://www.iitj.ac.in/Master-of-Science/en/Master-of-Science",
     "MSc Programs"),
    ("https://www.iitj.ac.in/Doctor-of-Philosophy/en/Doctor-of-Philosophy",
     "PhD Programs"),
    ("https://www.iitj.ac.in/m/Index/main-programs?lg=en",
     "All Programs Overview"),
    ("https://www.iitj.ac.in/m/Index/main-departments?lg=en",
     "All Departments Overview"),
]

# SOURCE 5: General / About pages
GENERAL_URLS = [
    ("https://www.iitj.ac.in/main/en/iitj", "About IIT Jodhpur"),
    ("https://www.iitj.ac.in/main/en/faculty-members", "Faculty Members Overview"),
    ("https://www.iitj.ac.in/main/en/recruitments", "Recruitments"),
    ("https://www.iitj.ac.in/main/en/contact", "Contact"),
    ("https://www.iitj.ac.in/office-of-students/en/office-of-students",
     "Office of Students"),
    ("https://www.iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
     "Admission to PG Programs"),
]

# SOURCE 6: Research pages — Office of R&D, CRF, research facilities
RESEARCH_URLS = [
    ("https://www.iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
     "Office of Research & Development"),
    ("https://www.iitj.ac.in/crf/en/crf",
     "Central Research Facility"),
    ("https://www.iitj.ac.in/aiot-fab-facility/en/aiot-fab-facility?",
     "AIOT Fab Facility"),
    ("https://www.iitj.ac.in/office-of-research-development/en/projects",
     "Sponsored Projects"),
    ("https://www.iitj.ac.in/office-of-research-development/en/information",
     "R&D Information"),
]

# SOURCE 7: Announcements, News & Events
ANNOUNCEMENT_URLS = [
    ("https://www.iitj.ac.in/main/en/news", "All News"),
    ("https://www.iitj.ac.in/main/en/all-announcement", "All Announcements"),
    ("https://www.iitj.ac.in/main/en/events", "All Events"),
]



# HELPER FUNCTIONS
def ensure_directory(path):
    """ Create the output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def scrape_webpage(url, description=""):
    """ Scrape visible text content from a single web page. """
    print(f"  Scraping: {description}")
    print(f"    URL: {url}")
    
    try:
        # Send GET request with browser-like headers and a 30-second timeout
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors (4xx, 5xx)
        
        # Parse the HTML content using lxml parser for speed
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove tags that do not contain useful textual content
        tags_to_remove = ['script', 'style', 'nav', 'noscript', 'iframe', 'svg']
        for tag_name in tags_to_remove:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Extract all visible text, using newline as separator between elements
        text = soup.get_text(separator='\n')
        
        # Clean up: remove empty lines and extra whitespace
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) > 2:  # Ignore very short fragments
                lines.append(stripped)
        
        cleaned_text = '\n'.join(lines)
        print(f"    ✓ Extracted {len(cleaned_text):,} characters")
        return cleaned_text
        
    except requests.RequestException as e:
        print(f"    ✗ Failed: {e}")
        return ""


def save_text(text, filename):
    """ Save extracted text to a file in the raw data directory."""
    filepath = os.path.join(RAW_DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"    → Saved to {filepath}")


def scrape_category(url_list, output_filename, description=""):
    """ Scrape all URLs in a category and combine them into a single text file."""
    all_texts = []
    
    for url, desc in url_list:
        text = scrape_webpage(url, desc)
        if text:
            # Add a section header to separate content from different pages
            all_texts.append(f"--- {desc} ---\n{text}")
        # Respectful delay between requests
        time.sleep(REQUEST_DELAY)
    
    # Combine all scraped text
    combined = '\n\n'.join(all_texts)
    
    if combined:
        save_text(combined, output_filename)
        print(f"  ✓ Total: {len(combined):,} characters saved to {output_filename}")
    else:
        print(f"  ⚠ No text collected for {output_filename}")
    
    return combined


# MAIN EXECUTION
def main():
    """ Main function to orchestrate the data collection process."""
    print("=" * 70)
    print("IIT JODHPUR DATA COLLECTION")
    print("Using new website: www.iitj.ac.in")
    print("=" * 70)
    
    # Ensure the output directory exists
    ensure_directory(RAW_DATA_DIR)
    
    # SOURCE 1: Academic Regulations (MANDATORY)
    # This is the most important source — contains all UG/PG/PhD regs
    print("\n[1/7] Collecting Academic Regulations (MANDATORY)...")
    scrape_category(ACADEMIC_REGULATION_URLS, 'academic_regulations.txt',
                    'Academic Regulations')
    
    # SOURCE 2: Department Pages
    # About, vision, mission, and research of each department
    print("\n[2/7] Collecting Department Pages...")
    scrape_category(DEPARTMENT_URLS, 'departments.txt', 'Departments')
    
    # SOURCE 3: Faculty Profile Pages
    # Faculty names, designations, research interests
    print("\n[3/7] Collecting Faculty Pages...")
    scrape_category(FACULTY_URLS, 'faculty_profiles.txt', 'Faculty')
    
    # SOURCE 4: Academic Program Pages
    # BTech, MTech, MSc, PhD program descriptions
    print("\n[4/7] Collecting Academic Program Pages...")
    scrape_category(PROGRAM_URLS, 'academic_programs.txt', 'Programs')
    
    # SOURCE 5: General / About Pages
    # Institute overview, recruitments, contact details, student office
    print("\n[5/7] Collecting General Pages...")
    scrape_category(GENERAL_URLS, 'general_pages.txt', 'General')
    
    # SOURCE 6: Research Pages
    # Office of R&D, CRF, AIOT, sponsored projects
    print("\n[6/7] Collecting Research Pages...")
    scrape_category(RESEARCH_URLS, 'research_pages.txt', 'Research')
    
    # SOURCE 7: Announcements, News & Events
    # News, announcements, circulars, events
    print("\n[7/7] Collecting Announcements & News...")
    scrape_category(ANNOUNCEMENT_URLS, 'announcements_news.txt',
                    'Announcements & News')
    
    # SUMMARY
    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    
    # List all collected files and their sizes
    print("\nCollected files:")
    total_size = 0
    for filename in sorted(os.listdir(RAW_DATA_DIR)):
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            print(f"  {filename:40s} {size:>10,} bytes")
    
    print(f"\n  {'TOTAL':40s} {total_size:>10,} bytes")
    print(f"\nRaw data saved to: {RAW_DATA_DIR}")


if __name__ == '__main__':
    main()
