import os
import random
import re
import sys
from decimal import Decimal as D

DAMPING = 0.85
SAMPLES = 100000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    N = len(corpus)
    d = damping_factor
    all_ = {key: (1 - d) / N for key in corpus}
    links = corpus[page] if corpus[page] else set(corpus.keys())
    
    for link in links:
        all_[link] += d / len(links)
        
    return all_


def pick_with_probabilities(pr_per_page):
    p = random.random()
    l = 0
    for key, prob in pr_per_page.items():
        if l < p < l + prob:
            return key

        else:
            l += prob

    return list(pr_per_page.keys())[-1]
    

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = n
    a = D(1) / D(n)  # increment every time we return to a page
    pr = {key: D() for key in corpus}
    curr_page = random.choice(list(corpus.keys()))
    for _ in range(n):
        curr_page = pick_with_probabilities(transition_model(corpus, curr_page, damping_factor))
        pr[curr_page] += a
    
    return pr


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    d = damping_factor
    pr = {key: 1/N for key in corpus}
    
    def numlinks(page):
        nonlocal corpus, N
        if len(corpus[page]) == 0:
            return N

        return len(corpus[page])

    def links_to(page):
        pages = []
        for item in corpus.items():
            if page in item[1] or not len(item[1]):
                pages.append(item[0])

        return pages

    def PR(page):
        nonlocal d, N, numlinks
        return (1 - d) / N + d * sum([pr[i] / numlinks(i) for i in links_to(page)])
    
    last = {key: 0 for key in corpus}
    while last != pr:
        for key in pr:
            last[key] = pr[key]
            
        for key in pr:
            pr[key] = PR(key)

    return pr


if __name__ == "__main__":
    main()
