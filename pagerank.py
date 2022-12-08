import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


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

    # Probability distribution of pages
    prob_distribution = dict()

    # Set of pages the current page links to
    links = corpus[page]

    # Each link gets an additional split_prob, which is
    # the probability we choose randomly among all three pages
    split_prob = (1 - damping_factor)/(len(corpus.keys()))
    equal_prob = 1/len(corpus.keys())

    if links:
        # Each link has a probability of damping_prob to start
        damping_prob = damping_factor/len(links)
        for key in corpus.keys():
            if key not in corpus[page]:
                prob_distribution[key] = split_prob
            else:
                prob_distribution[key] = damping_prob + split_prob
    else:
        for key in corpus.keys():
            prob_distribution[key] = equal_prob    
    
    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Counts number of visits for each page in corpus
    page_distribution = dict()
    for page in corpus.keys():
        page_distribution[page] = 0
    
    # starts on random page
    sample = random.choice(list(corpus.keys()))

    for i in range(1, n):
        page_distribution[sample] += 1
        prob_distribution = transition_model(corpus, sample, damping_factor)
        sample = random.choices(list(prob_distribution.keys()), list(prob_distribution.values()), k=1)[0]
        
    # Changes count to percentage of total visits
    for page in page_distribution:
        page_distribution[page] /= n

    return page_distribution
                
        
        

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # PageRank distribution for each page
    page_distribution = dict()

    # Total number of pages
    num_pages = len(corpus)
    
    for page in corpus.keys():
        page_distribution[page] = 1/num_pages
    

    while True:
        
        # Counts the number of pages that have reached
        # the convergence point
        convergence_count = 0

        # Finding PR for all pages in corpus
        for page in corpus:
            
            # Split the formula into new_rank and sum
            # new_rank = 1 - d / N
            new_rank = (1 - damping_factor)/num_pages

            # sum = Sigma(PR(i)/NumLinks(i))
            sum = 0
            
            for link in corpus:
                if page in corpus[link]:
                    num_links = len(corpus[link])
                    sum += page_distribution[link]/num_links
            
            sum *= damping_factor

            # new_rank will now equal PR(p)
            new_rank += sum

            # If PR(page) has reached it's convergence point
            # update convergence_count
            if abs(new_rank - page_distribution[page]) < 0.001:
                convergence_count += 1

            page_distribution[page] = new_rank
        
        if convergence_count == num_pages:
            break

        
    return page_distribution

    


if __name__ == "__main__":
    main()
