import re
import pandas as pd
from snorkel.labeling import labeling_function
ABSTAIN = -1

# Label codes
WORLD = 0
SPORTS = 1
BUSINESS = 2
SCITECH = 3

# ------------------------------
# Example Heuristic Labeling Functions
# ------------------------------

@labeling_function()
def lf_mention_trump(x):
    return WORLD if "trump" in x.clean_text.lower() else ABSTAIN

@labeling_function()
def lf_mentions_stock_market(x):
    return BUSINESS if re.search(r"\bstock\b|\bmarket\b|\binvestor\b", x.clean_text.lower()) else ABSTAIN

@labeling_function()
def lf_mentions_team_score(x):
    return SPORTS if re.search(r"\bteam\b|\bwin\b|\bscore\b", x.clean_text.lower()) else ABSTAIN

@labeling_function()
def lf_mentions_technology(x):
    return SCITECH if re.search(r"\bsoftware\b|\btech\b|\bAI\b|\balgorithm\b", x.clean_text.lower()) else ABSTAIN

@labeling_function()
def lf_mention_google(x):
    return SCITECH if "google" in x.clean_text.lower() else ABSTAIN

@labeling_function()
def lf_mention_ceo(x):
    return BUSINESS if "ceo" in x.clean_text.lower() else ABSTAIN

@labeling_function()
def lf_mentions_sports_terms(x):
    return SPORTS if re.search(r"\bmatch\b|\bleague\b|\bgoal\b", x.clean_text.lower()) else ABSTAIN

# Collect LFs
lfs = [
    lf_mention_trump,
    lf_mentions_stock_market,
    lf_mentions_team_score,
    lf_mentions_technology,
    lf_mention_google,
    lf_mention_ceo,
    lf_mentions_sports_terms,
]
