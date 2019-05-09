# Forked from https://github.com/sharan-naribole/reddit-sentiment-soccer-prediction
import unicodedata
import re

def string_normalize(x):
    """
    Function to normalize special characters in non-English languages

    Input: 
    x: Sentence

    Output: 
    Transformed sentence
    """
    if not x:
        return    
    x = unicodedata.normalize('NFD',x).encode('ascii','ignore')
    return x.decode("utf-8")

def get_match_info(submission_title):
    """
    Function to extract the team names and match score from the title

    Input: 
    submission_title: Reddit submission title

    Output: 
    match_info: list [Team A name, Team A score, Team B score, Team B name] 
    """
    # Regular expression to get groups (Team A), (Team A score), 
    # (Team B score), (Team B)
    match = re.search(r'Thread.\s(\w+)\s(\w*)\s*(\d+)\s*-\s*(\d+)\s(.+)',
                submission_title)
    if not match:
        return None
    else:
        match_info = list(match.groups())
        
        # If team A has two words, then join
        if match_info[1]:
            match_info[0] += ' ' + match_info[1]
        
        del match_info[1]
        
        # If second team name has additional Competition name 
        #e.g. (English Premier League)
        if "(" in match_info[3]:
            match_info[3] = match_info[3].split(" (")[0]
        elif "[" in match_info[3]:
            match_info[3] = match_info[3].split(" [")[0]
            
        return match_info
