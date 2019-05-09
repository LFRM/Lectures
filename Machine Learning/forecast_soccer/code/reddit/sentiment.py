# Forked from https://github.com/sharan-naribole/reddit-sentiment-soccer-prediction
import numpy as np
import preprocess
from textblob import TextBlob
import editdistance

def compute_sentiment(comment):
    """
    Returns the sentiment polarity of the comment body
    using textblob package
    input: comment instance

    Output: 
    sentiment property is a namedtuple of the form 
    Sentiment(polarity, subjectivity). The polarity 
    score is a float within the range [-1.0, 1.0]. 
    The subjectivity is a float within the range 
    [0.0, 1.0] where 0.0 is very objective and 1.0 
    is very subjective.
    """
    return TextBlob(comment.body).sentiment

def comments_sentiment(comments_list, Ntop = None):
    """
    Function to sort the comments in the given list of comments,
    extract the top Ntop comments and return list of sentiment
    polarities.

    Inputs: 
    comments_list            : list of comments
    Ntop                     : Number of top comments from each flair 
                               type for whom sentiment polarities 
                               will be collected. If None, all 
                               comments will be considered.

    Outputs:
    polarity_by_score        : polarity weighted by comments
    polarity_by_subjectivity : polarity weighted by subjectivity
    polarity_std             : standard deviation of polarities of 
                               individual comments
    polarity_overall         : Polarity obtained by combining top N 
                               comments together
    """
    
    # empty list, return NaN
    if not len(comments_list):
        return [np.nan]*3
    
    if not Ntop:
        Ntop = len(comments_list)
    
    comments_list.sort(key= lambda comment: comment.score, reverse=True)
    
    # Computing weighted polarity based on comment score and based 
    # on subjectivity
    comments_score = [comment.score for comment in comments_list[:Ntop]]
    comments_polarity = []
    comments_subjectivity = []
    
    for comment in comments_list[:Ntop]:
        sentiment = compute_sentiment(comment)
        comments_polarity.append(sentiment.polarity)
        #TODO: I think there is a mistake here, it should be 
        #      sentiment.subjectivity
        comments_subjectivity.append(sentiment.polarity)
    
    polarity_by_score = (np.sum(np.multiply(
                                comments_polarity,comments_score))/
                                np.sum(comments_score))
    polarity_by_subjectivity = (np.sum(np.multiply(
                                comments_polarity,comments_subjectivity))/
                                len(comments_subjectivity))
    polarity_std = np.std(comments_polarity)
    
    comments_overall = " ".join([comment.body for comment 
                                in comments_list[:Ntop]])
    polarity_overall = TextBlob(comments_overall).sentiment.polarity
    
    return [polarity_by_score, 
            polarity_by_subjectivity,
            polarity_std,
            polarity_overall]

def submission_sentiment(submission,team_a,team_b, Ntop = None):
    """
    Function to obtain the sentiment polarities for top N highest 
    scoring comments for each of Team A, Team B and Other commenter
    flairs.
    Inputs: 
    submission       : submission instance
    team_a           : Team A name
    team_b           : Team B name
    
    Output: 
    polarity_metrics : combined list of polarity metrics for all flair types
    """
    
    submission.comments.replace_more(limit=0)
    team_a_comments = []
    team_b_comments = []
    other_comments = []
    
    
    ## Classfication of comments based on flair
    for comment in submission.comments.list():
        
        flair = preprocess.string_normalize(comment.author_flair_text)
        
        """
        Failing to detect whether a comment's flair belongs to
        Team A or Team B when it actually belongs to one of the teams
        will lead to incorrect classification as Other Teams flair type.
        This results in an incorrect record generation. Examples include:
        Hoffenheim in title has flair 1899 Hoffenhiem, West Brom in title 
        has flair West Bromwich Albion etc. Simple approach to take care
        of above cases is to check if team name is stored in flair. However, 
        this does not address. Bayern Munich in title having flair 
        Bayern Munchen in comments.
        Therefore, an extra condition for checking edit distance lesser than
        equal to 3 is provided.
        """
        if not flair: 
            other_comments.append(comment)
        else:
            if team_a in flair:
                team_a_comments.append(comment)
            elif team_b in flair:
                team_b_comments.append(comment)
            elif editdistance.eval(team_a,flair) <= 3:
                team_a_comments.append(comment)
            elif editdistance.eval(team_b,flair) <= 3:
                team_b_comments.append(comment)
            else:
                other_comments.append(comment)
            
    # Combining the sentiment polarities and filling missing values
    polarity_metrics = []
    polarity_metrics += (comments_sentiment(team_a_comments, Ntop) +
                         comments_sentiment(team_b_comments, Ntop) +
                         comments_sentiment(other_comments, Ntop))          
                         
    return polarity_metrics 
