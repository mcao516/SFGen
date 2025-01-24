# coding=utf-8
import os
from googleapiclient import discovery

API_KEY = os.environ['PERSPECTIVE_API_KEY']
ATTRIBUTES = {
    'TOXICITY': {},
    'SEVERE_TOXICITY': {},
    'IDENTITY_ATTACK': {},
    'INSULT': {},
    'PROFANITY': {},
    'THREAT': {},
    'SEXUALLY_EXPLICIT': {},  # Experimental attributes
    'FLIRTATION': {},  # Experimental attributes
}


client = discovery.build(
    'commentanalyzer',
    'v1alpha1',
    developerKey=API_KEY,
    discoveryServiceUrl='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1',
    static_discovery=False,
)

def get_perspective_scores(sentence):
    analyze_request = {
        'comment': {'text': sentence},
        'requestedAttributes': ATTRIBUTES
    }
    response = client.comments().analyze(body=analyze_request).execute()
    return {k: v['summaryScore']['value'] for k, v in response['attributeScores'].items()}