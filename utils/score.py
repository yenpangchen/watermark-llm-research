## perspective api toxic score
## https://developers.perspectiveapi.com/s/docs-sample-requests?language=en_US
# pip install google-api-python-client

import os
import json
from googleapiclient import discovery
from dotenv import load_dotenv

# 加載 .env 文件中的環境變數
load_dotenv()

def cal_toxicity_score(comment_text):
    API_KEY = os.getenv('PERSPECTIVE_API_KEY')
    
    if not API_KEY:
        raise ValueError("API key not found. Please set the PERSPECTIVE_API_KEY environment variable.")
    
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    
    analyze_request = {
        'comment': {'text': comment_text},
        'requestedAttributes': {'TOXICITY': {}}
    }
    
    response = client.comments().analyze(body=analyze_request).execute()
    return response

# Example usage
if __name__ == "__main__":
    comment = "friendly greetings from python"
    result = cal_toxicity_score(comment)
    print(json.dumps(result, indent=2))
