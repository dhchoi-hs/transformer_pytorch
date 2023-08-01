import sys
import json
import requests


SENTENCE = 'i\'m __mask__ing __mask__'
resp = requests.post(f"http://192.168.1.70:8000/?text={SENTENCE}", timeout=10)

if resp.status_code != 200:
    print(f'{resp.status_code}: {resp.reason}')
    sys.exit()

PREDICTED_SENTENCE = SENTENCE
# print(f'Input sequence: {text}')
for i, s in enumerate(resp.json()):
    print(f'predected top-3 of mask #{i+1}:')
    print(json.dumps(s, indent=2))
    PREDICTED_SENTENCE = PREDICTED_SENTENCE.replace('__mask__', s[0]['text'], 1)
print(f'input:  {SENTENCE}')
print(f'output: {PREDICTED_SENTENCE}')
