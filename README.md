# Setup
- Download 100 dimensional GloVe embeddings from https://github.com/stanfordnlp/GloVe to `./glove.6B.100d.txt`
- Install dependencies from requirements.txt
- Create necessary resources from the `entity-linking-preprocessing` repo
  - Follow the setup in `README.md` of the `entity-linking-preprocessing` repo
  - Setup `el_en` db by running `create_entity_to_context.py` of the `entity-linking-preprocessing` repo
  - Create the candidates lookup by running `create_candidate_and_entity_lookups.py` of the `entity-linking-preprocessing` repo
- Create `.env` in this directory containing the following values:
``` shell
DBNAME=en_el # mysql db name
DBUSER= # mysql username
DBPASS= # mysql password
DBHOST=localhost
LOOKUPS_PATH= # path to candidates lookup
```
- Fetch the nltk requirements:
``` python
import nltk
nltk.download('punkt')
```
