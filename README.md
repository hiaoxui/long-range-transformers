# Long Range Transformers

Some variants of transformers are claimed to be able to process long contexts,
while most of them were ony tested on pseudo tasks, like LRA, or language modeling,
leaving their capability of comprehending long texts to be explored.
The goal of this project is to verify the effectiveness of long-range transformers on more practical NLP tasks:
Do they really work on NLP tasks concerning with long texts?
If not, why, and how can we make it work?

## Tasks

1. Coreference
2. NLI
3. Abstractive QA
4. Extractive QA
5. Summarization
6. 
## Datasets

- Ontonotes for coref
- DocNLI for NLI
- Qasper for abstractive QA
- Triviaqa for extractive QA
- SummFD and CNN

## Model

- Coarse2fine model for coref (located in [this folder](allennlp_modules))
- A baseline model for DocNLI (located in [this folder](allennlp_modules))
- A baseline model for abstractive QA (located in [this folder](abstractive_qa))
- A baseline model for extractive QA (located in [this folder](extractive_qa))
- A baseline model for summarization (located in [this folder](summarization))

# Experiments

## Coref

See [this doc](docs/coref.md)

### 2.2 NLI

See [this doc](docs/nli.md)
