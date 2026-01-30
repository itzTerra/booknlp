# BookNLP

BookNLP is a natural language processing pipeline that scales to books and other long documents (in English), including:

* Part-of-speech tagging
* Dependency parsing
* Entity recognition
* Supersense tagging (e.g., "animal", "artifact", "body", "cognition", etc.)
* Event tagging

BookNLP ships with two models, both with identical architectures but different underlying BERT sizes.  The larger and more accurate `big` model is fit for GPUs and multi-core computers; the faster `small` model is more appropriate for personal computers.  See the table below for a comparison of the difference, both in terms of overall speed and in accuracy for the tasks that BookNLP performs.


|                         | Small | Big  |
| ----------------------- | ----- | ---- |
| Entity tagging (F1)     | 88.2  | 90.0 |
| Supersense tagging (F1) | 73.2  | 76.2 |
| Event tagging (F1)      | 70.6  | 74.1 |

## Installation

* Create anaconda environment, if desired. First [download and install anaconda](https://www.anaconda.com/download/); then create and activate fresh environment.

```sh
conda create --name booknlp python=3.7
conda activate booknlp
```

* If using a GPU, install pytorch for your system and CUDA version by following installation instructions on  [https://pytorch.org](https://pytorch.org).


* Install booknlp and download Spacy model.

```sh
pip install booknlp
python -m spacy download en_core_web_sm
```

## Usage

```python
from booknlp.booknlp import BookNLP

model_params={
		"pipeline":"entity,supersense,event", 
		"model":"big"
	}
	
booknlp=BookNLP("en", model_params)

# Input file to process
input_file="input_dir/bartleby_the_scrivener.txt"

# Output directory to store resulting files in
output_directory="output_dir/bartleby/"

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id="bartleby"

booknlp.process(input_file, output_directory, book_id)
```

This runs the full BookNLP pipeline; you are able to run only some elements of the pipeline (to cut down on computational time) by specifying them in that parameter (e.g., to only run entity tagging and event tagging, change `model_params` above to include `"pipeline":"entity,event"`).

This process creates the directory `output_dir/bartleby` and generates the following files:

* `bartleby/bartleby.tokens` -- This encodes core word-level information.  Each row corresponds to one token and includes the following information:
	* paragraph ID
	* sentence ID
	* token ID within sentence
	* token ID within document
	* word
	* lemma
	* byte onset within original document
	* byte offset within original document
	* POS tag
	* dependency relation
	* token ID within document of syntactic head 
	* event

* `bartleby/bartleby.entities` -- This represents the typed entities within the document (e.g., people and places).
	* start token ID within document
	* end token ID within document
	* NOM (nominal), PROP (proper), or PRON (pronoun)
	* PER (person), LOC (location), FAC (facility), GPE (geo-political entity), VEH (vehicle), ORG (organization)
	* text of entity
* `bartleby/bartleby.supersense` -- This stores information from supersense tagging.
	* start token ID within document
	* end token ID within document
	* supersense category (verb.cognition, verb.communication, noun.artifact, etc.) 


# Annotations

## Entity annotations

The entity annotation layer covers six of the ACE 2005 categories in text:

* People (PER): *Tom Sawyer*, *her daughter*
* Facilities (FAC): *the house*, *the kitchen*
* Geo-political entities (GPE): *London*, *the village*
* Locations (LOC): *the forest*, *the river*
* Vehicles (VEH): *the ship*, *the car*
* Organizations (ORG): *the army*, *the Church*

The targets of annotation here include both named entities (e.g., Tom Sawyer), common entities (the boy) and pronouns (he).  These entities can be nested, as in the following:

<img src="img/nested_structure.png" alt="drawing" width="300"/>


For more, see: David Bamman, Sejal Popat and Sheng Shen, "[An Annotated Dataset of Literary Entities](http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/naacl2019_literary_entities.pdf)," NAACL 2019.

The entity tagging model within BookNLP is trained on an annotated dataset of 968K tokens, including the public domain materials in [LitBank](https://github.com/dbamman/litbank) and a new dataset of ~500 contemporary books, including bestsellers, Pulitzer Prize winners, works by Black authors, global Anglophone books, and genre fiction (article forthcoming).

## Event annotations

The event layer identifies events with asserted *realis* (depicted as actually taking place, with specific participants at a specific time) -- as opposed to events with other epistemic modalities (hypotheticals, future events, extradiegetic summaries by the narrator).

| Text                                                                                                                                                         | Events           | Source                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- | -------------------------- |
| My father’s eyes had **closed** upon the light of this world six months, when mine **opened** on it.                                                         | {closed, opened} | Dickens, David Copperfield |
| Call me Ishmael.                                                                                                                                             | {}               | Melville, Moby Dick        |
| His sister was a tall, strong girl, and she **walked** rapidly and resolutely, as if she knew exactly where she was going and what she was going to do next. | {walked}         | Cather, O Pioneers         |

For more, see: Matt Sims, Jong Ho Park and David Bamman, "[Literary Event Detection](http://people.ischool.berkeley.edu/~dbamman/pubs/pdf/acl2019_literary_events.pdf)," ACL 2019.

The event tagging model is trained on event annotations within [LitBank](https://github.com/dbamman/litbank).  The `small` model above makes use of a distillation process, by training on the predictions made by the `big` model for a collection of contemporary texts.

## Supersense tagging

[Supersense tagging](https://aclanthology.org/W06-1670.pdf) provides coarse semantic information for a sentence by tagging spans with 41 lexical semantic categories drawn from WordNet, spanning both nouns (including *plant*, *animal*, *food*, *feeling*, and *artifact*) and verbs (including *cognition*, *communication*, *motion*, etc.)

| Example                                                                                                                                                                                                          | Source                 |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| The [station wagons]<sub>artifact</sub> [arrived]<sub>motion</sub> at [noon]<sub>time</sub>, a long shining [line]<sub>group</sub> that [coursed]<sub>motion</sub> through the [west campus]<sub>location</sub>. | Delillo, *White Noise* |


The BookNLP tagger is trained on [SemCor](https://web.eecs.umich.edu/~mihalcea/downloads.html#semcor).

## Part-of-speech tagging and dependency parsing

BookNLP uses [Spacy](https://spacy.io) for part-of-speech tagging and dependency parsing.

# Acknowledgments

<table><tr><td><img width="250" src="https://www.neh.gov/sites/default/files/inline-files/NEH-Preferred-Seal820.jpg" /></td><td><img width="150" src="https://www.nsf.gov/images/logos/NSF_4-Color_bitmap_Logo.png" /></td><td>
BookNLP is supported by the National Endowment for the Humanities (HAA-271654-20) and the National Science Foundation (IIS-1942591).
</td></tr></table>
