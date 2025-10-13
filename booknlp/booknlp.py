import sys
import argparse
from typing import Any, Dict, Optional
from transformers import logging
from booknlp.english.english_booknlp import EnglishBookNLP, EnglishBookNLPConfig, Token
from booknlp.common.core import BookNLPResult

logging.set_verbosity_error()


class BookNLP:
    def __init__(
        self, language: str, model_params: Dict[str, Any] | EnglishBookNLPConfig
    ):
        if language == "en":
            self.booknlp = EnglishBookNLP(model_params)

    def process(
        self,
        filename: Optional[str] = None,
        text: Optional[str] = None,
        out_folder: Optional[str] = None,
        doc_id: str = "doc",
    ) -> BookNLPResult:
        return self.booknlp.process(filename, text, out_folder, doc_id)


def proc():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", help="Currently on {en}", required=True)
    parser.add_argument(
        "-i", "--inputFile", help="Filename to run BookNLP on", required=True
    )
    parser.add_argument(
        "-o", "--outputFolder", help="Folder to write results to", required=True
    )
    parser.add_argument(
        "--id",
        help="ID of text (for creating filenames within output folder)",
        required=True,
    )

    args = vars(parser.parse_args())

    language = args["language"]
    inputFile = args["inputFile"]
    outputFolder = args["outputFolder"]
    idd = args["id"]

    print("tagging %s" % inputFile)

    valid_languages = set(["en"])
    if language not in valid_languages:
        print(
            "%s not recognized; supported languages: %s" % (language, valid_languages)
        )
        sys.exit(1)

    model_params = {
        "pipeline": "entity,quote,supersense,event,coref",
        "model": "small",
    }

    booknlp = BookNLP(language, model_params)
    booknlp.process(inputFile, outputFolder, idd)


if __name__ == "__main__":
    proc()
