# MedSimplify

This project aims to extract and summarize medical reports using both extractive (BioBERT) and abstractive (PEGASUS) methods. Additionally, it performs Named Entity Recognition (NER) to identify key medical terms and provide relevant definitions.

Folder Structure
----------------

```
|-- training_data/                 # Folder containing medical reports in .txt format
|-- NER.py                         # Named Entity Recognition and meaning extraction
|-- pegasus.py                     # Abstractive summarization using PEGASUS
|-- bioBERT2.py                    # Extractive summarization using BioBERT
|-- extract_information2.py        # Extracts relevant sections from medical reports
|-- requirements.txt               # List of required dependencies
|-- README.md                      # Project instructions
```

Prerequisites
-------------

Ensure you have Python installed (preferably Python 3.8+).

Installation
------------

1.  Clone this repository:

    ```
    git clone https://github.com/SShreyas17/MedSimplify.git
    cd MedSimplify
    ```

2.  Install required dependencies:

    ```
    pip install -r requirements.txt
    ```

Running the Project
-------------------

Follow these steps to execute the project:

1.  **Run Extractive Summarization (BioBERT)**

    ```
    python bioBERT2.py
    ```

    -   This will process reports in `training_data/` and generate summaries in `BERT_summary/`.

    -   ROUGE scores will be saved as `summary_Scores.xlsx`.

2.  **Run Named Entity Recognition (NER)**

    ```
    python NER.py
    ```

    -   This script identifies key medical entities in the summaries and provides their meanings.

    -   The results will be stored as `summary_with_meanings.txt`.

3.  **Run Abstractive Summarization (PEGASUS)**

    ```
    python pegasus.py
    ```

    -   This will generate abstractive summaries in `Pegasus_summary/`.

    -   ROUGE scores will be saved as `summary_scores_pegasus.xlsx`.

Notes
-----

-   Ensure that the `training_data/` folder contains medical reports in `.txt` format before running the scripts.

-   The project uses `BioBERT` for extractive summarization, `PEGASUS` for abstractive summarization, and `scispaCy` for medical entity recognition.

License
-------

This project is for educational and research purposes. Modify and use as needed.

Acknowledgments
---------------

-   [BioBERT](https://github.com/dmis-lab/biobert)

-   [PEGASUS](https://huggingface.co/google/pegasus-pubmed)

-   [scispaCy](https://allenai.github.io/scispacy/)
