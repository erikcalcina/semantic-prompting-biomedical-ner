"""Entity labels per dataset, shared by training, extraction and evaluation."""

#: The 41 MACCROBAT2020 labels (fine-tuning schema), as written in the gold data.
MACCROBAT_LABELS = [
    "Activity",
    "Administration",
    "Age",
    "Area",
    "Biological attribute",
    "Biological structure",
    "Clinical event",
    "Color",
    "Coreference",
    "Date",
    "Detailed description",
    "Diagnostic procedure",
    "Disease disorder",
    "Distance",
    "Dosage",
    "Duration",
    "Family history",
    "Frequency",
    "Height",
    "History",
    "Lab value",
    "Mass",
    "Medication",
    "Nonbiological location",
    "Occupation",
    "Other entity",
    "Other event",
    "Outcome",
    "Personal background",
    "Qualitative concept",
    "Quantitative concept",
    "Severity",
    "Sex",
    "Shape",
    "Sign symptom",
    "Subject",
    "Texture",
    "Therapeutic procedure",
    "Time",
    "Volume",
    "Weight",
]

LABEL_SETS = {
    "maccrobat": {
        # Keys into the prompt JSON files; zero-/few-shot prompting runs one generation per key.
        "prompt_keys": [
            "age",
            "sex",
            "biological_structure",
            "sign_symptom",
            "diagnostic_procedure",
            "lab_value",
            "detailed_description",
        ],
        # Full schema used for fine-tuning and the instruction prompt.
        "all": MACCROBAT_LABELS,
        # The labels the paper reports scores on.
        "selected": [
            "Age",
            "Sex",
            "Biological structure",
            "Sign symptom",
            "Diagnostic procedure",
            "Lab value",
            "Detailed description",
        ],
    },
    "ncbi": {
        "prompt_keys": ["disease"],
        "all": ["DISEASE"],
        "selected": ["DISEASE"],
    },
}

DATASET_NAMES = list(LABEL_SETS)
