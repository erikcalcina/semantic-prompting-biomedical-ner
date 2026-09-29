import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

#: Prompt file per setting: plain label descriptions (baseline) or semantic definitions.
PROMPT_FILES = {
    "baseline": "baseline_prompts.json",
    "semantic": "semantic_prompts.json",
}

#: Dataset label -> key in the prompt JSON. In the semantic setting these labels carry
#: their definition in the instruction (fine-tuning) prompt; other labels stay bare.
LABEL_KEY_MAP = {
    "Age": "age",
    "Sex": "sex",
    "Biological structure": "biological_structure",
    "Diagnostic procedure": "diagnostic_procedure",
    "Lab value": "lab_value",
    "Sign symptom": "sign_symptom",
    "Detailed description": "detailed_description",
    "DISEASE": "disease",
}


class Prompts:
    def __init__(self, semantic: bool = False):
        """
        Args:
            semantic: use the semantic label definitions instead of the plain descriptions.
        """
        self.semantic = semantic
        prompt_path = os.path.join(CURRENT_DIR, PROMPT_FILES["semantic" if semantic else "baseline"])
        with open(prompt_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def _get_description(self, label, output_type="few_shot_prompting"):
        return self.data[output_type]["labels"][label]["description"]

    def _get_output_structure(self, label, output_type="few_shot_prompting"):
        return self.data[output_type]["labels"][label]["output_structure"]

    def _format_block(self, output_structure):
        """The answer format as shown in the paper: an indented JSON list ending with '...'."""
        entity = "\n".join("    " + line for line in json.dumps(output_structure, indent=4).split("\n"))
        return "[\n" + entity + "\n    ...\n]"

    def _content(self, description, output_structure):
        description = description[:1].lower() + description[1:]
        return "Please extract " + description + " Answer must follow the following format:\n" + self._format_block(output_structure)

    def _get_system_prompt(self, prompt_type):
        return self.data[prompt_type]["system_prompt"]

    def create_prompt_only_prompt(self, label, medical_text):
        """Returns the messages for prompt only

            Args:
                label: string - The label to be extracted.
                medical_text: string - The medical text the label is extracted.

            Returns:
                List[dict]: A list of system, user messages.

        """
        message = []
        description = self._get_description(label, output_type="prompt_only")
        output_structure = self._get_output_structure(label, output_type="prompt_only")
        message.append({"role": "system", "content": self._get_system_prompt("prompt_only")})
        message.append({"role": "user", "content": self._content(description, output_structure) + "\n\nMedical text:\n" + medical_text})
        return message

    def _get_example(self, label, example_number):
        return self.data["few_shot_prompting"]["labels"][label]["example_" + str(example_number)]

    def create_few_shot_prompt(self, label, medical_text):
        """Returns the messages for few shot prompting with 3 examples.

            Args:
                label: string - The label to be extracted.
                medical_text: string - The medical text the label is extracted.

            Returns:
                List[dict]: A list of system, user, assistant messages.

        """
        message = []
        description = self._get_description(label, output_type="few_shot_prompting")
        output_structure = self._get_output_structure(label, output_type="few_shot_prompting")
        message.append({"role": "system", "content": self._get_system_prompt("prompt_only")})
        # Examples 1, 4, 2: positive, negative, positive. The instruction is given once, before
        # the first example; the following turns contain only the medical text.
        for i, example_number in enumerate((1, 4, 2)):
            example = self._get_example(label, example_number)
            instruction = self._content(description, output_structure) + "\n\n" if i == 0 else ""
            message.append({"role": "user", "content": instruction + "Medical text:\n" + example["input"]})
            message.append({"role": "assistant", "content": json.dumps(example["output"], indent=4)})
        message.append({"role": "user", "content": "Medical text:\n" + medical_text})
        return message

    def _create_conversational_prompt(self, labels, output_structure, medical_text):
        output_structure = json.dumps(output_structure)
        labels_string = ', '.join(str(label) for label in labels)
        instruction_content = "Please extract the following entities: " + labels_string + ". Answer must follow the following format:\n[" + output_structure + "...]"
        prompt = instruction_content + "\n\nMedical text:\n" + medical_text
        return prompt

    def create_conversational_training_message_with_completion(self, labels, medical_text, system_output):
        # Function is for training the model on the conversational prompt
        message = []
        output_structure = self._get_output_structure("no_label", output_type="conversational_training")
        message.append({"role": "system", "content": self._get_system_prompt("conversational_training")})
        message.append({"role": "user", "content": self._create_conversational_prompt(labels, output_structure, medical_text)})
        message.append({"role": "assistant", "content": json.dumps(system_output)})
        return message

    def create_conversational_message(self, labels, medical_text):
        # Function is for evaluating the model on the conversational prompt
        message = []
        output_structure = self._get_output_structure("no_label", output_type="conversational_training")
        message.append({"role": "system", "content": self._get_system_prompt("conversational_training")})
        message.append({"role": "user", "content": self._create_conversational_prompt(labels, output_structure, medical_text)})
        return message

    def _create_instruction_content(self, labels_string, output_structure, n_labels):
        entities = "entity" if n_labels == 1 else "entities"
        return "Please extract the following " + entities + ": " + labels_string + ". Answer must follow the following format:\n" + output_structure

    def _labels_string(self, labels):
        parts = []
        for label in labels:
            key = LABEL_KEY_MAP.get(label)
            entry = self.data["prompt_only"]["labels"].get(key, {}) if key else {}
            name = entry.get("display_name", str(label))
            if self.semantic and key is not None:
                description = entry["description"]
                description = entry.get("instruction_description", description[:1].lower() + description[1:].removesuffix("."))
                parts.append(f"{name} ({description})")
            else:
                parts.append(name)
        return ", ".join(parts)

    def _create_instruction_prompt(self, labels, output_structure, medical_text, output_type):
        if len(labels) == 1:
            # A single-label dataset (NCBI) shows its own label in the output example.
            output_structure = {"text": "extracted value", "label": labels[0]}
        output_structure = self._format_block(output_structure)
        instruction_content = self._create_instruction_content(self._labels_string(labels), output_structure, len(labels))
        return self._get_system_prompt(output_type) + "\n\n" + instruction_content + "\n\nMedical text:\n" + medical_text

    def create_instruction_training_message_with_completion(self, labels, medical_text, system_output):
        # Function is for training the model on the instruction prompt
        message = {}
        output_structure = self._get_output_structure("no_label", output_type="instruction_training")
        message["prompt"] = self._create_instruction_prompt(labels, output_structure, medical_text, output_type="instruction_training")
        message["completion"] = json.dumps(system_output)
        return message

    def create_instruction_message(self, labels, medical_text):
        # Function is for evaluating the model on the instruction prompt
        message = {}
        output_structure = self._get_output_structure("no_label", output_type="instruction_training")
        message["prompt"] = self._create_instruction_prompt(labels, output_structure, medical_text, output_type="instruction_training")
        return message
