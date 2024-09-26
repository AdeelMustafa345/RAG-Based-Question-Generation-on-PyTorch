import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

class Generator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mohammedaly2222002/t5-small-squad-qg")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("mohammedaly2222002/t5-small-squad-qg").to("cpu")

    def prepare_instruction(self, highlighted_context, question_number):
        instruction_prompt = (
            f"Generate question {question_number}: Generate a question whose answer is highlighted by <h> from the context delimited by triple backticks.\n"
            "context:\n```\n"
            f"{highlighted_context}\n"
            "```"
        )
        return instruction_prompt

    def generate_question(self, instruction_prompt):
        inputs = self.tokenizer(instruction_prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.model.generate(
            inputs['input_ids'].to("cpu"),
            max_length=128,
            num_beams=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def rank_questions(self, questions):
        scores = {question: random.uniform(0, 1) for question in questions}
        sorted_questions = sorted(scores, key=scores.get, reverse=True)
        return sorted_questions
