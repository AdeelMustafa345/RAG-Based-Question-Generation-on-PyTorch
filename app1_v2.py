import wikipediaapi
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint

# Set the API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Class to fetch content from Wikipedia
class Retrieval:
    def __init__(self):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="MyWikiApp/1.0 (https://mywebsite.com; myemail@example.com)"
        )

    def fetch_wikipedia_page(self, page_name):
        """Fetch content from a Wikipedia page."""
        page = self.wiki_wiki.page(page_name)
        if page.exists():
            return page.text
        else:
            return None

    def highlight_answer(self, context, answer):
        """Highlight the answer in the context."""
        return context.replace(answer, f'<h> {answer} </h>')

# Class to generate questions using the HuggingFace API
class Generator:
    def __init__(self):
        # Initialize Hugging Face model for question generation
        self.llm = HuggingFaceEndpoint(
            repo_id="mohammedaly2222002/t5-small-squad-qg",
            temperature=0.7,  # Explicitly set temperature
            top_p=0.95,  # Explicitly set top_p
            top_k=50,  # Explicitly set top_k
            max_length=128  # Explicitly set max_length
        )

    def prepare_instruction(self, highlighted_context, question_number):
        """Prepare a more refined instruction for question generation."""
        instruction_prompt = (
            f"Generate a clear and concise question {question_number}: "
            "The answer is highlighted by <h> tags from the context provided below, delimited by triple backticks.\n"
            "Please ensure the question is precise, based on the context.\n"
            "context:\n```\n"
            f"{highlighted_context}\n"
            "```"
        )
        return instruction_prompt

    def generate_question(self, instruction_prompt):
        """Generate a question based on the refined prompt."""
        # Use LangChain's LLMChain with a prompt template
        chain = LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(instruction_prompt))
        # Instead of `run()`, use `invoke()` and pass the instruction prompt
        return chain.invoke(instruction_prompt)

# Streamlit App
def main():
    st.title("Wikipedia-based Question Generator")

    # Fetch Wikipedia content
    page_name = st.text_input("Enter Wikipedia page name:")
    retrieval = Retrieval()

    if page_name:
        wikipedia_content = retrieval.fetch_wikipedia_page(page_name)

        if wikipedia_content:
            st.subheader("Retrieved Wikipedia Content:")
            st.write(wikipedia_content[:500])  # Display the first 500 characters

            # Input answer from user
            input_answer = st.text_input("Enter a text from the context to generate questions:")
            if input_answer and input_answer in wikipedia_content:
                highlighted_context = retrieval.highlight_answer(wikipedia_content, input_answer)
                st.subheader("Highlighted Context:")
                st.write(highlighted_context[:500])  # Show part of the context

                # Ask how many questions to generate
                num_questions = st.number_input("How many questions do you want to generate?", min_value=1, max_value=10)

                # Safeguard for long enough context
                if len(wikipedia_content) > 100:
                    # Generate the questions
                    generator = Generator()
                    generated_questions = []
                    for i in range(num_questions):
                        instruction_prompt = generator.prepare_instruction(highlighted_context, i + 1)  # Pass the question number
                        generated_question = generator.generate_question(instruction_prompt)
                        if generated_question:
                            generated_questions.append(generated_question)
                        else:
                            st.warning(f"Failed to generate question {i + 1}")

                    # Display the generated questions
                    st.subheader("Generated Questions:")
                    if generated_questions:
                        for idx, question in enumerate(generated_questions, start=1):
                            st.write(f"{idx}. {question}")
                    else:
                        st.error("No questions generated. Please try again.")
                else:
                    st.error("Content is too short to generate questions.")
            else:
                st.error("Input answer not found in the Wikipedia content.")

        else:
            st.error("Page not found.")

if __name__ == "__main__":
    main()
