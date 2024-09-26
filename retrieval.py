import wikipediaapi
import spacy
import streamlit as st

class Retrieval:
    def __init__(self):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="MyWikiApp/1.0 (https://mywebsite.com; rafayofficial2022@gmail.com)"
        )
        self.nlp = spacy.load("en_core_web_sm")

    def fetch_wikipedia_page(self, page_name, num_paragraphs=3):
        page = self.wiki_wiki.page(page_name)
        if page.exists():
            paragraphs = page.text.split("\n\n")[:num_paragraphs]
            return "\n\n".join(paragraphs)
        else:
            st.warning(f"Original query '{page_name}' not found. Trying variations...")
            variations = [page_name.lower(), page_name.replace(" ", "_"), page_name.replace(" ", "")]
            for variation in variations:
                page = self.wiki_wiki.page(variation)
                if page.exists():
                    paragraphs = page.text.split("\n\n")[:num_paragraphs]
                    return "\n\n".join(paragraphs)
            return None

    def extract_answers(self, context):
        doc = self.nlp(context)
        answers = [ent.text for ent in doc.ents]
        return answers

    def highlight_answer(self, context, answer):
        return context.replace(answer, f'<h> {answer} <h>')
