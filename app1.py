import streamlit as st
from retrieval import Retrieval
from generator import Generator

def main():
    st.title("Wikipedia Question Generator")

    retrieval = Retrieval()
    generator = Generator()

    page_name = st.text_input("Enter Wikipedia page name:")
    num_paragraphs = st.number_input("Enter the number of paragraphs to retrieve:", min_value=1, max_value=10, value=3)

    if page_name:
        wikipedia_content = retrieval.fetch_wikipedia_page(page_name, num_paragraphs)

        if wikipedia_content:
            st.subheader("Retrieved Wikipedia Content:")
            st.write(wikipedia_content)

            answer_provided = st.radio("Do you want to provide an answer?", ("No", "Yes"))

            if answer_provided == "Yes":
                input_answer = st.text_input("Enter a text from the context to generate questions:")
                highlighted_context = retrieval.highlight_answer(wikipedia_content, input_answer)
            else:
                extracted_answers = retrieval.extract_answers(wikipedia_content)
                if extracted_answers:
                    input_answer = extracted_answers[0]
                    highlighted_context = retrieval.highlight_answer(wikipedia_content, input_answer)
                else:
                    st.warning("No answers found with NER.")
                    return

            num_questions = st.slider("How many questions do you want to generate?", 1, 10, 3)

            if st.button("Generate Questions"):
                generated_questions = []
                for i in range(num_questions):
                    instruction_prompt = generator.prepare_instruction(highlighted_context, i + 1)
                    generated_question = generator.generate_question(instruction_prompt)
                    generated_questions.append(generated_question)

                ranked_questions = generator.rank_questions(generated_questions)

                st.subheader("Ranked Questions:")
                for idx, question in enumerate(ranked_questions, start=1):
                    st.write(f"{idx}. {question}")

if __name__ == "__main__":
    main()
