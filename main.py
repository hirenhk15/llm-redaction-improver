# Import necessary packages
import streamlit as st
from enum import Enum
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate


CREATIVITY = 0.7
TEMPLATE = """
    Below is a draft text that may be poorly worded.
    Your goal is to:
    - Properly redact the draft text
    - Convert the draft text to a specified tone
    - Convert the draft text to a specified dialect

    Here are some examples different Tones:
    - Formal: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - Informal: Hey everyone, it's been a wild week! We've got some exciting news to share - Sam Altman is back at OpenAI, taking up the role of chief executive. After a bunch of intense talks, debates, and convincing, Altman is making his triumphant return to the AI startup he co-founded.  

    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, \
        cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flat, rubbish, biscuit, green fingers, \
        car park, trousers, windscreen

    Example Sentences from each dialect:
    - American: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - British: On Wednesday, OpenAI, the esteemed artificial intelligence start-up, announced that Sam Altman would be returning as its Chief Executive Officer. This decisive move follows five days of deliberation, discourse and persuasion, after Altman's abrupt departure from the company which he had co-established.

    Please start the redaction with a warm introduction. Add the introduction \
        if you need to.
    
    Below is the draft text, tone, and dialect:
    DRAFT: {draft}
    TONE: {tone}
    DIALECT: {dialect}

    YOUR {dialect} RESPONSE:
"""


class ModelType(Enum):
    GROQ='GroqCloud'
    OPENAI='OpenAI'


# Defining prompt template
class FinalPromptTemplate:
    def __init__(self, draft:str, tone:str, dialect: str) -> None:
        self.draft=draft
        self.tone=tone
        self.dialect=dialect

    def generate(self) -> str:
        prompt = PromptTemplate(
            input_variables=["draft", "tone", "dialect"],
            template=TEMPLATE
        )
        final_prompt = prompt.format(
            draft=self.draft,
            tone=self.tone,
            dialect=self.dialect
        )
        return final_prompt
    

class LLMModel:
    def __init__(self, model_provider: str) -> None:
        self.model_provider = model_provider

    def load(self, api_key=str):
        try:
            if self.model_provider==ModelType.GROQ.value:
                llm = ChatGroq(temperature=CREATIVITY, model="llama3-70b-8192", api_key=api_key) # model="mixtral-8x7b-32768"
            if self.model_provider==ModelType.OPENAI.value:
                llm = OpenAI(temperature=CREATIVITY, api_key=api_key)
            return llm
        
        except Exception as e:
            raise e

class LLMStreamlitUI:
    def __init__(self) -> None:
        pass

    def get_api_key(self):
        # Get the API Key to query the model
        input_text = st.text_input(
            label="Your API Key",
            placeholder="Ex: sk-2twmA8tfCb8un4...",
            key="api_key_input",
            type="password"
        )
        return input_text
    
    def get_draft(self):
            draft_text = st.text_area(
                label="Text",
                label_visibility="collapsed",
                placeholder="Your Text...",
                key="draft_input"
            )
            return draft_text

    def create(self):
        # Page title and header
        st.set_page_config(page_title="Re-write your text")
        st.markdown("<h1 style='text-align: center;'>Re-write your text</h1>", unsafe_allow_html=True)
        
        # Intro: instructions
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("Re-write and your text in different styles.")

        with col2:
            st.write("Contact Hiren Kelaiya to build your AI Projects")

        # Select the model provider
        st.markdown("## Which model provider you want to choose?")
        option_model_provider = st.selectbox(
                'Select the model provider',
                ('GroqCloud', 'OpenAI')
            )

        # Input API Key for model to query
        st.markdown("## Enter Your API Key")
        api_key = self.get_api_key()

        # Get the input text from user
        st.markdown("## Enter the text you want to re-write")
        draft_input = self.get_draft()

        if len(draft_input.split(" ")) > 700:
            st.write("Please enter a shorter text. The maximum length is 700 words.")
            st.stop()

        # Prompt template tunning options
        col1, col2 = st.columns(2)
        with col1:
            option_tone = st.selectbox(
                'Which tone would you like your redaction to have?',
                ('Formal', 'Informal'))
            
        with col2:
            option_dialect = st.selectbox(
                'Which English Dialect would you like?',
                ('American', 'British'))
        
        # Generate the output using LLM
        st.markdown("### Your Re-written text:")

        if draft_input:
            if not api_key:
                st.warning('Please insert a valid API Key.', 
                    icon="⚠️")
                st.stop()

            # Generate the final prompt
            final_prompt = FinalPromptTemplate(draft_input, option_tone, option_dialect)
            
            # Load the LLM model
            llm_model = LLMModel(model_provider=option_model_provider)
            llm = llm_model.load(api_key=api_key)

            # Invoke the LLM model
            improved_redaction = llm.invoke(final_prompt.generate())
            st.write(improved_redaction.content)


def main():
    # Create the streamlit UI
    st_ui = LLMStreamlitUI()
    st_ui.create()


if __name__ == "__main__":
    main()