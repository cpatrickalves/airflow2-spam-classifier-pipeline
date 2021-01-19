import streamlit as st
from NaiveBayesSolver import NaiveBayesSolver


hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """

spam_model = NaiveBayesSolver()
model_path = "models/testmodel"

def main():

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Spam Classifier")
    st.markdown("This app uses a Machine Learning model to classify **emails written in english** as spams or not.")
    st.markdown("---")

    text = st.text_area("Add the text of your e-mail here:")

    if st.button("Verify"):
        if not text:
            st.warning("Please type or paste the e-mail body in the text field and try again.")
        else:
            try:
                result = spam_model.predict_from_text(text, model_path)

                if "notspam" in result:
                    #st.success(f"There is a probability of {result['notspam']}% that this is not a spam")
                    st.success("This is not a spam")
                else:
                    #st.error(f"There is a probability of {result['spam']}% that this is a spam")
                    st.error(f"This is a spam")

            except Exception as e:
                    st.write("Error while scraping data.")
                    st.write(e)

    st.markdown("### About this project")
    st.info("""\
            * Create by: [Patrick Alves](https://cpatrickalves.github.io/)
            * source code: [GitHub](https://github.com/cpatrickalves/airflow2-spam-classifier-pipeline)
        """)


if __name__ == "__main__":
    main()