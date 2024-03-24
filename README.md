# Chat-with-PDF 🗎
### Prerequisites
Before you begin, ensure you have Python 3.10 or higher installed on your system.

### Setup and Run Locally
1. Clone the Repository:
```
https://github.com/PiSpace/Chat-with-PDF.git
```
2. Install Dependencies:
```
pip install -r requirements.txt
```
3. Environment Configuration:

Secure your Google API key for use in the application.
- Create a .env file in the root directory.
- Add your Google API key: GOOGLE_API_KEY=your_api_key_here.
4. Launch the Application:
Start the chatbot using Streamlit.
```
streamlit run app.py
```
5. Useful links
   - To get gemini pro API: https://ai.google.dev
   - To deploy your app: https://streamlit.io/
   - https://www.langchain.com/
   - For embedding and similarity search: https://ai.meta.com/tools/faiss/

6. Features
- PDF Upload: Simplify document handling by uploading multiple PDF files directly into the chatbot.
- Text Extraction: Utilize advanced algorithms to extract text accurately for analysis and interaction.
- Conversational AI: Engage with the Gemini AI model for insightful, context-aware answers drawn directly from your PDF content.
