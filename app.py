import streamlit as st
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import os 
import tempfile
from dotenv import load_dotenv
load_dotenv()
# os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")





st.set_page_config(page_title="ResumePal: AI-Powered Job Fit Analyzer")
st.title("ResumePal: AI-Powered Job Fit Analyzer")


uploaded_file = st.file_uploader("Please upload your resume (PDF)", type=['pdf'])

resume = ""
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load the resume content using PyPDFLoader
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        resume = "\n".join(doc.page_content for doc in documents)  # Extract text
    except Exception as e:
        st.error(f"Error loading resume: {e}")

job_description=st.text_area("Please enter the job decription: ")

prompt='''You are an expert in resume analysis and job matching. Your role is to evaluate the provided job description and resume for alignment and provide detailed feedback, including suggestions for improvement.  

    ### Input:  
    1. **Job Description:**  
    {job_description}  

    2. **Resume:**  
    {resume}  

    ### Output:  
    Provide the following:  
    1. **Percentage Match:**  
        - Match percentage based on skills, experience, and keywords.  

    2. **Relevant Skills:**  
        - List skills present in both the job description and resume.  

    3. **Missing Keywords or Skills:**  
        - Identify missing skills/keywords and rank their importance.  

    4. **Role Alignment:**  
        - How well does the applicant’s experience align with the job responsibilities?  

    5. **Experience Matching:**  
        - Compare the required years of experience to the applicant’s relevant experience.  

    6. **Education and Certification Analysis:**  
        - Does the resume meet educational requirements?  
        - Highlight missing certifications and suggest relevant ones.  

    7. **Soft Skills Analysis:**  
        - Compare soft skills in the job description and resume.  

    8. **Suggestions for Framing Projects and Achievements:**  
        - Feedback on how to highlight accomplishments better.  

    9. **Industry Alignment:**  
        - Assess whether the applicant’s experience aligns with the industry or domain.  

    10. **Writing Style Feedback:**  
        - Evaluate the professionalism and relevance of the resume’s tone and structure.  

    11. **Custom Recommendations:**  
        - Provide specific next steps for improvement.  
    '''

api_key=st.secrets["GOOGLE_API_KEY"]

def get_response(resume,job_description):
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=api_key)
    llm=ChatGoogleGenerativeAI(model='gemini-pro')

    chat_prompt=PromptTemplate(input_variables=["job_description","resume"],template=prompt)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run({"job_description": job_description, "resume": resume})
    return response


if st.button("ATS Search") and uploaded_file:
    response=get_response(resume=resume,job_description=job_description)
    st.write(response)
