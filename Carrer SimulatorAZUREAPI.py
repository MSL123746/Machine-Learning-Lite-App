import io
import openai
import streamlit as st
from datetime import datetime, timedelta
import datetime as dt
import re

# --- FIELD AND FOCUS AREA DEFINITIONS ---
fields = [
    "Business Analyst",
    "Marketing Specialist",
    "Human Resources Coordinator",
    "Financial Analyst",
    "Software Engineer",
    "AI Engineer",
    "Machine Learning Engineer",
    "Data Analyst",
    "Cybersecurity Specialist",
    "IT Support Specialist",
    "Project Management",
    "IT Director",
    "Executive Team Lead",
    "HR Manager",
    "Data Scientist", 
    "Product Manager" 
       ]


# --- Progressive Step-by-Step Sidebar ---

def generate_roadmap(missing_skills, fast_track=False):
    pass

fields_with_blank = ["-- Select Field --"] + fields + ["Other"]

current_field = st.sidebar.selectbox("Step 1: Select Your Target Role", fields_with_blank, key="step1_field")

custom_field = ""
effective_field = ""
if current_field == "Other":
    st.markdown(
        """
        <style>
        .custom-role-enabled input {
            background-color: #e5e7eb !important; /* gray-200 */
            color: #111827 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    custom_field = st.sidebar.text_input(
        label='Enter your custom target role when "Other" is selected',
        key="custom_target_role",
        placeholder="Press Enter to Continue",
        disabled=False
    )
    step1_done = bool(custom_field.strip())
    effective_field = custom_field
else:
    st.markdown(
        """
        <style>
        .custom-role-disabled input {
            background-color: #f3f4f6 !important; /* gray-100 */
            color: #6b7280 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.text_input(
        label="Enter your custom target role",
        key="custom_target_role",
        placeholder="Select 'Other' to enable",
        disabled=True
    )
    step1_done = current_field.strip() and current_field != "-- Select Field --" and current_field != "Other"
    effective_field = current_field

 # Removed sidebar debug info
print(f"DEBUG: current_field = {current_field}, step1_done = {step1_done}")

resume_text = ""
step2_done = False
if step1_done:
    st.sidebar.markdown("**Step 2: Upload Resume (PDF, DOCX, or TXT)**")
    uploaded_resume = st.sidebar.file_uploader(
        "Upload your resume (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        key="step2_resume"
    )
    if uploaded_resume is not None:
        if uploaded_resume.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_resume)
                resume_text = " ".join(page.extract_text() or "" for page in pdf_reader.pages)
            except Exception as e:
                st.sidebar.error(f"Could not read PDF: {e}")
        elif uploaded_resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                import docx
                doc = docx.Document(uploaded_resume)
                resume_text = " ".join([para.text for para in doc.paragraphs])
            except Exception as e:
                    # Silently remove 'Certainly!' without showing a message
                resume_text = uploaded_resume.read().decode("utf-8", errors="ignore")
        step2_done = True


# Step 3 removed; always set step3_done = True after step2
skillset = ""
step3_done = False
if step1_done and step2_done:
    step3_done = True

step4_done = False
if step1_done and step2_done and step3_done:
    step4_done = True

career_path = None
pace = 5


# Roadmap generation is only triggered by button
if step1_done and step2_done and step3_done and step4_done:
    generate = st.button("Generate My Roadmap", use_container_width=True, key="generate_roadmap_btn")
    if generate:
        st.warning("AI roadmap generation is currently disabled. All OpenAI API code has been removed as requested.")
    # (AI roadmap generation is disabled. No further processing here.)

def generate_roadmap(missing_skills, pace, fast_track=False):
    # Simulate a roadmap as a list of milestones/sprints
    sprints = []
    start_date = datetime.today()
    sprint_length = 7 if fast_track else 14  # days per sprint
    for i, skill in enumerate(missing_skills):
        sprint = {
            "Task": f"Learn {skill.title()}",
            "Start": (start_date + timedelta(days=i * sprint_length)).strftime("%Y-%m-%d"),
            "Finish": (start_date + timedelta(days=(i + 1) * sprint_length)).strftime("%Y-%m-%d"),
            "Resource": "Learning Sprint"
        }
        sprints.append(sprint)
    # Add a final milestone
    if sprints:
        sprints.append({
            "Task": "Capstone/Project/Certification",
            "Start": sprints[-1]["Finish"],
            "Finish": (datetime.strptime(sprints[-1]["Finish"], "%Y-%m-%d") + timedelta(days=sprint_length)).strftime("%Y-%m-%d"),
            "Resource": "Milestone"
        })
    return sprints

generate = False
# --- OUTPUT SECTION ---

if generate:
    pass

# --- STYLING ---
st.markdown(
    """
    <style>
    .stApp {background-color: #f8fafc;}
    section[data-testid='stSidebar'] {
        background-color: #f1f5f9;
        min-width: 400px !important;
        max-width: 450px !important;
        border: 2px solid #2563eb !important; /* blue-600 */
        border-radius: 12px !important;
        box-shadow: 0 0 0 2px #2563eb33 !important;
    }
    input, textarea, .stTextInput>div>div>input, .stTextArea>div>textarea, .stSelectbox>div>div {
        border: 2px solid #2563eb !important;
        border-radius: 8px !important;
        box-shadow: 0 0 0 1.5px #2563eb33 !important;
    }
    .stButton>button {font-weight: 600; border: 2px solid #2563eb !important;}
    .metric-square, .stMetric {background: #fff !important; border: 2px solid #2563eb !important; border-radius: 8px !important;}

    /* SMART table styling */
    #smart-table table {
        font-family: 'Times New Roman', Times, serif;
        font-size: 12pt;
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 2em;
    }
    #smart-table th, #smart-table td {
        border: 1px solid #444;
        padding: 8px 12px;
        text-align: left;
    }
    #smart-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    #smart-table th {
        background-color: #e0e7ef;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
