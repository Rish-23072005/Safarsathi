import streamlit as st
import os
from typing import TypedDict, Annotated, List
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment 
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="SafarSathi - AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Define PlannerState
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str
    start_date: str
    end_date: str
    num_men: int
    num_women: int
    num_others: int
    budget_men: float
    budget_women: float
    budget_others: float
    total_budget: float
    hotel_recommendations: str
    food_recommendations: str

# Initialize session state
if 'state' not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "city": "",
        "interests": [],
        "itinerary": "",
        "start_date": "",
        "end_date": "",
        "num_men": 0,
        "num_women": 0,
        "num_others": 0,
        "budget_men": 0.0,
        "budget_women": 0.0,
        "budget_others": 0.0,
        "total_budget": 0.0,
        "hotel_recommendations": "",
        "food_recommendations": ""
    }

@st.cache_resource
def get_llm():
    try:
        return ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )
    except Exception as e:
        st.error(f"Error initializing ChatGroq: {str(e)}")
        return None

# Define the itinerary prompt
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a detailed travel itinerary for {city} based on the user's interests: {interests}, dates: {start_date} to {end_date}, total members: {total_members}, and budget: {total_budget}. Include hotel recommendations and food suggestions."),
    ("human", "Create an itinerary for my trip."),
])

def create_itinerary(state: PlannerState) -> str:
    try:
        llm = get_llm()
        if llm is None:
            return "Failed to initialize the language model. Please check your API key and try again."
            
        response = llm.invoke(itinerary_prompt.format_messages(
            city=state["city"],
            interests=", ".join(state["interests"]),
            start_date=state["start_date"],
            end_date=state["end_date"],
            total_members=state["num_men"] + state["num_women"] + state["num_others"],
            total_budget=state["total_budget"]
        ))
        return response.content
    except Exception as e:
        return f"An error occurred while generating the itinerary: {str(e)}"

def main():
    st.title("✈️ SafarSathi - AI Travel Planner")
    
    # Sidebar for input
    with st.sidebar:
        st.header("Trip Details")
        city = st.text_input("Destination City")
        interests = st.text_input("Interests (comma-separated)")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        
        st.header("Group Details")
        num_men = st.number_input("Number of Men", min_value=0, value=1)
        num_women = st.number_input("Number of Women", min_value=0, value=1)
        num_others = st.number_input("Number of Others", min_value=0, value=0)
        
        st.header("Budget Details")
        budget_men = st.number_input("Budget per Man (USD)", min_value=0.0, value=100.0)
        budget_women = st.number_input("Budget per Woman (USD)", min_value=0.0, value=100.0)
        budget_others = st.number_input("Budget per Other (USD)", min_value=0.0, value=100.0)
        
        if st.button("Generate Itinerary"):
            with st.spinner("Creating your personalized itinerary..."):
                # Update state
                st.session_state.state.update({
                    "city": city,
                    "interests": [i.strip() for i in interests.split(",")],
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "num_men": num_men,
                    "num_women": num_women,
                    "num_others": num_others,
                    "budget_men": budget_men,
                    "budget_women": budget_women,
                    "budget_others": budget_others,
                    "total_budget": (num_men * budget_men) + (num_women * budget_women) + (num_others * budget_others)
                })
                
                # Generate itinerary
                itinerary = create_itinerary(st.session_state.state)
                st.session_state.state["itinerary"] = itinerary
    
    # Main content area
    if st.session_state.state["itinerary"]:
        st.markdown("### Your Personalized Itinerary")
        st.write(st.session_state.state["itinerary"])
        
        # Download button
        st.download_button(
            label="Download Itinerary",
            data=st.session_state.state["itinerary"],
            file_name="travel_itinerary.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main() 
