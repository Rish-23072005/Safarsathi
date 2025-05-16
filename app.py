import streamlit as st
import os
from typing import TypedDict, Annotated, List
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
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
        "num_men": 1,
        "num_women": 1,
        "num_others": 0,
        "budget_men": 100.0,
        "budget_women": 100.0,
        "budget_others": 100.0,
        "total_budget": 0.0,
        "hotel_recommendations": "",
        "food_recommendations": ""
    }

# Get LLM with caching
@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )

# Define the itinerary prompt
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful travel assistant. Create a detailed itinerary for {city} based on:
    - User's interests: {interests}
    - Trip dates: {start_date} to {end_date}
    - Total members: {total_members}
    - Total budget: ${total_budget}
    
    Please provide:
    1. A day-by-day itinerary with timings
    2. Hotel recommendations within budget
    3. Food recommendations
    4. Transportation options
    5. Local tips and cultural considerations
    6. Estimated costs breakdown"""),
    ("human", "Create an itinerary for my trip.")
])

def create_itinerary(state: PlannerState) -> str:
    try:
        llm = get_llm()
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
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Trip Details")
        
        # City input
        city = st.text_input("Destination City", placeholder="e.g., Paris")
        
        # Interests input
        interests = st.text_input("Interests (comma-separated)", placeholder="e.g., museums, food, art")
        
        # Date inputs
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
        
        st.header("Group Details")
        # Number of travelers
        col1, col2, col3 = st.columns(3)
        with col1:
            num_men = st.number_input("Men", min_value=0, value=1)
        with col2:
            num_women = st.number_input("Women", min_value=0, value=1)
        with col3:
            num_others = st.number_input("Others", min_value=0, value=0)
        
        st.header("Budget Details")
        # Budget inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            budget_men = st.number_input("Budget per Man ($)", min_value=0.0, value=100.0)
        with col2:
            budget_women = st.number_input("Budget per Woman ($)", min_value=0.0, value=100.0)
        with col3:
            budget_others = st.number_input("Budget per Other ($)", min_value=0.0, value=100.0)
        
        # Calculate total budget
        total_budget = (num_men * budget_men) + (num_women * budget_women) + (num_others * budget_others)
        st.metric("Total Budget", f"${total_budget:.2f}")
        
        # Generate button
        if st.button("Generate Itinerary", type="primary"):
            if not city or not interests:
                st.error("Please fill in all required fields")
                return
            
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
                "total_budget": total_budget
            })
            
            # Generate itinerary
            with st.spinner("Generating your personalized itinerary..."):
                itinerary = create_itinerary(st.session_state.state)
                st.session_state.state["itinerary"] = itinerary
    
    # Main content area
    if st.session_state.state["itinerary"]:
        st.markdown("### Your Personalized Itinerary")
        st.markdown(st.session_state.state["itinerary"])
        
        # Download button
        st.download_button(
            label="Download Itinerary",
            data=st.session_state.state["itinerary"],
            file_name=f"itinerary_{st.session_state.state['city']}_{st.session_state.state['start_date']}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main() 