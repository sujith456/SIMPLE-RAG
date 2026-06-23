import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from core.inference import get_hf_embedding, get_litellm
import os
import uuid
import json
import requests
from datetime import datetime
from sqlalchemy import (create_engine, Column, Integer, DateTime, func, String, Text, text)
from sqlalchemy.orm import (sessionmaker, declarative_base)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq import ChatGroq

DATABASE_URL = "sqlite:///langchain_chat_history_new.db"

# Initialize vector store
@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name="data_collection",
        embedding_function=get_hf_embedding(),
        persist_directory="./chroma_langchain_db",
    )

vector_store = get_vector_store()
base = declarative_base()

# ChatMessage schema matching conversation_rag.py
class ChatMessage(base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    references_data = Column(Text, nullable=True) # JSON string of retrieved references or tool metadata

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
base.metadata.create_all(engine)


class CustomeSqlHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, user_id: str = "default_user"):
        self.session_id = session_id
        self.user_id = user_id
        
    @property
    def messages(self):
        db = SessionLocal()
        try:
            rows = (db.query(ChatMessage)
                      .filter(ChatMessage.session_id == self.session_id)
                      .order_by(ChatMessage.id)
                      .all())
            result = []
            for row in rows:
                if row.role == 'human':
                    result.append(HumanMessage(content=row.content))
                elif row.role == 'system':
                    result.append(SystemMessage(content=row.content))
                elif row.role == 'ai':
                    # Parse tool calls if they exist in references_data
                    tool_calls = []
                    if row.references_data:
                        try:
                            meta = json.loads(row.references_data)
                            if isinstance(meta, dict) and "tool_calls" in meta:
                                tool_calls = meta["tool_calls"]
                        except Exception:
                            pass
                    result.append(AIMessage(content=row.content, tool_calls=tool_calls))
                elif row.role == 'tool':
                    tool_call_id = ""
                    name = ""
                    if row.references_data:
                        try:
                            meta = json.loads(row.references_data)
                            if isinstance(meta, dict):
                                tool_call_id = meta.get("tool_call_id", "")
                                name = meta.get("name", "")
                        except Exception:
                            pass
                    result.append(ToolMessage(content=row.content, tool_call_id=tool_call_id, name=name))
            return result
        except Exception as e:
            print(f"Exception occurred while fetching the messages: {e}")
            return []
        finally:
            db.close()

    async def aadd_messages(self, messages: list[BaseMessage]) -> None:
        pass 

    def add_messages(self, messages: list[BaseMessage]):
        db = SessionLocal()
        try:
            # Retrieve temporary references from session state if any
            temp_refs = []
            try:
                temp_refs = st.session_state.get("temp_retrieved_docs", [])
            except Exception:
                pass

            references_json = None
            if temp_refs:
                refs_list = []
                for doc, score in temp_refs:
                    refs_list.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
                references_json = json.dumps(refs_list)

            for message in messages:
                ref_json = None
                role = message.type  # 'human', 'ai', 'system', 'tool'
                
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        ref_json = json.dumps({"tool_calls": message.tool_calls})
                    else:
                        # Final response message: store retrieval references
                        ref_json = references_json
                elif isinstance(message, ToolMessage):
                    role = 'tool'
                    tool_args = getattr(message, 'tool_args', {})
                    ref_json = json.dumps({
                        "tool_call_id": message.tool_call_id,
                        "name": message.name,
                        "args": tool_args
                    })
                
                row = ChatMessage(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    role=role,
                    content=message.content,
                    references_data=ref_json
                )
                db.add(row)
            db.commit()
            
            # Clear stored refs
            try:
                st.session_state.pop("temp_retrieved_docs", None)
            except Exception:
                pass
        except Exception as e:
            print(f"Exception occurred while adding the messages: {e}")
        finally:
            db.close()

    def clear(self):
        db = SessionLocal()
        try:
            (db.query(ChatMessage)
               .filter(ChatMessage.session_id == self.session_id)
               .delete())
            db.commit()
        except Exception as e:
            print(f"Exception occurred while deleting the messages: {e}")
        finally:
            db.close()


def get_history():
    db = SessionLocal()
    try:
        rows = db.execute(text("""
        SELECT
            m.session_id,
            m.content,
            m.created_at
        FROM chat_messages m
        WHERE m.role='human'
        AND m.id = (
            SELECT MIN(id)
            FROM chat_messages
            WHERE session_id=m.session_id
            AND role='human'
        )
        ORDER BY m.created_at DESC
        """))
        return rows.fetchall()
    except Exception as e:
        print(f"Exception occurred while fetching history: {e}")
        return []
    finally:
        db.close()


def get_messages(session_id: str):
    db = SessionLocal()
    try:
        rows = (db.query(ChatMessage)
                  .filter(ChatMessage.session_id == session_id)
                  .order_by(ChatMessage.id.asc())
                  .all())
        return rows
    except Exception as e:
        print(f"Exception occurred while fetching messages: {e}")
        return []
    finally:
        db.close()


# -----------------------------
# Tool Definitions
# -----------------------------

@tool
def search_location(query: str) -> tuple:
    """
    Params: query:str
    query: details related to city or city name 
    Returns the tuple of lat,lon and other details
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': query,
        'format': 'jsonv2',
        'addressdetails': 1, 
        'limit': 3 
    }
    headers = {
        'User-Agent': 'MyWeatherApp/1.0 (contact@myemail.com)'
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data:
            top_result = data[0]
            return (
                float(top_result.get('lat')), 
                float(top_result.get('lon')), 
                top_result.get('address', {}).get('country'), 
                top_result.get('address', {}).get('state')
            )
    except Exception as e:
        print(f"An error occurred in search_location: {e}")
        return None


@tool
def get_weather_forecast(latitude: float, longitude: float, forecast_days: int = 3):
    """
    Retrieves the 3-day weather forecast for a given location using its coordinates.
    
    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        forecast_days (int, optional): Number of forecast days to retrieve. Defaults to 3.
        
    Returns:
        dict: A dictionary containing current, hourly, and daily forecasts.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,rain",
        "daily": "rain_sum,temperature_2m_max,temperature_2m_min",
        "current": "rain,temperature_2m,is_day",
        "timezone": "auto",
        "forecast_days": forecast_days
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return json.dumps({
            "current": {
                "time": data.get("current", {}).get("time"),
                "temperature_2m": data.get("current", {}).get("temperature_2m"),
                "rain": data.get("current", {}).get("rain"),
                "is_day": data.get("current", {}).get("is_day")
            },
            "hourly": {
                "time": data.get("hourly", {}).get("time"),
                "temperature_2m": data.get("hourly", {}).get("temperature_2m"),
                "rain": data.get("hourly", {}).get("rain")
            },
            "daily": {
                "time": data.get("daily", {}).get("time"),
                "rain_sum": data.get("daily", {}).get("rain_sum"),
                "temperature_2m_max": data.get("daily", {}).get("temperature_2m_max"),
                "temperature_2m_min": data.get("daily", {}).get("temperature_2m_min")
            }
        })
    except Exception as e:
        print(f"Error fetching forecast data: {e}")
        return None


@tool
def get_historical_weather(latitude: float, longitude: float, start_date: str, end_date: str):
    """
    Retrieves historical hourly weather data (temperature and rain) between two dates.
    
    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        start_date (str): The start date in 'YYYY-MM-DD' format (e.g. '2020-06-01').
        end_date (str): The end date in 'YYYY-MM-DD' format (e.g. '2020-06-05').
        
    Returns:
        dict: A dictionary containing daily historical weather summaries.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "rain_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return json.dumps({
            "daily": {
                "time": data.get("daily", {}).get("time"),
                "rain_sum": data.get("daily", {}).get("rain_sum"),
                "temperature_2m_max": data.get("daily", {}).get("temperature_2m_max"),
                "temperature_2m_min": data.get("daily", {}).get("temperature_2m_min")
            }
        })
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


def manual_retriver(query) -> list:
    if isinstance(query, dict):
        search_query = query.get("input") or query.get("query") or list(query.values())[0]
    else:
        search_query = query
        
    results = vector_store.similarity_search_with_relevance_scores(search_query, k=3)
    try:
        st.session_state["temp_retrieved_docs"] = results
    except Exception:
        pass
    return results


def format_retrival_score__docs(docs):
    return "\n\n".join(f"<Doc_Content>Doc content: {doc[0].page_content} Doc Score: {doc[1]} </Doc_Content>" for doc in docs)


class RetrievalInput(BaseModel):
    query: str = Field(description="The search query about Guidewire, insurance, or the AI ecosystem in India")



new_retrival_chain = RunnablePassthrough()|manual_retriver|format_retrival_score__docs

retrival_tool=new_retrival_chain.as_tool(
    name="Retrival_tool",description="information regrading Guidewire and insurance related question and also have info regrading the AI ecosystem in india", args_schema=RetrievalInput
)

# Initialize Agent Chain using ChatGroq for reliable tool calling
try:
    llm = ChatGroq(model='openai/gpt-oss-120b')
except Exception as e:
    print(f"Fallback to LiteLLM router due to: {e}")
    llm = get_litellm("reasoning")

models_with_tool = llm.bind_tools([search_location, get_historical_weather, get_weather_forecast, retrival_tool], parallel_tool_calls=False)

tool_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant with tools. based on the provided message history and tool results, answer the question by the user. Do not mention the tools used, only answer the question based on the facts provided. Do not assume anything. IMPORTANT: To answer weather/rain queries, you must first call search_location to obtain the latitude and longitude coordinates. Once the coordinates are returned from search_location, use them to call get_weather_forecast or get_historical_weather. Do not attempt to nest tool calls. Never pass function/tool calls as argument values."),
    ("placeholder", "{history}"),
    ("human", "{input}")
])


# Safe rerun helper function
def trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# Initialize session state for current chat session
if "session_id" not in st.session_state:
    try:
        history_sessions = get_history()
        if history_sessions:
            st.session_state["session_id"] = history_sessions[0][0]
        else:
            st.session_state["session_id"] = str(uuid.uuid4())
    except Exception:
        st.session_state["session_id"] = str(uuid.uuid4())


# -----------------------------
# UI Custom Styles
# -----------------------------
st.markdown("""
<style>
.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7f00ff 0%, #e100ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.stChatMessage {
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
}
div[data-testid="stSidebar"] {
    background-color: #1e222b;
}
div.stButton > button {
    border-radius: 8px;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR / HISTORY
# -----------------------------
with st.sidebar:
    st.title("🤖 Agentic RAG Dashboard")
    st.markdown("---")
    
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state.pop("temp_retrieved_docs", None)
        trigger_rerun()
        
    st.markdown("### 📜 Chat History")
    history_sessions = get_history()
    if history_sessions:
        for row in history_sessions:
            sess_id = row[0]
            first_msg = row[1]
            created_at = row[2]
            
            dt_str = ""
            if created_at:
                if isinstance(created_at, str):
                    try:
                        dt = datetime.fromisoformat(created_at)
                        dt_str = dt.strftime("%b %d, %H:%M")
                    except:
                        dt_str = str(created_at)[:16]
                elif hasattr(created_at, "strftime"):
                    dt_str = created_at.strftime("%b %d, %H:%M")
                    
            label = first_msg if len(first_msg) <= 30 else f"{first_msg[:27]}..."
            
            is_active = (sess_id == st.session_state["session_id"])
            btn_prefix = "👉 " if is_active else "💬 "
            
            if st.button(f"{btn_prefix}{label}\n({dt_str})", key=f"sess_{sess_id}", use_container_width=True):
                st.session_state["session_id"] = sess_id
                st.session_state.pop("temp_retrieved_docs", None)
                trigger_rerun()
    else:
        st.info("No past chats yet. Start asking questions!")

# -----------------------------
# MAIN CHAT AREA
# -----------------------------
st.markdown('<div class="header-title">🤖 Agentic Chat with Tools</div>', unsafe_allow_html=True)
st.markdown("Interact with a fully autonomous Agent. The assistant can search locations, fetch forecasts, query historical weather, and retrieve context from your PDFs.")
st.caption(f"Active Session: `{st.session_state['session_id']}`")
st.markdown("---")

# Render existing messages for current session
messages = get_messages(st.session_state["session_id"])
for msg in messages:
    role = "user" if msg.role == "human" else "assistant" if msg.role == "ai" else msg.role
    
    if role == "user":
        with st.chat_message("user"):
            st.write(msg.content)
            
    elif role == "assistant":
        if msg.content:
            with st.chat_message("assistant"):
                st.write(msg.content)
                # Show references inside container if available
                if msg.references_data:
                    try:
                        refs = json.loads(msg.references_data)
                        if isinstance(refs, list) and refs:
                            with st.container():
                                with st.expander("📑 View References", expanded=False):
                                    seen_contents = set()
                                    for i, ref in enumerate(refs):
                                        content = ref.get("page_content", "")
                                        if content in seen_contents:
                                            continue
                                        seen_contents.add(content)
                                        meta = ref.get("metadata", {})
                                        score = ref.get("score", 0.0)
                                        
                                        file_name = meta.get('file_name') or meta.get('source', 'Unknown Document')
                                        page_num = meta.get('page_number', 'N/A')
                                        
                                        st.markdown(f"**Reference {i+1}: {file_name} (Page {page_num})** — *Score: {score:.3f}*")
                                        st.write(content)
                                        
                                        if meta.get('has_image') and (meta.get('image_url') or meta.get('visual_crop_path')):
                                            img_path = meta.get('image_url') or meta.get('visual_crop_path')
                                            if os.path.exists(img_path):
                                                st.image(img_path, caption=f"Extracted graphic from Page {page_num}", use_container_width=True)
                    except Exception as e:
                        pass
                        
    elif role == "tool":
        tool_name = "Unknown Tool"
        tool_args = {}
        if msg.references_data:
            try:
                meta = json.loads(msg.references_data)
                if isinstance(meta, dict):
                    tool_name = meta.get("name", "Tool")
                    tool_args = meta.get("args", {})
            except Exception:
                pass
        
        with st.status(f"Used tool: `{tool_name}`", state="complete", expanded=False):
            if tool_args:
                st.write(f"**Arguments:** `{tool_args}`")
            st.write("**Result:**")
            st.write(msg.content)

# Accept user chat input
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.write(prompt)
        
    st.session_state.pop("temp_retrieved_docs", None)
    
    with st.chat_message("assistant"):
        history_imp = CustomeSqlHistory(session_id=st.session_state["session_id"], user_id="default_user")
        history = history_imp.messages
        
        # Initial message list including conversation history
        messages_input = tool_prompt.format_messages(history=history, input=prompt)
        
        new_messages = []
        tools_by_name = {
            "search_location": search_location,
            "get_weather_forecast": get_weather_forecast,
            "get_historical_weather": get_historical_weather,
            "Retrival_tool": retrival_tool
        }
        
        # Loop to run the agent step-by-step
        max_steps = 10
        step = 0
        while step < max_steps:
            step += 1
            with st.spinner("Thinking..."):
                try:
                    ai_message = models_with_tool.invoke(messages_input)
                except Exception as e:
                    st.error(f"Error invoking agent model: {e}")
                    break
            
            messages_input.append(ai_message)
            new_messages.append(ai_message)
            
            # If no tool call, agent has produced the final answer
            if not ai_message.tool_calls:
                if ai_message.content:
                    st.write(ai_message.content)
                break
                
            # Process tool calls
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args')
                tool_id = tool_call.get('id')
                
                # Show tool execution inside container like ChatGPT
                status_text = f"Using tool: `{tool_name}`"
                with st.status(status_text, expanded=False) as status:
                    st.write(f"**Arguments:** `{tool_args}`")
                    try:
                        tool_to_execute = tools_by_name.get(tool_name)
                        if not tool_to_execute:
                            raise ValueError(f"Tool `{tool_name}` is not registered.")
                            
                        tool_result = tool_to_execute.invoke(tool_args)
                        st.write("**Result:**")
                        st.write(tool_result)
                        status.update(label=f"Used tool: `{tool_name}`", state="complete")
                        
                        tool_msg = ToolMessage(content=str(tool_result), tool_call_id=tool_id, name=tool_name)
                        tool_msg.tool_args = tool_args  # Attach tool args for saving
                        messages_input.append(tool_msg)
                        new_messages.append(tool_msg)
                    except Exception as e:
                        st.write(f"**Error executing tool:** {e}")
                        status.update(label=f"Failed tool: `{tool_name}`", state="error")
                        
                        tool_msg = ToolMessage(content=f"Error executing tool: {e}", tool_call_id=tool_id, name=tool_name)
                        tool_msg.tool_args = tool_args
                        messages_input.append(tool_msg)
                        new_messages.append(tool_msg)
        
        # Save user message and the generated response loop messages to database
        user_msg = HumanMessage(content=prompt)
        history_imp.add_messages([user_msg] + new_messages)
        
        # # Rerun to update chat state and list
        # trigger_rerun()
