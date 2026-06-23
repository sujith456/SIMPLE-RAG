import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from core.inference import get_hf_embedding, get_litellm, get_image_description
import os
import uuid
import json
from datetime import datetime
from sqlalchemy import (create_engine, Column, Integer, DateTime, func, String, Text, text)
from sqlalchemy.orm import (sessionmaker, declarative_base)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

DATABASE_URL = "sqlite:///langchain_chat_history_new.db"
new_llm = get_litellm('medium')

# Global dictionary to transfer retrieved documents from the chain step to the listener callback safely across threads
global_retrieved_docs = {}

@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name="data_collection",
        embedding_function=get_hf_embedding(),
        persist_directory="./chroma_langchain_db",
    )

vector_store = get_vector_store()
base = declarative_base()

class ChatMessage(base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    references_data = Column(Text, nullable=True) # JSON string of retrieved references

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
base.metadata.create_all(engine)

# Migration: Check if references_data column exists, and if not, add it
try:
    with engine.begin() as conn:
        result = conn.execute(text("PRAGMA table_info(chat_messages)"))
        columns = [row[1] for row in result.fetchall()]
        if "references_data" not in columns:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN references_data TEXT"))
            print("Successfully added column 'references_data' to table 'chat_messages'")
except Exception as e:
    print(f"Error checking/adding column: {e}")


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
                elif row.role == 'ai':
                    result.append(AIMessage(content=row.content))
                elif row.role == 'system':
                    result.append(SystemMessage(content=row.content))
            return result
        except Exception as e:
            print(f"Exception occured while fetching the messsages {e}")
            return []
        finally:
            db.close()

    async def aadd_messages(self, messages: list[BaseMessage]) -> None:
        pass 

    def add_messages(self, messages: list[BaseMessage]):
        db = SessionLocal()
        try:
            # Retrieve temporary references from the global dictionary or fallback to session state
            temp_refs = global_retrieved_docs.pop(self.session_id, [])
            if not temp_refs:
                try:
                    temp_refs = st.session_state.get("temp_retrieved_docs", [])
                except Exception:
                    temp_refs = []

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
                msg_refs = references_json if message.type == "ai" else None
                row = ChatMessage(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    role=message.type,
                    content=message.content,
                    references_data=msg_refs
                )
                db.add(row)
            db.commit()
        except Exception as e:
            print(f"Exception occured while adding the messsages {e}")
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
            print(f"Exception occured while deleting the messsages {e}")
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
        print(f"Exception occured while fetching history {e}")
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
        print(f"Exception occured while fetching messages {e}")
        return []
    finally:
        db.close()


def manual_retriver(query: dict) -> list:
    session_id = query.get("session_id")
    results = vector_store.similarity_search_with_relevance_scores(query["input"], k=3)
    if session_id:
        global_retrieved_docs[session_id] = results
    try:
        st.session_state["temp_retrieved_docs"] = results
    except Exception:
        pass
    return results


def format_retrival_score__docs(docs):
    return "\n\n".join(f"<Doc_Content>Doc content: {doc[0].page_content} Doc Score: {doc[1]} </Doc_Content>" for doc in docs)


def get_session_history(inputs: dict) -> dict:
    messages = CustomeSqlHistory(session_id=inputs['session_id'], user_id=inputs['user_id']).messages
    return {**inputs, "history": messages}


conversion_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpfull assistant answer the user question based on the context {context}"),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

reasoning_llm = get_litellm("reasoning")
retrival_chain = (
    RunnableLambda(get_session_history) 
    | {
        "context": RunnablePassthrough() | manual_retriver | format_retrival_score__docs,
        "input": itemgetter("input"),
        "history": itemgetter("history"),
    }
    | conversion_prompt 
    | reasoning_llm
)


def on_end_function(run):
    try:
        inputs = run.inputs
        outputs = run.outputs
        
        # Write initial callback entry to log
        with open("debug_callback.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- Callback triggered at {datetime.now()} ---\n")
            f.write(f"inputs: {inputs}\n")
            f.write(f"outputs: {outputs}\n")
            
        content = ""
        if outputs:
            if isinstance(outputs, dict):
                out_val = outputs.get("output")
                if out_val:
                    if hasattr(out_val, "content"):
                        content = out_val.content
                    else:
                        content = str(out_val)
                else:
                    content = str(outputs)
            elif hasattr(outputs, "content"):
                content = outputs.content
            else:
                content = str(outputs)

        history_imp = CustomeSqlHistory(
            session_id=inputs["session_id"],
            user_id=inputs.get("user_id", "default_user")
        )
        history_imp.add_messages([
            HumanMessage(content=inputs["input"]),
            AIMessage(content=content)
        ])
        
        with open("debug_callback.log", "a", encoding="utf-8") as f:
            f.write("Successfully added messages to database.\n")
            
    except Exception as e:
        import traceback
        with open("debug_callback.log", "a", encoding="utf-8") as f:
            f.write(f"Error in on_end_function callback: {e}\n")
            f.write(traceback.format_exc())



listener_chain = retrival_chain.with_listeners(on_end=on_end_function)

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
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
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
    st.title("📚 RAG Chat Dashboard")
    st.markdown("---")
    
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state.pop("temp_retrieved_docs", None)
        trigger_rerun()
        
    st.markdown("### 📜 Chat History")
    history_sessions = get_history()
    print(history_sessions)
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
st.markdown('<div class="header-title">💬 Chat with Documents</div>', unsafe_allow_html=True)
st.markdown("Have a back-and-forth conversation with your documents. The assistant remembers context and retrieves relevant details from your embedded PDFs.")
st.caption(f"Active Session: `{st.session_state['session_id']}`")
st.markdown("---")

# Render existing messages for current session
messages = get_messages(st.session_state["session_id"])
for msg in messages:
    role = "user" if msg.role == "human" else "assistant" if msg.role == "ai" else msg.role
    with st.chat_message(role):
        st.write(msg.content)
        # Show references inside container if available
        if msg.role == 'ai' and msg.references_data:
            try:
                refs = json.loads(msg.references_data)
                if refs:
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
                                elem_type = meta.get('element_type', 'paragraph')
                                
                                st.markdown(f"**Reference {i+1}: {file_name} (Page {page_num})** — *Score: {score:.3f}*")
                                st.write(content)
                                
                                # Visual crop image check
                                if meta.get('has_image') and (meta.get('image_url') or meta.get('visual_crop_path')):
                                    img_path = meta.get('image_url') or meta.get('visual_crop_path')
                                    if os.path.exists(img_path):
                                        st.image(img_path, caption=f"Extracted graphic from Page {page_num}", use_container_width=True)
            except Exception as e:
                print(f"Error parsing references: {e}")

# Accept user chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display human message
    with st.chat_message("human"):
        st.write(prompt)
        
    # Clear any temporary retrieved docs in state
    st.session_state.pop("temp_retrieved_docs", None)
    
    # Run retrieval and generation
    with st.chat_message("ai"):
        with st.spinner("Searching documents and generating response..."):
            try:
                inputs = {
                    "input": prompt,
                    "session_id": st.session_state["session_id"],
                    "user_id": "default_user"
                }
                # Invoke the listener_chain (this automatically saves prompt & response to DB via on_end_function)
                response = listener_chain.invoke(inputs)
                
                # Rerun to update chat logs and history list
                trigger_rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")