import os
import datetime
import json
import time
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Define database connection
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///chatbot.db")

Base = declarative_base()

class User(Base):
    """User model to store user information."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"

class Conversation(Base):
    """Conversation model to group related messages."""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"

class Message(Base):
    """Message model to store chat messages."""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(50))  # 'user' or 'assistant'
    content = Column(Text)
    image_data = Column(LargeBinary, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role})>"

class KnowledgeItem(Base):
    """Model for storing knowledge items with embeddings."""
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    embedding = Column(Text)  # JSON serialized embedding
    source = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<KnowledgeItem(id={self.id}, source={self.source})>"

def initialize_database():
    """Create database tables if they don't exist with retry logic."""
    try:
        # Create tables with retry logic
        execute_with_retry(create_tables)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")

def execute_with_retry(func, *args, **kwargs):
    """Execute a database function with retry logic."""
    max_retries = 5
    base_wait_time = 1.5  # seconds
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:  # Don't wait on the last attempt
                wait_time = base_wait_time * (2 ** attempt)  # Exponential backoff
                print(f"Database connection error: {str(e)}. Retrying in {wait_time:.2f} seconds (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

def create_tables():
    """Create database tables."""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)

def get_session():
    """Get a database session."""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

def get_or_create_user(username=None):
    """Get or create a user."""
    def _get_or_create_user():
        with get_session() as session:
            # Try to get an existing user
            user = session.query(User).first()
            
            # If no user exists, create one
            if not user:
                user = User(username=username or "default_user")
                session.add(user)
                session.commit()
                
            return user
    
    return execute_with_retry(_get_or_create_user)

def get_or_create_conversation(user_id, title=None):
    """Get or create a conversation for a user."""
    def _get_or_create_conversation():
        with get_session() as session:
            # Try to get the most recent conversation for this user
            conversation = session.query(Conversation).filter(
                Conversation.user_id == user_id
            ).order_by(Conversation.created_at.desc()).first()
            
            # If no conversation exists or a new title is provided, create one
            if not conversation or title:
                conversation = Conversation(
                    user_id=user_id,
                    title=title or "New Conversation"
                )
                session.add(conversation)
                session.commit()
                # Refresh to ensure all attributes are loaded
                session.refresh(conversation)
            
            # Create a dictionary with all needed attributes
            return {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "title": conversation.title,
                "created_at": conversation.created_at
            }
    
    # Get conversation data
    conversation_data = execute_with_retry(_get_or_create_conversation)
    
    # Create a new conversation object with this data
    # (This object isn't bound to a session but has all the data we need)
    conversation = Conversation(
        id=conversation_data["id"],
        user_id=conversation_data["user_id"],
        title=conversation_data["title"]
    )
    
    # Set created_at manually since it might have a default value in the constructor
    conversation.created_at = conversation_data["created_at"]
    
    return conversation
    
    return execute_with_retry(_get_or_create_conversation)

def add_message_to_db(conversation_id, role, content, image_data=None):
    """Add a message to the database."""
    def _add_message():
        with get_session() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                image_data=image_data
            )
            session.add(message)
            session.commit()
            return message
    
    return execute_with_retry(_add_message)

def get_conversation_messages(conversation_id, limit=100):
    """Get messages for a conversation."""
    def _get_messages():
        with get_session() as session:
            messages = session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.timestamp).limit(limit).all()
            
            # Convert to dictionaries
            message_list = []
            for msg in messages:
                message_dict = {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                }
                message_list.append(message_dict)
                
            return message_list
    
    return execute_with_retry(_get_messages)

def add_knowledge_item(content, embedding, source=None):
    """Add a knowledge item with embedding to the database."""
    def _add_knowledge_item():
        with get_session() as session:
            # Convert embedding to JSON string
            embedding_json = json.dumps(embedding)
            
            item = KnowledgeItem(
                content=content,
                embedding=embedding_json,
                source=source
            )
            session.add(item)
            session.commit()
            return item
    
    return execute_with_retry(_add_knowledge_item)

def get_all_knowledge_items():
    """Get all knowledge items with embeddings."""
    def _get_items():
        with get_session() as session:
            items = session.query(KnowledgeItem).all()
            
            # Convert to dictionaries with parsed embeddings
            item_list = []
            for item in items:
                embedding = json.loads(item.embedding)
                item_dict = {
                    "id": item.id,
                    "content": item.content,
                    "embedding": embedding,
                    "source": item.source
                }
                item_list.append(item_dict)
                
            return item_list
    
    return execute_with_retry(_get_items)
