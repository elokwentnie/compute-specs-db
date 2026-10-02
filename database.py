"""
Database models and configuration for Compute Specs DB

Uses SQLAlchemy ORM with SQLite database for storing compute specification data.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./cpu_database.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CPUSpec(Base):
    """Compute specification database model"""
    __tablename__ = "cpu_specs"

    id = Column(Integer, primary_key=True, index=True)
    cpu_model_name = Column(String, index=True)
    family = Column(String)
    cpu_model = Column(String)
    codename = Column(String, index=True)
    cores = Column(Integer)
    threads = Column(Integer)
    tdp_watts = Column(Integer)
    launch_year = Column(Integer)
    max_turbo_frequency_ghz = Column(Float)
    l3_cache_mb = Column(Float)
    max_memory_tb = Column(Float)
    validated = Column(Boolean, nullable=False, default=False, index=True)


class GPUSpec(Base):
    """GPU specification database model"""
    __tablename__ = "gpu_specs"

    id = Column(Integer, primary_key=True, index=True)
    gpu_model_name = Column(String, index=True)
    vendor = Column(String, index=True)
    gpu_model = Column(String)
    form_factor = Column(String)
    memory_gb = Column(Integer)
    memory_type = Column(String)
    tdp_watts = Column(Integer)
    validated = Column(Boolean, nullable=False, default=False, index=True)


def _add_missing_validated_columns():
    """Add the validated column to tables created before it existed."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in ("cpu_specs", "gpu_specs"):
            columns = {col["name"] for col in inspector.get_columns(table)}
            if "validated" not in columns:
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN validated BOOLEAN NOT NULL DEFAULT 0"
                ))


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    _add_missing_validated_columns()


def get_db():
    """Database session dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
