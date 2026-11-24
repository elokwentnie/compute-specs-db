"""SQLAlchemy models and SQLite session setup."""

from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils import display_name, model_slug

SQLALCHEMY_DATABASE_URL = "sqlite:///./cpu_database.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CPUSpec(Base):
    """A CPU row."""
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

    @property
    def name(self) -> str:
        """Display name without trademark symbols and clock suffixes."""
        return display_name(self.cpu_model_name)

    @property
    def slug(self) -> str:
        """URL slug for the CPU's spec page."""
        return model_slug(self.cpu_model_name)

    @property
    def tdp_per_thread_w(self) -> Optional[float]:
        """TDP divided by thread count, in watts; None if either is missing."""
        if not self.tdp_watts or not self.threads:
            return None
        return round(self.tdp_watts / self.threads, 2)


class GPUSpec(Base):
    """A GPU row."""
    __tablename__ = "gpu_specs"

    id = Column(Integer, primary_key=True, index=True)
    gpu_model_name = Column(String, index=True)
    vendor = Column(String, index=True)
    gpu_model = Column(String)
    form_factor = Column(String)
    memory_gb = Column(Integer)
    memory_type = Column(String)
    tdp_watts = Column(Integer)
    architecture = Column(String, index=True)
    launch_year = Column(Integer)
    memory_bandwidth_tbs = Column(Float)
    fp64_tflops = Column(Float)
    fp32_tflops = Column(Float)
    fp16_bf16_tensor_tflops = Column(Float)
    fp8_tensor_tflops = Column(Float)
    fp4_tensor_tflops = Column(Float)
    interconnect = Column(String)
    interconnect_bandwidth_gbs = Column(Float)
    pcie_interface = Column(String)
    validated = Column(Boolean, nullable=False, default=False, index=True)

    @property
    def name(self) -> str:
        """Display name."""
        return display_name(self.gpu_model_name)

    @property
    def slug(self) -> str:
        """URL slug for the GPU's spec page."""
        return model_slug(self.gpu_model_name)


def _add_missing_columns():
    """Add columns that were introduced after a table was first created."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for model in (CPUSpec, GPUSpec):
            table = model.__table__
            existing = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if column.name == "validated":
                    ddl = "BOOLEAN NOT NULL DEFAULT 0"
                else:
                    ddl = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {ddl}"))


def init_db():
    """Create tables and apply small schema upgrades."""
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def get_db():
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
