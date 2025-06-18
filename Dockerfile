FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first for better caching
COPY pyproject.toml uv.lock ./

# Install uv and dependencies  
RUN pip install uv
RUN uv sync --frozen

# Copy application code
COPY . .

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:/app

# Create directory for SQLite database
RUN mkdir -p /app/data

# Set environment variables
ENV DATABASE_URL=sqlite:///data/dutch_vocab.db

# Expose port (not needed for Telegram bot, but good practice)
EXPOSE 8000

# Run the bot using uv
CMD ["uv", "run", "python", "src/main.py"]