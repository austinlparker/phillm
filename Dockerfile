FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Copy source code
COPY phillm/ ./phillm/

EXPOSE 3000

CMD ["python", "-m", "phillm.main"]