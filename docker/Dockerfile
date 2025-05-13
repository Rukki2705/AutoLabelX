# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all source files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
 && pip install pipenv \
 && pipenv install --system --deploy --ignore-pipfile

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit dashboard
CMD ["streamlit", "run", "active_learning/streamlit_dashboard.py", "--server.port=8501", "--server.enableCORS=false"]
