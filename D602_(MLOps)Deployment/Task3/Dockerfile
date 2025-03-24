# Use the Python 3.12.1 slim image based on Debian Bookworm.
FROM python:3.12.1-slim-bookworm

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose port 8000 for the API.
EXPOSE 8000

# Run the API with Uvicorn.
CMD ["uvicorn", "prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]