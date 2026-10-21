# Use a minimal base image with Python installed
FROM python:3.12-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the backend folder (app.py, requirements.txt, model) into /app
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The Flask API listens on port 7860
EXPOSE 7860

# Run the Flask app with gunicorn so it is reachable from outside the container
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "app:app"]
