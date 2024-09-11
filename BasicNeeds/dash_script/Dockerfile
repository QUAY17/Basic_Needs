# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY dash_app.py /app/
COPY requirements.txt /app/

# Copy the pre-computed data files
COPY sentiment_analysis/sentiment_analysis_results_*.csv /app/sentiment_analysis/
COPY keyword_analysis/keyword_analysis_results_*.csv /app/keyword_analysis/
COPY topic_analysis/topic_analysis_results_*.csv /app/topic_analysis/
COPY wordcloud_*.txt /app/
COPY topic_summary_*.txt /app/

# Copy the logo file (assuming it's a PNG file named ICASA_Logo.jpg)
COPY ICASA_Logo.jpg /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run dash_app.py when the container launches
CMD ["python", "dash_app.py"]