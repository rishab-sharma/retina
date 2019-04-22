FROM ubuntu:latest
MAINTAINER kraken2309 "rishabsharmaddn@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

# Add source file
COPY . /app
ENV HOME=/app
WORKDIR /app

#Install Python Dependencies
RUN pip install -r requirements.txt

ENV FLASK_APP=app.py

# Expose Port
EXPOSE 5000

ENTRYPOINT ["gunicorn" , "-b" , "0.0.0.0:5000" , "-w" , "4" , "wsgi:app"]