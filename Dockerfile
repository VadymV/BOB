FROM python:3.10

WORKDIR /app
COPY pyproject.toml poetry.lock .
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --only main
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 3100

CMD ["gunicorn", "app:app"]