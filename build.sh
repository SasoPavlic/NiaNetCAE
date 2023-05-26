#!/bin/bash

# Clean previous build artifacts
echo "Cleaning previous build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Install Poetry
echo "Installing Poetry..."
curl -sSL https://install.python-poetry.org | python -

#Sync poetry.lock and requirements.txt
poetry add $( cat requirements.txt )

# Install project dependencies
echo "Installing dependencies..."
poetry install

# Run tests
echo "Running tests..."
poetry run pytest

# Build the project
echo "Building the project..."
poetry build

# Build the Docker image
echo "Building the Docker image..."
docker build --tag spartan300/nianet:cae .

# Push the Docker image to Docker Hub
echo "Pushing the Docker image to Docker Hub..."
docker push spartan300/nianet:cae

echo "Build completed."

