APP_NAME=${APP_NAME:-$(basename $(pwd))}
# Build the container
build: ## Build the release and develoment container. The development
	docker-compose build --no-cache $(APP_NAME)

dev: ## Run container in development mode
	docker-compose build --no-cache $(APP_NAME) && docker-compose run $(APP_NAME)

# Build and run the container
up: ## Spin up the project
	docker-compose up --build $(APP_NAME)

stop: ## Stop running containers
	docker stop $(APP_NAME)
rm: stop ## Stop and remove running containers
	docker rm $(APP_NAME)
test:
	pytest tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y
