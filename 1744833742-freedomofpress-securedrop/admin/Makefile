DEFAULT_GOAL: help

.PHONY: test
test:  ## Run tox
	@echo "NB.  This can be VERY slow.  If you find yourself running this test suite multiple times, you may prefer to \"docker run -it securedrop-admin bash\", install the editor of your choice into the container and edit there, and run \"tox\" directly as you work."
	bin/dev-shell tox

.PHONY: update-pip-requirements
update-pip-requirements: ## Updates all Python requirements files via pip-compile.
	@echo "███ Updating admin pip requirements..."
	@bin/dev-shell pip-compile --allow-unsafe --generate-hashes --output-file requirements.txt requirements.in requirements-ansible.in
	@bin/dev-shell pip-compile --allow-unsafe --generate-hashes --output-file requirements-testinfra.txt requirements.in requirements-ansible.in requirements-testinfra.in
	@bin/dev-shell pip-compile --allow-unsafe --generate-hashes --output-file requirements-dev.txt requirements-dev.in

# Explanation of the below shell command should it ever break.
# 1. Set the field separator to ": ##" and any make targets that might appear between : and ##
# 2. Use sed-like syntax to remove the make targets
# 3. Format the split fields into $$1) the target name (in blue) and $$2) the target description
# 4. Pass this file as an arg to awk
# 5. Sort it alphabetically
# 6. Format columns with colon as delimiter.
.PHONY: help
help: ## Print this message and exit.
	@printf "Makefile for developing and testing securedrop-admin.\n"
	@printf "Subcommands:\n\n"
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%s\033[0m : %s\n", $$1, $$2}' $(MAKEFILE_LIST) \
		| sort \
		| column -s ':' -t
