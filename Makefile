.PHONY: docs

docs:
	docker run --rm --network host -v "$$PWD":/srv/jekyll -w /srv/jekyll ruby:3.3 bash -lc 'set -e; gem install github-pages; export PATH="$$(ruby -e '\''print Gem.bindir'\'')":$$PATH; jekyll serve --source docs --baseurl /justhtml --host 0.0.0.0'
