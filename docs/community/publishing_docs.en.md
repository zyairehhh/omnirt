# Publishing Docs

This repository publishes a static documentation site to GitHub Pages using MkDocs and Material for MkDocs.

## Local preview

Install the docs toolchain:

```bash
python -m pip install -e '.[docs]'
```

Run the local preview server:

```bash
mkdocs serve
```

Build the site exactly as CI does:

```bash
mkdocs build --strict
```

`--strict` is the default validation target for pull requests and the Pages build workflow because it catches broken links and nav mistakes early.

## GitHub Pages deployment

The repository includes a dedicated workflow at `.github/workflows/docs.yml`.

Deployment behavior:

- triggers on pushes to `main`
- can also be started manually through `workflow_dispatch`
- installs the docs dependencies with `pip install -e '.[docs]'`
- builds the current ref with `mkdocs build --strict`
- uses `mike` to publish the current docs into a versioned directory on the `gh-pages` branch
- exports the full versioned site from `gh-pages` into `site/`
- uploads the generated `site/` directory as the Pages artifact when GitHub Pages is enabled
- deploys through the official GitHub Pages actions flow when GitHub Pages is enabled

## Versioned docs

A normal `main` push publishes two entries:

- `/main/`: the current main-branch documentation
- `/latest/`: an alias pointing at the current main docs

For a formal release, start the docs workflow manually and set `version` to the release number, for example `v0.1.0`. To make a version the default landing point, set `default_version` to that version or alias. `mike` maintains `versions.json`, and Material for MkDocs renders the version selector in the site UI.

## Site URL and path model

This repository is configured as a project site, not a user site.

- repository: `datascale-ai/omnirt`
- public URL: `https://datascale-ai.github.io/omnirt/`

That means links and local previews should assume the docs live under the `/omnirt/` path when published.

## Maintenance guidelines

- keep `README.md` focused on the repository landing-page summary
- move longer user-facing guides into `docs/`
- prefer stable document paths and reorganize the nav through `mkdocs.yml`
- run `mkdocs build --strict` before merging significant doc changes

If GitHub Pages is not yet enabled for the repository, the workflow still maintains the versioned docs content on `gh-pages`, but it skips the Pages artifact deployment. Switch the Pages source to GitHub Actions in the repository settings once; after that, live deployments are handled by the workflow.
