# Contributing

Small bug reports and focused pull requests are welcome during this early preview.
Explain the behavior, provide a synthetic reproduction, and run the checks in README.md.
Use original synthetic fixtures and document changes to the public format.
Do not submit personal data, course-derived material, generation instructions,
private adapters, credentials, or proprietary templates.

Every commit must include a DCO sign-off. By signing, you certify the
[Developer Certificate of Origin 1.1](DCO) ([source](https://developercertificate.org/)).
Use `git commit -s` and an email you are comfortable publishing; a GitHub noreply
address is recommended. A sign-off is a certification, not a cryptographic signature.

Follow CODE_OF_CONDUCT.md. Report vulnerabilities using SECURITY.md.

## Changelogs and releases

Every merged user-facing change adds an entry to the relevant CHANGELOG.md (root, website/, or the public repo). Releases (TestFlight, site deploy, tags) fail without one. Batch site deploys into releases.

The Changelog PR check requires a new nonempty entry under `## [Unreleased]` or a new dated version, compared with the base branch. Existing entry text must be preserved; moving or editing an old entry is not a new entry. Documentation in docs/, tests, CI files and contribution policy are excluded; mixed product changes still require an entry. A maintainer can set `no-changelog` for changes with no user-facing effect. This label is not a release waiver.
