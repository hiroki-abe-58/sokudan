# Additional distribution channels

The Python package on PyPI remains the inference implementation. Additional
packages provide launchers, an HTTP SDK, or a serving container. Model weights
remain on Hugging Face and are downloaded when the server starts.

## Release status

Adding these files does **not** publish a package. Before announcing an install
command, verify the matching registry entry and a clean installation.

| Channel | Implementation in this repository | Remaining release step |
|---|---|---|
| PyPI | Existing Python package, 0.3.0 | Existing release |
| npm | `packages/npm`: CLI + typed ESM HTTP SDK | Initial authenticated publish; configure OIDC |
| Homebrew own tap | `Formula/sokudan.rb`: uv launcher and checksummed Python source | Merge Formula; verify Homebrew CI |
| GHCR | `Dockerfile.serve`, `container.yml`: Linux amd64 CPU server | Pass model smoke test; run manual publish; set package public |
| Docker Hub | Same tested serving image | Choose namespace; authenticate and push |
| Conda / conda-forge | Not implemented | Resolve MLX/PyTorch dependencies; build/test recipe; submit |
| Scoop / WinGet / Chocolatey | Not implemented | Test Windows distribution; prepare manifests/packages |
| Nix / AUR | Not implemented | Resolve native dependencies and test builds on target systems |
| JSR | SDK source is separate from Node-only CLI | Choose JSR scope, validate and publish SDK |

Official repository acceptance (Homebrew core, conda-forge, WinGet, Nixpkgs) is
separate from publishing to a maintainer-owned tap/channel. No submissions to
those repositories are made by these workflows.

## npm

Requirements: Node.js 22+, npm, Python 3 for launcher tests. Runtime consumers of
the SDK do not need Python. CLI consumers need uv, or `SOKUDAN_PYTHON` pointing
to an environment with sokudan and the appropriate extras already installed.

```sh
cd packages/npm
npm ci
npm run typecheck
npm test
npm pack
```

Install `sokudan-0.3.0.tgz` in a separate project to test the actual archive.
The build copies the root LICENSE and NOTICE and generates JS/declarations.
There is no postinstall hook and no npm runtime dependency.

For an HTTP integration check against the real Python server with a fake model:

```sh
# Use an environment with sokudan[serve] and pytest installed.
SOKUDAN_TEST_PYTHON=/path/to/venv/bin/python node scripts/check-python-server.mjs
```

Initial publication (from `packages/npm`, after verifying the account with
`npm whoami`):

```sh
npm login
npm publish --access public
```

The first-choice name is `sokudan`; an unregistered name is not a reservation.
Do not silently publish under a different name/account if registration fails.
The npm version pins the Python version used by the default CLI, so a matching
non-yanked release must already exist on PyPI. `scripts/check-release.mjs`
checks the tag and Python release before CI publishing.

For subsequent releases, configure an npm trusted publisher:

- GitHub owner: `hiroki-abe-58`
- Repository: `sokudan`
- Workflow filename: `npm.yml`
- GitHub environment: `npm` (create it; required reviewers are optional)

Use npm 11.5.1+ for trusted publishing (workflow uses npm 11 with Node 24).
Push `npm-v<version>` only after the package version and matching PyPI release
are ready. The workflow tests Node 22/24 on Linux, macOS and Windows, then
publishes with OIDC and provenance. It needs no stored npm token. An npm version
cannot be reused; SDK/launcher fixes also need a new matching Python release
under the current shared-version policy.

## Homebrew: third-party tap

Once the Formula is merged into the default branch:

```sh
brew tap hiroki-abe-58/sokudan https://github.com/hiroki-abe-58/sokudan
brew install hiroki-abe-58/sokudan/sokudan
sokudan serve --port 8000
```

Using this repository as a custom-URL tap avoids creating a second repository.
This is a **launcher formula**, not a Homebrew-core-style bundled Python runtime:
brew installs uv, Python 3.13 and the checksummed PyPI source. On first use uv
builds that source with the `serve` extra in its user cache. Research probes add
`train,bench,torch`. Transitive Python dependencies are resolved at first use,
not locked by the Formula. First use therefore needs network access.
Homebrew removes its files on uninstall; uv and Hugging Face user caches remain.

`homebrew.yml` checks Formula installation, `brew test`, and the first invocation
of `sokudan serve --help` on macOS. This does not claim that every Homebrew/Linux
configuration or GPU backend is tested. Do not submit this launcher to
Homebrew core as if it were an offline, fully bundled Python formula.

On each Python release update the Formula's URL and SHA-256 from PyPI, then rerun
the installation check. Keep its version aligned with the npm release.

## Serving container: GHCR and Docker Hub

The root `Dockerfile` is still the clean-install regression check.
`Dockerfile.serve` is the non-root CPU serving image. It initially targets
**Linux amd64**; Apple silicon can emulate it but will not use MLX/MPS inside it.
Native arm64 and CUDA images need their own build/runtime validation.

```sh
docker build --platform linux/amd64 -f Dockerfile.serve -t sokudan-serve:0.3.0 .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v sokudan-cache:/home/sokudan/.cache/huggingface sokudan-serve:0.3.0
```

The container binds to all interfaces internally; the example publishes only to
host loopback. Inference starts after the first model download. Cache that
download with the named volume. The server does not authenticate clients.

`container.yml` builds and tests on PRs and manual runs. It checks the packaged
CLI, `pip check`, actual model loading, `/health`, and an inference request.
Its manual `publish` option works only on `main` and pushes the **same tested
image** to `ghcr.io/hiroki-abe-58/sokudan:<version>-cpu` and `sha-<commit>`.
After the first push, confirm the GHCR package is public before documenting it
as an anonymous download. Neither a PR nor a tag automatically publishes it.

Docker Hub can receive the same tested image. Once the namespace has been chosen:

```sh
docker login
docker tag sokudan-serve:0.3.0 YOUR_NAMESPACE/sokudan:0.3.0-cpu
docker push YOUR_NAMESPACE/sokudan:0.3.0-cpu
```

For automation, store the Docker Hub username and a scoped write token in GitHub
Actions secrets and add a login/push step after the existing model test. Do not
commit credentials. Dependencies and the model's default Hub revision are not
fully frozen; the image tag describes the packaged code, not immutable model weights.

## Sources

- [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/)
- [uv tools](https://docs.astral.sh/uv/guides/tools/)
- [Homebrew taps](https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap)
- [Homebrew language-specific formulae](https://docs.brew.sh/Language-Specific-Formulae)
- [GHCR](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [conda-forge submissions](https://conda-forge.org/docs/maintainer/adding_pkgs/)
- [WinGet submissions](https://learn.microsoft.com/en-us/windows/package-manager/package/)
- [JSR publishing](https://jsr.io/docs/publishing-packages)
