// Resolve this repository's published GitHub Releases into the structured
// release history the deployment flow shows.
//
// Two sources of truth, nothing else: the release bodies GitHub already
// publishes (their field table carries the digest, the publication tag and the
// vLLM, B12X and builder commits) and `profiles/vllmb12x/profile.json` for the
// `lil_runtime` pin. The site never calls the GitHub API at run time, so this
// is where that data becomes a committed-shape artefact.
//
// The pull-request path must not touch the network, so `--fixture` copies a
// committed document instead — the same arrangement `latest-image.json` uses.
// Both paths run the identical validator, which names the offending tag.

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SITE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPOSITORY_ROOT = resolve(SITE_ROOT, "..");
const PROFILE = join(REPOSITORY_ROOT, "profiles/vllmb12x/profile.json");
const DEFAULT_OUTPUT = join(SITE_ROOT, "public/releases.json");
const DEFAULT_REPOSITORY = "randomvariable/vllm-multiarch-oci";
const ADDITIONS_HEADING = "## Base vLLM image additions";
const CHANGES_HEADING = "## Included upstream changes";

const COMMIT = /^[0-9a-f]{40}$/;
export const RELEASE_TAG = /^v\d{8}\.\d+$/;
const DIGEST = /^sha256:[0-9a-f]{64}$/;
const ISO_8601 = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/;
const IMMUTABLE_TAG = /^vllmb12x-[a-z0-9][a-z0-9-]*-[0-9a-f]{12}-[0-9a-f]{12}-[0-9]{8}-n[1-9][0-9]*$/;
const CHANGE_FIELDS = ["change", "component", "included_as", "url"];
const REQUIRED_FIELDS = [
  "tag",
  "published_at",
  "digest",
  "publication_tag",
  "vllm_commit",
  "b12x_commit",
  "lil_runtime_commit",
  "builder_commit",
  "highlights",
  "included_changes",
  "url",
];

// The same rejection build-data.mjs applies to recipes: nothing that names the
// private network may reach a public artefact — an internal hostname, a `.local`
// or `.internal` domain, a RFC 1918 address, or this lab's registry host.
const PRIVATE_REFERENCE =
  /(harbor\.services\.home\.internal|internal\.randomvariable|[a-z0-9-]+\.(?:internal|local)\b|(?<![0-9.])(?:10|192\.168)\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9.]))/i;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export function assertNoPrivateReference(serialised, source) {
  const match = PRIVATE_REFERENCE.exec(serialised);
  assert(!match, `${source}: private reference leaked into a public artefact: ${match && match[0]}`);
}

/** The `| Field | Value |` rows of a release body, unbackticked. */
function fieldTable(body) {
  const fields = {};
  for (const line of body.split("\n")) {
    const cells = line.match(/^\|\s*([^|]+?)\s*\|\s*(.*?)\s*\|\s*$/);
    if (!cells) continue;
    const label = cells[1].trim();
    if (label === "Field" || /^-+$/.test(label)) continue;
    fields[label.toLowerCase()] = cells[2].trim().replace(/^`(.*)`$/, "$1");
  }
  return fields;
}

/** One markdown section body, or the empty string when the heading is absent. */
function section(body, heading) {
  const start = body.indexOf(heading);
  if (start < 0) return "";
  const rest = body.slice(start + heading.length);
  const end = rest.indexOf("\n## ");
  return end < 0 ? rest : rest.slice(0, end);
}

function plainText(value) {
  return value.replace(/\[([^\]]+)\]\((?:[^)]*)\)/g, "$1").replace(/`/g, "").trim();
}

/** The bullets that describe what a release adds. */
function highlightList(body) {
  const highlights = [];
  for (const line of section(body, ADDITIONS_HEADING).split("\n")) {
    const bullet = line.match(/^-\s+(.*)$/);
    if (!bullet) continue;
    const text = plainText(bullet[1]);
    if (text) highlights.push(text);
  }
  return highlights;
}

/** Rows of the `## Included upstream changes` table. */
function includedChanges(body) {
  const changes = [];
  for (const line of section(body, CHANGES_HEADING).split("\n")) {
    const cells = line.match(/^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$/);
    if (!cells) continue;
    const [component, change, includedAs] = [1, 2, 3].map((index) => cells[index].trim());
    if (component === "Component" || /^-+$/.test(component)) continue;
    const link = change.match(/\((https:\/\/[^)]+)\)/);
    changes.push({
      component: plainText(component),
      change: plainText(change),
      included_as: plainText(includedAs),
      url: link ? link[1] : null,
    });
  }
  return changes;
}

function publishedAt(value) {
  assert(typeof value === "string" && ISO_8601.test(value), `published_at must be a UTC ISO 8601 timestamp: ${value}`);
  const timestamp = Date.parse(value);
  assert(!Number.isNaN(timestamp), `published_at is not an ISO 8601 timestamp: ${value}`);
  return value;
}

export function validateRecord(record, source) {
  assert(record && typeof record === "object" && !Array.isArray(record), `${source}: release must be an object`);
  for (const field of REQUIRED_FIELDS) {
    assert(Object.hasOwn(record, field), `${source}: release lacks ${field}`);
  }
  assert(RELEASE_TAG.test(record.tag), `${source}: tag ${record.tag} does not follow vYYYYMMDD.N`);
  publishedAt(record.published_at);
  assert(DIGEST.test(record.digest), `${source}: digest must be sha256-prefixed: ${record.digest}`);
  assert(IMMUTABLE_TAG.test(record.publication_tag), `${source}: publication tag ${record.publication_tag} does not follow the immutable publication contract`);
  for (const field of ["vllm_commit", "b12x_commit", "lil_runtime_commit", "builder_commit"]) {
    assert(COMMIT.test(record[field]), `${source}: ${field} must be a full commit OID: ${record[field]}`);
  }
  assert(
    Array.isArray(record.highlights) && record.highlights.every((value) => typeof value === "string" && value.trim()),
    `${source}: highlights must be an array of non-empty strings`,
  );
  assert(Array.isArray(record.included_changes), `${source}: included_changes must be an array`);
  for (const [index, change] of record.included_changes.entries()) {
    const where = `${source}: included_changes[${index}]`;
    assert(
      change && typeof change === "object" && JSON.stringify(Object.keys(change).sort()) === JSON.stringify(CHANGE_FIELDS),
      `${where} must carry exactly component, change, included_as and url`,
    );
    for (const field of ["component", "change", "included_as"]) {
      assert(typeof change[field] === "string" && change[field].trim(), `${where}.${field} must be a non-empty string`);
    }
    assert(
      change.url === null || (/^https:\/\/[^\s]+$/.test(change.url) && !change.url.includes("internal")),
      `${where}.url must be null or a public HTTPS URL: ${change.url}`,
    );
  }
  assert(/^https:\/\/[^\s]+$/.test(record.url), `${source}: url must be an HTTPS URL: ${record.url}`);
  assertNoPrivateReference(JSON.stringify(record), source);
  return record;
}

export function validateReleases(document, source) {
  assert(document && typeof document === "object" && !Array.isArray(document), `${source}: releases.json must be an object`);
  assert(JSON.stringify(Object.keys(document)) === JSON.stringify(["releases"]), `${source}: releases.json must carry exactly one top-level key, releases`);
  assert(Array.isArray(document.releases), `${source}: releases.json must carry a top-level releases array`);
  assert(document.releases.length > 0, `${source}: releases.json lists no releases`);
  const tags = new Set();
  let previous = Infinity;
  for (const record of document.releases) {
    const tag = record && typeof record.tag === "string" && record.tag ? record.tag : "<unknown tag>";
    validateRecord(record, `${source}: ${tag}`);
    assert(!tags.has(record.tag), `${source}: duplicate release tag ${record.tag}`);
    tags.add(record.tag);
    const timestamp = Date.parse(record.published_at);
    assert(timestamp <= previous, `${source}: releases must be ordered newest first (${record.tag})`);
    previous = timestamp;
  }
  return document;
}

/** The `lil_runtime` pin and the lineage commits this build is anchored to. */
export function pinnedSources(path = PROFILE) {
  const profile = JSON.parse(readFileSync(path, "utf8"));
  const sources = profile.sources;
  assert(sources && typeof sources === "object", `${path}: sources mapping is required`);
  for (const name of ["vllm", "b12x", "lil_runtime"]) {
    assert(sources[name] && COMMIT.test(sources[name].commit), `${path}: sources.${name}.commit must be a full commit OID`);
    assert(/^https:\/\/github\.com\/[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+\.git$/.test(sources[name].remote || ""), `${path}: sources.${name}.remote must be a public GitHub URL`);
  }
  return {
    lilRuntime: sources.lil_runtime.commit,
    lilRemote: sources.lil_runtime.remote,
    vllm: sources.vllm.commit,
    b12x: sources.b12x.commit,
  };
}

function field(fields, label, source) {
  const value = fields[label.toLowerCase()];
  assert(typeof value === "string" && value.trim(), `${source}: release body has no ${label} row`);
  return value;
}

/** One GitHub release object plus the pinned profile in -> one site record. */
export function toRecord(release, pinned) {
  const source = `release ${release && release.tag_name ? release.tag_name : "<unknown tag>"}`;
  assert(typeof release.tag_name === "string" && RELEASE_TAG.test(release.tag_name), `${source}: unexpected tag shape`);
  assert(typeof release.body === "string" && release.body.trim(), `${source}: body is empty`);
  assert(release.draft !== true, `${source}: a draft release was published into the site data`);
  const fields = fieldTable(release.body);
  const publication = field(fields, "Publication tag", source);
  const record = {
    tag: release.tag_name,
    published_at: publishedAt(release.published_at),
    digest: field(fields, "Digest", source),
    publication_tag: publication.slice(publication.lastIndexOf(":") + 1),
    vllm_commit: field(fields, "vLLM", source),
    b12x_commit: field(fields, "B12X", source),
    // The body records the build's lineage commits; the policy pin is this
    // repository's own state, so a release whose body predates the row reports
    // the pin the current build resolves against.
    lil_runtime_commit: fields.lil_runtime || fields["runtime data"] || fields["upstream runtime"] || pinned.lilRuntime,
    builder_commit: field(fields, "Builder", source),
    highlights: highlightList(release.body),
    included_changes: includedChanges(release.body),
    url: release.html_url,
  };
  assert(record.digest.startsWith("sha256:"), `${source}: digest is not a manifest digest`);
  assert(release.prerelease !== true, `${source}: a pre-release was published into the site data`);
  return validateRecord(record, source);
}

function fetchReleases(repository) {
  const stdout = execFileSync("gh", ["api", "--paginate", `repos/${repository}/releases`], {
    cwd: REPOSITORY_ROOT,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const releases = [];
  for (const chunk of stdout.split("\n")) {
    const text = chunk.trim();
    if (!text) continue;
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) releases.push(...parsed);
    else releases.push(parsed);
  }
  assert(releases.length > 0, `repos/${repository}/releases: GitHub returned no releases`);
  return releases;
}

function parseArguments(argv) {
  const options = { output: DEFAULT_OUTPUT, repository: DEFAULT_REPOSITORY, fixture: null };
  for (let index = 0; index < argv.length; index += 1) {
    const value = (name) => {
      assert(argv[index + 1], `--${name} needs a value`);
      index += 1;
      return argv[index];
    };
    switch (argv[index]) {
      case "--fixture":
        options.fixture = value("fixture");
        break;
      case "--output":
        options.output = value("output");
        break;
      case "--repository":
        options.repository = value("repository");
        break;
      default:
        assert(false, `unknown argument ${argv[index]}`);
    }
  }
  return options;
}

/**
 * A path relative to the caller, the site or the repository, whichever exists.
 * `pnpm --dir recipes-site exec` runs from the site while a workflow runs from
 * the repository root, and both spell the fixture path the natural way.
 */
export function locate(given) {
  if (isAbsolute(given)) return given;
  const candidates = [resolve(process.cwd(), given), resolve(SITE_ROOT, given), resolve(REPOSITORY_ROOT, given)];
  for (const candidate of candidates) if (existsSync(candidate)) return candidate;
  assert(false, `${given}: no such file, tried ${candidates.join(", ")}`);
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const pinned = pinnedSources();
  let document;
  let source;
  if (options.fixture) {
    // The pull-request path: a committed document, no GitHub API, no token.
    source = locate(options.fixture);
    document = JSON.parse(readFileSync(source, "utf8"));
  } else {
    source = "github api";
    const seen = fetchReleases(options.repository)
      .filter((release) => !release.draft && release.prerelease !== true)
      .map((release) => toRecord(release, pinned));
    seen.sort((left, right) => Date.parse(right.published_at) - Date.parse(left.published_at));
    document = { releases: seen };
  }
  validateReleases(document, source);
  mkdirSync(dirname(options.output), { recursive: true });
  writeFileSync(options.output, `${JSON.stringify(document, null, 2)}\n`);
  process.stdout.write(`${options.output}: ${document.releases.length} releases, newest ${document.releases[0].tag}\n`);
}

if (process.argv[1] && import.meta.url === `file://${resolve(process.argv[1])}`) {
  await main();
}
