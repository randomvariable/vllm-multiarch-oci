// Gates for the Pages data pipeline: the three artefacts the deployment flow
// renders from, and the self-hosted Pyodide tree. These run on pull requests,
// where the workflow supplies the committed fixtures, so nothing here may reach
// the network: every case either reads `tests/fixtures/` or builds a synthetic
// distribution in a scratch directory beside the site.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

import { buildRoute, publishBuildPages, publishDeploymentData } from "../scripts/build-data.mjs";
import { assemble, pins, verifyPublished } from "../scripts/prepare-pyodide.mjs";
import { assertNoPrivateReference, validateRecord, validateReleases } from "../scripts/resolve-releases.mjs";
import { dockerInvocation, recipeInventory, validateConfigs, validateOptions, validateRecord as validateConfigRecord } from "../scripts/resolve-configs.mjs";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const fixtureRoot = join(siteRoot, "tests/fixtures");
const repositoryRoot = resolve(siteRoot, "..");
const PRIVATE_REGISTRY = "harbor.services.home.internal.randomvariable.co.uk/custom/worker";

function fixture(name) {
  return JSON.parse(readFileSync(join(fixtureRoot, name), "utf8"));
}

function scratch(prefix) {
  const directory = mkdtempSync(join(siteRoot, prefix));
  test.after(() => rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function publicWith(...names) {
  const directory = scratch(".site-data-public-");
  for (const name of names) copyFileSync(join(fixtureRoot, name), join(directory, name));
  return directory;
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function rejection(operation) {
  try {
    await operation();
  } catch (error) {
    return error.message;
  }
  assert.fail("the build was expected to fail");
}

// The committed fixtures must pass the gate the Pages build applies.
test("the pull-request fixtures satisfy the Pages build gate", async () => {
  await publishDeploymentData(publicWith("releases.json", "configs.json", "options.json"));
});

test("a malformed release record fails and names its tag", () => {
  const record = fixture("releases.json").releases[0];
  assert.throws(
    () => validateRecord({ ...record, digest: "sha256:not-a-digest" }, `release ${record.tag}`),
    new RegExp(`release ${record.tag}: digest must be sha256-prefixed`),
  );
  const missing = { ...record };
  delete missing.included_changes;
  assert.throws(() => validateRecord(missing, `release ${record.tag}`), /lacks included_changes/);
  assert.throws(
    () => validateRecord({ ...record, builder_commit: "db2e3be" }, `release ${record.tag}`),
    /builder_commit must be a full commit OID/,
  );
  const orphan = { ...record, included_changes: [{ component: "vLLM", change: "x", included_as: "y" }] };
  assert.throws(() => validateRecord(orphan, `release ${record.tag}`), /must carry exactly component, change, included_as and url/);

  const unsorted = structuredClone(fixture("releases.json"));
  unsorted.releases.reverse();
  assert.throws(() => validateReleases(unsorted, "releases.json"), /ordered newest first/);

  const duplicated = structuredClone(fixture("releases.json"));
  duplicated.releases[1] = { ...duplicated.releases[0] };
  assert.throws(() => validateReleases(duplicated, "releases.json"), /duplicate release tag/);
});

test("an internal registry reference fails the build in every generated artefact", async () => {
  const injected = `${PRIVATE_REGISTRY}@sha256:${"0".repeat(64)}`;
  const poison = {
    "releases.json": (document) => {
      document.releases[0].highlights[0] = `${document.releases[0].highlights[0]} ${injected}`;
    },
    "configs.json": (document) => {
      document["recipe:qwen38-27b"].warnings = [injected];
    },
    "options.json": (document) => {
      document.recipe_docs["qwen38-27b"].HOME.why = `${document.recipe_docs["qwen38-27b"].HOME.why} ${injected}`;
    },
  };
  for (const [name, corrupt] of Object.entries(poison)) {
    const directory = publicWith("releases.json", "configs.json", "options.json");
    const document = fixture(name);
    corrupt(document);
    writeFileSync(join(directory, name), `${JSON.stringify(document, null, 2)}\n`);
    const message = await rejection(() => publishDeploymentData(directory));
    assert.match(message, /private reference leaked into a public artefact/);
    assert.ok(message.includes(name), `${name} should be named in the failure: ${message}`);
  }

  // The recipe lane refused a private registry before this pipeline existed;
  // prove the shared guard is what enforces it now.
  assert.throws(() => assertNoPrivateReference(`{"image":"${injected}"}`, "recipe.yaml"), /private reference leaked/);
  assert.throws(() => assertNoPrivateReference('{"a":"10.20.30.40"}', "recipe.yaml"), /private reference leaked/);
  assert.throws(() => assertNoPrivateReference('{"a":"nfs.home.internal:/exports"}', "recipe.yaml"), /private reference leaked/);
  assert.doesNotThrow(() => assertNoPrivateReference(`{"a":"${injected.replace("harbor.services.home.internal.randomvariable.co.uk", "ghcr.io/randomvariable")}","b":"0.29.0+rv.mxfp8"}`, "recipe.yaml"));
});

test("a stale or mismatched configuration record fails the build", () => {
  const document = fixture("configs.json");
  const { declarations } = recipeInventory();
  const check = (candidate) => validateConfigs(candidate, "configs.json", declarations);
  assert.ok(Object.keys(document).length > 0);
  assert.ok(document["resolver_version"] === undefined, "resolver_version belongs to the record, not the map");
  check(structuredClone(document));

  const moved = structuredClone(document);
  moved["recipe:qwen38-27b"].profile = "ds4-flash";
  assert.throws(() => check(moved), /recipe:qwen38-27b resolved profile ds4-flash, recipe qwen38-27b declares qwen38-flash-next/);

  // The recipe file is the source of truth; a record that no longer matches it
  // is a cache left behind by an edit.
  const drifted = structuredClone(document);
  drifted["recipe:qwen38-flash-next-gb10-tp2"].topology.nodes = 4;
  assert.throws(() => check(drifted), /resolved 4 nodes, recipe qwen38-flash-next-gb10-tp2 declares 2/);

  const removed = structuredClone(document);
  delete removed["recipe:qwen38-27b"];
  assert.throws(() => check(removed), /no configuration for recipe qwen38-27b/);

  const unsourceled = structuredClone(document);
  const settings = unsourceled["recipe:qwen38-27b"].settings;
  settings["max-model-len"] = { value: settings["max-model-len"].value };
  assert.throws(() => check(unsourceled), /must carry exactly value and source/);

  const stale = structuredClone(document);
  stale["recipe:qwen38-27b"].resolver_version = { lil_runtime: "0".repeat(40), builder: "not-a-commit" };
  assert.throws(() => check(stale), /resolver_version\.builder must be a full commit OID/);

  const invented = structuredClone(document);
  invented["selection:whatever"] = structuredClone(document["profile:ds4-flash#native"]);
  assert.throws(() => check(invented), /is not a selection key/);

  const mismatched = structuredClone(document);
  mismatched["profile:ds41-flash#native"] = structuredClone(document["profile:ds4-flash#native"]);
  assert.throws(() => check(mismatched), /profile:ds41-flash#native carries profile ds4-flash/);
});

test("a recipe value that upstream does not document must be documented", () => {
  const { values } = recipeInventory();
  validateOptions(fixture("options.json"), "options.json", values);

  const undocumented = structuredClone(fixture("options.json"));
  delete undocumented.recipe_docs["qwen38-27b"].HF_XET_CACHE;
  assert.throws(() => validateOptions(undocumented, "options.json", values), /sets HF_XET_CACHE with no documentation/);

  const stale = structuredClone(fixture("options.json"));
  stale.recipe_docs["qwen38-27b"].A_VALUE_NOBODY_SETS = { group: "runtime", summary: "x", why: "y" };
  assert.throws(() => validateOptions(stale, "options.json", values), /documents A_VALUE_NOBODY_SETS, which the recipe never sets/);

  const duplicated = structuredClone(fixture("options.json"));
  duplicated.recipe_docs["qwen38-27b"]["max-model-len"] = { group: "memory", summary: "x", why: "y" };
  assert.throws(() => validateOptions(duplicated, "options.json", values), /duplicates documentation upstream already carries/);

  const ungrouped = structuredClone(fixture("options.json"));
  ungrouped.recipe_docs["qwen38-27b"].HOME.group = "not-a-group";
  assert.throws(() => validateOptions(ungrouped, "options.json", values), /group not-a-group is not a declared group/);

  const unjustified = structuredClone(fixture("options.json"));
  delete unjustified.recipe_docs["qwen38-27b"].HOME.why;
  assert.throws(() => validateOptions(unjustified, "options.json", values), /recipe_docs\.qwen38-27b\.HOME needs why/);

  const undocumentedEnvironment = structuredClone(fixture("options.json"));
  undocumentedEnvironment.parameter_docs.environment.CUDA_VISIBLE_DEVICES = { group: "not-a-group", summary: "x" };
  assert.throws(() => validateOptions(undocumentedEnvironment, "options.json", values), /group not-a-group is not a declared group/);

  const phantomOption = structuredClone(fixture("options.json"));
  phantomOption.parameter_docs.options["a-value-nobody-manages"] = { group: "memory", summary: "x", why: "y" };
  assert.throws(() => validateOptions(phantomOption, "options.json", values), /documents an option upstream does not manage/);
});

// A synthetic distribution, so the checksum paths are exercised without a
// download: `assemble` verifies every cached entry before it uses it.
function synthetic({ omitCoreFile = null } = {}) {
  const declared = pins();
  const root = scratch(".site-data-pyodide-");
  const staging = join(root, "members");
  mkdirSync(join(staging, "pyodide"), { recursive: true });
  for (const name of declared.coreFiles) writeFileSync(join(staging, "pyodide", name), `${name} contents\n`);
  if (omitCoreFile) rmSync(join(staging, "pyodide", omitCoreFile));
  const wheel = `${declared.packages[0].name}-${declared.packages[0].version}-pyemscripten_wasm32.whl`;
  const wheelBytes = Buffer.from("synthetic wheel\n");
  const lock = {
    info: { abi_version: "2026_0", arch: "wasm32", platform: "emscripten_5_0_3", python: "3.14.2" },
    packages: { [declared.packages[0].name]: { name: declared.packages[0].name, version: declared.packages[0].version, file_name: wheel, sha256: digest(wheelBytes) } },
  };
  writeFileSync(join(staging, "pyodide", "pyodide-lock.json"), `${JSON.stringify(lock)}\n`);
  const archive = join(root, declared.archive.name);
  execFileSync("tar", ["--create", "--bzip2", "--file", archive, "--directory", staging, "pyodide"]);
  // The synthetic archive is pinned by the digest it actually carries, so
  // verification runs the production code path rather than a stub.
  const dist = { ...declared, archive: { ...declared.archive, sha256: digest(readFileSync(archive)) } };

  mkdirSync(join(root, "cache", dist.version), { recursive: true });
  copyFileSync(archive, join(root, "cache", dist.version, dist.archive.name));
  writeFileSync(join(root, "cache", dist.version, wheel), wheelBytes);

  const runtimeRoot = join(root, "runtime");
  for (const name of ["profiles", "hardware", "templates"]) mkdirSync(join(runtimeRoot, name), { recursive: true });
  for (const name of dist.policyFiles) writeFileSync(join(runtimeRoot, name), `${name}\n`);
  writeFileSync(join(runtimeRoot, "profiles", "qwen38-flash-next.yaml"), "kind: model\n");
  writeFileSync(join(runtimeRoot, "hardware", "gb10-roce.yaml"), "kind: hardware\n");
  writeFileSync(join(runtimeRoot, "templates", "glm53-flash.jinja"), "template\n");
  return { dist, root, runtimeRoot, wheel };
}

function options({ dist, root, runtimeRoot }, output) {
  return { output, cacheDir: join(root, "cache"), runtimeRoot, workDir: join(root, "work"), offline: true };
}

test("a Pyodide checksum mismatch fails the build", async () => {
  const case1 = synthetic();
  const cachedArchive = join(case1.root, "cache", case1.dist.version, case1.dist.archive.name);
  writeFileSync(cachedArchive, Buffer.concat([readFileSync(cachedArchive), Buffer.from("tampered\n")]));
  assert.match(await rejection(() => assemble(case1.dist, options(case1, join(case1.root, "out")))), /checksum mismatch, expected [0-9a-f]{64} and got [0-9a-f]{64}/);

  // A forged wheel is caught against the digest the verified archive's own
  // lock records, not a second hand-written list.
  const case2 = synthetic();
  writeFileSync(join(case2.root, "cache", case2.dist.version, case2.wheel), Buffer.from("forged wheel\n"));
  assert.match(await rejection(() => assemble(case2.dist, options(case2, join(case2.root, "out")))), /checksum mismatch/);
});

test("a missing Pyodide member or an unbuilt policy tree fails the build", async () => {
  const missing = synthetic({ omitCoreFile: "pyodide.asm.wasm" });
  assert.match(await rejection(() => assemble(missing.dist, options(missing, join(missing.root, "out")))), /the archive carries no pyodide\.asm\.wasm/);

  const prepared = synthetic();
  assert.match(
    await rejection(() => assemble(prepared.dist, { ...options(prepared, join(prepared.root, "out")), runtimeRoot: join(prepared.root, "absent") })),
    /the merged policy tree is missing/,
  );

  const noPolicy = synthetic();
  rmSync(join(noPolicy.runtimeRoot, "options.yaml"));
  assert.match(await rejection(() => assemble(noPolicy.dist, options(noPolicy, join(noPolicy.root, "out")))), /options\.yaml is missing from the policy tree/);
});

test("two runs with the same pins produce an identical manifest", async () => {
  const prepared = synthetic();
  const first = await assemble(prepared.dist, options(prepared, join(prepared.root, "out-a")));
  const second = await assemble(prepared.dist, options(prepared, join(prepared.root, "out-b")));
  assert.equal(
    readFileSync(join(prepared.root, "out-a", "manifest.json"), "utf8"),
    readFileSync(join(prepared.root, "out-b", "manifest.json"), "utf8"),
    "a second run must not perturb the manifest",
  );
  assert.deepEqual(first, second);

  const published = join(prepared.root, "out-a");
  const paths = first.files.map((entry) => entry.path);
  // An unexpected extra file, or one whose bytes moved after the manifest was
  // written, must not survive as far as a reader's browser.
  writeFileSync(join(published, "unexpected.js"), "surprise\n");
  assert.match(await rejection(() => verifyPublished(published, first)), /published tree and manifest disagree: unexpected\.js/);
  rmSync(join(published, "unexpected.js"));
  writeFileSync(join(published, "pyodide.js"), "corrupted\n");
  assert.match(await rejection(() => verifyPublished(published, first)), /pyodide\.js: digest does not match the manifest/);
  writeFileSync(join(published, "pyodide.js"), "pyodide.js contents\n");
  verifyPublished(published, first);
  assert.deepEqual(paths, [...paths].sort(), "the manifest must be ordered by path");
  assert.ok(first.files.every((entry) => entry.source.length > 0), "every published file must record where it came from");
  assert.ok(paths.includes("image_tools/launcher/resolver.py"));
  assert.ok(paths.includes("image_tools/launcher/probes.py"));
  assert.ok(paths.includes("profiles/qwen38-flash-next.yaml"), "the policy layout must survive packaging");
  assert.ok(paths.includes("hardware/gb10-roce.yaml"), "this repository's hardware profile must ship too");
  assert.ok(paths.some((path) => path.endsWith(".whl")), "PyYAML must be packaged, not fetched at run time");
  assert.equal(first.packages.length, 1);
});

test("the pinned Pyodide artefacts are versioned and digest-pinned", () => {
  const dist = pins();
  assert.match(dist.version, /^\d+\.\d+\.\d+$/);
  assert.match(dist.archive.sha256, /^[0-9a-f]{64}$/);
  assert.equal(dist.archive.name, `pyodide-core-${dist.version}.tar.bz2`);
  assert.match(dist.archive.url, new RegExp(`^https://github\\.com/pyodide/pyodide/releases/download/${dist.version}/pyodide-core-`));
  assert.deepEqual(dist.packages, [{ name: "pyyaml", version: "6.0.3" }]);
  assert.ok(dist.launcherFiles.includes("image_tools/launcher/resolver.py"));
  assert.ok(dist.launcherFiles.includes("image_tools/launcher/probes.py"));
  assert.ok(dist.policyFiles.includes("options.yaml") && dist.policyFiles.includes("parameter-docs.yaml"));
});

test("the resolver the browser runs needs nothing the browser lacks", () => {
  // The image and the browser must execute the same file. If this lane ever
  // needs more than the standard library and PyYAML, the command tab cannot.
  const source = readFileSync(join(repositoryRoot, "image_tools/launcher/resolver.py"), "utf8");
  const imports = [...source.matchAll(/^ *(?:import|from) ([A-Za-z0-9_.]+)/gm)].map((match) => match[1].split(".")[0]);
  const allowed = new Set(["__future__", "ast", "copy", "dataclasses", "hashlib", "importlib", "json", "math", "os", "pathlib", "re", "sys", "typing", "yaml", "image_tools"]);
  assert.deepEqual([...new Set(imports)].filter((name) => !allowed.has(name)).sort(), []);
});

test("a build route keeps a separator where a dot would be dropped", () => {
  // Astro strips dots from a route segment, so /builds/v20261003.1/ is served
  // nowhere and a plain file name collapses to v202610031. The dash is what the
  // reader can bookmark.
  assert.equal(buildRoute("v20261003.1"), "v20261003-1");
  assert.throws(() => buildRoute("v20261003"), /not a vYYYYMMDD\.N build tag/);
  assert.throws(() => buildRoute("../../etc"), /not a vYYYYMMDD\.N build tag/);
});

test("build pages are regenerated for the published set and stale ones removed", async () => {
  const directory = scratch(".site-data-builds-");
  const document = fixture("releases.json");
  const repository = fixture("latest-image.json").repository;
  mkdirSync(join(directory, "v20991231-1"), { recursive: true });
  writeFileSync(join(directory, "v20991231-1", "index.md"), "---\ntitle: stale\n---\n\nGone.\n");

  await publishBuildPages(document, repository, directory);

  const routes = document.releases.map((release) => buildRoute(release.tag)).sort();
  assert.deepEqual(
    readdirSync(directory, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort(),
    routes,
  );
  const page = readFileSync(join(directory, routes.at(-1), "index.md"), "utf8");
  const newest = document.releases[0];
  assert.ok(page.includes(`title: "${newest.tag} build"`), "the title carries the exact tag");
  assert.ok(page.includes(`docker pull ${repository}@${newest.digest}`), "the pull command is digest-pinned");
  assert.ok(page.includes(newest.publication_tag), "the publication tag is recorded");
  assert.ok(!page.includes("harbor."), "a generated page holds no private registry");
});

test("a record must say what it was resolved against", () => {
  // The browser re-runs the resolver to show an edited vLLM command and cannot
  // import vLLM itself. Without these fields it would resolve as if the container's
  // packages were absent, and the printed command would differ from what runs.
  const key = "recipe:qwen38-flash-next-gb10-tp2";
  const good = fixture("configs.json")[key];
  assert.doesNotThrow(() => validateConfigRecord(structuredClone(good), key));

  for (const [mutate, expected] of [
    [(record) => delete record.resolution_context, /resolution_context/],
    [(record) => (record.resolution_context = { ...record.resolution_context, source: "runner" }), /source must be image or host/],
    [(record) => (record.resolution_context = { ...record.resolution_context, vllm_environment: ["not-a-name"] }), /must be null or a list of environment names/],
    [(record) => (record.resolution_context = { ...record.resolution_context, b12x_mxfp8_moe: "yes" }), /must be a boolean/],
    [(record) => (record.resolution_context = { ...record.resolution_context, runtime_identity: "latest" }), /hex digest/],
  ]) {
    const broken = structuredClone(good);
    mutate(broken);
    assert.throws(() => validateConfigRecord(broken, key), expected, `expected ${expected}`);
  }
});

test("asking the image forwards only the allowlisted variables", () => {
  const reference = "ghcr.io/randomvariable/vllm-b12x-multi@sha256:" + "a".repeat(64);
  const docker = dockerInvocation(
    {
      PATH: "/should/not/appear",
      PYTHONPATH: "/repository/should/not/appear",
      HOME: "/runner",
      AWS_SECRET_ACCESS_KEY: "leak",
      LWS_WORKER_INDEX: "0",
      LWS_GROUP_SIZE: "2",
      LWS_LEADER_ADDRESS: "$(LWS_LEADER_ADDRESS)",
      POD_IP: "192.0.2.10",
      // The host lane's own values: paths in the checkout, which the container is
      // given no mount for. They must not be forwarded; see DOCKER_PASSTHROUGH.
      VLLM_IMAGE_DATA_ROOT: "/home/runner/work/_temp/recipes-site-data/runtime",
      VLLM_IMAGE_RECIPE_ROOT: "/home/runner/work/vllm-multiarch-oci/vllm-multiarch-oci/recipes",
    },
    reference,
    ["--recipe", "qwen38-flash-next-gb10-tp2"],
  );
  const forwarded = docker.filter((_, index) => docker[index - 1] === "-e");
  assert.deepEqual(
    forwarded.map((entry) => entry.split("=")[0]).sort(),
    ["LWS_GROUP_SIZE", "LWS_LEADER_ADDRESS", "LWS_WORKER_INDEX", "POD_IP"],
  );
  // Nothing forwarded may name a path: every one is either the literal a pod
  // controller injects or a value the manifest itself sets.
  for (const entry of forwarded) assert.doesNotMatch(entry, /=.*\//, `${entry}: a filesystem path reached the container`);
  assert.doesNotMatch(docker.join(" "), /vllm-multiarch-oci/, "the runner's checkout path reached the container");
  assert.equal(docker[0], "run");
  assert.equal(docker[2], "--entrypoint");
  assert.equal(docker[3], "/opt/python/bin/python");
  assert.equal(docker[docker.indexOf(reference) + 1], "-m");
  assert.equal(docker.at(-1), "--print-config");
  assert.throws(() => dockerInvocation({}, "ghcr.io/x/y:latest", []), /digest-pinned reference/);
});

// The trunk lane of the Pages workflow is the only place the published releases are
// read from GitHub, and it reads them through the `gh` CLI, which refuses to call the
// API there at all: a runner carries no credential file of its own, so the step needs
// the run's token. A pull request builds green from committed fixtures without ever
// reaching that step, so the invariant is pinned here, where every lane runs.
test("the release resolver reaches GitHub authenticated", () => {
  const workflow = parseYaml(readFileSync(join(repositoryRoot, ".github/workflows/recipes-pages.yaml"), "utf8"));
  const job = workflow.jobs.build;
  const runText = (step) => String(step.run ?? "");
  // The pull-request lane resolves from committed fixtures and never calls the API, so
  // the trunk lane is exactly the step that runs the resolver without --fixture.
  const trunk = job.steps.filter((step) => runText(step).includes("resolve-releases.mjs") && !runText(step).includes("--fixture"));
  assert.equal(trunk.length, 1, "exactly one step resolves the published releases from GitHub");
  const token = trunk[0].env?.GH_TOKEN ?? job.env?.GH_TOKEN;
  assert.match(String(token), /\$\{\{\s*github\.token\s*\}\}/, "gh needs the run's token to call the API on a runner");
  // The token is only useful with the permission that authorises the listing, so a
  // step that is authenticated but unauthorised is the same red build.
  assert.equal(job.permissions?.["contents"], "read", "listing releases needs contents:read");
});
