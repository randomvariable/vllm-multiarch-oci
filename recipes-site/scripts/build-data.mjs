import { readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

const SITE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPOSITORY_ROOT = resolve(SITE_ROOT, "..");
const RECIPES_ROOT = join(REPOSITORY_ROOT, "recipes");
const PUBLIC_ROOT = join(SITE_ROOT, "public");
const COMMIT = /^[0-9a-f]{40}$/;
const DIGEST_REFERENCE = /^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?(?:\/[a-z0-9]+(?:[._-][a-z0-9]+)*)+@sha256:[0-9a-f]{64}$/;
const IMMUTABLE_TAG = /^vllmb12x-[a-z0-9][a-z0-9-]*-[0-9a-f]{12}-[0-9a-f]{12}-[0-9]{8}-n[1-9][0-9]*$/;
const TOP_LEVEL_KEYS = ["meta", "model", "runtime", "deployment", "validation", "guide"];
const PARAMETER_TYPES = new Set(["string", "integer", "stringMap"]);
const REQUIRED_DEPENDENCY_FIELDS = ["id", "name", "version", "source", "owner", "depends_on", "readiness", "tested"];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function recipeFiles(directory) {
  const paths = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) paths.push(...await recipeFiles(path));
    else if (entry.isFile() && /\.ya?ml$/.test(entry.name)) paths.push(path);
  }
  return paths.sort();
}

function validateRecipe(recipe, source) {
  assert(recipe && typeof recipe === "object" && !Array.isArray(recipe), `${source}: recipe must be a mapping`);
  assert(JSON.stringify(Object.keys(recipe)) === JSON.stringify(TOP_LEVEL_KEYS), `${source}: top-level keys must be exactly ${TOP_LEVEL_KEYS.join(", ")}`);
  assert(typeof recipe.meta.title === "string" && typeof recipe.meta.slug === "string" && typeof recipe.meta.description === "string", `${source}: meta fields are required`);
  assert(typeof recipe.model.model_id === "string" && COMMIT.test(recipe.model.revision) && typeof recipe.model.served_name === "string", `${source}: model id, full commit OID and served name are required`);
  assert(Array.isArray(recipe.runtime.command) && recipe.runtime.command.every((value) => typeof value === "string"), `${source}: runtime.command must be a string array`);
  assert(Array.isArray(recipe.runtime.base_args) && recipe.runtime.base_args.every((value) => typeof value === "string"), `${source}: runtime.base_args must be a string array`);
  assert(recipe.runtime.base_env && typeof recipe.runtime.base_env === "object" && !Array.isArray(recipe.runtime.base_env), `${source}: runtime.base_env must be a mapping`);
  assert(Object.values(recipe.runtime.base_env).every((value) => typeof value === "string"), `${source}: runtime.base_env values must be strings`);
  assert(recipe.deployment.nodes === 2 && recipe.deployment.tensor_parallel_size === 2, `${source}: seed topology must remain two-node TP=2`);
  assert(typeof recipe.deployment.model_path === "string" && recipe.deployment.model_path.startsWith("/"), `${source}: deployment.model_path must be absolute`);
  assert(Number.isSafeInteger(recipe.deployment.storage_min_free_gib) && recipe.deployment.storage_min_free_gib > 0, `${source}: storage minimum must be a positive integer`);
  assert(Number.isSafeInteger(recipe.deployment.download_workers) && recipe.deployment.download_workers > 0, `${source}: download workers must be a positive integer`);
  assert(Array.isArray(recipe.deployment.ignore_patterns) && recipe.deployment.ignore_patterns.every((value) => typeof value === "string"), `${source}: ignore patterns must be strings`);
  assert(recipe.deployment.parameters && typeof recipe.deployment.parameters === "object", `${source}: parameter mapping is required`);
  for (const [name, parameter] of Object.entries(recipe.deployment.parameters)) {
    assert(typeof parameter.label === "string" && PARAMETER_TYPES.has(parameter.type) && typeof parameter.required === "boolean" && Object.hasOwn(parameter, "default") && typeof parameter.description === "string", `${source}: malformed parameter ${name}`);
    if (parameter.type === "string") assert(typeof parameter.default === "string", `${source}: ${name} default must be a string`);
    if (parameter.type === "integer") assert(Number.isSafeInteger(parameter.default), `${source}: ${name} default must be an integer`);
    if (parameter.type === "stringMap") assert(parameter.default && typeof parameter.default === "object" && !Array.isArray(parameter.default) && Object.values(parameter.default).every((value) => typeof value === "string"), `${source}: ${name} default must be a string mapping`);
  }
  assert(DIGEST_REFERENCE.test(recipe.validation.image) && !recipe.validation.image.includes("internal.randomvariable"), `${source}: validation image must be a public digest-qualified reference`);
  assert(recipe.validation.lws === "verified" && recipe.validation.docker === "unverified" && typeof recipe.validation.evidence === "string", `${source}: validation status and evidence are required`);
  assert(!JSON.stringify(recipe).includes("harbor.services.home.internal"), `${source}: private registry leaked into public recipe`);
  return recipe;
}

function validateLatest(latest) {
  assert(latest && typeof latest === "object" && !Array.isArray(latest), "latest-image.json must be an object");
  assert(typeof latest.repository === "string" && !latest.repository.includes("@") && !latest.repository.includes("internal.randomvariable"), "latest image repository must be public and unqualified");
  assert(typeof latest.tag === "string" && IMMUTABLE_TAG.test(latest.tag), "latest image tag must follow the immutable vllmb12x publication contract");
  assert(typeof latest.reference === "string" && DIGEST_REFERENCE.test(latest.reference), "latest image reference must be digest-qualified");
  assert(latest.reference.startsWith(`${latest.repository}@sha256:`), "latest image reference must belong to repository");
  assert(typeof latest.resolved_at === "string" && !Number.isNaN(Date.parse(latest.resolved_at)), "latest image resolved_at must be an ISO timestamp");
  return latest;
}

function validateDependencies(data) {
  assert(data && typeof data === "object" && Array.isArray(data.dependencies), "platform-dependencies.yaml must contain dependencies");
  const ids = new Set();
  for (const dependency of data.dependencies) {
    for (const field of REQUIRED_DEPENDENCY_FIELDS) assert(Object.hasOwn(dependency, field), `dependency ${dependency.id || "<unknown>"} lacks ${field}`);
    assert(typeof dependency.id === "string" && !ids.has(dependency.id), `dependency id is missing or duplicated: ${dependency.id}`);
    ids.add(dependency.id);
    assert(typeof dependency.version === "string" && dependency.version.trim(), `dependency ${dependency.id} lacks a version pin`);
    assert(typeof dependency.source === "string" && /^https:\/\//.test(dependency.source), `dependency ${dependency.id} lacks an HTTPS source`);
    assert(typeof dependency.readiness === "string" && dependency.readiness.trim(), `dependency ${dependency.id} lacks a readiness command`);
    assert(Array.isArray(dependency.depends_on) && dependency.depends_on.every((value) => typeof value === "string"), `dependency ${dependency.id} has invalid edges`);
    assert(typeof dependency.tested === "boolean", `dependency ${dependency.id} must record tested status`);
  }
  for (const dependency of data.dependencies) for (const edge of dependency.depends_on) assert(ids.has(edge), `dependency ${dependency.id} references unknown dependency ${edge}`);
  return data;
}

async function main() {
  const recipes = [];
  for (const path of await recipeFiles(RECIPES_ROOT)) recipes.push(validateRecipe(parseYaml(await readFile(path, "utf8")), path));
  assert(recipes.length > 0, "no recipes were found");
  const slugs = recipes.map((recipe) => recipe.meta.slug);
  assert(new Set(slugs).size === slugs.length, "recipe slugs must be unique");
  validateLatest(JSON.parse(await readFile(join(PUBLIC_ROOT, "latest-image.json"), "utf8")));
  validateDependencies(parseYaml(await readFile(join(SITE_ROOT, "src/data/platform-dependencies.yaml"), "utf8")));
  await writeFile(join(PUBLIC_ROOT, "recipes.json"), `${JSON.stringify({ recipes }, null, 2)}\n`);
}

await main();
