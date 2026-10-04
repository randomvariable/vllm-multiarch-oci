// The URL is the flow's state. A shared link is only trustworthy if the query
// string round-trips exactly, and a value the reader set back to its default must
// not be reported as a change, because the manifest would otherwise carry a flag
// the accepted recipe never had.

import assert from "node:assert/strict";
import test from "node:test";

// The module imports the renderer and the browser resolver at load time, so the
// globals those expect have to exist before the import resolves.
globalThis.window = { location: { pathname: "/vllm-multiarch-oci/", hash: "" } };
const { changesFor, payload, readState } = await import("../src/components/deployment-flow.js");

const record = {
  settings: {
    "max-num-seqs": { value: 8, source: "recipe:qwen38-flash-next-gb10-tp2" },
    "gpu-memory-utilization": { value: 0.74, source: "model:qwen38-flash-next" },
  },
  environment: {
    NCCL_NET: { value: "IB", source: "hardware:gb10-roce" },
    HF_HOME: { value: "/models", source: "recipe:qwen38-flash-next-gb10-tp2" },
  },
};

function search(state) {
  return new URL(payload(state), "https://example.test").search;
}

test("a state round-trips through the query string", () => {
  const state = {
    build: "v20261003.1",
    model: "recipe:qwen38-flash-next-gb10-tp2",
    target: "compose",
    // Values chosen for what breaks a naive encoding: an endpoint carries : * /,
    // a JSON object carries braces, quotes and commas, and one has a space.
    settings: { "max-num-seqs": "16", "default-chat-template-kwargs": '{"thinking":true}' },
    environment: { NCCL_NET: "Socket", CUDA_LAUNCH_BLOCKING: "1" },
    parameters: { hca: "rocep1s0f1,roceP2p1s0f1", topology_values: "server17 server19" },
    explain: true,
    advanced: true,
  };
  assert.deepEqual(readState(search(state)), state);
});

test("defaults are left out of the URL", () => {
  const url = new URL(
    payload({ build: "nightly", model: "recipe:x", target: "lws", settings: {}, environment: {}, parameters: {} }),
    "https://example.test",
  );
  assert.equal(url.searchParams.get("build"), null, "nightly is the default channel");
  assert.equal(url.searchParams.get("target"), null, "lws is the default target");
  assert.equal(url.searchParams.get("model"), "recipe:x");
  const read = readState(url.searchParams.toString());
  assert.deepEqual([read.settings, read.environment, read.parameters], [{}, {}, {}]);
  assert.equal(read.explain, false);
  assert.equal(read.advanced, false);
});

test("a value set back to its default is not a change", () => {
  // The reader typed the same thing the recipe already had. Emitting `--flag
  // value` here would put a flag in the manifest that the accepted command does
  // not contain, and "Your changes" would report an edit that did not happen.
  assert.deepEqual(changesFor(record, { "max-num-seqs": "8" }, {}), []);
  assert.deepEqual(changesFor(record, {}, { NCCL_NET: "IB" }), []);
  assert.deepEqual(changesFor(record, {}, {}), []);
  // The number and its text spelling are one value, because a query string can
  // only ever carry text.
  assert.deepEqual(changesFor(record, { "gpu-memory-utilization": "0.74" }, {}), []);
});

test("a differing or new value is a change", () => {
  assert.deepEqual(changesFor(record, { "max-num-seqs": "16" }, {}), [
    { kind: "setting", name: "max-num-seqs", value: "16" },
  ]);
  assert.deepEqual(changesFor(record, {}, { OMP_NUM_THREADS: "2" }), [
    { kind: "environment", name: "OMP_NUM_THREADS", value: "2" },
  ]);
});

test("no record yields no changes instead of an error", () => {
  assert.deepEqual(changesFor(null, { "max-num-seqs": "16" }, { NCCL_NET: "Socket" }), []);
});
