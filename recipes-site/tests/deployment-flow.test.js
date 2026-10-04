// The deployment flow's state lives in the URL, so these tests check the two
// properties that make a shared link trustworthy: a state survives a round trip
// through the query string unchanged, and a value the reader set back to its
// default is not reported as a change. Neither depends on the DOM, and the flow
// module is imported with the browser globals it needs defined first.

import assert from "node:assert/strict";
import test from "node:test";

globalThis.window = { location: { pathname: "/vllm-multiarch-oci/", hash: "" } };
globalThis.document = { createElement: () => ({}) };

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

test("a state round-trips through the query string", () => {
  const state = {
    build: "v20261003.1",
    model: "recipe:qwen38-flash-next-gb10-tp2",
    target: "compose",
    settings: { "max-num-seqs": "16", "enable-prefix-caching": "false" },
    environment: { NCCL_NET: "Socket" },
    parameters: {
      namespace: "llm-d",
      // A KV-event endpoint and a JSON blob are exactly the values that break a
      // naive encoding: they carry :, /, {, }, quotes and commas.
      "kv-events-endpoint": "tcp://*:5556",
      "speculative-config": '{"method":"mtp","num_speculative_tokens":3}',
    },
    explain: true,
    advanced: true,
  };
  assert.deepEqual(readState(new URL(payload(state), "https://example.test").search), state);
});

test("defaults are left out of the URL and the flags come back", () => {
  const url = new URL(
    payload({ build: "nightly", model: "recipe:x", target: "lws", settings: {}, environment: {}, parameters: {} }),
    "https://example.test",
  );
  assert.equal(url.searchParams.get("build"), null, "nightly is the default channel");
  assert.equal(url.searchParams.get("target"), null, "lws is the default target");
  assert.equal(url.searchParams.get("model"), "recipe:x");
  assert.deepEqual(readState(url.searchParams.toString()).settings, {});
});

test("a value set back to its default is not a change", () => {
  assert.deepEqual(changesFor(record, { "max-num-seqs": "8" }, {}), []);
  assert.deepEqual(changesFor(record, {}, { NCCL_NET: "IB" }), []);
  assert.deepEqual(changesFor(record, {}, {}), []);
});

test("a differing or new value is a change, and the record decides", () => {
  assert.deepEqual(changesFor(record, { "max-num-seqs": "16" }, {}), [
    { kind: "setting", name: "max-num-seqs", value: "16" },
  ]);
  assert.deepEqual(changesFor(record, {}, { OMP_NUM_THREADS: "2" }), [
    { kind: "environment", name: "OMP_NUM_THREADS", value: "2" },
  ]);
  // A number and its string spelling are the same value: the resolver compares
  // types loosely here because the URL can only ever carry text.
  assert.deepEqual(changesFor(record, { "gpu-memory-utilization": "0.74" }, {}), []);
});

test("no record means no changes rather than an error", () => {
  assert.deepEqual(changesFor(null, { "max-num-seqs": "16" }, { NCCL_NET: "Socket" }), []);
});
