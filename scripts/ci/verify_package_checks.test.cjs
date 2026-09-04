// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

const { test } = require('node:test');
const assert = require('node:assert/strict');
const verify = require('./verify_package_checks.cjs');

function fixture() {
  const context = {
    eventName: 'pull_request', repo: { owner: 'HYGON-AI', repo: 'rocSHMEM-das' },
    payload: { action: 'closed', pull_request: {
      number: 7, merged: true, base: { ref: 'develop' },
      head: { sha: 'a'.repeat(40) }, merge_commit_sha: 'b'.repeat(40),
    } },
  };
  const runs = {};
  const jobs = {};
  for (const [index, workflow] of ['quality-gate.yml', 'rocshmem-ci.yml'].entries()) {
    runs[workflow] = [{ id: index + 1, event: 'pull_request', head_sha: 'a'.repeat(40),
      pull_requests: [{ number: 7 }], status: 'completed', conclusion: 'success',
      run_attempt: 2, html_url: `https://example.test/runs/${index + 1}` }];
    jobs[index + 1] = [{ status: 'completed', conclusion: 'success',
      name: index === 0 ? 'Checks / All required checks' : 'Test (standard)' }];
  }
  const calls = [];
  const github = {
    rest: { actions: { listWorkflowRuns: 'runs', listJobsForWorkflowRun: 'jobs' } },
    paginate: async (method, args) => {
      calls.push({ method, args });
      return method === 'runs' ? runs[args.workflow_id] : jobs[args.run_id];
    },
  };
  return { context, github, runs, jobs, calls };
}

test('merged PR with successful latest checks records both exact SHAs', async () => {
  const f = fixture();
  const result = await verify(f);
  assert.equal(result.merge_sha, 'b'.repeat(40));
  assert.equal(result.head_sha, 'a'.repeat(40));
  assert.equal(result.checks.length, 2);
  assert.equal(result.checks[0].attempt, 2);
  assert.equal(f.calls[0].args.head_sha, 'a'.repeat(40));
  assert.equal(f.calls[1].args.filter, 'latest');
});

for (const [name, change] of [
  ['closed without merge', f => { f.context.payload.pull_request.merged = false; }],
  ['PR opened', f => { f.context.payload.action = 'opened'; }],
  ['PR synchronize', f => { f.context.payload.action = 'synchronize'; }],
  ['push event', f => { f.context.eventName = 'push'; }],
  ['other target branch', f => { f.context.payload.pull_request.base.ref = 'main'; }],
]) {
  test(`rejects ${name} without querying checks`, async () => {
    const f = fixture(); change(f);
    await assert.rejects(verify(f), /requires a PR merged/);
    assert.equal(f.calls.length, 0);
  });
}

for (const status of ['failure', 'cancelled', 'skipped', null]) {
  test(`rejects latest ${status} run despite an older green run`, async () => {
    const f = fixture();
    const list = f.runs['rocshmem-ci.yml'];
    list.push({ ...list[0], id: 9, conclusion: status,
      status: status === null ? 'in_progress' : 'completed' });
    await assert.rejects(verify(f), /latest run.*must succeed/);
  });
}

test('does not accept runs from another PR or another source commit', async () => {
  const f = fixture();
  const original = f.runs['quality-gate.yml'][0];
  f.runs['quality-gate.yml'] = [
    { ...original, pull_requests: [{ number: 8 }] },
    { ...original, head_sha: 'c'.repeat(40) },
  ];
  await assert.rejects(verify(f), /no matching run/);
});

test('a green workflow containing a skipped Standard job is not sufficient', async () => {
  const f = fixture();
  f.jobs[2][0].conclusion = 'skipped';
  await assert.rejects(verify(f), /required successful job missing/);
});

test('a green full/manual test is not a substitute for PR Standard', async () => {
  const f = fixture();
  f.jobs[2][0].name = 'Test (full)';
  await assert.rejects(verify(f), /required successful job missing/);
});

test('API errors fail closed', async () => {
  const f = fixture();
  f.github.paginate = async () => { throw new Error('403'); };
  await assert.rejects(verify(f), /403/);
});
