// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

// Fail closed: an older green run must not mask a failed or pending newer run.
module.exports = async function verifyPackageChecks({ github, context }) {
  const pr = context.payload.pull_request;
  if (context.eventName !== 'pull_request' || context.payload.action !== 'closed' ||
      !pr?.merged || pr.base.ref !== 'develop' || !pr.merge_commit_sha) {
    throw new Error('DEB packaging requires a PR merged into develop.');
  }
  const checks = [];
  for (const workflow of ['quality-gate.yml', 'rocshmem-ci.yml']) {
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...context.repo, workflow_id: workflow, event: 'pull_request',
      head_sha: pr.head.sha, per_page: 100,
    });
    const matching = runs.filter(run =>
      run.event === 'pull_request' && run.head_sha === pr.head.sha &&
      run.pull_requests?.some(item => item.number === pr.number));
    matching.sort((a, b) => b.id - a.id);
    const run = matching[0];
    if (!run || run.status !== 'completed' || run.conclusion !== 'success') {
      throw new Error(`${workflow}: latest run for PR #${pr.number} at ${pr.head.sha} ` +
        `must succeed; got ${run ? `${run.status}/${run.conclusion}: ${run.html_url}` : 'no matching run'}. ` +
        'Inspect the PR checks; do not bypass this guard.');
    }
    // A workflow can be green when every job is skipped. Require the actual
    // aggregate quality check / Standard test job in the latest run attempt.
    const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
      ...context.repo, run_id: run.id, filter: 'latest', per_page: 100,
    });
    const expectedName = workflow === 'quality-gate.yml'
      ? /(^|\/ )All required checks$/ : /^Test \(standard\)$/;
    if (!jobs.some(job => expectedName.test(job.name) &&
        job.status === 'completed' && job.conclusion === 'success')) {
      throw new Error(`${workflow}: required successful job missing in latest attempt: ${run.html_url}`);
    }
    checks.push({ workflow, id: run.id, attempt: run.run_attempt,
      url: run.html_url, head_sha: run.head_sha, conclusion: run.conclusion });
  }
  return { pr: pr.number, head_sha: pr.head.sha, merge_sha: pr.merge_commit_sha, checks };
};
