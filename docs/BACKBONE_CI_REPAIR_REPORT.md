# Backbone CI Repair Report

Repository: OMNIAMIND

Timestamp UTC: 2026-05-21T19:05:35Z

Purpose:
Repair remaining red GitHub Actions caused by missing online backbone package installation.

No release was created.
No tag was created.
Only CI workflow files and this repair report were changed.

Boundary:
measurement != inference != decision

Before:
{
  "green": false,
  "status": "failed",
  "reason": "At least one Actions run for current HEAD failed.",
  "runs": [
    {
      "id": 26246966527,
      "name": "CI",
      "status": "completed",
      "conclusion": "failure",
      "html_url": "https://github.com/Tuttotorna/OMNIAMIND/actions/runs/26246966527",
      "created_at": "2026-05-21T19:01:21Z",
      "updated_at": "2026-05-21T19:01:42Z",
      "head_sha": "3449cb757768ecd579b37b0d2944f91667d70d4a"
    }
  ]
}

Patch:
{
  "ci_changed": true,
  "legacy_non_dot_github_removed": [],
  "duplicate_test_workflows_removed": [],
  "python_version_policy": "3.12 only",
  "backbone_installs": {
    "OMNIA": true,
    "omnia-limit": true,
    "OMNIA-INVARIANCE": true
  },
  "required_omnia_doi_command_present": null
}

Local tests:
{
  "status": "pass",
  "passed": 2,
  "failed": 0,
  "errors": 0,
  "returncode": 0,
  "summary": "2 passed in 1.34s"
}

Push:
null

After online check:
null

After failed logs:
null
