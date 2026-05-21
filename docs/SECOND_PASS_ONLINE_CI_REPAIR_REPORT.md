# Second-Pass Online CI Repair Report

Repository: OMNIAMIND

Timestamp UTC: 2026-05-21T18:58:19Z

Purpose:
Repair remaining red GitHub Actions after first ecosystem CI repair.

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
      "id": 26246592306,
      "name": "CI",
      "status": "completed",
      "conclusion": "failure",
      "html_url": "https://github.com/Tuttotorna/OMNIAMIND/actions/runs/26246592306",
      "created_at": "2026-05-21T18:54:05Z",
      "updated_at": "2026-05-21T18:54:29Z",
      "head_sha": "94d149f225db13bef2d13ff3e6ccbc7c1938b049"
    }
  ]
}

Failed online log samples before repair:
{
  "ok": true,
  "failed_runs": [
    {
      "id": 26246592306,
      "name": "CI",
      "status": "completed",
      "conclusion": "failure",
      "html_url": "https://github.com/Tuttotorna/OMNIAMIND/actions/runs/26246592306",
      "created_at": "2026-05-21T18:54:05Z",
      "updated_at": "2026-05-21T18:54:29Z",
      "head_sha": "94d149f225db13bef2d13ff3e6ccbc7c1938b049"
    }
  ],
  "samples": [
    {
      "run_id": 26246592306,
      "download_ok": true,
      "html_url": "https://github.com/Tuttotorna/OMNIAMIND/actions/runs/26246592306",
      "samples": [
        {
          "file": "0_test _ python-3.11.txt",
          "lines": [
            "2026-05-21T18:54:12.9022578Z \u001b[36;1mpython -m pip install pytest numpy matplotlib jsonschema\u001b[0m",
            "2026-05-21T18:54:14.3996489Z Collecting pytest",
            "2026-05-21T18:54:14.5048832Z   Downloading pytest-9.0.3-py3-none-any.whl.metadata (7.6 kB)",
            "2026-05-21T18:54:15.0406576Z Collecting iniconfig>=1.0.1 (from pytest)",
            "2026-05-21T18:54:15.0905720Z Collecting packaging>=22 (from pytest)",
            "2026-05-21T18:54:15.1385434Z Collecting pluggy<2,>=1.5 (from pytest)",
            "2026-05-21T18:54:15.1922957Z Collecting pygments>=2.7.2 (from pytest)",
            "2026-05-21T18:54:16.5216810Z Downloading pytest-9.0.3-py3-none-any.whl (375 kB)",
            "2026-05-21T18:54:17.3766933Z Installing collected packages: typing-extensions, six, rpds-py, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, attrs, referencing, python-dateutil, pytest, contourpy, matplotlib, jsonschema-specifications, jsonschema",
            "2026-05-21T18:54:22.9732942Z Successfully installed attrs-26.1.0 contourpy-1.3.3 cycler-0.12.1 fonttools-4.63.0 iniconfig-2.3.0 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 kiwisolver-1.5.0 matplotlib-3.10.9 numpy-2.4.6 packaging-26.2 pillow-12.2.0 pluggy-1.6.0 pygments-2.20.0 pyparsing-3.3.2 pytest-9.0.3 python-dateutil-2.9.0.post0 referencing-0.37.0 rpds-py-0.30.0 six-1.17.0 typing-extensions-4.15.0",
            "2026-05-21T18:54:25.4392178Z \u001b[36;1m  python -m pytest -q\u001b[0m",
            "2026-05-21T18:54:25.9953673Z ==================================== ERRORS ====================================",
            "2026-05-21T18:54:25.9954349Z _____________ ERROR collecting tests/test_backbone_orchestrator.py _____________",
            "2026-05-21T18:54:25.9955687Z ImportError while importing test module '/home/runner/work/OMNIAMIND/OMNIAMIND/tests/test_backbone_orchestrator.py'.",
            "2026-05-21T18:54:25.9956990Z Traceback:",
            "2026-05-21T18:54:25.9962773Z E   ModuleNotFoundError: No module named 'omnia'",
            "2026-05-21T18:54:25.9964202Z ERROR tests/test_backbone_orchestrator.py",
            "2026-05-21T18:54:25.9965280Z !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!",
            "2026-05-21T18:54:25.9966032Z 1 error in 0.11s",
            "2026-05-21T18:54:26.0153045Z ##[error]Process completed with exit code 2."
          ]
        },
        {
          "file": "1_test _ python-3.12.txt",
          "lines": [
            "2026-05-21T18:54:12.0813832Z \u001b[36;1mpython -m pip install pytest numpy matplotlib jsonschema\u001b[0m",
            "2026-05-21T18:54:14.0673741Z Collecting pytest",
            "2026-05-21T18:54:14.1426743Z   Downloading pytest-9.0.3-py3-none-any.whl.metadata (7.6 kB)",
            "2026-05-21T18:54:14.5856186Z Collecting iniconfig>=1.0.1 (from pytest)",
            "2026-05-21T18:54:14.6143219Z Collecting packaging>=22 (from pytest)",
            "2026-05-21T18:54:14.6407850Z Collecting pluggy<2,>=1.5 (from pytest)",
            "2026-05-21T18:54:14.6738059Z Collecting pygments>=2.7.2 (from pytest)",
            "2026-05-21T18:54:15.6765435Z Downloading pytest-9.0.3-py3-none-any.whl (375 kB)",
            "2026-05-21T18:54:16.6352883Z Installing collected packages: typing-extensions, six, rpds-py, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, attrs, referencing, python-dateutil, pytest, contourpy, matplotlib, jsonschema-specifications, jsonschema",
            "2026-05-21T18:54:21.9844673Z Successfully installed attrs-26.1.0 contourpy-1.3.3 cycler-0.12.1 fonttools-4.63.0 iniconfig-2.3.0 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 kiwisolver-1.5.0 matplotlib-3.10.9 numpy-2.4.6 packaging-26.2 pillow-12.2.0 pluggy-1.6.0 pygments-2.20.0 pyparsing-3.3.2 pytest-9.0.3 python-dateutil-2.9.0.post0 referencing-0.37.0 rpds-py-0.30.0 six-1.17.0 typing-extensions-4.15.0",
            "2026-05-21T18:54:24.2356027Z \u001b[36;1m  python -m pytest -q\u001b[0m",
            "2026-05-21T18:54:24.7254427Z ==================================== ERRORS ====================================",
            "2026-05-21T18:54:24.7254894Z _____________ ERROR collecting tests/test_backbone_orchestrator.py _____________",
            "2026-05-21T18:54:24.7255517Z ImportError while importing test module '/home/runner/work/OMNIAMIND/OMNIAMIND/tests/test_backbone_orchestrator.py'.",
            "2026-05-21T18:54:24.7256417Z Traceback:",
            "2026-05-21T18:54:24.7259822Z E   ModuleNotFoundError: No module named 'omnia'",
            "2026-05-21T18:54:24.7260888Z ERROR tests/test_backbone_orchestrator.py",
            "2026-05-21T18:54:24.7261212Z !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!",
            "2026-05-21T18:54:24.7261537Z 1 error in 0.11s",
            "2026-05-21T18:54:24.7499132Z ##[error]Process completed with exit code 2."
          ]
        },
        {
          "file": "test _ python-3.11/4_Install tooling.txt",
          "lines": [
            "2026-05-21T18:54:12.9022572Z \u001b[36;1mpython -m pip install pytest numpy matplotlib jsonschema\u001b[0m",
            "2026-05-21T18:54:14.3996412Z Collecting pytest",
            "2026-05-21T18:54:14.5048765Z   Downloading pytest-9.0.3-py3-none-any.whl.metadata (7.6 kB)",
            "2026-05-21T18:54:15.0406403Z Collecting iniconfig>=1.0.1 (from pytest)",
            "2026-05-21T18:54:15.0905683Z Collecting packaging>=22 (from pytest)",
            "2026-05-21T18:54:15.1385385Z Collecting pluggy<2,>=1.5 (from pytest)",
            "2026-05-21T18:54:15.1922914Z Collecting pygments>=2.7.2 (from pytest)",
            "2026-05-21T18:54:16.5216777Z Downloading pytest-9.0.3-py3-none-any.whl (375 kB)",
            "2026-05-21T18:54:17.3766878Z Installing collected packages: typing-extensions, six, rpds-py, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, attrs, referencing, python-dateutil, pytest, contourpy, matplotlib, jsonschema-specifications, jsonschema",
            "2026-05-21T18:54:22.9732623Z Successfully installed attrs-26.1.0 contourpy-1.3.3 cycler-0.12.1 fonttools-4.63.0 iniconfig-2.3.0 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 kiwisolver-1.5.0 matplotlib-3.10.9 numpy-2.4.6 packaging-26.2 pillow-12.2.0 pluggy-1.6.0 pygments-2.20.0 pyparsing-3.3.2 pytest-9.0.3 python-dateutil-2.9.0.post0 referencing-0.37.0 rpds-py-0.30.0 six-1.17.0 typing-extensions-4.15.0"
          ]
        },
        {
          "file": "test _ python-3.11/6_Run tests when tests exist.txt",
          "lines": [
            "2026-05-21T18:54:25.4392176Z \u001b[36;1m  python -m pytest -q\u001b[0m",
            "2026-05-21T18:54:25.9953668Z ==================================== ERRORS ====================================",
            "2026-05-21T18:54:25.9954346Z _____________ ERROR collecting tests/test_backbone_orchestrator.py _____________",
            "2026-05-21T18:54:25.9955682Z ImportError while importing test module '/home/runner/work/OMNIAMIND/OMNIAMIND/tests/test_backbone_orchestrator.py'.",
            "2026-05-21T18:54:25.9956987Z Traceback:",
            "2026-05-21T18:54:25.9962767Z E   ModuleNotFoundError: No module named 'omnia'",
            "2026-05-21T18:54:25.9964198Z ERROR tests/test_backbone_orchestrator.py",
            "2026-05-21T18:54:25.9965266Z !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!",
            "2026-05-21T18:54:25.9966027Z 1 error in 0.11s",
            "2026-05-21T18:54:26.0153028Z ##[error]Process completed with exit code 2."
          ]
        },
        {
          "file": "test _ python-3.12/4_Install tooling.txt",
          "lines": [
            "2026-05-21T18:54:12.0813828Z \u001b[36;1mpython -m pip install pytest numpy matplotlib jsonschema\u001b[0m",
            "2026-05-21T18:54:14.0673651Z Collecting pytest",
            "2026-05-21T18:54:14.1426666Z   Downloading pytest-9.0.3-py3-none-any.whl.metadata (7.6 kB)",
            "2026-05-21T18:54:14.5856144Z Collecting iniconfig>=1.0.1 (from pytest)",
            "2026-05-21T18:54:14.6143183Z Collecting packaging>=22 (from pytest)",
            "2026-05-21T18:54:14.6407800Z Collecting pluggy<2,>=1.5 (from pytest)",
            "2026-05-21T18:54:14.6738003Z Collecting pygments>=2.7.2 (from pytest)",
            "2026-05-21T18:54:15.6765415Z Downloading pytest-9.0.3-py3-none-any.whl (375 kB)",
            "2026-05-21T18:54:16.6352843Z Installing collected packages: typing-extensions, six, rpds-py, pyparsing, pygments, pluggy, pillow, packaging, numpy, kiwisolver, iniconfig, fonttools, cycler, attrs, referencing, python-dateutil, pytest, contourpy, matplotlib, jsonschema-specifications, jsonschema",
            "2026-05-21T18:54:21.9844296Z Successfully installed attrs-26.1.0 contourpy-1.3.3 cycler-0.12.1 fonttools-4.63.0 iniconfig-2.3.0 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 kiwisolver-1.5.0 matplotlib-3.10.9 numpy-2.4.6 packaging-26.2 pillow-12.2.0 pluggy-1.6.0 pygments-2.20.0 pyparsing-3.3.2 pytest-9.0.3 python-dateutil-2.9.0.post0 referencing-0.37.0 rpds-py-0.30.0 six-1.17.0 typing-extensions-4.15.0"
          ]
        },
        {
          "file": "test _ python-3.12/6_Run tests when tests exist.txt",
          "lines": [
            "2026-05-21T18:54:24.2356025Z \u001b[36;1m  python -m pytest -q\u001b[0m",
            "2026-05-21T18:54:24.7254421Z ==================================== ERRORS ====================================",
            "2026-05-21T18:54:24.7254891Z _____________ ERROR collecting tests/test_backbone_orchestrator.py _____________",
            "2026-05-21T18:54:24.7255513Z ImportError while importing test module '/home/runner/work/OMNIAMIND/OMNIAMIND/tests/test_backbone_orchestrator.py'.",
            "2026-05-21T18:54:24.7256414Z Traceback:",
            "2026-05-21T18:54:24.7259819Z E   ModuleNotFoundError: No module named 'omnia'",
            "2026-05-21T18:54:24.7260884Z ERROR tests/test_backbone_orchestrator.py",
            "2026-05-21T18:54:24.7261210Z !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!",
            "2026-05-21T18:54:24.7261535Z 1 error in 0.11s",
            "2026-05-21T18:54:24.7499102Z ##[error]Process completed with exit code 2."
          ]
        }
      ]
    }
  ]
}

Patch:
{
  "ci_changed": true,
  "legacy_non_dot_github_removed": [],
  "duplicate_test_workflows_removed": [],
  "python_version_policy": "3.12 only",
  "omnia_required_doi_command_present": null
}

Local tests:
{
  "status": "pass",
  "passed": 2,
  "failed": 0,
  "errors": 0,
  "returncode": 0,
  "summary": "2 passed in 2.62s"
}

Push:
null

After online check:
null
