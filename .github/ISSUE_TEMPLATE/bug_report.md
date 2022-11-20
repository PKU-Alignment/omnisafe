---
name: Bug Report
about: Create a report to help us improve
title: "[BUG]"
labels: ["bug"]
assignees: zmsn-2077

---

## Describe the bug

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior.

Please try to provide a minimal example to reproduce the bug. Error messages and stack traces are also helpful.

Please use the markdown code blocks for both code and stack traces.

```python
import omnisafe
```

```pytb
Traceback (most recent call last):
  File ...
```

## Expected behavior

A clear and concise description of what you expected to happen.

## Screenshots

If applicable, add screenshots to help explain your problem.

## System info

Describe the characteristic of your environment:

- Describe how the library was installed (pip, source, ...)
- Python version
- Versions of any other relevant libraries

```python
import omnisafe, sys
print(omnisafe.__version__, sys.version, sys.platform)
```

## Additional context

Add any other context about the problem here.

## Reason and Possible fixes

If you know or suspect the reason for this bug, paste the code lines and suggest modifications.

## Checklist

- [ ] I have checked that there is no similar issue in the repo. (**required**)
- [ ] I have read the [documentation](https://omnisafe.readthedocs.io). (**required**)
- [ ] I have provided a minimal working example to reproduce the bug. (**required**)
