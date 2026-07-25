# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.0.1.x | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities.
2. **Email**: Send details to [juliendelaunay35000@gmail.com](mailto:juliendelaunay35000@gmail.com)
3. **Include**: A description of the vulnerability, steps to reproduce, and potential impact.

We will acknowledge your report within 48 hours and aim to provide a fix within 7 days for critical issues.

## Security Measures

### Pickle Deserialization

Model files (`.pkl`) are loaded using a **restricted unpickler** that only allows classes from `sklearn`, `numpy`, and standard library modules. This prevents arbitrary code execution from untrusted pickle files.

**Recommendation**: Only load model files from trusted sources.

### HTML Output (XSS Prevention)

All dynamic data inserted into HTML visualizations is serialized through `json.dumps()`, which properly escapes special characters and prevents cross-site scripting (XSS) attacks.

### No Runtime JavaScript Dependencies

The JavaScript visualization bundle (`bundle.js`) is pre-built and committed to the repository. There are no npm runtime dependencies, eliminating supply-chain risks from the JavaScript ecosystem.

### File Handling

All file operations use context managers (`with` statements) to ensure proper resource cleanup and prevent resource leaks.

## Dependencies

This project depends on well-maintained scientific Python libraries:

- **numpy** — Numerical computing
- **scipy** — Scientific computing
- **scikit-learn** — Machine learning
- **spaCy** — Natural language processing
- **pandas** — Data manipulation
- **matplotlib** — Visualization
- **nltk** — Text processing

We recommend keeping all dependencies up to date. Use `pip audit` or similar tools to check for known vulnerabilities in your dependency tree.
