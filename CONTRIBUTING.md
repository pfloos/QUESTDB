# 🌟 Contributing to QUESTDB

Welcome, and thank you for your interest in contributing to **QUESTDB** — an open resource for quantum and electronic structure theory benchmarks. Whether you're fixing a typo, adding new benchmark data, or proposing new features, we’re excited to have you here!

---

## 🧭 Table of Contents

- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Code Guidelines](#code-guidelines)
- [Data Contributions](#data-contributions)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Community Standards](#community-standards)

---

## 🚀 Getting Started

1. **Fork the repository** to your GitHub account.
2. Clone your forked copy locally:
   ```bash
   git clone https://github.com/your-username/QUESTDB.git
   ```
3. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes, commit, and push.
5. Open a pull request (PR) to the `main` or `dev` branch.

---

## 🛠️ Ways to Contribute

- 📦 Add new molecule entries or quantum chemistry datasets  
- 🐛 Fix bugs in data processing or scripts  
- 🧪 Add or improve test coverage  
- 📚 Improve documentation or fix typos  
- 🌐 Translate documentation (future feature)  
- 💡 Propose new benchmark protocols  

---

## 🧑‍💻 Code Guidelines

- Write **clean, readable** Python (PEP8 preferred).
- Place new scripts in `scripts/` or `tools/` folders as appropriate.
- Add **docstrings** to all functions and modules.
- Include **unit tests** where applicable (in the `tests/` directory).
- Use relative imports and avoid hardcoding paths.

---

## 📊 Data Contributions

If you're contributing new benchmark data:

- Include a short metadata file (`.json`, `.yaml`, or `.md`) describing:
  - Molecule/system name
  - Method and basis set
  - Source/reference (DOI, paper)
- Prefer CSV or JSON formats for raw data.
- Follow the folder structure under `data/` and `benchmarks/`.

---

## 📝 Commit Messages

Use clear, descriptive commit messages. Follow this convention:

```
type: short summary (max 72 characters)

More detailed explanation (optional).
```

**Examples:**

- `fix: correct dipole moment for H2O`  
- `feat: add CASSCF benchmark for benzene`  
- `docs: update README with citation info`  

---

## 🔀 Pull Request Process

1. Ensure your branch is up to date with `main`.
2. Run any relevant scripts or tests before submitting.
3. Keep pull requests focused on **one change or topic**.
4. Add reviewers if appropriate or tag maintainers.
5. PR titles should be meaningful and follow the commit format.

---

## 🐞 Reporting Issues

Please use the [Issues](https://github.com/your-org/QUESTDB/issues) tab for:

- Bug reports  
- Feature requests  
- Questions or discussion topics  

Include details, logs, and reproduction steps when applicable.

---

## 🤝 Community Standards

This project follows a [Code of Conduct](./CODE_OF_CONDUCT.md). Please be respectful and inclusive in all interactions. We welcome contributors of all backgrounds and experience levels.

---

Thank you for helping build and improve QUESTDB! 🧪🔬✨
