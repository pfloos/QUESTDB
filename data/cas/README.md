## ✨ Active Space Information

This directory provides detailed information on the **active spaces** used in each CASSCF calculation. For every molecular system and transition, it includes a description of the active space configuration across symmetry representations.

---

### 🎯 Why Active Spaces Matter

In CASSCF-based methods, selecting an appropriate active space is **crucial**. It ensures that the most relevant electron correlation effects are captured while keeping computational demands within reasonable bounds. Designing a good active space requires a careful balance:

- ✅ **Include enough orbitals** to represent essential correlation effects
- ⚖️ **Avoid unnecessary orbitals** that increase the computational cost without added value

---

### 🧩 What's Included

For each system and transition, the following information is provided:

- The **number of active orbitals** used per **irreducible representation**
- Details on the **state-averaging procedure**, specifying the number of averaged states per irreducible representation
- A consistent inclusion of the **ground state** in the state-averaging procedure, even if it belongs to a different symmetry representation than the excited states

This documentation aims to provide full transparency and reproducibility for the active space choices in all multiconfigurational calculations.

---

### 📁 Repository Structure

```bash
cas/
├── docx/       # Human-readable active space info (.docx format)
├── json/       # JSON-formatted versions of the DOCX files
│   └── docx2json.py  # Script to convert DOCX files into JSON
├── out/        # Compressed CASSCF/MRPT output files (.tar.gz)
└── README.md   # This file
```

---

### 📄 File Naming Convention

All files follow a \`{molecule}_mrpt\` prefix convention to ensure consistency across \`.docx\`, \`.json\`, and \`.tar.gz\` outputs. For example:

- \`ch2o+_mrpt.docx\` → Active space description
- \`ch2o+_mrpt.json\` → Parsed JSON version of the above
- \`ch2o+_mrpt_output.tar.gz\` → Raw calculation outputs

---

### 📌 Notes

- All \`.docx\` files are manually curated and serve as the primary source of information.
- The \`.json\` files are intended for programmatic use (e.g., in workflows or pipelines).
- The \`.tar.gz\` files contain raw MRPT outputs, suitable for reference or re-analysis.
