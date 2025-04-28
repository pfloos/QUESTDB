# 🧩 Geometries

Welcome to the **Geometries** section of the QUEST Database!

This directory provides the **molecular structures** used in QUESTDB in two convenient formats, designed for both **publication** and **computational use**:

---

## 📄 Contents

- 📚 **TeX Format** (`TeX/`)  
  Contains a `.tex` source file along with its compiled `.pdf`, presenting the **Cartesian coordinates** of all molecules in a **clean, citable format**.  
  Ideal for inclusion in scientific articles and reports.

- 🧪 **XYZ Format** (`xyz/`)  
  Includes individual `.xyz` files for **each molecule**, ready for **direct input** into computational chemistry software (e.g., Gaussian, ORCA, Q-Chem).

---

## 📏 Units

> All Cartesian coordinates are expressed in **atomic units (bohr)**.

Please **convert to Ångströms** if needed, by applying the standard conversion factor:  
**1 bohr ≈ 0.529177 Å**.

---

## 📢 Important Notes

- All geometries have been **carefully curated** to ensure **consistency** and **reproducibility**.
- For benchmarking, validation, or method development, we **strongly recommend** using these geometries **without modification**.

---

## 🚀 Quick Start

```bash
# Navigate to xyz directory
cd xyz/

# View a specific molecule
cat molecule_name.xyz
