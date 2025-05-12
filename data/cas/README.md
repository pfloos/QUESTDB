## ✨ Active Space Information

This directory provides detailed information on the **active spaces** used in each CASSCF calculation. For every molecular system and transition, it includes a description of the active space configuration across symmetry representations.

### 🎯 Why Active Spaces Matter

In CASSCF-based methods, selecting an appropriate active space is **crucial**. It ensures that the most relevant electron correlation effects are captured while keeping computational demands within reasonable bounds. Designing a good active space requires a careful balance:

- ✅ **Include enough orbitals** to represent essential correlation effects  
- ⚖️ **Avoid unnecessary orbitals** that increase the computational cost without added value

### 🧩 What's Included

For each system and transition, the following information is provided:

- The **number of active orbitals** used per **irreducible representation**
- Details on the **state-averaging procedure**, specifying the number of averaged states per irreducible representation
- A consistent inclusion of the **ground state** in the state-averaging procedure, even if it belongs to a different symmetry representation than the excited states

This documentation aims to provide full transparency and reproducibility for the active space choices in all multiconfigurational calculations.
