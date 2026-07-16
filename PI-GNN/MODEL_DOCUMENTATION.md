# Physics-Informed Graph Neural Network (PI-GNN) for Storm Surge Modeling

## 1. Overview
This project replaces the computationally expensive ADCIRC+SWAN hydrodynamic simulator with a fast, Physics-Informed Spatio-Temporal Graph Neural Network (ST-GCRN). 

The model acts as a **pure surrogate solver**. It takes raw hurricane track data (`fort.22`) and the static coastal mesh (`fort.14`), internally calculates the wind forcing using an analytical parametric model, and predicts the physical storm surge hydrodynamics iteratively.

---

## 2. Input Data & Features
The model operates independently of ADCIRC's pre-computed meteorological grids (`fort.73/74`). The inputs are:
1. **Static Bathymetry (Depth):** Extracted from `fort.14`.
2. **Open Boundaries:** Parsed from `fort.14` (NETA/NOPE parameters) to identify the deep ocean forcing nodes.
3. **Storm Track:** Parsed from ATCF format (`fort.22`).

For every time step, the following features are dynamically calculated for every mesh node:
- **Depth ($h$)**
- **Atmospheric Pressure ($P$)**
- **Wind X-Velocity ($U$)**
- **Wind Y-Velocity ($V$)**

---

## 3. Parametric Wind Model (Holland 1980)
To enforce strict meteorological physics, the `process_adcirc.py` script analytically generates the $U, V, P$ fields using the Holland (1980) equations before feeding them to the GNN.

### A. Holland B-Parameter
$$ B = \frac{V_{max}^2 \cdot \rho_{air} \cdot e}{\Delta P} $$
Where:
* $V_{max}$ is the maximum sustained wind speed (from `fort.22`).
* $\Delta P = P_n - P_c$ (Ambient pressure minus Central Pressure).
* $\rho_{air}$ is air density (1.15 kg/m³).

### B. Pressure Drop Field
$$ P(r) = P_c + (P_n - P_c) e^{-(R_{max}/r)^B} $$
Where $r$ is the Haversine distance from the node to the storm's moving eye.

### C. Gradient Wind Velocity
$$ V_{gradient} = \sqrt{\frac{B}{\rho_{air}} \Delta P \left(\frac{R_{max}}{r}\right)^B e^{-(R_{max}/r)^B} + \left(\frac{r \cdot f}{2}\right)^2} - \frac{r \cdot |f|}{2} $$
Where $f$ is the Coriolis parameter ($2 \Omega \sin(\theta)$).

### D. Cartesian Wind Vectors ($U, V$)
An inflow angle ($\beta = 15^\circ$) is applied to simulate the cyclonic inward spiral of the hurricane.
$$ U_{wind} = -V_{gradient} \cdot \sin(\theta_{node} + \beta) $$
$$ V_{wind} = V_{gradient} \cdot \cos(\theta_{node} + \beta) $$

---

## 4. The Neural Network Architecture
* **Spatial Processing (GCN):** Graph Convolutional layers utilize the `fort.14` elemental connectivity to pass water between neighboring triangles.
* **Temporal Processing (GRU):** Gated Recurrent Units maintain the momentum of the moving storm surge over a sliding time window (e.g., 6 time steps).
* **Autoregressive Rollout:** The network predicts the water level $\zeta_{t+1}$, and feeds that prediction back into itself as the input for calculating $\zeta_{t+2}$, mimicking a time-stepping differential equation solver.

---

## 5. Physics-Informed Loss Functions
The model's weights are optimized by balancing three simultaneous constraints in `physics_loss.py`:

### A. Data Loss ($L_{data}$)
Ensures the GNN predicts the same surge as the ADCIRC benchmark (`fort.63.nc`).
$$ L_{data} = \frac{1}{N} \sum (\zeta_{GNN} - \zeta_{ADCIRC})^2 $$

### B. Boundary Condition Loss ($L_{BC}$)
Enforces the Dirichlet tidal boundary conditions. The network is severely penalized if the water level at the open ocean boundary nodes diverges from the forced astronomical tide.
$$ L_{BC} = \text{MSE}(\zeta_{GNN\_Boundary}, \text{Tide}_{actual}) $$

### C. Physics Loss ($L_{physics}$)
Approximates the Shallow Water Equations (Mass Continuity) across the graph. It penalizes non-physical gradients (water stacking up unnaturally).
$$ L_{physics} = \text{Mean} \left( \left| \frac{\partial \zeta}{\partial t} \right|^2 \right) + \lambda \cdot \text{Mean} \left( (\nabla \zeta)^2 \right) $$
*(Future Implementation: Fully embedding the Generalized Wave Continuity Equation (GWCE) into the automatic differentiation graph).*
