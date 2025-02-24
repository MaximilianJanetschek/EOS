
from typing import Any, Optional

from pydantic import Field
from pyscipopt import Model, quicksum, Variable

from akkudoktoreos.core.coreabc import (
    ConfigMixin,
    DevicesMixin,
    EnergyManagementSystemMixin,
)
from akkudoktoreos.core.pydantic import ParametersBaseModel
from akkudoktoreos.optimization.genetic import OptimizationParameters
from dataclasses import dataclass

class ExactSolutionResponse(ParametersBaseModel):
    """Response model for the exact optimization solution."""

    akku_charge: list[float] = Field(
        description="Array with target charging / Discharging values in wh."
    )
    eauto_charge: Optional[list[float]] = Field(
        default=None, description="Array containing electric vehicle charging values in wh."
    )


from dataclasses import dataclass, field
from typing import Dict, List

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class ModelVariables:
    charge: Dict[tuple[str, int], Variable] = field(default_factory=dict)
    discharge: Dict[tuple[str, int], Variable] = field(default_factory=dict)
    soc: Dict[tuple[str, int], Variable] = field(default_factory=dict)
    grid_import: Dict[int, Variable] = field(default_factory=dict)
    grid_export: Dict[int, Variable] = field(default_factory=dict)
    flow_direction: Dict[int, Variable] = field(default_factory=dict)

@dataclass
class ModelParameters:
    """
    Class for modeling grid battery storage systems with multiple battery types.
    Manages parameters like state of charge (SoC), power limits, and efficiencies.
    """
    # Minimum state of charge allowed for each battery type (in percentage 0-100)
    soc_min: Dict[str, float] = field(default_factory=dict)

    # Maximum state of charge allowed for each battery type (in percentage 0-100)
    soc_max: Dict[str, float] = field(default_factory=dict)

    # Initial state of charge for each battery type (in percentage 0-100)
    soc_init: Dict[str, float] = field(default_factory=dict)

    # Maximum charging/discharging power for each battery type (in watts)
    power_max: Dict[str, float] = field(default_factory=dict)

    # Energy storage capacity for each battery type (in watt-hours)
    capacity: Dict[str, float] = field(default_factory=dict)

    # Charging efficiency for each battery type (as decimal 0-1)
    # Represents energy stored / energy input
    eff_charge: Dict[str, float] = field(default_factory=dict)

    # Discharging efficiency for each battery type (as decimal 0-1)
    # Represents energy output / energy discharged
    eff_discharge: Dict[str, float] = field(default_factory=dict)

    # List of available battery types in the system
    battery_set: List[str] = field(default_factory=list)

    total_load: List[float] = field(default_factory=list)
    pv_forecast: List[float] = field(default_factory=list)
    price_import: List[float] = field(default_factory=list)
    price_export: List[float] = field(default_factory=list)
    price_storage: List[float] = field(default_factory=list)




    @classmethod
    def init_from_parameters(cls, parameters):
        """
        Initialize GridModel from a parameters object.

        Args:
            parameters: Object containing battery configurations
                       Expected to have attributes 'pv_akku' and 'eauto'

        Returns:
            ModelParameters: New instance populated with battery parameters
        """
        grid_model = cls()

        # Extract parameters from input
        grid_model.add_ems_parameters(parameters)

        # Define supported battery types
        battery_types = ["pv_akku", "eauto"]  # PV battery storage and electric vehicle

        # Add each battery type if it exists in parameters
        for batt_type in battery_types:
            grid_model.add_battery(parameters, batt_type)

        return grid_model

    def add_ems_parameters(self, parameters):
        self.total_load = parameters.ems.gesamtlast  # Required total energy
        self.pv_forecast = parameters.ems.pv_prognose_wh  # Forecasted production
        # Price parameters
        self.price_import = parameters.ems.strompreis_euro_pro_wh  # Price for buying from grid
        self.price_export = parameters.ems.einspeiseverguetung_euro_pro_wh  # Price for selling to grid
        self.price_storage = (
            parameters.ems.preis_euro_pro_wh_akku
        )  # Value of stored energy at end of horizon

    def add_battery(self, parameters, batt_type: str):
        """
        Add battery parameters to the grid model.

        Args:
            parameters: Object containing battery configurations
            batt_type (str): Battery type identifier (e.g., 'pv_akku' or 'eauto')

        Note:
            If battery attributes are not found, default values are used:
            - min_soc_percentage: 0%
            - max_soc_percentage: 100%
            - init_soc_percentage: 50%
            - max_charge_power_w: 0W
            - capacity_wh: 0Wh
            - charging_efficiency: 1.0
            - discharging_efficiency: 1.0
        """
        # Get battery configuration if it exists
        battery = getattr(parameters, batt_type, None)
        if battery is not None:
            # Add all battery parameters with their default values if not specified
            self.soc_min[batt_type] = getattr(battery, "min_soc_percentage", 0)
            self.soc_max[batt_type] = getattr(battery, "max_soc_percentage", 100)
            self.soc_init[batt_type] = getattr(battery, "init_soc_percentage", 50)
            self.power_max[batt_type] = getattr(battery, "max_charge_power_w", 0)
            self.capacity[batt_type] = getattr(battery, "capacity_wh", 0)
            self.eff_charge[batt_type] = getattr(battery, "charging_efficiency", 1)
            self.eff_discharge[batt_type] = getattr(battery, "discharging_efficiency", 1)

            # Add battery type to the set of available batteries
            self.battery_set.append(batt_type)

@dataclass
class ModelSolution:
    pass

class MILPOptimization(ConfigMixin, DevicesMixin, EnergyManagementSystemMixin):
    """Mixed-Integer Linear Programming Optimization for Energy Management Systems.

    This class implements a Mixed-Integer Linear Programming (MILP) formulation that
    minimizes energy costs while satisfying system constraints. It considers multiple
    energy sources and storage devices, including PV systems, batteries, and electric vehicles.

    The optimization problem is solved using the SCIP solver through the PySCIPOpt interface.

    Attributes:
        opti_param (Dict[str, Any]): Dictionary storing optimization parameters.
        possible_charge_values (List[float]): List of available charge rates as percentages.
        verbose (bool): Flag to control logging verbosity.
    """

    def __init__(
        self,
        verbose: bool = False,
    ):
        """Initialize the MILP optimization problem.

        Args:
            verbose (bool, optional): Enable verbose output. Defaults to False.
        """
        self.opti_param: dict[str, Any] = {}
        self.verbose = verbose

    def optimize_ems(
        self,
        parameters: OptimizationParameters,
    ) -> ExactSolutionResponse:
        """Solve the energy management system optimization problem using MILP.

        This method formulates and solves a MILP problem to minimize energy costs while satisfying
        system constraints. The optimization considers:
        - Grid power exchange (import/export)
        - Battery storage systems
        - PV generation
        - Electric vehicle charging
        - Time-varying electricity prices

        Args:
            parameters (OptimizationParameters): Input parameters containing:
                - Load profiles (total_load)
                - PV generation forecast (pv_forecast_wh)
                - Battery parameters (capacity, efficiency, power limits)
                - Price data (grid import/export prices)
                - Initial conditions

        Returns:
            ExactSolutionResponse: Optimization results containing optimal charging schedules.

        Raises:
            ValueError: If no optimal solution is found.

        Note:
            The optimization problem includes the following key components:

            Variables:
                - c[i,t]: Charging power for storage device i at time t
                - d[i,t]: Discharging power for storage device i at time t
                - s[i,t]: State of charge for storage device i at time t
                - n[t]: Grid import power at time t
                - e[t]: Grid export power at time t

            Constraints:
                1. Power balance at each timestep
                2. Battery dynamics (state of charge evolution)
                3. Operating limits (power, energy capacity)
                4. Grid power flow directionality

            Objective:
                Maximize: sum(-n[t]*p_N[t] + e[t]*p_E[t]) + sum(s[i,T]*p_a)
                where:
                - p_N: Grid import price
                - p_E: Grid export price
                - p_a: Final state of charge value
                - T: Final timestep
        """
        # Create optimization model
        model = Model("energy_management")

        # Define sets
        time_steps = range(self.config.optimization_hours)  # Time steps



        grid_model = ModelParameters.init_from_parameters(parameters)



        # Create variables
        charge = {}  # Charging power
        discharge = {}  # Discharging power
        soc = {}  # State of charge
        for batt_type in grid_model.battery_set:
            for t in time_steps:
                charge[batt_type, t] = model.addVar(
                    name=f"charge_{batt_type}_{t}", vtype="C", lb=0, ub=grid_model.power_max[batt_type]
                )
                discharge[batt_type, t] = model.addVar(
                    name=f"discharge_{batt_type}_{t}", vtype="C", lb=0, ub=grid_model.power_max[batt_type]
                )
                soc[batt_type, t] = model.addVar(
                    name=f"soc_{batt_type}_{t}",
                    vtype="C",
                    lb=grid_model.soc_min[batt_type],
                    ub=grid_model.soc_max[batt_type],
                )

        grid_import = {}  # Grid import power
        grid_export = {}  # Grid export power
        for t in time_steps:
            grid_import[t] = model.addVar(name=f"grid_import_{t}", vtype="C", lb=0)
            grid_export[t] = model.addVar(name=f"grid_export_{t}", vtype="C", lb=0)

        # Add constraints
        # Grid balance constraint
        for t in time_steps:
            model.addCons(
                quicksum(
                    -charge[batt_type, t] + discharge[batt_type, t] for batt_type in grid_model.battery_set
                )
                + grid_model.pv_forecast[t]
                + grid_import[t]
                == grid_export[t] + grid_model.total_load[t],
                name=f"grid_balance_{t}",
            )

        # Battery dynamics constraints
        for batt_type in grid_model.battery_set:
            for t in time_steps:
                if t == time_steps[0]:
                    model.addCons(
                        grid_model.soc_init[batt_type] * grid_model.capacity[batt_type] / 100
                        + grid_model.eff_charge[batt_type] * charge[batt_type, t]
                        - (1 / grid_model.eff_discharge[batt_type]) * discharge[batt_type, t]
                        == soc[batt_type, t] * grid_model.capacity[batt_type] / 100,
                        name=f"battery_dynamics_{batt_type}_{t}",
                    )
                else:
                    model.addCons(
                        soc[batt_type, t - 1] * grid_model.capacity[batt_type] / 100
                        + grid_model.eff_charge[batt_type] * charge[batt_type, t]
                        - (1 / grid_model.eff_discharge[batt_type]) * discharge[batt_type, t]
                        == soc[batt_type, t] * grid_model.capacity[batt_type] / 100,
                        name=f"battery_dynamics_{batt_type}_{t}",
                    )

        # Prevent simultaneous import and export when import price is less than or equal to export price
        flow_var = {}
        for t in time_steps:
            if isinstance(grid_model.price_export, float):
                enforce_flow = grid_model.price_import[t] <= grid_model.price_export
            else:
                enforce_flow = grid_model.price_import[t] <= grid_model.price_export[t]

            if enforce_flow:
                flow_var[t] = model.addVar(name=f"flow_direction_{t}", vtype="B", lb=0, ub=1)
                max_bezug = sum(
                    grid_model.eff_charge[batt_type] * grid_model.power_max[batt_type] for batt_type in grid_model.battery_set
                ) + max(grid_model.total_load)
                max_einspeise = sum(
                    grid_model.eff_discharge[batt_type] * grid_model.power_max[batt_type] for batt_type in grid_model.battery_set
                ) + max(grid_model.pv_forecast)
                big_m = max(max_bezug, max_einspeise)
                model.addCons(
                    grid_export[t] <= big_m * flow_var[t], name=f"export_constraint_{t}"
                )
                model.addCons(
                    grid_import[t] <= big_m * (1 - flow_var[t]), name=f"import_constraint_{t}"
                )

        # Set objective
        objective = quicksum(
            -grid_import[t] * grid_model.price_import[t] + grid_export[t] * grid_model.price_export for t in time_steps
        ) + quicksum(
            soc[batt_type, time_steps[-1]] * grid_model.price_storage * grid_model.capacity[batt_type]
            for batt_type in grid_model.battery_set
        )
        model.setObjective(objective, "maximize")

        model_vars = ModelVariables(
            charge=charge,
            discharge=discharge,
            soc=soc,
            grid_import=grid_import,
            grid_export=grid_export,
            flow_direction=flow_var,
        )


        # set warm start
        self.set_warm_start(model, model_vars, time_steps, grid_model)


        model.optimize()

        # Solve the model
        if self.verbose:
            print("Number of variables:", len(model.getVars()))
            print("Number of constraints:", len(model.getConss()))
            print("Objective value:", model.getObjVal())

        if model.getStatus() != "optimal":
            raise ValueError("No optimal solution found")

        # Extract solution
        if "pv_akku" in grid_model.battery_set:
            akku_charge = [
                model.getVal(charge["pv_akku", t]) - model.getVal(discharge["pv_akku", t])
                for t in time_steps
            ]
        else:
            akku_charge = []

        if "eauto" in grid_model.battery_set:
            ev_charge = [model.getVal(charge["eauto", t]) for t in time_steps]
        else:
            ev_charge = None

        return ExactSolutionResponse(
            akku_charge=akku_charge,
            eauto_charge=ev_charge,
        )

    def visualize_warm_start(
            self,
            charge: dict,
            discharge: dict,
            soc: dict,
            time_steps: range,
            model_params: ModelParameters,
            save_path: str = None
    ):
        """
        Visualize the battery state of charge, charging, and discharging patterns
        for each battery type from the warm start solution.

        Args:
            charge (dict): Dictionary with battery charging values for each timestep
            discharge (dict): Dictionary with battery discharging values for each timestep
            soc (dict): Dictionary with state of charge values for each timestep
            time_steps (range): Range of optimization timesteps
            model_params (ModelParameters): Model parameters containing battery configurations
            save_path (str, optional): Path to save the plots. If None, plots are displayed.

        Returns:
            None
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import numpy as np

        # Create a figure for each battery type
        for batt_type in model_params.battery_set:
            # Extract data for this battery
            soc_values = [soc.get((batt_type, t), 0) for t in time_steps]
            charge_values = [charge.get((batt_type, t), 0) for t in time_steps]
            discharge_values = [discharge.get((batt_type, t), 0) for t in time_steps]
            net_power = [charge_values[i] - discharge_values[i] for i in range(len(time_steps))]

            # Convert time steps to hour labels
            hours = list(time_steps)

            # Create figure with 2 subplots stacked vertically
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

            # Plot 1: State of Charge
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(hours, soc_values, 'b-', marker='o', linewidth=2, label='State of Charge (%)')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('State of Charge (%)')
            ax1.set_title(f'Battery {batt_type} - State of Charge over Time')
            ax1.grid(True)
            ax1.set_ylim([
                max(0, min(soc_values) - 5),  # Min with 5% padding
                min(100, max(soc_values) + 5)  # Max with 5% padding
            ])

            # Add horizontal lines for min and max SoC limits
            min_soc = model_params.soc_min.get(batt_type, 0)
            max_soc = model_params.soc_max.get(batt_type, 100)
            ax1.axhline(y=min_soc, color='r', linestyle='--', alpha=0.7, label=f'Min SoC ({min_soc}%)')
            ax1.axhline(y=max_soc, color='g', linestyle='--', alpha=0.7, label=f'Max SoC ({max_soc}%)')
            ax1.legend(loc='best')

            # Plot 2: Charging and Discharging Power
            ax2 = fig.add_subplot(gs[1])

            # Create bar chart for charge and discharge
            bar_width = 0.35
            x = np.arange(len(hours))

            # Plot charging as positive values
            charging_bars = ax2.bar(x - bar_width / 2, charge_values, bar_width, label='Charging Power (W)',
                                    color='green', alpha=0.7)

            # Plot discharging as negative values
            discharge_bars = ax2.bar(x + bar_width / 2, discharge_values, bar_width, label='Discharging Power (W)',
                                     color='red', alpha=0.7)

            # Plot net power as a line
            ax2_twin = ax2.twinx()
            net_line = ax2_twin.plot(x, net_power, 'b-', marker='*', linewidth=2, label='Net Power (W)')

            # Add labels and legend for both axes
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Power (Watts)')
            ax2_twin.set_ylabel('Net Power (Watts)', color='blue')

            # Combine legends from both y-axes
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

            ax2.set_title(f'Battery {batt_type} - Charging and Discharging Power')
            ax2.set_xticks(x)
            ax2.set_xticklabels(hours)
            ax2.grid(True)

            # Add max power limit line
            max_power = model_params.power_max.get(batt_type, 0)
            ax2.axhline(y=max_power, color='purple', linestyle='-.', alpha=0.7,
                        label=f'Max Power ({max_power}W)')

            # Add annotations for capacity
            capacity = model_params.capacity.get(batt_type, 0)
            plt.figtext(0.02, 0.02, f"Battery Capacity: {capacity} Wh", fontsize=10)

            plt.tight_layout()

            # Save or show the plot
            if save_path:
                plt.savefig(f"{save_path}/battery_{batt_type}_visualization.png", dpi=300, bbox_inches='tight')
            else:
                plt.show()

        # Create an additional plot showing all batteries' SoC on the same graph for comparison
        if len(model_params.battery_set) > 1:
            plt.figure(figsize=(12, 6))
            for batt_type in model_params.battery_set:
                soc_values = [soc.get((batt_type, t), 0) for t in time_steps]
                plt.plot(hours, soc_values, marker='o', linewidth=2, label=f'{batt_type} SoC (%)')

            plt.xlabel('Time (hours)')
            plt.ylabel('State of Charge (%)')
            plt.title('Comparison of All Batteries - State of Charge')
            plt.grid(True)
            plt.legend(loc='best')

            if save_path:
                plt.savefig(f"{save_path}/all_batteries_soc_comparison.png", dpi=300, bbox_inches='tight')
            else:
                plt.show()

    def set_warm_start(self, model: Model, vars: ModelVariables, time_steps: range, model_params: ModelParameters):
        import time
        start = time.time()

        # Calculate warm start solution
        warm_charge, warm_discharge, warm_soc, warm_grid_import, warm_grid_export, warm_flow_direction = self.generate_warm_start(
            time_steps=time_steps,
            model_params=model_params
        )

        print(f"Warm start calculation time: {time.time() - start}")

        # Create a solution object
        solution = model.createSol()

        # Set variable values in the solution object
        for batt_type in model_params.battery_set:
            for t in time_steps:
                # Set battery-related variables
                model.setSolVal(solution, vars.charge[batt_type, t], warm_charge[batt_type, t])
                model.setSolVal(solution, vars.discharge[batt_type, t], warm_discharge[batt_type, t])
                model.setSolVal(solution, vars.soc[batt_type, t], warm_soc[batt_type, t])

        # Set grid import/export values
        for t in time_steps:
            model.setSolVal(solution, vars.grid_import[t], warm_grid_import[t])
            model.setSolVal(solution, vars.grid_export[t], warm_grid_export[t])

            # Set flow direction binary variables if they exist
            if t in vars.flow_direction.keys():
                model.setSolVal(solution, vars.flow_direction[t], warm_flow_direction[t])

        # Try to add the solution to the model
        print(solution)
        print('checking solution')
        accepted = model.trySol(solution)

        if accepted:
            print("Warm start solution was accepted")
        else:
            print("Warm start solution was rejected - check solution feasibility")

        # Free the solution object
        model.freeSol(solution)

    def generate_warm_start(
            self,
            time_steps: range,
            model_params: ModelParameters,
    ) -> tuple[dict, dict, dict, dict, dict, dict]:
        """Generate warm start solution for the MILP optimization.

        Args:
            time_steps: Range of optimization timesteps
            model_params: Model parameters containing load profiles, PV forecast, etc.

        Returns:
            Tuple of dictionaries containing initial values for:
            - charging power
            - discharging power
            - state of charge (as percentage 0-100)
            - grid import power
            - grid export power
            - flow direction binary variables
        """
        # Initialize solution dictionaries
        charge = {(b, t): 0.0 for b in model_params.battery_set for t in time_steps}
        discharge = {(b, t): 0.0 for b in model_params.battery_set for t in time_steps}
        soc = {(b, t): 0.0 for b in model_params.battery_set for t in time_steps}
        grid_import = {t: 0.0 for t in time_steps}
        grid_export = {t: 0.0 for t in time_steps}
        flow_direction = {t: 0 for t in time_steps}  # 0 for import, 1 for export

        # First pass: Direct PV usage and simple battery charging
        for batt_type in model_params.battery_set:
            # Start with initial SoC percentage
            current_soc_pct = model_params.soc_init[batt_type]

            for t in time_steps:
                # Store the current SoC percentage
                soc[batt_type, t] = current_soc_pct

                # Calculate net power (positive means excess PV)
                net_power = model_params.pv_forecast[t] - model_params.total_load[t]

                if net_power > 0:  # Excess PV available
                    # Calculate how much more energy the battery can store (in Wh)
                    max_energy_to_store = (model_params.soc_max[batt_type] - current_soc_pct) * model_params.capacity[
                        batt_type] / 100

                    # Convert to charging power considering efficiency
                    max_charge_power = max_energy_to_store / model_params.eff_charge[batt_type]

                    # Calculate maximum charging power considering all constraints
                    max_charge = min(
                        model_params.power_max[batt_type],  # Power limit
                        net_power,  # Available PV excess
                        max_charge_power  # SoC limit
                    )

                    if max_charge > 0:
                        charge[batt_type, t] = max_charge

                        # Calculate energy gained (in Wh)
                        energy_gained = max_charge * model_params.eff_charge[batt_type]

                        # Update SoC percentage
                        soc_gained_pct = (energy_gained / model_params.capacity[batt_type]) * 100
                        current_soc_pct += soc_gained_pct

        # Second pass: Ensure minimum SOC
        for batt_type in model_params.battery_set:
            # Process timesteps in reverse to ensure we meet minimum SoC
            for t in reversed(time_steps):
                if soc[batt_type, t] < model_params.soc_min[batt_type]:
                    # Calculate required energy to reach minimum SoC (in Wh)
                    energy_required = (model_params.soc_min[batt_type] - soc[batt_type, t]) * model_params.capacity[
                        batt_type] / 100

                    # Convert to charging power considering efficiency
                    charge_power_required = energy_required / model_params.eff_charge[batt_type]

                    # Limit by maximum power
                    charge_power = min(model_params.power_max[batt_type], charge_power_required)

                    # Update charging for this timestep
                    charge[batt_type, t] += charge_power

                    # Recalculate SoC for all timesteps
                    temp_soc = {}
                    current_soc_pct = model_params.soc_init[batt_type]

                    for update_t in time_steps:
                        # Calculate energy change (in Wh)
                        energy_gained = charge[batt_type, update_t] * model_params.eff_charge[batt_type]
                        energy_lost = discharge[batt_type, update_t] / model_params.eff_discharge[batt_type]

                        # Update SoC percentage
                        soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100
                        current_soc_pct += soc_change_pct
                        temp_soc[update_t] = current_soc_pct

                    # Update SoC dictionary
                    for update_t, soc_pct in temp_soc.items():
                        soc[batt_type, update_t] = soc_pct

        # Third pass: Optimize for price differences
        for batt_type in model_params.battery_set:
            for t in time_steps:
                # Check if we're importing from grid
                net_power = model_params.pv_forecast[t] - model_params.total_load[t]
                if net_power < 0:  # Grid import needed
                    # Find future timestep with higher price where we could discharge
                    for future_t in range(t + 1, len(time_steps)):
                        if model_params.price_import[future_t] > model_params.price_import[t]:
                            # Simulate SoC evolution up to future timestep
                            sim_soc_pct = model_params.soc_init[batt_type]
                            for sim_t in range(len(time_steps)):
                                if sim_t < future_t:
                                    energy_gained = charge[batt_type, sim_t] * model_params.eff_charge[batt_type]
                                    energy_lost = discharge[batt_type, sim_t] / model_params.eff_discharge[batt_type]
                                    soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[
                                        batt_type]) * 100
                                    sim_soc_pct += soc_change_pct

                            # Calculate max discharge possible at future timestep
                            max_soc_drop_pct = sim_soc_pct - model_params.soc_min[batt_type]
                            max_energy_discharge = (max_soc_drop_pct * model_params.capacity[batt_type]) / 100

                            # Convert to discharge power considering efficiency
                            max_discharge_power = max_energy_discharge * model_params.eff_discharge[batt_type]

                            # Limit by maximum power
                            available_discharge = min(
                                model_params.power_max[batt_type],
                                max_discharge_power
                            )

                            if available_discharge > 0:
                                # Update discharge for future timestep
                                discharge[batt_type, future_t] += available_discharge

                                # Recalculate all SoC values
                                temp_soc = {}
                                current_soc_pct = model_params.soc_init[batt_type]

                                for update_t in time_steps:
                                    # Calculate energy change (in Wh)
                                    energy_gained = charge[batt_type, update_t] * model_params.eff_charge[batt_type]
                                    energy_lost = discharge[batt_type, update_t] / model_params.eff_discharge[batt_type]

                                    # Update SoC percentage
                                    soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[
                                        batt_type]) * 100
                                    current_soc_pct += soc_change_pct
                                    temp_soc[update_t] = current_soc_pct

                                # Check if any SoC violates constraints
                                if any(soc_pct < model_params.soc_min[batt_type] for soc_pct in temp_soc.values()):
                                    # Revert discharge if it would violate minimum SoC
                                    discharge[batt_type, future_t] -= available_discharge
                                else:
                                    # Update SoC dictionary with new values
                                    for update_t, soc_pct in temp_soc.items():
                                        soc[batt_type, update_t] = soc_pct

                                    # We successfully scheduled a discharge, so stop looking for more opportunities
                                    break

        # Final validation pass
        for batt_type in model_params.battery_set:
            current_soc_pct = model_params.soc_init[batt_type]

            for t in time_steps:
                # Calculate SoC change from charge/discharge
                energy_gained = charge[batt_type, t] * model_params.eff_charge[batt_type]
                energy_lost = discharge[batt_type, t] / model_params.eff_discharge[batt_type]
                soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100

                # Check if next SoC would be valid
                next_soc_pct = current_soc_pct + soc_change_pct

                if next_soc_pct < model_params.soc_min[batt_type]:
                    # Calculate shortfall in percentage terms
                    shortfall_pct = model_params.soc_min[batt_type] - next_soc_pct

                    # Convert to energy (Wh)
                    shortfall_energy = (shortfall_pct * model_params.capacity[batt_type]) / 100

                    if discharge[batt_type, t] > 0:
                        # Calculate discharge power reduction needed
                        discharge_reduction = min(
                            discharge[batt_type, t],
                            shortfall_energy * model_params.eff_discharge[batt_type]
                        )

                        # Reduce discharge
                        discharge[batt_type, t] -= discharge_reduction

                        # Recalculate change
                        energy_gained = charge[batt_type, t] * model_params.eff_charge[batt_type]
                        energy_lost = discharge[batt_type, t] / model_params.eff_discharge[batt_type]
                        soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                    # If still below minimum, increase charge
                    if next_soc_pct < model_params.soc_min[batt_type]:
                        # Recalculate shortfall
                        shortfall_pct = model_params.soc_min[batt_type] - next_soc_pct
                        shortfall_energy = (shortfall_pct * model_params.capacity[batt_type]) / 100

                        # Calculate additional charge needed
                        additional_charge = shortfall_energy / model_params.eff_charge[batt_type]

                        # Add charge (limited by max power)
                        additional_charge = min(
                            additional_charge,
                            model_params.power_max[batt_type] - charge[batt_type, t]
                        )

                        charge[batt_type, t] += additional_charge

                        # Final recalculation
                        energy_gained = charge[batt_type, t] * model_params.eff_charge[batt_type]
                        energy_lost = discharge[batt_type, t] / model_params.eff_discharge[batt_type]
                        soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                # Update SOC for this timestep
                soc[batt_type, t] = next_soc_pct
                current_soc_pct = next_soc_pct

        # Calculate grid import/export and set flow direction
        for t in time_steps:
            # Calculate net battery power for this timestep
            battery_net_power = sum(
                discharge[batt_type, t] - charge[batt_type, t]
                for batt_type in model_params.battery_set
            )

            # Calculate overall power balance
            net_power = model_params.pv_forecast[t] + battery_net_power - model_params.total_load[t]

            if net_power < 0:
                # Need to import from grid
                grid_import[t] = -net_power  # Convert negative value to positive import
                grid_export[t] = 0.0
                flow_direction[t] = 0  # 0 means import
            else:
                # Exporting to grid
                grid_import[t] = 0.0
                grid_export[t] = net_power
                flow_direction[t] = 1  # 1 means export

            # If prices dictate that we shouldn't have both import and export
            # when import price <= export price, enforce flow direction
            if isinstance(model_params.price_export, float):
                enforce_flow = model_params.price_import[t] <= model_params.price_export
            else:
                enforce_flow = model_params.price_import[t] <= model_params.price_export[t]

            if enforce_flow and grid_import[t] > 0 and grid_export[t] > 0:
                # This shouldn't happen with our calculations, but just to be safe:
                # Choose the dominant direction based on which is larger
                if grid_import[t] > grid_export[t]:
                    grid_export[t] = 0.0
                    flow_direction[t] = 0
                else:
                    grid_import[t] = 0.0
                    flow_direction[t] = 1

        return charge, discharge, soc, grid_import, grid_export, flow_direction
