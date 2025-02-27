from operator import truediv
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
from akkudoktoreos.optimization.utils import visualize_warm_start
import time

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
    price_storage: float = field(default_factory=list)
    no_discharge: List[float] = field(default_factory=list)



    @classmethod
    def init_from_parameters(cls, parameters: OptimizationParameters):
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

    def add_ems_parameters(self, parameters: OptimizationParameters):
        self.total_load = parameters.ems.gesamtlast  # Required total energy
        self.pv_forecast = parameters.ems.pv_prognose_wh  # Forecasted production
        # Price parameters
        p_import = parameters.ems.strompreis_euro_pro_wh  # Price for buying from grid

        if isinstance(p_import, list):
            self.price_import = p_import
        else:
            self.price_import = [p_import] * len(self.total_load)

        p_export = parameters.ems.einspeiseverguetung_euro_pro_wh  # Price for selling to grid
        if isinstance(p_export, list):
            self.price_export = p_export
        else:
            self.price_export = [p_export] * len(self.total_load)


        self.price_storage = parameters.ems.preis_euro_pro_wh_akku
        # Value of stored energy at end of horizon

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


@dataclass
class HeuristicSolution:
    charge: dict[tuple[str, int], float]
    discharge: dict[tuple[str, int], float]
    soc: dict[tuple[str, int], float]
    grid_import: dict[int, float]
    grid_export: dict[int, float]
    flow_direction: dict[int, int]

    @classmethod
    def from_params(cls, model_params: ModelParameters, time_steps: range):
        charge = {(b, t): 0.0 for b in model_params.battery_set for t in time_steps}
        discharge = {(b, t): 0.0 for b in model_params.battery_set for t in time_steps}
        soc = {(b, t): model_params.soc_init[b] for b in model_params.battery_set for t in time_steps}
        grid_import = {t: 0.0 for t in time_steps}
        grid_export = {t: 0.0 for t in time_steps}
        flow_direction = {t: 0 for t in time_steps}  # 0 for import, 1 for export

        return cls(charge, discharge, soc, grid_import, grid_export, flow_direction)


    def _update_grid_values(
            self,
            time_steps,
            model_params
    ):
        """Update grid import/export values based on current battery charge/discharge."""
        for t in time_steps:
            # Calculate net battery power for this timestep
            battery_net_power = sum(
                self.discharge[batt_type, t] - self.charge[batt_type, t]
                for batt_type in model_params.battery_set
            )

            # Calculate overall power balance
            net_power = model_params.pv_forecast[t] + battery_net_power - model_params.total_load[t]

            if net_power < 0:
                # Need to import from grid
                self.grid_import[t] = -net_power  # Convert negative value to positive import
                self.grid_export[t] = 0.0
                self.flow_direction[t] = 0  # 0 means import
            else:
                # Exporting to grid
                self.grid_import[t] = 0.0
                self.grid_export[t] = net_power
                self.flow_direction[t] = 1  # 1 means export

    def _calculate_and_print_objective(
            self,
            pass_name,
            time_steps,
            model_params: ModelParameters
    ):
        """Calculate and print the objective value for the current solution."""
        # Calculate the objective value
        grid_costs = 0
        battery_value = 0

        # Grid costs/revenue
        for t in time_steps:
            # Grid import costs (negative contribution to objective)
            grid_costs -= self.grid_import[t] * model_params.price_import[t]
            grid_costs += self.grid_export[t] * model_params.price_export[t]


        # Battery end state value
        for batt_type in model_params.battery_set:
            # Value of energy stored in battery at end of horizon
            final_timestep = time_steps[-1]
            battery_value += (self.soc[batt_type, final_timestep] * model_params.capacity[
                batt_type] / 100) * model_params.price_storage

        # Total objective value
        total_objective = grid_costs + battery_value

        # Print the objective value with detailed breakdown
        print(f"\n--- {pass_name} ---")
        print(f"Grid Costs/Revenue: {grid_costs:.4f}")
        print(f"Battery End Value: {battery_value:.4f}")
        print(f"Total Objective Value: {total_objective:.4f}")

        # Print additional metrics
        total_imported = sum(self.grid_import[t] for t in time_steps)
        total_exported = sum(self.grid_export[t] for t in time_steps)
        print(f"Total Energy Imported: {total_imported:.2f} Wh")
        print(f"Total Energy Exported: {total_exported:.2f} Wh")

        # Print final SOC for each battery
        print("Final Battery SOC Values:")
        for batt_type in model_params.battery_set:
            print(f"  {batt_type}: {self.soc[batt_type, time_steps[-1]]:.2f}%")
        print("-" * 30)


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
        cannot_discharge: list = ['eauto']
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
            discharge_factor = 1 if batt_type not in cannot_discharge else 0
            for t in time_steps:
                charge[batt_type, t] = model.addVar(
                    name=f"charge_{batt_type}_{t}", vtype="C", lb=0, ub=grid_model.power_max[batt_type]
                )
                discharge[batt_type, t] = model.addVar(
                    name=f"discharge_{batt_type}_{t}", vtype="C", lb=0, ub=grid_model.power_max[batt_type] * discharge_factor
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
            -grid_import[t] * grid_model.price_import[t] + grid_export[t] * grid_model.price_export[t] for t in time_steps
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


    def set_warm_start(self,
                       model: Model,
                       vars: ModelVariables,
                       time_steps: range,
                       model_params: ModelParameters
    ):
        start = time.time()

        # Calculate warm start solution
        greedy_start = self.generate_warm_start(
            time_steps=time_steps,
            model_params=model_params
        )

        print(f"Warm start calculation time: {time.time() - start}")
        if False:
            visualize_warm_start(
                heur_sol=greedy_start,
                model_params=model_params,
                time_steps=time_steps
            )

        # Create a solution object
        solution = model.createSol()

        # Set variable values in the solution object
        for batt_type in model_params.battery_set:
            for t in time_steps:
                # Set battery-related variables
                model.setSolVal(solution, vars.charge[batt_type, t], greedy_start.charge[batt_type, t])
                model.setSolVal(solution, vars.discharge[batt_type, t], greedy_start.discharge[batt_type, t])
                model.setSolVal(solution, vars.soc[batt_type, t], greedy_start.soc[batt_type, t])

        # Set grid import/export values
        for t in time_steps:
            model.setSolVal(solution, vars.grid_import[t], greedy_start.grid_import[t])
            model.setSolVal(solution, vars.grid_export[t], greedy_start.grid_export[t])

            # Set flow direction binary variables if they exist
            if t in vars.flow_direction.keys():
                model.setSolVal(solution, vars.flow_direction[t], greedy_start.flow_direction[t])

        # Try to add the solution to the model

        try:
            accepted = model.checkSol(solution, completely=True, original=True)

            if accepted:
                print("Warm start solution was accepted")
                try:
                    model.addSol(solution)
                    print("Solution successfully added")
                except Exception as e:
                    print(f"Error adding solution: {e}")
            else:
                print("Warm start solution was rejected - check solution feasibility")

                # Print some details to debug
                for i, var in enumerate(model.getVars()):
                    if solution[i] < var.getLbOriginal() or solution[i] > var.getUbOriginal():
                        print(
                            f"Variable {var.name} has value {solution[i]} outside bounds [{var.getLbOriginal()}, {var.getUbOriginal()}]")

        except Exception as e:
            print(f"Error checking solution: {e}")

    def generate_warm_start(
            self,
            time_steps: range,
            model_params: ModelParameters,
    ) -> HeuristicSolution:
        """Generate improved warm start solution for the MILP optimization.

        This implementation follows three main passes:
        1. First pass: Prioritize charging the last battery type with excess solar until min SoC
           is reached, then move to the second last battery, and so on.
        2. Second pass: Ensure minimum SoC requirements by charging at lowest grid import cost.
        3. Third pass: Optimize by charging the first battery at low price times and
           discharging at high price times, accounting for efficiency losses.

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
        greedy_sol = HeuristicSolution.from_params(model_params= model_params, time_steps=time_steps)

        # Initialize current SoC for all batteries
        current_soc = {batt_type: model_params.soc_init[batt_type] for batt_type in model_params.battery_set}

        # ----- FIRST PASS: Prioritized charging with excess solar, starting from first battery -----

        # Process each timestep
        for t in time_steps:

            # Calculate initial power balance (positive means excess PV)
            remaining_power = model_params.pv_forecast[t] - model_params.total_load[t]

            # If excess PV available, try to charge batteries starting from the last one
            if remaining_power > 0:
                for batt_type in model_params.battery_set:
                    # Skip if battery is already at or above min SoC
                    if current_soc[batt_type] >= 90:
                        continue

                    # Calculate maximum charging power considering all constraints
                    max_charge = min(
                        model_params.power_max[batt_type],  # Power limit
                        (model_params.soc_max[batt_type] - current_soc[batt_type]) *model_params.capacity[batt_type] / model_params.eff_charge[batt_type],
                        remaining_power,  # Available PV excess
                    )

                    if max_charge > 0:
                        # Set the charge for this battery at this timestep
                        greedy_sol.charge[batt_type, t] = max_charge

                        # Calculate energy gained (in Wh)
                        energy_gained = max_charge * model_params.eff_charge[batt_type]

                        # Update SoC percentage
                        soc_gained_pct = (energy_gained / model_params.capacity[batt_type]) * 100
                        current_soc[batt_type] += soc_gained_pct

                        # Update SoC for this timestep
                        greedy_sol.soc[batt_type, t] = current_soc[batt_type]

                        # Reduce remaining power
                        remaining_power -= max_charge

                        # If no more power to allocate, exit loop
                        if remaining_power <= 0:
                            break
            else:
                for batt_type in model_params.battery_set:
                    greedy_sol.soc[batt_type, t] = current_soc[batt_type]

        # Update grid import/export after first pass
        greedy_sol._update_grid_values(time_steps, model_params)

        # Calculate and print objective after first pass
        greedy_sol._calculate_and_print_objective(
            "After First Pass (Prioritized PV Charging)",  time_steps,
            model_params
        )

        # ----- SECOND PASS: Ensure minimum SOC at EVERY time step -----
        # For each battery, check if minimum SoC is met at all time steps
        import numpy as np
        for batt_type in model_params.battery_set:


            # Second pass: Find violations and fix them
            for t in time_steps:
                # Check if SoC violates minimum requirement
                if greedy_sol.soc[batt_type, t] < model_params.soc_min[batt_type]:
                    # Calculate shortfall
                    shortfall_pct = model_params.soc_min[batt_type] - greedy_sol.soc[batt_type, t]
                    # Convert to energy (Wh)
                    shortfall_energy = (shortfall_pct * model_params.capacity[batt_type]) / 100

                    # Try to charge at the lowest price timesteps before this timestep
                    remaining_shortfall = shortfall_energy

                    # get the prices sorted
                    earlier_times = model_params.price_import[0:t]
                    cheapest_prices = np.argsort(earlier_times)

                    for earlier_t in cheapest_prices:
                        # Calculate how much more we can charge at this timestep
                        available_charge_power = model_params.power_max[batt_type] - greedy_sol.charge[
                            batt_type, earlier_t]

                        if available_charge_power <= 0:
                            continue  # Already charging at maximum power

                        # Calculate energy we can gain with efficiency
                        max_energy_gain = available_charge_power * model_params.eff_charge[batt_type]

                        # Limit by the remaining shortfall
                        energy_to_add = min(max_energy_gain, remaining_shortfall)
                        power_to_add = energy_to_add / model_params.eff_charge[batt_type]

                        if power_to_add > 0:
                            # Add charge
                            greedy_sol.charge[batt_type, earlier_t] += power_to_add

                            # we need to check if the soc allows to transfer the power
                            soc_increase = energy_to_add / model_params.capacity[batt_type]

                            for t_test in range(earlier_t, t):
                                enough_gap = greedy_sol.soc[batt_type, t_test] + soc_increase <= model_params.soc_max[batt_type]
                                if not enough_gap:
                                    # there is not enough energy
                                    break

                            # Reduce remaining shortfall
                            remaining_shortfall -= energy_to_add

                            # If shortfall is eliminated, break
                            if remaining_shortfall <= 0:
                                break

                    # If we couldn't eliminate the shortfall by charging in earlier timesteps,
                    # we need to adjust the discharge decisions at earlier timesteps
                    if remaining_shortfall > 0:
                        for earlier_t in range(0,t):
                            # Calculate how much we can reduce discharge
                            reducible_discharge = greedy_sol.discharge[batt_type, earlier_t]

                            if reducible_discharge <= 0:
                                continue  # No discharge to reduce

                            # Calculate energy we can save by reducing discharge (accounting for efficiency)
                            max_energy_save = reducible_discharge / model_params.eff_discharge[batt_type]

                            # Limit by the remaining shortfall
                            energy_to_save = min(max_energy_save, remaining_shortfall)
                            discharge_to_reduce = energy_to_save * model_params.eff_discharge[batt_type]

                            if discharge_to_reduce > 0:
                                # Reduce discharge
                                greedy_sol.discharge[batt_type, earlier_t] -= discharge_to_reduce

                                # Reduce remaining shortfall
                                remaining_shortfall -= energy_to_save

                                # If shortfall is eliminated, break
                                if remaining_shortfall <= 0:
                                    break

            # Final pass: Recalculate SoC profile for all timesteps after modifications
            current_soc_pct = model_params.soc_init[batt_type]

            for t in time_steps:
                # Calculate energy change (in Wh)
                energy_gained = greedy_sol.charge[batt_type, t] * model_params.eff_charge[batt_type]
                energy_lost = greedy_sol.discharge[batt_type, t] / model_params.eff_discharge[batt_type]

                # Update SoC percentage
                soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100
                current_soc_pct += soc_change_pct

                # Update SoC for this timestep
                greedy_sol.soc[batt_type, t] = current_soc_pct

                # Double-check that minimum SOC is now met
                if greedy_sol.soc[batt_type, t] < model_params.soc_min[batt_type]:
                    print(f"Could not meet minimum SOC for battery {batt_type} at timestep {t}. "
                                    f"Current: {greedy_sol.soc[batt_type, t]:.2f}%, Minimum: {model_params.soc_min[batt_type]:.2f}%")

        # Update grid import/export after second pass
        greedy_sol._update_grid_values( time_steps, model_params)

        # Calculate and print objective after second pass
        greedy_sol._calculate_and_print_objective(
            "After Second Pass (Min SoC Enforcement)", time_steps,
            model_params
        )

        # ------ FOURTH PASS: Use the excess battery SoC in times where prices are super high
        import numpy as np

        # Get time indices sorted by price (highest first)
        if isinstance(model_params.price_import, list):
            des_prices = np.argsort([-p for p in model_params.price_import])
        else:
            # If price_import is a single value, we can't optimize based on price differences
            des_prices = list(time_steps)

        # Process high-price times first
        for t_idx in des_prices:
            t = t_idx if isinstance(t_idx, int) else int(t_idx)

            # Skip if time step is out of range
            if t not in time_steps:
                continue

            # Check if we are importing
            if greedy_sol.grid_import[t] > 0:
                # Check if we have excess battery capacity at the end (above min_soc)
                for batt_type in model_params.battery_set:
                    # Skip electric vehicle battery if it exists
                    if batt_type == 'eauto':
                        continue

                    # Check if we have excess SoC at the end
                    if greedy_sol.soc[batt_type, time_steps[-1]] > model_params.soc_min[batt_type]:
                        # Calculate how much we can discharge without violating min SoC
                        excess_soc_pct = greedy_sol.soc[batt_type, time_steps[-1]] - model_params.soc_min[batt_type]
                        excess_energy_wh = (excess_soc_pct * model_params.capacity[batt_type]) / 100

                        # Convert to potential discharge power (accounting for efficiency)
                        potential_discharge = excess_energy_wh * model_params.eff_discharge[batt_type]

                        # Limit by maximum discharge power, available excess, and current grid import
                        available_discharge_power = min(
                            model_params.power_max[batt_type] - greedy_sol.discharge[batt_type, t],  # Power limit
                            potential_discharge,  # Energy from excess SoC
                            greedy_sol.grid_import[t]  # Don't discharge more than we're importing
                        )

                        if available_discharge_power > 0:
                            # Add discharge at this timestep
                            greedy_sol.discharge[batt_type, t] += available_discharge_power

                            # Recalculate SoC profile for all timesteps
                            recalc_soc = {}
                            current_soc_pct = model_params.soc_init[batt_type]

                            for update_t in time_steps:
                                # Calculate energy change (in Wh)
                                energy_gained = greedy_sol.charge[batt_type, update_t] * model_params.eff_charge[batt_type]
                                energy_lost = greedy_sol.discharge[batt_type, update_t] / model_params.eff_discharge[batt_type]

                                # Update SoC percentage
                                soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[
                                    batt_type]) * 100
                                current_soc_pct += soc_change_pct
                                recalc_soc[update_t] = current_soc_pct

                            # Update SoC dictionary with new values
                            for update_t, soc_pct in recalc_soc.items():
                                greedy_sol.soc[batt_type, update_t] = soc_pct

                            # Update grid import/export after this change
                            greedy_sol._update_grid_values(
                                                     time_steps, model_params)

        # Calculate and print objective after third pass
        greedy_sol._calculate_and_print_objective(
            "After Third Pass (Excess SoC Utilization)", time_steps,
            model_params
        )


        # ----- THIRD PASS: Price optimization for the first battery -----


        # Assume the first battery in the list is capable of both charging and discharging
        # This pass is only applicable if we have at least one battery
        if model_params.battery_set:
            # Get the first battery (assuming it's the main battery that can both charge and discharge)
            main_battery = model_params.battery_set[0]

            # Create a list of timesteps with grid import costs
            if isinstance(model_params.price_import, list):
                time_price_pairs = [(t, model_params.price_import[t]) for t in time_steps]
            else:
                time_price_pairs = [(t, model_params.price_import) for t in time_steps]

            # Sort by price (highest first for potential discharge opportunities)
            high_price_times = sorted(time_price_pairs, key=lambda x: x[1], reverse=True)

            # Keep track of improvements
            improvement_found = True
            iteration = 0
            max_iterations = 10  # Limit the number of iterations to prevent infinite loops

            while improvement_found and iteration < max_iterations:
                improvement_found = False
                iteration += 1

                # For each high price time where we're importing from grid
                for high_t, high_price in high_price_times:
                    # Check if we're importing from grid
                    if greedy_sol.grid_import[high_t] <= 0:
                        continue  # No grid import at this time, no opportunity for improvement

                    # Calculate maximum discharge potential at this timestep
                    current_battery_discharge = greedy_sol.discharge[main_battery, high_t]
                    additional_discharge_power = min(
                        model_params.power_max[main_battery] - current_battery_discharge,  # Power limit
                        greedy_sol.grid_import[high_t]  # Only discharge up to the current grid import amount
                    )

                    if additional_discharge_power <= 0:
                        continue  # No additional discharge possible

                    # Find earlier timesteps with lower prices where we could charge
                    earlier_times = [(t, p) for t, p in time_price_pairs if t < high_t and p < high_price]
                    earlier_times.sort(key=lambda x: x[1])  # Sort by price (lowest first)

                    for low_t, low_price in earlier_times:
                        # Calculate how much energy would be needed for the discharge, accounting for efficiency
                        energy_needed = additional_discharge_power / model_params.eff_discharge[main_battery]

                        # Calculate charging power needed, accounting for efficiency
                        charging_power_needed = energy_needed / model_params.eff_charge[main_battery]

                        # Check if we have capacity to charge at this time
                        current_battery_charge = greedy_sol.charge[main_battery, low_t]
                        available_charge_capacity = model_params.power_max[main_battery] - current_battery_charge

                        if available_charge_capacity <= 0:
                            continue  # No capacity to charge more at this time

                        # Calculate actual charging power we can add
                        charge_power_to_add = min(charging_power_needed, available_charge_capacity)

                        if charge_power_to_add <= 0:
                            continue

                        # Calculate how much we can actually discharge with this amount of charge
                        discharge_power_possible = charge_power_to_add * model_params.eff_charge[main_battery] * \
                                                   model_params.eff_discharge[main_battery]

                        # Check if this arbitrage would be profitable
                        cost_to_charge = charge_power_to_add * low_price
                        savings_from_discharge = discharge_power_possible * high_price

                        if savings_from_discharge <= cost_to_charge:
                            continue  # Not profitable

                        # Check if SoC constraints would be violated
                        # Simulate SoC evolution
                        sim_soc = {}
                        sim_current_soc = model_params.soc_init[main_battery]

                        for t in time_steps:
                            sim_charge = greedy_sol.charge[main_battery, t]
                            sim_discharge = greedy_sol.discharge[main_battery, t]

                            # Add our potential charge/discharge
                            if t == low_t:
                                sim_charge += charge_power_to_add
                            elif t == high_t:
                                sim_discharge += discharge_power_possible

                            # Calculate energy change (in Wh)
                            energy_gained = sim_charge * model_params.eff_charge[main_battery]
                            energy_lost = sim_discharge / model_params.eff_discharge[main_battery]

                            # Update SoC percentage
                            soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[main_battery]) * 100
                            sim_current_soc += soc_change_pct
                            sim_soc[t] = sim_current_soc

                        # Check if SoC constraints are violated
                        if any(soc_val < model_params.soc_min[main_battery] or soc_val > model_params.soc_max[
                            main_battery]
                               for soc_val in sim_soc.values()):
                            continue  # SoC constraint would be violated

                        # If we get here, we can make an improvement
                        greedy_sol.charge[main_battery, low_t] += charge_power_to_add
                        greedy_sol.discharge[main_battery, high_t] += discharge_power_possible

                        # Update SoC for all timesteps
                        current_soc_pct = model_params.soc_init[main_battery]
                        for t in time_steps:
                            # Calculate energy change (in Wh)
                            energy_gained = greedy_sol.charge[main_battery, t] * model_params.eff_charge[main_battery]
                            energy_lost = greedy_sol.discharge[main_battery, t] / model_params.eff_discharge[main_battery]

                            # Update SoC percentage
                            soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[main_battery]) * 100
                            current_soc_pct += soc_change_pct

                            # Update SoC for this timestep
                            greedy_sol.soc[main_battery, t] = current_soc_pct

                        # Update grid values
                        greedy_sol._update_grid_values(
                                                 time_steps, model_params)

                        improvement_found = True
                        break  # Found a charging time for this discharge opportunity

                    if improvement_found:
                        break  # Found an improvement, restart the search with updated values

        # Update grid import/export after Fourth pass
        greedy_sol._update_grid_values(time_steps, model_params)

        # Calculate and print objective after third pass
        greedy_sol._calculate_and_print_objective(
            "After Fourth Pass (Price Arbitrage)", time_steps, model_params
        )

        # Final validation pass to ensure constraints are met
        last_timestep = time_steps[-1]
        for batt_type in model_params.battery_set:
            current_soc_pct = model_params.soc_init[batt_type]

            for t in time_steps:
                # Calculate SoC change from charge/discharge
                energy_gained = greedy_sol.charge[batt_type, t] * model_params.eff_charge[batt_type]
                energy_lost = greedy_sol.discharge[batt_type, t] / model_params.eff_discharge[batt_type]
                soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100

                # Check if next SoC would be valid
                next_soc_pct = current_soc_pct + soc_change_pct

                # Only enforce minimum SoC constraint at the last timestep
                if t == last_timestep and next_soc_pct < model_params.soc_min[batt_type]:
                    # Adjust charging/discharging to meet minimum SoC
                    if greedy_sol.discharge[batt_type, t] > 0:
                        # First try reducing discharge
                        discharge_reduction = min(
                            greedy_sol.discharge[batt_type, t],  # Cannot reduce more than current discharge
                            (model_params.soc_min[batt_type] - next_soc_pct) * model_params.capacity[batt_type] / 100 *
                            model_params.eff_discharge[batt_type]  # Energy needed to meet min SoC
                        )

                        greedy_sol.discharge[batt_type, t] -= discharge_reduction

                        # Recalculate next SoC
                        energy_gained = greedy_sol.charge[batt_type, t] * model_params.eff_charge[batt_type]
                        energy_lost = greedy_sol.discharge[batt_type, t] / model_params.eff_discharge[batt_type]
                        soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                    if next_soc_pct < model_params.soc_min[batt_type]:
                        # If still below min, increase charging
                        shortfall_pct = model_params.soc_min[batt_type] - next_soc_pct
                        shortfall_energy = (shortfall_pct * model_params.capacity[batt_type]) / 100
                        additional_charge = shortfall_energy / model_params.eff_charge[batt_type]

                        # Limit by maximum power
                        additional_charge = min(
                            additional_charge,
                            model_params.power_max[batt_type] - greedy_sol.charge[batt_type, t]  # Remaining charge capacity
                        )

                        greedy_sol.charge[batt_type, t] += additional_charge

                        # Final recalculation
                        energy_gained = greedy_sol.charge[batt_type, t] * model_params.eff_charge[batt_type]
                        energy_lost = greedy_sol.discharge[batt_type, t] / model_params.eff_discharge[batt_type]
                        soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                # Also check for exceeding maximum SoC
                if next_soc_pct > model_params.soc_max[batt_type]:
                    # First try reducing charging
                    if greedy_sol.charge[batt_type, t] > 0:
                        charge_reduction = min(
                            greedy_sol.charge[batt_type, t],  # Cannot reduce more than current charge
                            (next_soc_pct - model_params.soc_max[batt_type]) * model_params.capacity[batt_type] / 100 /
                            model_params.eff_charge[batt_type]  # Excess energy causing overfill
                        )

                        greedy_sol.charge[batt_type, t] -= charge_reduction

                        # Recalculate next SoC
                        energy_gained = greedy_sol.charge[batt_type, t] * model_params.eff_charge[batt_type]
                        energy_lost = greedy_sol.discharge[batt_type, t] / model_params.eff_discharge[batt_type]
                        soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                    # If still above max, increase discharging
                    if next_soc_pct > model_params.soc_max[batt_type]:
                        excess_pct = next_soc_pct - model_params.soc_max[batt_type]
                        excess_energy = (excess_pct * model_params.capacity[batt_type]) / 100
                        additional_discharge = excess_energy * model_params.eff_discharge[batt_type]

                        # Limit by maximum power
                        additional_discharge = min(
                            additional_discharge,
                            model_params.power_max[batt_type] - greedy_sol.discharge[batt_type, t]  # Remaining discharge capacity
                        )

                        greedy_sol.discharge[batt_type, t] += additional_discharge

                        # Final recalculation
                        energy_gained = greedy_sol.charge[batt_type, t] * model_params.eff_charge[batt_type]
                        energy_lost = greedy_sol.discharge[batt_type, t] / model_params.eff_discharge[batt_type]
                        soc_change_pct = ((energy_gained - energy_lost) / model_params.capacity[batt_type]) * 100
                        next_soc_pct = current_soc_pct + soc_change_pct

                # Update SOC for this timestep
                greedy_sol.soc[batt_type, t] = next_soc_pct
                current_soc_pct = next_soc_pct

        # Final update of grid import/export
        greedy_sol._update_grid_values( time_steps, model_params)

        # Calculate and print objective after final validation
        greedy_sol._calculate_and_print_objective(
            "After Final Validation",  time_steps, model_params
        )

        return greedy_sol