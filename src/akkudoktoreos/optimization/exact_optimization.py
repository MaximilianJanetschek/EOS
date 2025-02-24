
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
        for t in time_steps:
            if isinstance(grid_model.price_export, float):
                enforce_flow = grid_model.price_import[t] <= grid_model.price_export
            else:
                enforce_flow = grid_model.price_import[t] <= grid_model.price_export[t]

            if enforce_flow:
                flow_direction = model.addVar(name=f"flow_direction_{t}", vtype="B", lb=0, ub=1)
                max_bezug = sum(
                    grid_model.eff_charge[batt_type] * grid_model.power_max[batt_type] for batt_type in grid_model.battery_set
                ) + max(grid_model.total_load)
                max_einspeise = sum(
                    grid_model.eff_discharge[batt_type] * grid_model.power_max[batt_type] for batt_type in grid_model.battery_set
                ) + max(grid_model.pv_forecast)
                big_m = max(max_bezug, max_einspeise)
                model.addCons(
                    grid_export[t] <= big_m * flow_direction, name=f"export_constraint_{t}"
                )
                model.addCons(
                    grid_import[t] <= big_m * (1 - flow_direction), name=f"import_constraint_{t}"
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
            grid_export=grid_export
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

    def set_warm_start(
        self,
        model: Model,
        vars: ModelVariables,
        time_steps: range,
        model_params: ModelParameters,
    ):
        import time
        start = time.time()
        # Calculate warm start solution
        warm_charge, warm_discharge, warm_soc = self.generate_warm_start(
            time_steps=time_steps,
            model_params = model_params
        )
        print(f"Warm start calculation time: {time.time() - start}")
        print(warm_charge, warm_discharge, warm_soc)
        # Create a solution object
        solution = model.createSol()

        # Set variable values in the solution object
        for batt_type in model_params.battery_set:
            for t in time_steps:
                # Set battery-related variables
                model.setSolVal(solution, vars.charge[batt_type, t], warm_charge[batt_type, t])
                model.setSolVal(solution, vars.discharge[batt_type, t], warm_discharge[batt_type, t])
                model.setSolVal(solution, vars.soc[batt_type, t], warm_soc[batt_type, t])

                # Calculate and set grid import/export values
                net_power = model_params.pv_forecast[t] + sum(
                    warm_discharge[batt_type, t] - warm_charge[batt_type, t]
                    for batt_type in model_params.battery_set
                ) - model_params.total_load[t]

                if net_power < 0:
                    model.setSolVal(solution, vars.grid_import[t], -net_power)
                    model.setSolVal(solution, vars.grid_export[t], 0.0)
                else:
                    model.setSolVal(solution, vars.grid_import[t], 0.0)
                    model.setSolVal(solution, vars.grid_export[t], net_power)

        # Set flow direction binary variables if they exist
        for t in time_steps:
            if isinstance(model_params.price_export, float):
                enforce_flow = model_params.price_import[t] <= model_params.price_export
            else:
                enforce_flow = model_params.price_import[t] <= model_params.price_export[t]

            if enforce_flow:
                flow_var = model.getVarByName(f"flow_direction_{t}")
                if flow_var is not None:
                    flow_val = 1.0 if net_power > 0 else 0.0
                    model.setSolVal(solution, flow_var, flow_val)

        # Try to add the solution to the model
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
    ) -> tuple[dict, dict, dict]:
        """Generate warm start solution for the MILP optimization.

        Args:
            parameters: Input parameters containing load profiles, PV forecast, etc.
            time_steps: Range of optimization timesteps


        Returns:
            Tuple of dictionaries containing initial values for:
            - charging power
            - discharging power
            - state of charge
        """
        # Initialize solution dictionaries
        charge = {(b, t): 0.0 for b in model_params.battery_set for t in time_steps}
        discharge = {(b, t): 0.0 for b in model_params.battery_set for t in time_steps}
        soc = {(b, t): 0.0 for b in model_params.battery_set for t in time_steps}


        # First pass: Direct PV usage and simple battery charging
        for batt_type in model_params.battery_set:
            current_soc = model_params.soc_init[batt_type] * model_params.capacity[batt_type] / 100

            for t in time_steps:
                # Calculate net power (positive means excess PV)
                net_power = model_params.pv_forecast[t] - model_params.total_load[t]
                soc[batt_type, t] = current_soc * 100 / model_params.capacity[batt_type]

                if net_power > 0:  # Excess PV available
                    # Calculate maximum charging power considering SOC limit
                    max_charge = min(
                        model_params.power_max[batt_type],
                        net_power,
                        (model_params.soc_max[batt_type] * model_params.capacity[batt_type] / 100 - current_soc) / model_params.eff_charge[batt_type]
                    )

                    if max_charge > 0:
                        charge[batt_type, t] = max_charge
                        current_soc += max_charge * model_params.eff_charge[batt_type]

        # Second pass: Ensure minimum SOC
        for batt_type in model_params.battery_set:
            for t in reversed(time_steps):
                if soc[batt_type, t] < model_params.soc_min[batt_type]:
                    required_charge = (model_params.soc_min[batt_type] - soc[batt_type, t]) * model_params.capacity[batt_type] / 100
                    charge_power = min(model_params.power_max[batt_type], required_charge / model_params.eff_charge[batt_type])

                    # Update charging for this timestep
                    charge[batt_type, t] = charge_power

                    # Update SOC backwards
                    for prev_t in range(t, -1, -1):
                        if prev_t == 0:
                            soc[batt_type, prev_t] = model_params.soc_init[batt_type]
                        else:
                            soc[batt_type, prev_t] = (
                                                             soc[batt_type, prev_t - 1] * model_params.capacity[batt_type] / 100 +
                                                             charge[batt_type, prev_t] * model_params.eff_charge[batt_type] -
                                                             discharge[batt_type, prev_t] / model_params.eff_discharge[batt_type]
                                                     ) * 100 / model_params.capacity[batt_type]

        # Third pass: Optimize for price differences
        for batt_type in model_params.battery_set:
            for t in time_steps:
                # Check if we're importing from grid
                net_power = model_params.pv_forecast[t] - model_params.total_load[t]
                if net_power < 0:  # Grid import needed
                    # Find future timestep with higher price where we could discharge
                    for future_t in range(t + 1, len(time_steps)):
                        if model_params.price_import[future_t] > model_params.price_import[t]:
                            # Calculate available discharge capacity
                            current_soc = soc[batt_type, t] * model_params.capacity[batt_type] / 100
                            available_discharge = min(
                                model_params.power_max[batt_type],
                                (current_soc - model_params.soc_min[batt_type] * model_params.capacity[batt_type] / 100) * model_params.eff_discharge[
                                    batt_type]
                            )

                            if available_discharge > 0:
                                discharge[batt_type, future_t] = available_discharge
                                # Update SOC for future timesteps
                                for update_t in range(t, len(time_steps)):
                                    if update_t == 0:
                                        soc[batt_type, update_t] = model_params.soc_init[batt_type]
                                    else:
                                        soc[batt_type, update_t] = (
                                                                           soc[batt_type, update_t - 1] * model_params.capacity[
                                                                       batt_type] / 100 +
                                                                           charge[batt_type, update_t] * model_params.eff_charge[
                                                                               batt_type] -
                                                                           discharge[batt_type, update_t] /
                                                                           model_params.eff_discharge[batt_type]
                                                                   ) * 100 / model_params.capacity[batt_type]

                                break

        return charge, discharge, soc
